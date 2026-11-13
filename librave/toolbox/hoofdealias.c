/* --------------------------------------------------------------------
Copyright (C) 2026 Swedish Meteorological and Hydrological Institute, SMHI,

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/
/**
 * HOOF dealiasing of radial wind data (torus mapping).
 *
 * For each bin the measurement vr is mapped to the torus quantities
 *   A = cos(el) cos(az) sin(pi vr/NI), B = cos(el) sin(az) sin(pi vr/NI),
 *   D = d/daz (NI cos(pi vr/NI) / pi),
 * which are independent of aliasing. For each height layer of the whole volume a wind model
 * (u, v) is fitted with least squares from D = -A u + B v. The measurement is then unfolded
 * with the Nyquist multiplier that brings it closest to the wind model
 *   vm = cos(el) (u sin(az) + v cos(az)).
 * Bins in height layers without a wind model can not be dealiased and are set to nodata.
 * @file
 * @author Peter Smerkol, Vito Švagelj, ARSO
 * @date 2026-11-13
 */
#include "hoofdealias.h"
#include <math.h>
#include <string.h>
#include "rave_attribute.h"
#include "rave_field.h"
#include "rave_object.h"
#include "rave_alloc.h"
#include "rave_types.h"
#include "rave_debug.h"

#define HOOFDEALIAS_EARTH_RADIUS  6371200.0   /* Earth radius [m] */
#define HOOFDEALIAS_KE            (4.0/3.0)   /* Equivalent Earth factor */

/**
 * Data of one scan used in dealiasing. Arrays have nrays*nbins elements, index ray*nbins+bin,
 * unless noted otherwise.
 */
typedef struct HoofDealiasScan {
  PolarScan_t* scan;            /**< the scan */
  PolarScanParam_t* param;      /**< the parameter to dealias */
  long nrays;                   /**< number of rays */
  long nbins;                   /**< number of bins */
  double ni;                    /**< Nyquist velocity [m/s] */
  double cosel;                 /**< cosine of the elevation angle */
  double* cosaz;                /**< cosines of the ray azimuths (nrays) */
  double* sinaz;                /**< sines of the ray azimuths (nrays) */
  double* vr;                   /**< measurements [m/s], NaN = nodata or undetect */
  unsigned char* undetect;      /**< 1 where the measurement is undetect */
  double* z;                    /**< heights above sea level of the range bins (nbins) [m], same for all rays */
  int* layer;                   /**< height layers of the range bins (nbins), -1 or nlayers if outside the layers */
  double* d;                    /**< D quantity of the torus mapping, NaN if it can not be calculated */
  double* dv;                   /**< dealiased measurements [m/s], NaN = not dealiased */
} HoofDealiasScan;

/*@{ Private functions */

/**
 * Releases the memory of a scan structure.
 */
static void HoofDealiasInternal_releaseScan(HoofDealiasScan* hs)
{
  RAVE_FREE(hs->cosaz);
  RAVE_FREE(hs->sinaz);
  RAVE_FREE(hs->vr);
  RAVE_FREE(hs->undetect);
  RAVE_FREE(hs->z);
  RAVE_FREE(hs->layer);
  RAVE_FREE(hs->d);
  RAVE_FREE(hs->dv);
  RAVE_OBJECT_RELEASE(hs->param);
  RAVE_OBJECT_RELEASE(hs->scan);
}

/**
 * Returns the maximum raw value of an unsigned integer data type, or 0 for other types.
 */
static double HoofDealiasInternal_maxRaw(RaveDataType type)
{
  if (type == RaveDataType_UCHAR) {
    return 255.0;
  } else if (type == RaveDataType_USHORT) {
    return 65535.0;
  }
  return 0.0;
}

/**
 * Reads a scan into the scan structure. Converts VRAD normalized to the Nyquist velocity
 * (encoding range only covering values between -1 and 1) to m/s, and calculates the bin heights
 * and the D quantity.
 * @returns 1 if the scan can be dealiased, 0 if it is skipped or on memory errors
 */
static int HoofDealiasInternal_readScan(HoofDealiasScan* hs, PolarScan_t* scan, const char* quantity, double height)
{
  RaveAttribute_t* attr = NULL;
  RaveDataType type;
  long ir, ib, nrays, nbins;
  double gain, offset, nodata, undetect, val, maxRaw, rstart, rscale, step;
  double KR, KRsq, sinEl, r;
  double* f3 = NULL;
  int result = 0;

  memset(hs, 0, sizeof(HoofDealiasScan));

  if (!PolarScan_hasParameter(scan, quantity) || hoofdealiased_by_quantity(scan, quantity)) {
    goto done;
  }
  attr = PolarScan_getAttribute(scan, "how/NI");
  if (attr == NULL || !RaveAttribute_getDouble(attr, &hs->ni) || hs->ni <= 0.0) {
    RAVE_WARNING1("Scan has no valid how/NI, skipping it in dealiasing of %s", quantity);
    goto done;
  }
  hs->param = PolarScan_getParameter(scan, quantity);
  type = PolarScanParam_getDataType(hs->param);
  maxRaw = HoofDealiasInternal_maxRaw(type);
  if (maxRaw == 0.0 && type != RaveDataType_FLOAT && type != RaveDataType_DOUBLE) {
    RAVE_WARNING1("Unsupported data type of %s, skipping the scan in dealiasing", quantity);
    goto done;
  }

  hs->scan = RAVE_OBJECT_COPY(scan);
  nrays = hs->nrays = PolarScan_getNrays(scan);
  nbins = hs->nbins = PolarScan_getNbins(scan);
  hs->cosel = cos(PolarScan_getElangle(scan));
  hs->cosaz = RAVE_MALLOC(sizeof(double) * nrays);
  hs->sinaz = RAVE_MALLOC(sizeof(double) * nrays);
  hs->vr = RAVE_MALLOC(sizeof(double) * nrays * nbins);
  hs->undetect = RAVE_CALLOC((size_t)(nrays * nbins), sizeof(unsigned char));
  hs->z = RAVE_MALLOC(sizeof(double) * nbins);
  hs->layer = RAVE_MALLOC(sizeof(int) * nbins);
  hs->d = RAVE_MALLOC(sizeof(double) * nrays * nbins);
  hs->dv = RAVE_MALLOC(sizeof(double) * nrays * nbins);
  f3 = RAVE_MALLOC(sizeof(double) * nrays * nbins);
  if (hs->cosaz == NULL || hs->sinaz == NULL || hs->vr == NULL || hs->undetect == NULL ||
      hs->z == NULL || hs->layer == NULL || hs->d == NULL || hs->dv == NULL || f3 == NULL) {
    RAVE_ERROR0("Memory allocation error");
    goto done;
  }

  /* measurements, comparing raw values to nodata and undetect so that the comparison is exact */
  gain = PolarScanParam_getGain(hs->param);
  offset = PolarScanParam_getOffset(hs->param);
  nodata = PolarScanParam_getNodata(hs->param);
  undetect = PolarScanParam_getUndetect(hs->param);
  for (ir = 0; ir < nrays; ir++) {
    for (ib = 0; ib < nbins; ib++) {
      PolarScanParam_getValue(hs->param, ib, ir, &val);
      if (val == nodata) {
        hs->vr[ir*nbins+ib] = NAN;
      } else if (val == undetect) {
        hs->vr[ir*nbins+ib] = NAN;
        hs->undetect[ir*nbins+ib] = 1;
      } else {
        hs->vr[ir*nbins+ib] = gain * val + offset;
      }
    }
  }

  /* VRAD normalized to the Nyquist velocity is converted to m/s */
  if (maxRaw > 0.0 && fmax(fabs(offset), fabs(offset + gain*maxRaw)) <= 1.0 + 2.0*fabs(gain)) {
    RAVE_INFO1("%s is normalized to the Nyquist velocity, converting to m/s", quantity);
    for (ir = 0; ir < nrays*nbins; ir++) {
      hs->vr[ir] *= hs->ni;
    }
  }

  /* azimuths of the ray centres, first ray at north */
  step = 2.0 * M_PI / (double)nrays;
  for (ir = 0; ir < nrays; ir++) {
    hs->cosaz[ir] = cos((double)ir * step);
    hs->sinaz[ir] = sin((double)ir * step);
  }

  /* heights from the equivalent Earth model. */
  rstart = PolarScan_getRstart(scan) * 1000.0;
  rscale = PolarScan_getRscale(scan);
  KR = HOOFDEALIAS_KE * HOOFDEALIAS_EARTH_RADIUS;
  KRsq = HOOFDEALIAS_EARTH_RADIUS * HOOFDEALIAS_EARTH_RADIUS * HOOFDEALIAS_KE * HOOFDEALIAS_KE;
  sinEl = sin(PolarScan_getElangle(scan));
  for (ib = 0; ib < nbins; ib++) {
    r = rstart + (double)ib * rscale;
    hs->z[ib] = (sqrt(r*r + KRsq + 2.0*r*HOOFDEALIAS_EARTH_RADIUS*HOOFDEALIAS_KE*sinEl) - KR) + height;
  }

  /* D quantity, derivative of f3 = NI cos(pi vr/NI)/pi in azimuth, wrapping around north */
  for (ir = 0; ir < nrays*nbins; ir++) {
    f3[ir] = hs->ni * cos(M_PI * hs->vr[ir] / hs->ni) / M_PI;
  }
  for (ir = 0; ir < nrays; ir++) {
    long next = (ir + 1 == nrays) ? 0 : ir + 1;
    long prev = (ir == 0) ? nrays - 1 : ir - 1;
    double daz = (double)next * step - (double)prev * step;
    if (ir == 0 || ir == nrays - 1) {
      daz = daz + 2.0 * M_PI;
    }
    for (ib = 0; ib < nbins; ib++) {
      hs->d[ir*nbins+ib] = (f3[next*nbins+ib] - f3[prev*nbins+ib]) / daz;
    }
  }

  for (ir = 0; ir < nrays*nbins; ir++) {
    hs->dv[ir] = NAN;
  }
  result = 1;
done:
  RAVE_FREE(f3);
  RAVE_OBJECT_RELEASE(attr);
  if (!result) {
    HoofDealiasInternal_releaseScan(hs);
  }
  return result;
}

/**
 * Returns the height layer of a height, the largest k with edges[k] <= z. Returns -1 for heights
 * below the lowest edge and nlayers for heights at or above the highest edge (outside the layers).
 * @param[in] edges - the layer edges, nlayers+1 increasing values
 */
static int HoofDealiasInternal_layer(const double* edges, int nlayers, double z)
{
  int lo = 0, hi = nlayers + 1;   
  if (!(z >= edges[0])) {
    return (z < edges[0]) ? -1 : nlayers;   
  }
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    if (edges[mid] <= z) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/**
 * Removes quality fields with the HOOF dealiasing task from a parameter.
 */
static void HoofDealiasInternal_removeQualityField(PolarScanParam_t* param)
{
  int i;
  for (i = PolarScanParam_getNumberOfQualityFields(param) - 1; i >= 0; i--) {
    RaveField_t* field = PolarScanParam_getQualityField(param, i);
    RaveAttribute_t* attr = (field != NULL) ? RaveField_getAttribute(field, "how/task") : NULL;
    char* task = NULL;
    if (attr != NULL && RaveAttribute_getString(attr, &task) && task != NULL && strcmp(task, HOOFDEALIAS_TASK) == 0) {
      PolarScanParam_removeQualityField(param, i);
    }
    RAVE_OBJECT_RELEASE(attr);
    RAVE_OBJECT_RELEASE(field);
  }
}

/**
 * Writes the dealiased measurements back to the parameter. For 8 and 16 bit unsigned data the
 * gain and offset are recalculated so that the data minimum maps to raw 1 and the maximum to
 * raw maxRaw-1, raw 0 is undetect and raw maxRaw is nodata. Float data is written directly.
 * Adds how/dealiased and the quality field (255 = dealiased, 0 = not dealiased).
 * @returns 1 on success, otherwise 0
 */
static int HoofDealiasInternal_writeScan(HoofDealiasScan* hs)
{
  RaveField_t* field = NULL;
  RaveAttribute_t* attr = NULL;
  long i, ir, ib, n = hs->nrays * hs->nbins;
  double maxRaw = HoofDealiasInternal_maxRaw(PolarScanParam_getDataType(hs->param));
  double nodata, undetect, gain = 1.0, offset = 0.0, vmin = INFINITY, vmax = -INFINITY, raw;
  int result = 0;

  if (maxRaw > 0.0) {
    nodata = maxRaw;
    undetect = 0.0;
    for (i = 0; i < n; i++) {
      if (!isnan(hs->dv[i])) {
        vmin = fmin(vmin, hs->dv[i]);
        vmax = fmax(vmax, hs->dv[i]);
      }
    }
    if (vmin <= vmax) {
      gain = (vmax - vmin) / (maxRaw - 2.0);
      if (fabs(gain) < 1e-12) {
        gain = 1.0;
      }
      offset = vmin - gain;
    }
  } else {
    nodata = PolarScanParam_getNodata(hs->param);
    undetect = PolarScanParam_getUndetect(hs->param);
  }
  PolarScanParam_setGain(hs->param, gain);
  PolarScanParam_setOffset(hs->param, offset);
  PolarScanParam_setNodata(hs->param, nodata);
  PolarScanParam_setUndetect(hs->param, undetect);

  /* data and quality field */
  field = RAVE_OBJECT_NEW(&RaveField_TYPE);
  if (field == NULL || !RaveField_createData(field, hs->nbins, hs->nrays, RaveDataType_UCHAR)) {
    RAVE_ERROR0("Failed to create quality field");
    goto done;
  }
  for (ir = 0; ir < hs->nrays; ir++) {
    for (ib = 0; ib < hs->nbins; ib++) {
      double v = hs->dv[ir*hs->nbins+ib];
      if (!isnan(v)) {
        raw = (maxRaw > 0.0) ? fmin(fmax(round((v - offset) / gain), 1.0), maxRaw - 1.0) : v;
        RaveField_setValue(field, ib, ir, 255.0);
      } else if (hs->undetect[ir*hs->nbins+ib]) {
        raw = undetect;
      } else {
        raw = nodata;
      }
      PolarScanParam_setValue(hs->param, ib, ir, raw);
    }
  }

  attr = RaveAttributeHelp_createString("how/task", HOOFDEALIAS_TASK);
  if (attr == NULL || !RaveField_addAttribute(field, attr)) goto done;
  RAVE_OBJECT_RELEASE(attr);
  attr = RaveAttributeHelp_createDouble("what/gain", 1.0/255.0);
  if (attr == NULL || !RaveField_addAttribute(field, attr)) goto done;
  RAVE_OBJECT_RELEASE(attr);
  attr = RaveAttributeHelp_createDouble("what/offset", 0.0);
  if (attr == NULL || !RaveField_addAttribute(field, attr)) goto done;
  RAVE_OBJECT_RELEASE(attr);
  HoofDealiasInternal_removeQualityField(hs->param);
  if (!PolarScanParam_addQualityField(hs->param, field)) goto done;

  /* parameter attributes */
  attr = RaveAttributeHelp_createString("how/dealiased", "True");
  if (attr == NULL || !PolarScanParam_addAttribute(hs->param, attr)) goto done;
  RAVE_OBJECT_RELEASE(attr);
  attr = RaveAttributeHelp_createString("how/task", HOOFDEALIAS_TASK);
  if (attr == NULL || !PolarScanParam_addAttribute(hs->param, attr)) goto done;

  result = 1;
done:
  if (!result) {
    RAVE_ERROR0("Failed to write dealiased data");
  }
  RAVE_OBJECT_RELEASE(attr);
  RAVE_OBJECT_RELEASE(field);
  return result;
}

/*@} End of Private functions */

/*@{ Interface functions */

int hoofdealiased_by_quantity(PolarScan_t* scan, const char* quantity)
{
  PolarScanParam_t* param = NULL;
  RaveAttribute_t* attr = NULL;
  char* da = NULL;
  int result = 0;

  if (PolarScan_hasParameter(scan, quantity)) {
    param = PolarScan_getParameter(scan, quantity);
    attr = PolarScanParam_getAttribute(param, "how/dealiased");
    if (attr != NULL && RaveAttribute_getString(attr, &da) && da != NULL && strncmp(da, "True", 4) == 0) {
      result = 1;
    }
  }
  RAVE_OBJECT_RELEASE(attr);
  RAVE_OBJECT_RELEASE(param);
  return result;
}

int hoofdealias_pvol_by_quantity(PolarVolume_t* pvol, const char* quantity, double zsector,
                                 double zmax, int minpoints, double maxwind)
{
  HoofDealiasScan* scans = NULL;
  double *saa = NULL, *sbb = NULL, *sab = NULL, *sad = NULL, *sbd = NULL, *us = NULL, *vs = NULL;
  double* edges = NULL;
  long* counts = NULL;
  int nscans, ns = 0, is, nl, l, result = 0;
  long ir, ib, i;
  double height, zdatamax = -INFINITY, ztop, vnymin = INFINITY, nymax;

  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  RAVE_ASSERT((quantity != NULL), "quantity == NULL");
  if (zsector <= 0.0 || maxwind <= 0.0) {
    RAVE_ERROR0("Height layer thickness and maximum wind must be positive");
    return 0;
  }

  /* read all scans that can be dealiased */
  height = PolarVolume_getHeight(pvol);
  nscans = PolarVolume_getNumberOfScans(pvol);
  scans = RAVE_CALLOC((size_t)(nscans > 0 ? nscans : 1), sizeof(HoofDealiasScan));
  if (scans == NULL) {
    RAVE_ERROR0("Memory allocation error");
    goto done;
  }
  for (is = 0; is < nscans; is++) {
    PolarScan_t* scan = PolarVolume_getScan(pvol, is);
    if (scan != NULL && HoofDealiasInternal_readScan(&scans[ns], scan, quantity, height)) {
      vnymin = fmin(vnymin, scans[ns].ni);
      for (ib = 0; ib < scans[ns].nbins; ib++) {
        zdatamax = fmax(zdatamax, scans[ns].z[ib]);
      }
      ns++;
    }
    RAVE_OBJECT_RELEASE(scan);
  }
  if (ns == 0) {
    RAVE_INFO1("No scans with %s to dealias", quantity);
    goto done;
  }

  /* height layers of thickness zsector, from the radar height up to the first layer edge at or
     above the highest bin (or zmax, if lower) */
  ztop = fmin(zdatamax, zmax);
  nl = 0;
  edges = RAVE_MALLOC(sizeof(double) * 2);
  if (edges == NULL) {
    RAVE_ERROR0("Memory allocation error");
    goto done;
  }
  edges[0] = height;
  do {
    double* tmp = RAVE_REALLOC(edges, sizeof(double) * (nl + 2));
    if (tmp == NULL) {
      RAVE_ERROR0("Memory allocation error");
      goto done;
    }
    edges = tmp;
    edges[nl + 1] = edges[nl] + zsector;
    nl++;
  } while (edges[nl] < ztop);

  /* height layer of each range bin, bins below the radar or above the highest edge are outside */
  for (is = 0; is < ns; is++) {
    for (ib = 0; ib < scans[is].nbins; ib++) {
      scans[is].layer[ib] = HoofDealiasInternal_layer(edges, nl, scans[is].z[ib]);
    }
  }

  counts = RAVE_CALLOC((size_t)nl, sizeof(long));
  saa = RAVE_CALLOC((size_t)nl, sizeof(double));
  sbb = RAVE_CALLOC((size_t)nl, sizeof(double));
  sab = RAVE_CALLOC((size_t)nl, sizeof(double));
  sad = RAVE_CALLOC((size_t)nl, sizeof(double));
  sbd = RAVE_CALLOC((size_t)nl, sizeof(double));
  us = RAVE_MALLOC(sizeof(double) * nl);
  vs = RAVE_MALLOC(sizeof(double) * nl);
  if (counts == NULL || saa == NULL || sbb == NULL || sab == NULL || sad == NULL || sbd == NULL || us == NULL || vs == NULL) {
    RAVE_ERROR0("Memory allocation error");
    goto done;
  }

  /* sums for the least squares fit of D = -A u + B v in each height layer */
  for (is = 0; is < ns; is++) {
    HoofDealiasScan* hs = &scans[is];
    for (ir = 0; ir < hs->nrays; ir++) {
      for (ib = 0; ib < hs->nbins; ib++) {
        i = ir*hs->nbins + ib;
        l = hs->layer[ib];
        if (isnan(hs->vr[i]) || isnan(hs->d[i]) || l < 0 || l >= nl) {
          continue;
        }
        double s = sin(M_PI * hs->vr[i] / hs->ni);
        double a = hs->cosel * hs->cosaz[ir] * s;
        double b = hs->cosel * hs->sinaz[ir] * s;
        counts[l]++;
        saa[l] += a*a;
        sbb[l] += b*b;
        sab[l] += a*b;
        sad[l] += a*hs->d[i];
        sbd[l] += b*hs->d[i];
      }
    }
  }

  /* wind models determined by the closed-form least-squares solution,
    accepted if there are more than minpoints points and the wind speed is below the maximum */
  for (l = 0; l < nl; l++) {
    double det = saa[l]*sbb[l] - sab[l]*sab[l];
    us[l] = vs[l] = NAN;
    if (counts[l] > minpoints && det > 1e-12 * saa[l] * sbb[l]) {
      double u = (sab[l]*sbd[l] - sbb[l]*sad[l]) / det;
      double v = (saa[l]*sbd[l] - sab[l]*sad[l]) / det;
      if (hypot(u, v) < maxwind) {
        us[l] = u;
        vs[l] = v;
      }
    }
  }

  /* unfold the bins that have a wind model with the Nyquist multiplier in [-nymax, nymax] closest
     to the model: the closest multiplier is calculated directly, then it and its neighbours are
     compared */
  nymax = floor(maxwind / vnymin);
  for (is = 0; is < ns; is++) {
    HoofDealiasScan* hs = &scans[is];
    for (ir = 0; ir < hs->nrays; ir++) {
      for (ib = 0; ib < hs->nbins; ib++) {
        i = ir*hs->nbins + ib;
        l = hs->layer[ib];
        if (isnan(hs->vr[i]) || l < 0 || l >= nl || isnan(us[l])) {
          continue;
        }
        double wm = hs->cosel * (us[l]*hs->sinaz[ir] + vs[l]*hs->cosaz[ir]);
        double nearest = fmin(fmax(nearbyint((wm - hs->vr[i]) / (2.0*hs->ni)), -nymax), nymax);
        double nbest = NAN, mnbest = 1000000.0;
        for (int dn = -1; dn <= 1; dn++) {
          double n = fmin(fmax(nearest + dn, -nymax), nymax) + 0.0;
          double mn = fabs(hs->vr[i] + 2.0*hs->ni*n - wm);
          if (mn < mnbest) {
            mnbest = mn;
            nbest = n;
          }
        }
        if (!isnan(nbest)) {
          hs->dv[i] = hs->vr[i] + 2.0*nbest*hs->ni;
        }
      }
    }
  }

  /* write the dealiased data */
  for (is = 0; is < ns; is++) {
    if (!HoofDealiasInternal_writeScan(&scans[is])) {
      goto done;
    }
  }
  result = 1;
done:
  if (scans != NULL) {
    for (is = 0; is < ns; is++) {
      HoofDealiasInternal_releaseScan(&scans[is]);
    }
  }
  RAVE_FREE(scans);
  RAVE_FREE(edges);
  RAVE_FREE(counts);
  RAVE_FREE(saa);
  RAVE_FREE(sbb);
  RAVE_FREE(sab);
  RAVE_FREE(sad);
  RAVE_FREE(sbd);
  RAVE_FREE(us);
  RAVE_FREE(vs);
  return result;
}

int hoofdealias_pvol(PolarVolume_t* pvol)
{
  return hoofdealias_pvol_by_quantity(pvol, "VRADH", HOOFDEALIAS_ZSECTOR, HOOFDEALIAS_ZMAX,
                                      HOOFDEALIAS_MINPOINTS, HOOFDEALIAS_MAXWIND);
}

/*@} End of Interface functions */
