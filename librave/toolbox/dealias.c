/* --------------------------------------------------------------------
Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI

This is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with HLHDF.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/

/** Function for dealiasing weather radar winds.
 * @file
 * @author Gunther Haase, SMHI
 * @date 2013-02-06
 */

#include "dealias.h"

double max_vector (double *a, int n) {
	int i;
	double max = -32000;
	for (i=0; i<n; i++) {
		if (*(a+i) > max) max = *(a+i);
	}
	return max;
}


double min_vector (double *a, int n) {
	int i;
	double min = 32000;
	for (i=0; i<n; i++) {
		if (*(a+i) < min) min = *(a+i);
	}
	return min;
}


int dealiased(PolarScan_t* scan) {
  PolarScanParam_t* vrad = NULL;
  RaveAttribute_t* attr = NULL;
  int ret = 0;
  int retda = 0;
  char* da;

  if (PolarScan_hasParameter(scan, "VRAD")) {
    vrad = PolarScan_getParameter(scan, "VRAD");
    attr = PolarScanParam_getAttribute(vrad, "how/dealiased");
    if (attr != NULL) {
      retda = RaveAttribute_getString(attr, &da);
      if (retda) {
        if (!strncmp(da, "True", (size_t)4)) {
          ret = 1;
        }
      }
    }
  }
  RAVE_OBJECT_RELEASE(attr);
  RAVE_OBJECT_RELEASE(vrad);
  return ret;
}


int dealias_pvol(PolarVolume_t* inobj) {
  PolarScan_t* scan = NULL;
  int is, nscans;
  int retval = 0;  /* Using this feels a bit artificial */

  nscans = PolarVolume_getNumberOfScans(inobj);

  for (is=0; is<nscans; is++) {
    scan = PolarVolume_getScan(inobj, is);
    retval = dealias_scan(scan);
    RAVE_OBJECT_RELEASE(scan);
  }

  return retval;
}


int dealias_scan(PolarScan_t* scan) {
  PolarScanParam_t* vrad = NULL;
  RaveAttribute_t* attr = NULL;
  RaveAttribute_t* dattr = RAVE_OBJECT_NEW(&RaveAttribute_TYPE);;
  int nbins, nrays, i, j, n, m, ib, ir, eind;
  int retval = 0;
  double elangle, gain, offset, nodata, undetect, NI, val, vm, min1, esum, u1, v1, min2, dmy, vmin, vmax;

  nbins = PolarScan_getNbins(scan);
  nrays = PolarScan_getNrays(scan);
  elangle = PolarScan_getElangle(scan);

  if ( (PolarScan_hasParameter(scan, "VRAD")) && (!dealiased(scan)) ) {
    if (elangle*RAD2DEG<=EMAX) {
      vrad = PolarScan_getParameter(scan, "VRAD");
      gain = PolarScanParam_getGain(vrad);
      offset = PolarScanParam_getOffset(vrad);
      nodata = PolarScanParam_getNodata(vrad);
      undetect = PolarScanParam_getUndetect(vrad);
      attr = PolarScan_getAttribute(scan, "how/NI");  /* only location? */
      if (attr != NULL) {
        RaveAttribute_getDouble(attr, &NI);
      } else {
        NI = abs(offset);
      }
      // number of rows
      m = floor (VAF/NI*VMAX);
      // number of columns
      n = NF;

      int *vrad_nodata = RAVE_CALLOC ((size_t)nrays*nbins, sizeof(int));
      int *vrad_undetect = RAVE_CALLOC ((size_t)nrays*nbins, sizeof(int));
      double *x = RAVE_CALLOC ((size_t)nrays*nbins, sizeof(double));
      double *y = RAVE_CALLOC ((size_t)nrays*nbins, sizeof(double));
      double *vo = RAVE_CALLOC ((size_t)nrays*nbins, sizeof(double));
      double *vd = RAVE_CALLOC ((size_t)nrays*nbins, sizeof(double));
      double *uh = RAVE_CALLOC ((size_t)(m*n), sizeof(double));
      double *vh = RAVE_CALLOC ((size_t)(m*n), sizeof(double));
      double *e = RAVE_CALLOC ((size_t)(m*n*nrays), sizeof(double));
      double *xt = RAVE_CALLOC ((size_t)(m*n*nrays), sizeof(double));
      double *yt = RAVE_CALLOC ((size_t)(m*n*nrays), sizeof(double));
      double *vt1 = RAVE_CALLOC ((size_t)nrays, sizeof(double));
      double *dv = RAVE_CALLOC ((size_t)(MVA+1), sizeof(double));
      double *v = RAVE_CALLOC ((size_t)(MVA+1)*nrays, sizeof(double));

      // read and re-arrange data (ray -> bin)
      for (ir=0; ir<nrays; ir++) {
        for (ib=0; ib<nbins; ib++) {
          PolarScanParam_getValue(vrad, ib, ir, &val);
          if (val==nodata) *(vrad_nodata+ir+ib*nrays) = 1;
          if (val==undetect) *(vrad_undetect+ir+ib*nrays) = 1;
          if ((val!=nodata) && (val!=undetect)) *(vo+ir+ib*nrays) = offset+gain*val;
          else *(vo+ir+ib*nrays) = NAN;

          // map measured data to 3D
          *(x+ir+ib*nrays) = NI/M_PI * cos(*(vo+ir+ib*nrays)*M_PI/NI);
          *(y+ir+ib*nrays) = NI/M_PI * sin(*(vo+ir+ib*nrays)*M_PI/NI);
        }
      }

      for (i=0; i<n; i++) {
        for (j=0; j<m; j++) {
          *(uh+i*m+j) = NI/VAF*(j+1) * sin(2*M_PI/NF*i);
          *(vh+i*m+j) = NI/VAF*(j+1) * cos(2*M_PI/NF*i);
        }
      }

      for (ir=0; ir<nrays; ir++) {
        for (i=0; i<n; i++) {
          for (j=0; j<m; j++) {
            vm = *(uh+i*m+j) * sin(360./nrays*ir*DEG2RAD) +
                 *(vh+i*m+j) * cos(360./nrays*ir*DEG2RAD);
            *(xt+i*m+j+ir*m*n) = NI/M_PI * cos(vm*M_PI/NI);
            *(yt+i*m+j+ir*m*n) = NI/M_PI * sin(vm*M_PI/NI);
          }
        }
      }

      for (ib=0; ib<nbins; ib++) {
        for (ir=0; ir<nrays; ir++) {
          for (i=0; i<m*n; i++) {
            *(e+i+ir*m*n) = fabs(*(xt+i+ir*m*n)-*(x+ir+ib*nrays)) +
                            fabs(*(yt+i+ir*m*n)-*(y+ir+ib*nrays));
          }
        }

        min1 = 1e32;
        eind = 0;
        u1 = 0;
        v1 = 0;
        for (i=0; i<m*n; i++) {
          esum = 0;
          for (ir=0; ir<nrays; ir++) {
            if (!isnan(*(e+i+ir*m*n))) {
              esum = esum + *(e+i+ir*m*n);
            }
          }
          if (esum<min1) {
            min1 = esum;
            eind = i;
          }
          u1 = *(uh+eind);
          v1 = *(vh+eind);
        }

        for (ir=0; ir<nrays; ir++) {
          *(vt1+ir) = u1*sin(360./nrays*ir*DEG2RAD) + v1*cos(360./nrays*ir*DEG2RAD);
        }

        for (i=0; i<MVA+1; i++) {
          *(dv+i) = NI*(2*i-MVA);
        }

        for (ir=0; ir<nrays; ir++) {
          for (i=0; i<MVA+1; i++) {
            *(v+i+ir*(MVA+1)) = *(dv+i);
          }
        }

        for (ir=0; ir<nrays; ir++) {
          min2 = 1e32;
          dmy = 0;
          for (i=0; i<MVA+1; i++) {
            dmy = fabs(*(v+i+ir*(MVA+1))-(*(vt1+ir)-*(vo+ir+ib*nrays)));
            if ((dmy<min2) && (!isnan(dmy))) {
              *(vd+ir+ib*nrays) = *(vo+ir+ib*nrays) + *(dv+i);
              min2 = dmy;
            }
          }
        }
      }

      //offset = 2*offset;
      //gain = 2*gain;
      vmax = max_vector(vd, nrays*nbins);
      vmin = min_vector(vd, nrays*nbins);
      if (vmin<offset+gain || vmax>=offset+255*gain) {
          gain = (vmax-vmin)/253;
          offset = vmin-gain-EPSILON;
      }
      PolarScanParam_setOffset (vrad, offset);
      PolarScanParam_setGain (vrad, gain);

      for (ir=0 ; ir<nrays ; ir++) {
        for (ib=0; ib<nbins; ib++) {
          *(vd+ir+ib*nrays) = (*(vd+ir+ib*nrays)-offset)/gain;
          if (*(vrad_nodata+ir+ib*nrays)) *(vd+ir+ib*nrays) = nodata;
          if (*(vrad_undetect+ir+ib*nrays)) *(vd+ir+ib*nrays) = undetect;

          PolarScanParam_setValue (vrad, ib, ir, *(vd+ir+ib*nrays));
        }
      }

      RaveAttribute_setName(dattr, "how/dealiased");
      RaveAttribute_setString(dattr, "True");
      PolarScanParam_addAttribute(vrad, dattr);

      RAVE_FREE(vrad_nodata);
      RAVE_FREE(vrad_undetect);
      RAVE_FREE(x);
      RAVE_FREE(y);
      RAVE_FREE(vo);
      RAVE_FREE(vd);
      RAVE_FREE(uh);
      RAVE_FREE(vh);
      RAVE_FREE(e);
      RAVE_FREE(xt);
      RAVE_FREE(yt);
      RAVE_FREE(vt1);
      RAVE_FREE(dv);
      RAVE_FREE(v);
      RAVE_OBJECT_RELEASE(vrad);
      RAVE_OBJECT_RELEASE(attr);
    }
    retval = 1;

  } else {
    retval = 0;  /* No VRAD or already dealiased */
  }
  RAVE_OBJECT_RELEASE(dattr);
  return retval;
}

