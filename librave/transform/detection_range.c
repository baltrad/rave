/* --------------------------------------------------------------------
Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Provides functionality for creating composites.
 * @file
 * @author Harri Hohti, FMI
 * @author Daniel Michelson, SMHI (Intgration)
 * @author Anders Henja, SMHI (Adaption to rave framework)
 * @date 2010-01-19
 */
#include "detection_range.h"
#include "raveobject_list.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <math.h>

#define ER 8495000.0  /**< effective earth radius for bin height calculations */

#define CONSTGRAD 50.0 /**< 10 dB/km = 1000 m / 20 dBN */

/**
 * Represents the detection range generator.
 */
struct _DetectionRange_t {
  RAVE_OBJECT_HEAD /** Always on top */
  char* lookupPath; /**< where lookup files are located, default is /tmp */
};

/*@{ Private functions */
/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int DetectionRange_constructor(RaveCoreObject* obj)
{
  DetectionRange_t* this = (DetectionRange_t*)obj;
  this->lookupPath = NULL;
  if (!DetectionRange_setLookupPath(this, "/tmp")) {
    return 0;
  }
  return 1;
}

/**
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int DetectionRange_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  DetectionRange_t* this = (DetectionRange_t*)obj;
  DetectionRange_t* src = (DetectionRange_t*)srcobj;
  this->lookupPath = NULL;
  if (!DetectionRange_setLookupPath(this, DetectionRange_getLookupPath(src))) {
    return 0;
  }
  return 1;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void DetectionRange_destructor(RaveCoreObject* obj)
{
  DetectionRange_t* this = (DetectionRange_t*)obj;
  RAVE_FREE(this->lookupPath);
}

/**
 * Calculates the height of the bin assuming
 * a ideal sphere.
 * @param[in] m - the range near surface (in meters)
 * @param[in] e - the elevation angle
 * @param[in] h0 - the height above ground for the radar
 * @return the height in meter
 */
static int DetectionRangeInternal_binheight(double m,double e,double h0)
{
   double rh;
   int h;

   rh=m/cos(e);
   h=(int)(h0+(rh*sin(e)+0.5*rh*rh/ER));
   return(h);
}

/**
 * Calculate the range of the bin.
 * @param[in] h - altitude
 * @param[in] e - the elevation in radians
 * @param[in] h0 - altitude0
 * @return the range
 */
static double DetectionRangeInternal_bindist(double h,double e,double h0)
{
   double r;
   r = cos(e)*ER*(sqrt(sin(e)*sin(e)+0.002*(h-h0)/ER) - sin(e));
   return(r);
}

/**
 * Locates the lowest elevation angle in the volume and returns it.
 */
static double DetectionRangeInternal_getLowestElevationAngle(PolarVolume_t* pvol)
{
  int nscans = 0, i = 0;
  double result = 9999999.0;
  nscans = PolarVolume_getNumberOfScans(pvol);
  for (i = 0; i < nscans; i++) {
    PolarScan_t* scan = PolarVolume_getScan(pvol, i);
    if (scan != NULL) {
      if (PolarScan_getElangle(scan) < result) {
        result = PolarScan_getElangle(scan);
      }
    }
    RAVE_OBJECT_RELEASE(scan);
  }
  return result;
}

/**
 * Sector weight factors generation. Linear weights with highest value
 * at center of the sector normalized to 1 and sector edges to 0.
 * @param[in] weightsector - width of weighting sector
 * @param[in] maxweight - maximum unnormalized sector weight
 * @param[in] inW - weight sector width input value
 * @param[out] Wsecsum - sum of sector weight factors
 * @return an array of sector weight factors with length = weightsector
 */
static double* DetectionRangeInternal_createSectorWeightFactors(
  int weightsector, double maxweight, int inW, double* Wsecsum)
{
  int i = 0;
  double wg = 0.0, mx2 = 0.0;
  double* weightarr = NULL;
  double sum = 0.0;

  RAVE_ASSERT((Wsecsum != NULL), "Wsecsum == NULL");

  weightarr = RAVE_MALLOC(sizeof(double)*weightsector);
  if (weightarr == NULL) {
    RAVE_CRITICAL0("Failed to allocate weight array");
    return NULL;
  }
  memset(weightarr, 0, sizeof(double)*weightsector);

  mx2 = maxweight*maxweight;
  for(i = 0; i < weightsector; i++) {
    wg = (double)((inW+1)-abs(inW-i));
    weightarr[i] = wg/maxweight;
    sum += weightarr[i];
  }
  *Wsecsum = sum;
  return weightarr;
}

/**
 * Creates the previous top file name.
 * @param[in] self - self
 * @param[in] source - the source name (MAY NOT BE NULL))
 * @param[in,out] name - the generated file name
 * @param[in] len - the allocated length of name
 * @return 1 on success or 0 if something erroneous occured
 */
static int DetectionRangeInternal_createPreviousTopFilename(
  DetectionRange_t* self, const char* source, char* name, int len)
{
  int slen = 0;
  int result = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (source == NULL) {
    RAVE_ERROR0("Providing a NULL value for source");
    goto done;
  }
  slen = strlen(source);

  if ((strlen(DetectionRange_getLookupPath(self)) + slen + 13) > len) {
    RAVE_WARNING0("Not enough memory allocated for top file name");
    goto done;
  }
  sprintf(name, "%s/%s_oldtop.txt", DetectionRange_getLookupPath(self), source);

  result = 1;
done:
  return result;
}

/**
 * Reads the previous background top value from the cached file. If no source
 * has been specified or if the lookup file does not exist, the climatological value
 * (5.5) will be returned.
 */
static double DetectionRangeInternal_readPreviousBackgroundTop(DetectionRange_t* self, const char* source)
{
  FILE* fp = NULL;
  char filename[1024];
  double TOPrev = 5.5;
  int nritems = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");

  if (!DetectionRangeInternal_createPreviousTopFilename(self, source, filename, 1024)) {
    goto done;
  }
  fp = fopen(filename, "r");
  if (fp == NULL) {
    RAVE_INFO1("Could not locate lookup file %s, defaulting to TOPrev = 5.5", filename);
    goto done;
  } else {
    nritems = fscanf(fp,"%lf",&TOPrev);
  }

done:
  if (fp != NULL) {
    fclose(fp);
  }
  return TOPrev;
}

static time_t DetectionRangeInternal_getPreviousTopFiletime(DetectionRange_t* self, const char* source)
{
  time_t result;
  char filename[1024];
  struct stat filestat;

  time(&result); // Initialize time to now if any file operation fails.

  if (!DetectionRangeInternal_createPreviousTopFilename(self, source, filename, 1024)) {
    goto done;
  }
  if (stat(filename,&filestat) != 0) {
    goto done;
  }
  result = filestat.st_mtime;

done:
  return result;
}

/**
 * Backround TOP values generation based on previous TOP and its age.
 * TOP backround value converges from value of previous valid TOP to
 * climatological guess of 5.5 km during 48 hour ageing period beginning
 * if previous TOP is older than 2 hours
 * @param[in] self - self
 * @param[in] source - the source
 * @param[in] TOPprev
 * @returns the aged TOPprev
 */
static double DetectionRangeInternal_generateAgedTop(DetectionRange_t* self, const char* source, double TOPprev)
{
  time_t prevtoptime, curtime;
  double prevtop_age=0.0, max_prevtop_age=48.0, rel_age=0.0, TOPdiff=0.0, newTOPprev=0.0;

  time(&curtime);
  prevtoptime = DetectionRangeInternal_getPreviousTopFiletime(self, source);

  /* ageing of previous TOP begins if it's older than two hours */
  prevtop_age=(double)(curtime - prevtoptime)/3600.0-2.25;
  if(prevtop_age > 0.0) {
    rel_age = prevtop_age / max_prevtop_age;
    if(rel_age>1.0) {
      rel_age=1.0;
    }
    TOPdiff=TOPprev-5.5;
    newTOPprev=TOPprev-rel_age*TOPdiff;
    TOPprev=newTOPprev;
  }
  return TOPprev;
}

/*@} End of Private functions */

/*@{ Interface functions */
int DetectionRange_setLookupPath(DetectionRange_t* self, const char* path)
{
  int result = 0;
  char* tmp = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (path == NULL) {
    return 0;
  }
  tmp = RAVE_STRDUP(path);
  if (tmp != NULL) {
    RAVE_FREE(self->lookupPath);
    self->lookupPath = tmp;
    result = 1;
  }
  return result;
}

const char* DetectionRange_getLookupPath(DetectionRange_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->lookupPath;
}

PolarScan_t* DetectionRange_top(DetectionRange_t* self, PolarVolume_t* pvol, double scale, double threshold_dBZN)
{
  PolarScan_t* maxdistancescan = NULL;
  PolarScan_t* result = NULL;
  PolarScan_t* retval = NULL;
  PolarScanParam_t* param = NULL;
  int nrscans = 0;
  double scaleFactor = 0.0;
  long nbins = 0, nrays = 0;
  int rayi = 0, bini = 0, elevi = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");

  if (pvol == NULL) {
    RAVE_ERROR0("Can not determine top from a volume that is NULL");
    goto error;
  } else if (scale <= 0.0) {
    RAVE_ERROR0("Scale must be > 0.0");
    goto error;
  }

  PolarVolume_sortByElevations(pvol, 0); // Descending
  nrscans = PolarVolume_getNumberOfScans(pvol);
  maxdistancescan = PolarVolume_getScanWithMaxDistance(pvol);
  if (maxdistancescan == NULL) {
    goto error;
  }
  result = RAVE_OBJECT_CLONE(maxdistancescan);
  if (result == NULL) {
    RAVE_ERROR0("Failed to clone max distance scan");
    goto error;
  }
  PolarScan_removeAllParameters(result);

  // Calculate bins from scale
  scaleFactor = PolarScan_getRscale(maxdistancescan) / scale;
  nbins = (long)(scaleFactor * (double)PolarScan_getNbins(maxdistancescan));

  param = RAVE_OBJECT_NEW(&PolarScanParam_TYPE);
  if (param == NULL || !PolarScanParam_createData(param, nbins, PolarScan_getNrays(maxdistancescan), RaveDataType_UCHAR)) {
    RAVE_ERROR0("Failed to create polar scan parameter for echo top");
    goto error;
  }

  PolarScanParam_setQuantity(param, "HGHT");
  PolarScanParam_setGain(param, (double)0.1);  /* Email from Harri 2010-09-30 */
  PolarScanParam_setOffset(param, (double)-0.1);
  PolarScanParam_setNodata(param, (double)255.0);
  PolarScanParam_setUndetect(param, (double)0.0);
  PolarScan_addParameter(result, param);
  PolarScan_setElangle(result, DetectionRangeInternal_getLowestElevationAngle(pvol));
  PolarScan_setBeamwidth(result, PolarScan_getBeamwidth(maxdistancescan));
  PolarScan_setDefaultParameter(result, "HGHT");
  PolarScan_setRscale(result, scale);
  nrays = PolarScan_getNrays(result);

  for(rayi = 0; rayi < nrays; rayi++) {
    for (bini = 0; bini < nbins; bini++) {
      int topfound = 0;
      int overMaxelev = 0;
      int highest_ei = 0;
      double range = PolarScan_getRange(result, bini);
      int bi = 0, lower_bi = 0;
      double toph = 0.0;
      int found = 0; /* Used to break elevation loop when value has been found */

      for (elevi = 0; !found && elevi < (nrscans - 1); elevi++) {
        PolarScan_t* scan = PolarVolume_getScan(pvol, elevi);
        PolarScan_t* lowscan = PolarVolume_getScan(pvol, elevi+1);
        double elangle = 0.0, lower_elangle = 0.0, height = 0.0, lower_height = 0.0;
        double binh = 0.0, lower_binh = 0.0, Dh = 0.0, dBZN = 0.0, lower_dBZN = 0.0;
        RaveValueType dBZN_type = RaveValueType_UNDEFINED, lower_dBZN_type = RaveValueType_UNDEFINED;

        bi = PolarScan_getRangeIndex(scan, range);
        lower_bi = PolarScan_getRangeIndex(lowscan, range);

        elangle = PolarScan_getElangle(scan);
        lower_elangle = PolarScan_getElangle(lowscan);

        height = PolarScan_getHeight(scan);
        lower_height = PolarScan_getHeight(lowscan);

        if (bi < 0) {
          highest_ei = elevi + 1;
        } else {
          binh = DetectionRangeInternal_binheight(range, elangle, height);
          lower_binh = DetectionRangeInternal_binheight(range, lower_elangle, lower_height);
          Dh=(double)(binh-lower_binh);

          dBZN_type = PolarScan_getConvertedValue(scan, bi, rayi, &dBZN);
          lower_dBZN_type = PolarScan_getConvertedValue(lowscan, lower_bi, rayi, &lower_dBZN);

          if (dBZN_type == RaveValueType_DATA || lower_dBZN_type == RaveValueType_DATA) {
            if (!found && dBZN > threshold_dBZN && elevi == highest_ei)
            {
              if (lower_dBZN)
              {
                overMaxelev = 1;
              }
              found = 1;
            }
            if (!found && dBZN == threshold_dBZN)
            {
              if(lower_dBZN)
              {
                topfound = 1;
                toph=(double)binh;
              }
              found = 1;
            }
            if (!found && lower_dBZN == threshold_dBZN)
            {
              topfound = 1;
              toph = (double)lower_binh;
              found = 1;
            }
            if (!found && lower_dBZN > threshold_dBZN)
            {
              topfound=1;
              if(!dBZN) {
                toph=lower_binh+(double)(lower_dBZN - threshold_dBZN)*CONSTGRAD;
              } else {
                toph=lower_binh+(double)(lower_dBZN - threshold_dBZN) * Dh/(double)(lower_dBZN - dBZN);
              }
              found = 1;
            }
          }
        }
        RAVE_OBJECT_RELEASE(scan);
        RAVE_OBJECT_RELEASE(lowscan);
      }

      if(overMaxelev) {
        PolarScan_setValue(result, bini, rayi, 254.0);
      } else if (topfound) {
        PolarScan_setValue(result, bini, rayi, toph/100.0 + 1.0);
      }
    }
  }

  retval = RAVE_OBJECT_COPY(result);
error:
  RAVE_OBJECT_RELEASE(maxdistancescan);
  RAVE_OBJECT_RELEASE(param);
  RAVE_OBJECT_RELEASE(result);
  return retval;
}

PolarScan_t* DetectionRange_filter(DetectionRange_t* self, PolarScan_t* scan)
{
  PolarScan_t* result = NULL;
  PolarScan_t* clone = NULL;
  PolarScanParam_t* param = NULL;

  int bi = 0, ri = 0;
  int nbins = 0, nrays = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");

  if (scan == NULL) {
    RAVE_ERROR0("No point to filter a NULL field\n");
    goto done;
  }

  clone = RAVE_OBJECT_CLONE(scan);
  if (clone == NULL) {
    RAVE_CRITICAL0("Failed to clone scan");
    goto done;
  }
  nbins = PolarScan_getNbins(clone);
  nrays = PolarScan_getNrays(clone);

  if (!PolarScan_setDefaultParameter(clone, "HGHT")) {
    RAVE_ERROR0("Could not set default parameter to HGHT");
    goto done;
  }

  param = PolarScan_getParameter(scan, "HGHT");
  if (param == NULL) {
    RAVE_ERROR0("No HGHT parameter in scan");
    goto done;
  }

  for (bi = 0; bi < nbins; bi++) {
    for (ri = 0; ri < nrays; ri++) {
      double dob = 0.0, dnb = 0.0, db = 0.0;
      unsigned char ob = 0, nb = 0, b = 0;

      if (ri == 0) {
        PolarScanParam_getValue(param, bi, nrays-1, &dob);
      } else {
        PolarScanParam_getValue(param, bi, ri-1, &dob);
      }
      if (ri == nrays-1) {
        PolarScanParam_getValue(param, bi, 0, &dnb);
      } else {
        PolarScanParam_getValue(param, bi, ri+1, &dnb);
      }
      PolarScanParam_getValue(param, bi, ri, &db);

      ob = (unsigned char)dob;
      nb = (unsigned char)dnb;
      b = (unsigned char)db;

      if((!(ob | nb) && b) || (b > (ob+nb))) {
        PolarScan_setValue(clone, bi, ri, 0.0);
      }
    }
  }

  result = RAVE_OBJECT_COPY(clone);

done:
  RAVE_OBJECT_RELEASE(clone);
  RAVE_OBJECT_RELEASE(param);
  return result;
}

/**
 * @param[in] avgsector - width of the floating average azimuthal sector
 * @param[in] sortage - defining the higher portion of sorted ray to be analysed, typically 0.05 - 0.2
 * @param[in] samplepoint - define the position to pick a representative TOP value from highest
 *                          valid TOPs, typically near 0.5 (median) lower values (nearer to
 *                          highest TOP, 0.15) used in noisier radars like KOR.
 */
PolarScan_t* DetectionRange_analyze(DetectionRange_t* self,
  PolarScan_t* scan, int avgsector, double sortage, double samplepoint)
{
#ifdef KALLE
  int i,                        /* common index variable                                     */
      A,B,                      /* azimuth and bin (range gate) indices                      */
      weightsector,             /* width of weighting sector [deg]                           */
      StartBin,BinCount,        /* starting bin and bin count of top "ray" analysis,
                                   500 m bins assumed !                                      */
      sortpart_ray,             /* the defined part of height-sorted ray (see sortage)       */
      valid_raytop_count=0,     /* count of "valid TOP rays" e.g. rays having
                                   top_count==sortpart_ray (=ray weight is > 0.99)           */
      wI,                       /* index of weight factors array                             */
      rayN,                     /* count of rays having valid top value                      */
      inW,                      /* weight sector width input value,  weightsector=inW*2+1    */
      outbyte,                  /* output array (outarr) value (0-255)                       */
      items,                    /* number of items read with sscanf or fscanf                */
      picbin;                   /* bin number to pick from valid TOP values of sorted ray
                                   topcount * samplepoint [median would be 0.5]              */

  char *p,                      /* common pointer                                            */
       hdr[200]={0},            /* TOP pgm header string                                     */
       *prevTOPfile;

  double half_bw,               /* half beam width [rad]                                     */
         minrange,maxrange,     /* minimum and maximum radial ranges of analysis [km]        */
         Wsum,                  /* sum of total weight factors                               */
         Wsecsum,               /* sum of sector weight factors                              */
         maxweight,             /* maximum unnormalized sector weight                        */
         *weightarr = NULL,     /* array of sector weight factors, dimension=weightsector    */
         TOPprev,               /* background TOP value (based on previous TOP)              */
         *ray_pickhightop = NULL, /* TOP picked from highest TOPs defined by sortage
                                   and samplepoint */
         lowest_elev,
         *Final_WTOP = NULL,    /* Final ray-, sector- and background weighted TOP for rays  */

         *ray_maxtop = NULL,    /* array of maximum TOPs of rays                             */
         *rayweight = NULL,     /* weight ray depends on how many TOPs are existing
                                   in sort part of ray */
         maxR_analyzed_highbeam=250.0, /* final analysed maximum range of upper edge of ray  */
         maxR_analyzed_lowbeam=250.0,  /* final analysed maximum range of lower edge of ray  */
         *sorttop = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (scan == NULL) {
    RAVE_ERROR0("Trying to analyse a NULL field");
    return NULL;
  }
  half_bw = PolarScan_getBeamwidth(scan) / 2.0;
  inW = avgsector / 2;
  weightsector = inW*2 + 1;
  maxweight=(double)inW+1.0;

  lowest_elev = PolarScan_getElangle(scan); // Original code uses 0.3 here and then later on the real lowest elangle !?

  /* Sector weight factors generation. Linear weights with highest value
   * at center of the sector normalized to 1 and sector edges to 0.
   */
  weightarr = DetectionRangeInternal_createSectorWeightFactors(weightsector, maxweight, inW, &Wsecsum);

  /** Calculate previous top value */
  TOPprev = DetectionRangeInternal_readPreviousBackgroundTop(self, PolarScan_getSource(scan));
  TOPprev = DetectionRangeInternal_generateAgedTop(self, PolarScan_getSource(scan), TOPprev);

  /* maximum range of previous top just under beam */
  prev_maxr=DetectionRangeInternal_bindist(TOPprev*1000.0,lowest_elev-half_bw,0.0);

  /* Sorting of TOP values and selection of representative TOPs per ray */

  /* determine StartBin and BinCount assuming 10km start on each side */
  {
    double scale = PolarScan_getRscale(scan);
    if (scale == 0.0) {
      StartBin = 0;
      BinCount = PolarScan_getNbins(scan);
    } else {
      int gapnbins = (int)(10000.0/scale);
      StartBin = gapnbins;
      BinCount = PolarScan_getNbins(scan) - gapnbins;
    }
  }
#endif

  return NULL;
}

//set BEAMWIDTH = 1.0
//set AVERAGING_SECTOR = 60  # width of the floating average azimuthal sector
//         # HELP: Number of azimuthal gates or degrees??
//set OLDTOP = "$SITE"_oldtop.txt
//set HIGHPART = 0.1         # What part of the sorted are processed (1 = all)
//set SAMPLEPOINT = 0.5      # What values of previous existing sorted height
//#                            is selected to represent the valid TOP value of
//#                            a ray (0.5 is the median, 0.0 refers to highest value)

/*@} End of Interface functions */

RaveCoreObjectType DetectionRange_TYPE = {
    "DetectionRange",
    sizeof(DetectionRange_t),
    DetectionRange_constructor,
    DetectionRange_destructor,
    DetectionRange_copyconstructor
};

