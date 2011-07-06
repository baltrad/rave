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
#include <string.h>


#define ER 8495000.0  /**< effective earth radius for bin height calculations */

#define CONSTGRAD 50.0 /**< 10 dB/km = 1000 m / 20 dBN */

/**
 * Represents the detection range generator.
 */
struct _DetectionRange_t {
  RAVE_OBJECT_HEAD /** Always on top */
};

/*@{ Private functions */
/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int DetectionRange_constructor(RaveCoreObject* obj)
{
  return 1;
}

/**
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int DetectionRange_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  return 1;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void DetectionRange_destructor(RaveCoreObject* obj)
{
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

/*@} End of Private functions */

/*@{ Interface functions */
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
  PolarScan_setElangle(result, 0.0); // Might need lowest elevation angle if I found need for it
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

/*@} End of Interface functions */

RaveCoreObjectType DetectionRange_TYPE = {
    "DetectionRange",
    sizeof(DetectionRange_t),
    DetectionRange_constructor,
    DetectionRange_destructor,
    DetectionRange_copyconstructor
};

