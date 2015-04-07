/* --------------------------------------------------------------------
Copyright (C) 2012 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Implementation of the vpr correction functionality
 * @file
 * @author Lars Norin (SMHI) - Original implementation
 * @author Anders Henja (SMHI) - Adapted and modified to follow the rave tool box concept
 * @date 2013-11-19
 */
#include "rave_vpr_correction.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include <string.h>

/**
 * Represents the vpr correction algorithm
 */
struct _RaveVprCorrection_t {
  RAVE_OBJECT_HEAD /** Always on top */
  double minZLimitStratiformCloud;
  double minReflectivity;
  double heightLimit;
  double profileHeight;
  double minDistance;
  double maxDistance;
  int minNumberOfObservations;
};

/*@{ Private functions */
/**
 * Constructor
 */
static int RaveVprCorrection_constructor(RaveCoreObject* obj)
{
  RaveVprCorrection_t* self = (RaveVprCorrection_t*)obj;
  self->minZLimitStratiformCloud = -20.0;
  self->minReflectivity = 10.0;
  self->heightLimit = 1300.0;
  self->profileHeight = 200.0;
  self->minDistance = 1000.0;
  self->maxDistance = 25000.0;
  self->minNumberOfObservations = 10;
  return 1;
}

/**
 * Copy constructor
 */
static int RaveVprCorrection_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RaveVprCorrection_t* self = (RaveVprCorrection_t*)obj;
  RaveVprCorrection_t* src = (RaveVprCorrection_t*)srcobj;
  self->minZLimitStratiformCloud = src->minZLimitStratiformCloud;
  self->minReflectivity = src->minReflectivity;
  self->heightLimit = src->heightLimit;
  self->profileHeight = src->profileHeight;
  self->minDistance = src->minDistance;
  self->maxDistance = src->maxDistance;
  self->minNumberOfObservations = src->minNumberOfObservations;
  return 1;
}

/**
 * Destructor
 */
static void RaveVprCorrection_destructor(RaveCoreObject* obj)
{
  /*
  RaveVprCorrection_t* self = (RaveVprCorrection_t*)obj;
  */
}

/*@} End of Private functions */

/*@{ Interface functions */
void RaveVprCorrection_setMinZLimitStratiformCloud(RaveVprCorrection_t* self, double limit)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->minZLimitStratiformCloud = limit;
}

/**
 * Returns the min limit for when the reflectivity should be seen as stratiform. All values
 * about this limit will be defined to be stratiform
 * @param[in] self - self
 * @returns - the min reflectivity for stratiform rain
 */
double RaveVprCorrection_getMinZLimitStratiformCloud(RaveVprCorrection_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->minZLimitStratiformCloud;
}

void RaveVprCorrection_setMinReflectivity(RaveVprCorrection_t* self, double limit)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->minReflectivity = limit;
}

double RaveVprCorrection_getMinReflectivity(RaveVprCorrection_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->minReflectivity;
}

void RaveVprCorrection_setHeightLimit(RaveVprCorrection_t* self, double limit)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->heightLimit = limit;
}

double RaveVprCorrection_getHeightLimit(RaveVprCorrection_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->heightLimit;
}

int RaveVprCorrection_setProfileHeight(RaveVprCorrection_t* self, double height)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (height == 0.0) {
    return 0;
  }
  self->profileHeight = height;
  return 1;
}

double RaveVprCorrection_getProfileHeight(RaveVprCorrection_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->profileHeight;
}

void RaveVprCorrection_setMinDistance(RaveVprCorrection_t* self, double limit)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->minDistance = limit;
}

double RaveVprCorrection_getMinDistance(RaveVprCorrection_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->minDistance;
}

void RaveVprCorrection_setMaxDistance(RaveVprCorrection_t* self, double limit)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->maxDistance = limit;
}

double RaveVprCorrection_getMaxDistance(RaveVprCorrection_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->maxDistance;
}

RaveField_t* RaveVprCorrectionInternal_createIndices(RaveVprCorrection_t* self, PolarVolume_t* pvol)
{
  long nscans = 0, /*i = 0, j = 0, k = 0, */lnbins = 0;
  RaveField_t* indices = NULL;
  RaveField_t* result = NULL;

  PolarVolume_sortByElevations(pvol, 1); /* Ascending order */

  nscans = PolarVolume_getNumberOfScans(pvol);

  /* We assume that the lowest scan has the dimension/scale that is wanted in the result */
  indices = RAVE_OBJECT_NEW(&RaveField_TYPE);
  if (indices == NULL || !RaveField_createData(indices, lnbins, nscans, RaveDataType_INT)) {
    RAVE_ERROR0("Failed to allocate indices");
    goto done;
  }

  result = RAVE_OBJECT_COPY(indices);
done:
  RAVE_OBJECT_RELEASE(indices);
  return result;
}

PolarVolume_t* RaveVprCorrection_separateSC(RaveVprCorrection_t* self, PolarVolume_t* pvol)
{
  double* reflectivityProfile = RAVE_MALLOC(sizeof(double) * self->heightLimit / self->profileHeight);
  RAVE_FREE(reflectivityProfile);
  return NULL;
#ifdef KALLE
  PolarScan_t* lowestscan = NULL;
  long nscans = 0, lnbins = 0, lnrange = 0, i = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");

  PolarVolume_sortByElevations(pvol, 1); /* Ascending order */
  nscans = PolarVolume_getNumberOfScans(pvol);
  lowestscan = PolarVolume_getScan(pvol, 0);
  if (lowestscan == NULL) {
    RAVE_ERROR0("Failed to get the lowest scan");
    goto done;
  }
  lnbins = PolarScan_getNbins(lowestscan);
  lnrange = PolarScan_getRange(lowestscan);

done:
  return NULL;
  long nscans = 0, i = 0, j = 0, k = 0, lnbins = 0;
  double aa = 0.0;
  double lscanrange = 0.0;
  PolarScan_t* scan = NULL;
  PolarScan_t* lowestscan = NULL;
  RaveField_t* indices = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");

  PolarVolume_sortByElevations(pvol, 1); /* Ascending order */
  nscans = PolarVolume_getNumberOfScans(pvol);
  lowestscan = PolarVolume_getScan(pvol, 0);
  if (lowestscan == NULL) {
    RAVE_ERROR0("Failed to get the lowest scan");
    goto done;
  }
  lnbins = PolarScan_getNbins(lowestscan);

  /* We assume that the lowest scan has the dimension/scale that is wanted in the result */
  indices = RAVE_OBJECT_NEW(&RaveField_TYPE);
  if (indices == NULL || !RaveField_createData(indices, lnbins, nscans, RaveDataType_INT)) {
    RAVE_ERROR0("Failed to allocate indices");
    goto done;
  }
#endif
#ifdef KALLE
  for (k = 0; k < nscans; k++) {
    long nbins = 0;
    scan = PolarVolume_getScan(pvol, k);
    if (scan == NULL) {
      RAVE_ERROR1("No scan found at position %d", k);
      goto done;
    }
    nbins = PolarScan_getNbins(scan);
    for (i = 0; i < nbins; i++) {
      for (j = 0; j < lnbins; j++) {
        if (j == 0) {
          aa = fabs(PolarScan_getGroundRange(lowestscan, j) - PolarScan_getGroundRange(scan, i));
          if (aa < 1e3) {
            RaveField_setValue(indices, j, k, i);
          }
        } else if (fabs(PolarScan_getGroundRange(lowestscan, j) - PolarScan_getGroundRange(scan, i)) < aa) {
          if (aa < 1e3) {
            RaveField_setValue(indices, j, k, i);
          }
        }
      }
    }
    RAVE_OBJECT_RELEASE(scan);
  }
#endif
#ifdef KALLE
done:
  RAVE_OBJECT_RELEASE(lowestscan);
  RAVE_OBJECT_RELEASE(indices);
#endif
  return NULL;
#ifdef KALLE
  // Find indices of bins in higher tilts closest in ground range to bins in lowest tilt
  for (i=0;i<nbins;i++)
  {
      for (k=0;k<ntilts;k++)
      {
          for (j=0;j<nbins;j++)
          {
              if (j==0)
              {
                  aa = fabs(ground_range[j][0]-ground_range[i][k]);
                  if (aa<1e3)
                      ind[i][k] = j;
              }
              else if (fabs(ground_range[j][0]-ground_range[i][k]) < aa)
              {
                  aa = fabs(ground_range[j][0]-ground_range[i][k]);
                  if (aa<1e3)
                      ind[i][k] = j;
              }
          }
      }
  }
#endif
#ifdef KALLE
  int i = 0, j = 0, k = 0;
  int nbins = 0, nscans = 0;
  RaveObjectList_t* distanceVectors = NULL;
  RaveField_t *distanceField = NULL, *heightField = NULL;
  PolarVolume_t* outvol = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");

  PolarVolume_sortByElevations(pvol, 1); /* Want elevations to be sorted in ascending order */

  distanceField = PolarVolume_getDistanceField(pvol);

  heightField = PolarVolume_getHeightField(pvol);

  nbins = RaveField_getXsize(distanceField);
  nscans = PolarVolume_getNumberOfScans(pvol);

  outvol = RAVE_OBJECT_NEW(&PolarVolume_TYPE);
#endif

//  for (k = 0; k < nscans; k++) {
//    for (i = 0; i < nbins; i++) {
//      double h = 0.0, d = 0.0;
//      RaveField_getValue(heightField, i, k, &h);
//      RaveField_getValue(distanceField, i, k, &d);
//      if (h <= -99999.0)
//
//    }
//  }
//
#ifdef KALLE
  //heightField = PolarVolume_getHeightField(pvol);

//  // Find indices of bins in higher tilts closest in ground range to bins in lowest tilt
//  for (i=0;i<nbins;i++)
//  {
//      for (k=0;k<ntilts;k++)
//      {
//          for (j=0;j<nbins;j++)
//          {
//              if (j==0)
//              {
//                  aa = fabs(ground_range[j][0]-ground_range[i][k]);
//                  if (aa<1e3)
//                      ind[i][k] = j;
//              }
//              else if (fabs(ground_range[j][0]-ground_range[i][k]) < aa)
//              {
//                  aa = fabs(ground_range[j][0]-ground_range[i][k]);
//                  if (aa<1e3)
//                      ind[i][k] = j;
//              }
//          }
//      }
//  }
#endif
}

static PolarObservation* RaveVprCorrectionInternal_filterObservations(RaveVprCorrection_t* self, PolarObservation* obses, int nobses, int* nFilteredObses)
{
  int ndataobservations = 0;
  int nfiltered = 0;
  int i = 0;
  PolarObservation* dataobservations = NULL;
  PolarObservation* result = NULL;

  *nFilteredObses = 0;
  dataobservations = RaveTypes_FilterPolarObservationDataValues(obses, nobses, &ndataobservations);
  if (dataobservations != NULL && ndataobservations > 0) {
    result = RAVE_MALLOC(sizeof(PolarObservation) * ndataobservations);
    if (result == NULL) {
      RAVE_ERROR0("Failed to allocate memory for filtered observations");
      goto done;
    }
    for (i = 0; i < ndataobservations; i++) {
      if (dataobservations[i].distance >= self->minDistance &&
          dataobservations[i].distance <= self->maxDistance &&
          dataobservations[i].v > self->minReflectivity) {
        result[nfiltered++] = dataobservations[i];
      }
    }
    *nFilteredObses = nfiltered;
  }
done:
  RAVE_FREE(dataobservations);
  return result;
}

double* RaveVprCorrection_getReflectivityArray(RaveVprCorrection_t* self, PolarVolume_t* pvol, int* nElements)
{
  double* result = NULL;
  int nelem = 0, i = 0;
  double nextHeight = 0.0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  nelem = (int)(self->heightLimit / self->profileHeight);
  result = RAVE_MALLOC(sizeof(double) * nelem);
  nextHeight = self->profileHeight / 2.0;
  for (i = 0; i < nelem; i++) {
    int nobservations = 0;
    PolarObservation* observations = PolarVolume_getCorrectedValuesAtHeight(pvol, nextHeight, self->profileHeight, &nobservations);
    result[i] = -9999.0;
    if (observations != NULL) {
      int ndataobservations = 0;
      PolarObservation* dataobservations = RaveVprCorrectionInternal_filterObservations(self, observations, nobservations, &ndataobservations);
      if (dataobservations != NULL && ndataobservations > self->minNumberOfObservations) {
        RaveTypes_SortPolarObservations(dataobservations, ndataobservations);
        if (ndataobservations % 2 == 0 && ndataobservations > 0) {
          result[i] = (dataobservations[ndataobservations/2].v + dataobservations[ndataobservations/2 - 1].v)/2.0;
        } else {
          result[i] = dataobservations[ndataobservations/2].v;
        }
      }
      RAVE_FREE(dataobservations);
    }
    RAVE_FREE(observations);
    nextHeight = nextHeight + self->profileHeight;
  }
  *nElements = nelem;
  return result;
}

/*@} End of Interface functions */

RaveCoreObjectType RaveVprCorrection_TYPE = {
    "RaveVprCorrection",
    sizeof(RaveVprCorrection_t),
    RaveVprCorrection_constructor,
    RaveVprCorrection_destructor,
    RaveVprCorrection_copyconstructor
};
