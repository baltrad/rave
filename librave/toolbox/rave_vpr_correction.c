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
#include "tnc.h"
#include <math.h>

#define TMINUS -5.0
#define TPLUS 4.0
#define DZDH -0.003
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
  double minusTemperature;
  double plusTemperature;
  double dzdh;
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
  self->heightLimit = 10000.0;
  self->profileHeight = 100.0;
  self->minDistance = 1000.0;
  self->maxDistance = 25000.0;
  self->minNumberOfObservations = 10;
  self->minusTemperature = TMINUS;
  self->plusTemperature = TPLUS;
  self->dzdh = DZDH;
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
  self->minusTemperature = src->minusTemperature;
  self->plusTemperature = src->plusTemperature;
  self->dzdh = src->dzdh;
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

void RaveVprCorrection_setMinusTemperature(RaveVprCorrection_t* self, double temp)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->minusTemperature = temp;
}

double RaveVprCorrection_getMinusTemperature(RaveVprCorrection_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->minusTemperature;
}


void RaveVprCorrection_setPlusTemperature(RaveVprCorrection_t* self, double temp)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->plusTemperature = temp;
}


double RaveVprCorrection_getPlusTemperature(RaveVprCorrection_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->plusTemperature;
}

int RaveVprCorrection_setDzdh(RaveVprCorrection_t* self, double dzdh)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (dzdh >= 0.0) {
    RAVE_ERROR0("DZDH must be negative");
    return 0;
  }
  self->dzdh = dzdh;
  return 1;
}


double RaveVprCorrection_getDzdh(RaveVprCorrection_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->dzdh;
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

int RaveVprCorrection_getNumberOfHeightIntervals(RaveVprCorrection_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (int)(self->heightLimit / self->profileHeight);
}

double* RaveVprCorrection_getHeights(RaveVprCorrection_t* self, int* nrHeights)
{
  double* result = NULL;
  int n = 0, i = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  n = RaveVprCorrection_getNumberOfHeightIntervals(self);
  result = RAVE_MALLOC(sizeof(double) * n);
  if (result != NULL) {
    for (i = 0; i < n; i++) {
      result[i] = ((double)i + 0.5) * self->profileHeight;
    }
    *nrHeights = n;
  }
  return result;
}

RaveVprValue_t* RaveVprCorrection_getReflectivityArray(RaveVprCorrection_t* self, PolarVolume_t* pvol, int* nElements)
{
  RaveVprValue_t* result = NULL;
  int nelem = 0, i = 0;
  double nextHeight = 0.0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  nelem = RaveVprCorrection_getNumberOfHeightIntervals(self);
  result = RAVE_MALLOC(sizeof(RaveVprValue_t) * nelem);
  if (result == NULL) {
    RAVE_CRITICAL0("Failed to allocate memory");
    return NULL;
  }

  nextHeight = self->profileHeight / 2.0;
  for (i = 0; i < nelem; i++) {
    int nobservations = 0;
    PolarObservation* observations = PolarVolume_getCorrectedValuesAtHeight(pvol, nextHeight, self->profileHeight, &nobservations);
    result[i].nrpoints = 0;
    result[i].value = 0.0;
    if (observations != NULL) {
      int ndataobservations = 0;
      PolarObservation* dataobservations = RaveVprCorrectionInternal_filterObservations(self, observations, nobservations, &ndataobservations);
      if (dataobservations != NULL) {
        RaveTypes_SortPolarObservations(dataobservations, ndataobservations);
        if (ndataobservations % 2 == 0 && ndataobservations > 0) {
          result[i].value = (dataobservations[ndataobservations/2].v + dataobservations[ndataobservations/2 - 1].v)/2.0;
        } else {
          result[i].value = dataobservations[ndataobservations/2].v;
        }
        result[i].nrpoints = ndataobservations;
      }
      RAVE_FREE(dataobservations);
    }
    RAVE_FREE(observations);
    nextHeight = nextHeight + self->profileHeight;
  }
  *nElements = nelem;
  return result;
}

static int RaveVprCorrectionInternal_lsqFirstOrder(RaveVprCorrection_t* self, RaveVprValue_t* refArray, int start, int nelems, double* a, double* b, double* mean)
{
  double *x = NULL, *y = NULL;
  double nextHeight = self->profileHeight / 2.0;
  int i = 0, index = 0;
  int result = 0;
  double vsum = 0.0;

  if (start > nelems) {
    RAVE_ERROR0("start > nelems");
    return 0;
  }
  x = RAVE_MALLOC(sizeof(double)*(nelems - start));
  y = RAVE_MALLOC(sizeof(double)*(nelems - start));
  if (x == NULL || y == NULL) {
    RAVE_ERROR0("Failed to allocate memory");
    goto done;
  }
  for (i = start; i < nelems; i++) {
    if (isnan(refArray[i].value) == 0) {
      x[index] = nextHeight;
      y[index] = refArray[i].value;
      vsum += y[index];
      index++;
    }
    nextHeight = nextHeight + self->profileHeight;
  }
  if (index < 2) {
    RAVE_ERROR0("Impossible to determine lsq curve fit with less than 2 points");
    goto done;
  }
  *mean = vsum / (double)index;
  result = RaveVprCorrectionHelper_lsqFirstOrder(index, x, y, a, b);
done:
  RAVE_FREE(x);
  RAVE_FREE(y);
  return result;
}

typedef struct VprArrayWrapper {
  int nrelements;
  RaveVprValue_t* vpr;
  int minindex;
  int maxindex;
  double profileHeight;
} VprArrayWrapper;

static int RaveVprCorrectionInternal_evaluate(double x[], double *f, double height)
{
  if (x[3] != 0.0) {
    *f = x[0] + x[1]*pow(exp(1), -pow(((height - x[2])/x[3]), 2.0));
    return 0;
  }
  return -1;
}

static double RaveVprCorrectionInternal_vprCalculateError(double x[], VprArrayWrapper* harray)
{
  int i = 0;
  double sum = 0.0;
  double v = 0.0;

  for (i = harray->minindex; i < harray->maxindex; i++) {
    if (RaveVprCorrectionInternal_evaluate(x, &v, ((double)i + 0.5)*harray->profileHeight) < 0) {
      continue;
    }
    if (harray->vpr[i].value != 0.0) {
      sum = sum + pow((v / harray->vpr[i].value) - 1.0, 2.0);
    }
  }

  return sum;
}

static void RaveVprCorrectionInternal_createGradient(double f0, double x[], double g[], VprArrayWrapper* harray)
{
  double epsilon = 1e-8;
  int i = 0;

  double xn[4];
  for (i = 0; i < 4; i++) {
    memcpy(xn, x, sizeof(double)*4);
    xn[i] += epsilon;
    g[i] = (RaveVprCorrectionInternal_vprCalculateError(xn, harray) - f0)/epsilon;
  }
}


static int RaveVprCorrectionInternal_vprFitnessFunction(double x[], double *f, double g[], void* state)
{
  VprArrayWrapper* harray = (VprArrayWrapper*)state;

  *f = RaveVprCorrectionInternal_vprCalculateError(x, harray);

  RaveVprCorrectionInternal_createGradient(*f, x, g, harray);

  return 0;
}

static int RaveVprCorrectionInternal_getMinTemperatureFromHTArray(int n, RaveHeightTemperature_t* arr, double* minv)
{
  int i = 0;
  int result = -1;

  if (n <= 0 || arr == NULL) {
    goto done;
  }

  *minv = arr[0].temperature;
  result = 0;

  for (i = 1; i < n; i++) {
    if (arr[i].temperature < *minv) {
      *minv = arr[i].temperature;
      result = i;
    }
  }

done:
  return result;
}

static int RaveVprCorrectionInternal_getMaxTemperatureFromHTArray(int n, RaveHeightTemperature_t* arr, double* maxv)
{
  int i = 0;
  int result = -1;

  if (n <= 0 || arr == NULL) {
    goto done;
  }

  *maxv = arr[0].temperature;
  result = 0;

  for (i = 1; i < n; i++) {
    if (arr[i].temperature > *maxv) {
      *maxv = arr[i].temperature;
      result = i;
    }
  }

done:
  return result;
}

static int RaveVprCorrectionInternal_getHeightClosestToTemperature(RaveVprCorrection_t* self, int n, RaveHeightTemperature_t* arr, double temp, double* height)
{
  int i = 0;
  int result = -1;
  double delta;

  if (n <= 0 || arr == NULL) {
    goto done;
  }

  *height = arr[0].height;
  delta = abs(arr[0].temperature - temp);
  result = 0;

  for (i = 1; i < n; i++) {
    if (abs(arr[i].temperature - temp) < delta) {
      delta = abs(arr[i].temperature - temp);
      *height = arr[i].height;
      result = i;
    }
  }

done:
  return result;
}

/**
 * Calculate the weighted mean value below the bright band
 */
static int RaveVprCorrectionInternal_getWeightedMeanBelowBrightband(RaveVprCorrection_t* self, int nelems, RaveVprValue_t* refArray, double hplus, double* w_mean)
{
  int i = 0;
  double w_sum = 0.0;
  double n_sum = 0.0;
  for (i = 0; i < nelems; i++) {
    if (refArray[i].nrpoints >= self->minNumberOfObservations) { /* At least N points contributed */
      if ((double)(i+0.5)*self->profileHeight < hplus) { /* Only heights below hplus */
        w_sum = refArray[i].value * ((double)refArray[i].nrpoints);
        n_sum += ((double)refArray[i].nrpoints);
      }
    }
  }
  if (n_sum > 0) {
    *w_mean = w_sum / n_sum;
    return 1;
  }
  return 0;
}

/**
 * Returns the index of the maximum value in the array that are within the boundaries min and maxindex.
 * @param[in] self - self
 * @param[in] refArray - the reflectivity array
 * @param[in] nelems - number of elements in refArray
 * @param[in] minindex - the start index
 * @param[in] maxindex - the stop index
 * @return the index of the maximum value within the index minindex..maxindex
 */
static int RaveVprCorrectionInternal_getMaxInRange(RaveVprCorrection_t* self, RaveVprValue_t* refArray, int nelems, int minindex, int maxindex)
{
  int i = 0;
  int result = -1;

  double maxv = 0.0;
  if (minindex < 0 || minindex > nelems || maxindex < 0 || maxindex > nelems) {
    RAVE_ERROR0("Calling getMaxInRange with index out of range");
    return result;
  }
  maxv = refArray[minindex].value;
  result = minindex;
  for (i = minindex + 1; i < maxindex; i++) {
    if (refArray[minindex].value > maxv) {
      maxv = refArray[minindex].value;
      result = i;
    }
  }
  return result;
}

#define MAXNFEVAL 100000 /* Max number of evaluations */

double* RaveVprCorrection_getIdealVpr(RaveVprCorrection_t* self, PolarVolume_t* pvol, int htsize, RaveHeightTemperature_t* htarray, int* nElements)
{
  double *result = NULL, *tresult = NULL, *x = NULL, *g = NULL, *low = NULL, *up = NULL, *firstguess = NULL;
  RaveVprValue_t* refArray = NULL;
  int nelems = 0;
  int tncResult;
  int niter = 0, nfeval = 0, nrHeights = 0;
  double f;
  double mint = 0.0, maxt = 0.0;
  int minindex = 0, maxindex = 0;
  int i;

  RAVE_ASSERT((self != NULL), "self == NULL");

  refArray = RaveVprCorrection_getReflectivityArray(self, pvol, &nelems);
  if (refArray == NULL) {
    goto done;
  }

  minindex = RaveVprCorrectionInternal_getMinTemperatureFromHTArray(htsize, htarray, &mint);
  maxindex = RaveVprCorrectionInternal_getMaxTemperatureFromHTArray(htsize, htarray, &maxt);
  if (minindex < 0 || maxindex < 0) {
    RAVE_ERROR0("Could not identify min or max temperature");
    goto done;
  }

  tresult = RaveVprCorrection_getHeights(self, &nrHeights); /* We use the heights as template for the results since there should be as many offsets as heights */
  if (tresult == NULL) {
    RAVE_CRITICAL0("Failed to allocate heights array");
    goto done;
  }

  low = RAVE_CALLOC(4, sizeof(double));
  up = RAVE_CALLOC(4, sizeof(double));
  x = RAVE_MALLOC(4 * sizeof(double));
  g = RAVE_CALLOC(4, sizeof(double));
  firstguess = RAVE_MALLOC(4 * sizeof(double));

  if (low == NULL || up == NULL || x == NULL || g == NULL || firstguess == NULL) {
    RAVE_CRITICAL0("Failed to allocate memory");
    goto done;
  }
  if (mint < self->minusTemperature && maxt > self->plusTemperature /*&& temps[nTemps-1] > 0.0 TODO */) {
    double hplus = 0.0, hminus = 0.0;
    double w_mean = 0.0;
    int i1, i2;

    RaveVprCorrectionInternal_getHeightClosestToTemperature(self, htsize, htarray, self->plusTemperature, &hplus);
    RaveVprCorrectionInternal_getHeightClosestToTemperature(self, htsize, htarray, self->minusTemperature, &hminus);

    if(!RaveVprCorrectionInternal_getWeightedMeanBelowBrightband(self, nelems, refArray, hplus, &w_mean)) {
      RAVE_ERROR0("Could not calculate weighted_mean");
      goto done;
    }

    i1 = (int) floor(hplus / self->profileHeight - 0.5);
    i2 =  (int) ceil(hminus / self->profileHeight - 0.5);
    RAVE_INFO3("I1 = %d, I2 = %d, nelems = %d", i1, i2, nelems);

    if (i1 < 0 || i1 >= nelems || i2 < 0 || i2 >= nelems) {
      RAVE_ERROR3("i1 = %d, i2 = %d. One or both outside range 0 .. %d", i1, i2, nelems);
      goto done;
    }

    if (i1 + 1 < i2 - 1) {
      VprArrayWrapper wrapper;
      double a, b; /* LSQ constants */
      double mean;
      int imax = RaveVprCorrectionInternal_getMaxInRange(self, refArray, nelems, i1 + 1, i2 - 1);

      // Fit VPR below bright band
      /**
       * Invoke truncated newton rapson to find the best curve fit.
       */
      wrapper.nrelements = nelems;
      wrapper.vpr = refArray;
      wrapper.minindex = 0;
      wrapper.maxindex = imax;
      wrapper.profileHeight = self->profileHeight;
      // First guess and upper / lower bound coefficients
      x[0] = low[0] = up[0] = firstguess[0] = w_mean;
      if (refArray[imax].value >= w_mean) {
        x[1] = firstguess[0] = refArray[imax].value - w_mean;
        low[1] = x[1] * 0.5;
        up[1] = x[1] * 2.0;
      } else {
        x[1] = low[1] = up[1] = firstguess[0] = 0.0;
      }
      x[2] = low[2] = up[2] = firstguess[2] = ((double)imax + 0.5) * self->profileHeight;
      x[3] = firstguess[3] = ((double)(imax - i1)) * self->profileHeight / 2.0 + 300;
      low[3] = ((double)(imax - i1)) * self->profileHeight / 4.0;
      up[3] = ((double)(imax - i1)) * self->profileHeight / 0.5;

      RAVE_DEBUG4("LOW = %f, %f, %f, %f", low[0], low[1], low[2], low[3]);
      RAVE_DEBUG4("UP  = %f, %f, %f, %f", up[0], up[1], up[2], up[3]);
      RAVE_DEBUG4("X   = %f, %f, %f, %f", x[0], x[1], x[2], x[3]);

      tncResult = tnc(4, x, &f, NULL,
                      RaveVprCorrectionInternal_vprFitnessFunction, (void*)&wrapper,
                      low, up, NULL, NULL,
                      TNC_MSG_NONE /*TNC_MSG_NONE*/, -1, MAXNFEVAL, -1.0, 0.0,
                      0.0, 0.0, 1.0e-6, 1.0e-6, -1.0,
                      -1.0, &nfeval, &niter, NULL);

      RAVE_INFO7("TNCRESULT=%d, X AFTER = %f, %f, %f, %f, F = %f, NITER = %d", tncResult, x[0], x[1], x[2], x[3], f, niter);
      if (tncResult == TNC_LOCALMINIMUM || tncResult == TNC_FCONVERGED || tncResult == TNC_XCONVERGED) {
        int n = i1+1+imax+1;
        for (i = 0; i < n && i < nrHeights; i++) {
          double v = -9999.0;
          RaveVprCorrectionInternal_evaluate(x, &v, tresult[i]);
          tresult[i] = v;
        }
      } else {
        int n = i1+1+imax+1;
        for (i = 0; i < n && i < nrHeights; i++) {
          double v = -9999.0;
          RaveVprCorrectionInternal_evaluate(firstguess, &v, tresult[i]);
          tresult[i] = v;
        }
      }

      /* Fit highest part of vpr */
      if (!RaveVprCorrectionInternal_lsqFirstOrder(self, refArray, i2, nelems, &a, &b, &mean)) {
        RAVE_ERROR0("Failed to determine first degree polynom");
        goto done;
      }
      RAVE_INFO2("ax + b with a=%f, b=%f", a, b);

      if (a > 0) {
        /* We dont allow a positive slope */
        for (i = i2; i < nrHeights; i++) {
          tresult[i] = mean;
        }
      } else if (a > self->dzdh) {
        /* Ideal VPR */
        for (i = i2; i < nrHeights; i++) {
          tresult[i] = a * ((self->profileHeight * (double)(i+1)) / 2.0) + b;
        }
      } else {
        /* No profile is generated if slope is to low*/
        for (i = i2; i < nrHeights; i++) {
          tresult[i] = -9999.0;
        }
      }

      /* Fit vpr above bright band with a straight line */
      if ((i2+1) - (i1+imax+1) == 0) {
        RAVE_ERROR0("i2+1 == i1+imax+1 will result in division by zero");
        goto done;
      }
      a = (tresult[i2] - tresult[i1+imax+1])/(((i2+1) - (i1+imax+1))*self->profileHeight/2.0);
      b = (tresult[i2] - a*((i2+1)*self->profileHeight/2.0));
      for (i = i1+imax+2; i < i2; i++) {
        tresult[i] = a*((i+1)*self->profileHeight/2.0) + b;
      }
    }
  }

  result = tresult;
  tresult = NULL;
  *nElements = nelems;

done:
  RAVE_FREE(refArray);
  RAVE_FREE(x);
  RAVE_FREE(g);
  RAVE_FREE(low);
  RAVE_FREE(up);
  RAVE_FREE(firstguess);
  RAVE_FREE(tresult);
  return result;
}

int RaveVprCorrectionHelper_lsqFirstOrder(int nelem, double* x, double* y, double* a, double* b)
{
  int i = 0;
  double u = 0.0, v = 0.0, s = 0.0, t = 0.0;
  for (i = 0; i < nelem; i++) {
    u += x[i] * x[i];
    v += x[i];
    s += y[i] * x[i];
    t += y[i];
  }
  if ((nelem*u - v*v) == 0.0 || v == 0.0) {
    RAVE_ERROR0("Impossible to determine lsq fit");
    return 0;
  }
  *b = (t*u - s*v)/(nelem*u-v*v);
  *a = (t - (*b)*nelem)/v;
  return 1;
}

RaveHeightTemperature_t* RaveVprCorrectionHelper_readH1D(const char* filename, int* ntemps)
{
  FILE* fp = NULL;
  int i = 0, j = 0, N = 0, f = 0;
  double c;
  RaveHeightTemperature_t* ht = NULL;
  RaveHeightTemperature_t* result = NULL;

  if (filename == NULL) {
    RAVE_ERROR0("Need to specify a filename");
    return NULL;
  }

  fp = fopen(filename, "r");
  i=0;

  if (fp == NULL) {
    RAVE_ERROR1("Failed to open %s", filename);
    return NULL;
  }

  // Read header
  f = fscanf(fp,"%d %d %lf %lf %lf %lf",&j, &N, &c, &c, &c, &c);
  if (f != 6) {
    RAVE_ERROR0("Failure when reading header");
    goto done;
  }

  // Allocate 2D array
  ht = (RaveHeightTemperature_t*)RAVE_CALLOC(N, sizeof(RaveHeightTemperature_t));
  if (ht == NULL) {
    RAVE_CRITICAL0("Failed to allocate memory");
    goto done;
  }

  // Read height vector
  for (i=0;i<N;i++) {
    if (!feof(fp)) {
      f=fscanf(fp,"%lf",&c);
      ht[i].height=c;
    } else {
      RAVE_ERROR0("Reached end of file. Exiting...");
      goto done;
    }
  }

  // Read dummy data
  for (i=0;i<N;i++) {
    if (!feof(fp)) {
      f=fscanf(fp,"%lf",&c);
    } else {
      RAVE_ERROR0("Reached end of file. Exiting...");
      goto done;
    }
  }

  // Read temperature data
  for (i=0;i<N;i++) {
    if (!feof(fp)) {
      f=fscanf(fp,"%lf",&c);
      ht[i].temperature=c-273.15;
    } else {
      RAVE_ERROR0("Reached end of file. Exiting...");
      goto done;
    }
  }

  result = ht;
  *ntemps = N;
  ht = NULL;

done:
  RAVE_FREE(ht);
  if (fp != NULL) {
    fclose (fp);
  }
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
