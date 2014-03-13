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
  double scHeightLimit;
  double scDistanceLimit;
};

/*@{ Private functions */
/**
 * Constructor
 */
static int RaveVprCorrection_constructor(RaveCoreObject* obj)
{
  RaveVprCorrection_t* self = (RaveVprCorrection_t*)obj;
  self->minZLimitStratiformCloud = -20.0;
  self->scHeightLimit = 1.3e3;
  self->scDistanceLimit = 25e3;
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
  self->scHeightLimit = src->scHeightLimit;
  self->scDistanceLimit = src->scDistanceLimit;
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

void RaveVprCorrection_setSCHeightLimit(RaveVprCorrection_t* self, double limit)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->scHeightLimit = limit;
}

double RaveVprCorrection_getSCHeightLimit(RaveVprCorrection_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->scHeightLimit;
}

void RaveVprCorrection_setSCDistanceLimit(RaveVprCorrection_t* self, double limit)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->scDistanceLimit = limit;
}

double RaveVprCorrection_getSCDistanceLimit(RaveVprCorrection_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->scDistanceLimit;
}

PolarVolume_t* RaveVprCorrection_separateSC(RaveVprCorrection_t* self, PolarVolume_t* pvol)
{
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
  return 0;
}

/*@} End of Interface functions */

RaveCoreObjectType RaveVprCorrection_TYPE = {
    "RaveVprCorrection",
    sizeof(RaveVprCorrection_t),
    RaveVprCorrection_constructor,
    RaveVprCorrection_destructor,
    RaveVprCorrection_copyconstructor
};