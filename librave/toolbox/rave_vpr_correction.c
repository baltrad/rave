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
};

/*@{ Private functions */
/**
 * Constructor
 */
static int RaveVprCorrection_constructor(RaveCoreObject* obj)
{
  /*RaveVprCorrection_t* self = (RaveVprCorrection_t*)obj;*/
  return 1;
}

/**
 * Copy constructor
 */
static int RaveVprCorrection_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  /*
  RaveVprCorrection_t* self = (RaveVprCorrection_t*)obj;
  RaveVprCorrection_t* src = (RaveVprCorrection_t*)obj;
  */
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
int RaveVprCorrection_separateSC(RaveVprCorrection_t* self, PolarVolume_t* pvol)
{
#ifdef KALLE
  int i = 0, j = 0, k = 0;
  int nbins = 0, nscans = 0;
  RaveObjectList_t* distanceVectors = NULL;
  RaveField_t *distanceField = NULL, *heightField = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");


  PolarVolume_sortByElevations(pvol, 1); /* Want elevations to be sorted in ascending order */

  distanceField = PolarVolume_getDistanceField(pvol);
  heightField = PolarVolume_getHeightField(pvol);

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
