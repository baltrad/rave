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

/** Tutorial functionality for re-projecting Cartesian data.
 * Serves as a working example of RAVE C APIs.
 * @file
 * @author Daniel Michelson, SMHI
 * @date 2011-06-06
 */
#include "reproj.h"


void CopyMetaData(Cartesian_t* source, Cartesian_t* dest) {
  Cartesian_setDate(dest, Cartesian_getDate(source));
  Cartesian_setTime(dest, Cartesian_getTime(source));
  Cartesian_setStartDate(dest, Cartesian_getStartDate(source));
  Cartesian_setStartTime(dest, Cartesian_getStartTime(source));
  Cartesian_setEndDate(dest, Cartesian_getEndDate(source));
  Cartesian_setEndTime(dest, Cartesian_getEndTime(source));
  Cartesian_setSource(dest, Cartesian_getSource(source));
  Cartesian_setObjectType(dest, Cartesian_getObjectType(source));
  Cartesian_setProduct(dest, Cartesian_getProduct(source));
  Cartesian_setQuantity(dest, Cartesian_getQuantity(source));

  Cartesian_setNodata(dest, Cartesian_getNodata(source));
  Cartesian_setUndetect(dest, Cartesian_getUndetect(source));
  Cartesian_setOffset(dest, Cartesian_getOffset(source));
  Cartesian_setGain(dest, Cartesian_getGain(source));
}


Cartesian_t* reproj(Cartesian_t* inobj, const char* areaid) {
  Cartesian_t* result = NULL;
  ProjectionRegistry_t* preg = NULL;
  AreaRegistry_t* areg = NULL;
  Projection_t* iproj = NULL;
  Projection_t* oproj = NULL;
  Area_t* oarea = NULL;
  RaveDataType dt;
  RaveValueType rvt;

  long ix=0, iy=0, ixsize=0, iysize=0;  /* input */
  long ox=0, oy=0, oxsize=0, oysize=0;  /* output */
  double herex=0.0, herey=0.0, therex=0.0, therey=0.0;
  double llX=0.0, llY=0.0, urX=0.0, urY=0.0;
  double result_val;

  /* Determine input geometry */
  Cartesian_getAreaExtent(inobj, &llX, &llY, &urX, &urY);
  dt = Cartesian_getDataType(inobj);
  iproj = Cartesian_getProjection(inobj);
  ixsize = Cartesian_getXSize(inobj);
  iysize = Cartesian_getYSize(inobj);

  /* Initialize output geometry and object.
   * File strings for registries are hardwired here only for simplicity. */
  preg = ProjectionRegistry_load("/opt/rave/config/projections.xml");
  areg = AreaRegistry_load("/opt/rave/config/polish_areas.xml", preg);
  oarea = AreaRegistry_getByName(areg, areaid);
  result = RAVE_OBJECT_NEW(&Cartesian_TYPE);
  if (!Cartesian_init(result, oarea, dt)) {
    printf("Failed to initialize output object. Bailing ...\n");
  }
  oproj = Area_getProjection(oarea);

  CopyMetaData(inobj, result);

  oxsize = Cartesian_getXSize(result);
  oysize = Cartesian_getYSize(result);

  if (!(Cartesian_isTransformable(inobj)) && (Cartesian_isTransformable(inobj))) {
    printf("objects are not transformable\n");
  }

  for (oy=0;oy<oysize;oy++) {
    herey = Cartesian_getLocationY(result, oy);

    for (ox=0;ox<oxsize;ox++) {
      herex = Cartesian_getLocationX(result, ox);

      /* Where are we in output space? */
      if (!Projection_transformx(iproj, oproj, herex, herey, 0.0, &therex, &therey, NULL)) {
        printf("Failed to transform\n");
        exit(0);
      }

      /* Get pixel indices of input image */
      ix = Cartesian_getIndexX(inobj, therex);
      iy = Cartesian_getIndexY(inobj, therey);

      /* If we're in bounds, get source value and plug
       * it into the destination image. */
      if ((ix>=0) && (iy>=0) && (ix<ixsize) && (iy<iysize)) {
        rvt = Cartesian_getValue(inobj, ix, iy, &result_val);

        if (!Cartesian_setValue(result, ox, oy, result_val)) {
          printf("Failed to set output value\n");
        }
      }
    }
  }
  RAVE_OBJECT_RELEASE(preg);
  RAVE_OBJECT_RELEASE(areg);
  RAVE_OBJECT_RELEASE(iproj);
  RAVE_OBJECT_RELEASE(oproj);
  RAVE_OBJECT_RELEASE(oarea);
  return result;
}
