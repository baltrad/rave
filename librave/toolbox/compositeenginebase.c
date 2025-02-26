
/* --------------------------------------------------------------------
Copyright (C) 2025 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Provides some utilities
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-31
 */
#include "compositeenginebase.h"
#include "composite_utils.h"
#include "compositearguments.h"
#include "polarvolume.h"
#include "rave_properties.h"
#include "raveobject_hashtable.h"
#include "raveobject_list.h"
#include "rave_types.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_datetime.h"
#include "projection_pipeline.h"
#include <string.h>
#include "rave_field.h"
#include <float.h>
#include <stdio.h>
#include <math.h>

CompositeEngineObjectBinding_t* CompositeEngineObjectBinding_createObjectBinding(CompositeArguments_t* arguments, Cartesian_t* cartesian, int* nobjects, OdimSources_t* sources)
{
  CompositeEngineObjectBinding_t* result = NULL;
  Projection_t* projection = NULL;

  int i = 0, nlen = 0;

  if (arguments == NULL || cartesian == NULL || nobjects == NULL) {
    RAVE_ERROR0("Must provide arguments, cartesian and nobjects must be provided");
    return NULL;
  }
  projection = Cartesian_getProjection(cartesian);
  if (projection == NULL) {
    RAVE_ERROR0("Cartesian must have a projection");
    return NULL;
  }

  nlen = CompositeArguments_getNumberOfObjects(arguments);
  result = RAVE_MALLOC(sizeof(CompositeEngineObjectBinding_t)* nlen);
  if (result == NULL) {
    RAVE_ERROR0("Failed to allocate array for object bindings");
    goto fail;
  }
  memset(result, 0, sizeof(CompositeEngineObjectBinding_t)*nlen);
  for (i = 0; i < nlen; i++) {
    RaveCoreObject* object = CompositeArguments_getObject(arguments, i);
    if (object != NULL) {
      Projection_t* objproj = CompositeUtils_getProjection(object);
      ProjectionPipeline_t* pipeline = NULL;
      OdimSource_t* source = NULL;
      char strsrc[512];

      if (objproj == NULL) {
        RAVE_ERROR0("Object does not have a projection");
        RAVE_OBJECT_RELEASE(object);
        goto fail;
      }
      pipeline = ProjectionPipeline_createPipeline(projection, objproj);
      if (pipeline == NULL) {
        RAVE_ERROR0("Failed to create pipeline");
        RAVE_OBJECT_RELEASE(object);
        RAVE_OBJECT_RELEASE(objproj);
        goto fail;
      }
      if (sources != NULL && CompositeUtils_getObjectSource(object, strsrc, 512)) {
        source = OdimSources_identify(sources, (const char*)strsrc);
        if (source != NULL) {
          RAVE_INFO2("Identified source string: %s -> NOD:%s", strsrc, OdimSource_getNod(source));
        } else {
          RAVE_WARNING1("Could not identify source string: %s", strsrc);
        }
      }

      result[i].object = RAVE_OBJECT_COPY(object);
      result[i].pipeline = RAVE_OBJECT_COPY(pipeline);
      result[i].source = RAVE_OBJECT_COPY(source);
      RAVE_OBJECT_RELEASE(pipeline);
      RAVE_OBJECT_RELEASE(objproj);
      RAVE_OBJECT_RELEASE(source);
    }
    RAVE_OBJECT_RELEASE(object);
  }
  *nobjects = nlen;
  RAVE_OBJECT_RELEASE(projection);
  return result;
fail:
  RAVE_OBJECT_RELEASE(projection);
  CompositeEngineObjectBinding_releaseObjectBinding(&result, nlen);
  return NULL;
}

void CompositeEngineObjectBinding_releaseObjectBinding(CompositeEngineObjectBinding_t** arr, int nobjects)
{
  int i = 0;
  if (arr == NULL) {
    RAVE_ERROR0("Nothing to release");
    return;
  }
  if (*arr != NULL) {
    for (i = 0; i < nobjects; i++) {
      RAVE_OBJECT_RELEASE((*arr)[i].object);
      RAVE_OBJECT_RELEASE((*arr)[i].pipeline);
      RAVE_OBJECT_RELEASE((*arr)[i].source)
      RAVE_OBJECT_RELEASE((*arr)[i].value)
    }
    RAVE_FREE(*arr);
    *arr = NULL;
  }
}
