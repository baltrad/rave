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
 * Contains definitions useful when working with the composite engine.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-02-26
 */
 #ifndef COMPOSITE_ENGINE_BASE_H
 #define COMPOSITE_ENGINE_BASE_H
 #include "cartesian.h"
 #include "cartesianvolume.h"
 #include "compositearguments.h"
 #include "rave_attribute.h"
 #include "rave_object.h"
 #include "rave_types.h"
 #include "rave_value.h"
 #include "raveobject_list.h"
 #include "raveobject_hashtable.h"
 #include "projection_pipeline.h"
 #include <strings.h>
 
/**
 * Binding for associating rave objects with pipelines, sources and other miscellaneous information.
 */
typedef struct CompositeEngineObjectBinding_t {
  RaveCoreObject* object; /**< the rave object */
  ProjectionPipeline_t* pipeline; /**< the projection pipeline */
  OdimSource_t* source; /**< the source associated with the object */
  RaveValue_t* value;   /**< a rave value, can be used to cache information */
} CompositeEngineObjectBinding_t;
 
/*@{ Composite engine functions for working with composite engine object bindings */
/**
 * Creates the binding between radar objects, pipelines, sources and other values that are relevant when creating composites.
 * The order of the binding will be the same as the objects in the arguments at time the object is 
 * @param[in] arguments - the arguments (containing the radar objects)
 * @param[in] cartesian - the target composite 
 * @param[out] nobject - the number of items in the returned array
 * @param[in] sources - an OPTIONAL odim sources (MAY BE NULL). When creating binding, if possible to identify the odim source it will be attached to the binding.
 * @return the array of bindings or NULL on failure
 */
 CompositeEngineObjectBinding_t* CompositeEngineObjectBinding_createObjectBinding(CompositeArguments_t* arguments, Cartesian_t* cartesian, int* nobjects, OdimSources_t* sources);
 
 /**
 * Releases the objects and then deallocates the array
 * @param[in,out] arr - the array to release
 * @param[in] nobjects - number of items in array
 */
 void CompositeEngineObjectBinding_releaseObjectBinding(CompositeEngineObjectBinding_t** arr, int nobjects);

 /*@} End of Composite engine functions for working with composite engine object bindings */

 #endif /* COMPOSITE_ENGINE_BASE_H */
 