
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
 * Useful functions when working with the composite engine
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-02-26
 */
#include "compositeenginefunctions.h"
#include "composite_utils.h"
#include "compositearguments.h"
#include "polarvolume.h"
#include "rave_attribute.h"
#include "rave_object.h"
#include "rave_properties.h"
#include "rave_value.h"
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
#include "raveutil.h"

int CompositeEngineFunctions_prepareRATE(CompositeEngine_t* engine, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, int nbindings)
{
  RaveProperties_t* properties = NULL;
  int i = 0;

  properties = CompositeEngine_getProperties(engine);
  if (properties != NULL) {
    for (i = 0; i < nbindings; i++) {
      if (binding[i].source != NULL) {
        RAVE_OBJECT_RELEASE(binding[i].value);
        RaveValue_t* value = RaveProperties_get(properties, "rave.rate.zr.coefficients");
        if (value != NULL && RaveValue_type(value) == RaveValue_Type_Hashtable) {
          RaveObjectHashTable_t* rates = RaveValue_toHashTable(value);
          if (RaveObjectHashTable_exists(rates, OdimSource_getNod(binding[i].source))) {
            RaveValue_t* zr = (RaveValue_t*)RaveObjectHashTable_get(rates, OdimSource_getNod(binding[i].source));
            if (zr != NULL && RaveValue_isDoubleArray(zr)) {
              double* v = NULL;
              int len = 0;
              RaveValue_getDoubleArray(zr, &v, &len);
              if (len == 2) {
                binding[i].value = RAVE_OBJECT_COPY(zr);
              } else {
                RAVE_ERROR1("rave.rate.zr.coefficients coefficient for %s could not be read, should be (zr_a, zr_b)", OdimSource_getNod(binding[i].source));
              }
              RAVE_FREE(v);
            }
            RAVE_OBJECT_RELEASE(zr);
          }
          RAVE_OBJECT_RELEASE(rates);
        }
        RAVE_OBJECT_RELEASE(value);
      }
    }
  }
  RAVE_OBJECT_RELEASE(properties);
  return 1;
}

/**
 * If there is a rave value hash in hashtable for source-nod, this hash is returned. Otherwise a hashtable is created as a rave value, added and then returned.
 */
static RaveValue_t* CompositeEngineFunctionsInternal_ensureRaveValueHashExistsForNod(OdimSource_t* source, RaveValue_t* jsonObject)
{
  RaveValue_t *result=NULL;
  const char* nod = OdimSource_getNod(source);
  if (RaveValueHash_exists(jsonObject, nod)) {
    RaveValue_t* obj = RaveValueHash_get(jsonObject, nod);
    if (obj != NULL && RaveValue_type(obj) == RaveValue_Type_Hashtable) {
      result = RAVE_OBJECT_COPY(obj);
    } else {
      RAVE_ERROR1("Bad rave value for %s", nod);
    }
    RAVE_OBJECT_RELEASE(obj);
  } else {
    RaveValue_t* obj = RaveValue_createHashTable(NULL);
    if (obj == NULL || !RaveValueHash_put(jsonObject, nod, obj)) {
      RAVE_ERROR0("Failed to create rave value hash");
      RAVE_OBJECT_RELEASE(obj);
      goto fail;
    }
    result = RAVE_OBJECT_COPY(obj);
    RAVE_OBJECT_RELEASE(obj);
  }

fail:
  return result;
}

int CompositeEngineFunctions_updateRATECoefficients(CompositeArguments_t* arguments, Cartesian_t* cartesian, CompositeEngineObjectBinding_t* bindings, int nbindings)
{
#ifdef RAVE_JSON_SUPPORTED
  CartesianParam_t* param = NULL;
  RaveAttribute_t* attr = NULL;
  RaveValue_t *jsonhash = NULL, *nodhash = NULL;
  char* jsonstr = NULL;
  int result = 0;

  if (cartesian == NULL) {
    return 0;
  }
  param = Cartesian_getParameter(cartesian, "RATE");
  if (param != NULL) {
    int i = 0;
    if (!CartesianParam_hasAttribute(param, "how/product_parameters/json")) {
      jsonhash = RaveValue_createHashTable(NULL);
      if (jsonhash == NULL) {
        RAVE_ERROR0("Failed to create json hash");
        goto fail;
      }
    } else {
      char* tmps = NULL;
      attr = CartesianParam_getAttribute(param, "how/product_parameters/json");
      if (attr == NULL || RaveAttribute_getFormat(attr) != RaveAttribute_Format_String) {
        RAVE_ERROR0("how/product_parameters/json is not a json string");
        goto fail;
      }

      RaveAttribute_getString(attr, &tmps);
      jsonhash = RaveValue_fromJSON(tmps);
      if (jsonhash == NULL || RaveValue_type(jsonhash) != RaveValue_Type_Hashtable) {
        RAVE_ERROR0("how/product_parameters/json is not a json object");
        goto fail;
      }
      RAVE_OBJECT_RELEASE(attr);
    }
    
    for (i = 0; i < nbindings; i++) {
      if (bindings[i].source != NULL && bindings[i].value != NULL) {
        const char* nod = OdimSource_getNod(bindings[i].source);
        nodhash = CompositeEngineFunctionsInternal_ensureRaveValueHashExistsForNod(bindings[i].source, jsonhash);
        if (nodhash == NULL) {
          RAVE_ERROR0("NO HASH, aborting");
          goto fail;
        }

        if (RaveValue_isDoubleArray(bindings[i].value) && RaveValueList_size(bindings[i].value) == 2) {
          if (!RaveValueHash_put(nodhash, "zr", bindings[i].value)) {
            RAVE_ERROR1("Failed to add zr-relation to hash for %s", nod);
          }
        }
        RAVE_OBJECT_RELEASE(nodhash);
      } else if (bindings[i].source != NULL) {
        const char* nod = OdimSource_getNod(bindings[i].source);
        RaveValue_t* defaultZr = NULL;
        double zrarr[2] = {DEFAULT_ZR_A, DEFAULT_ZR_B};

        nodhash = CompositeEngineFunctionsInternal_ensureRaveValueHashExistsForNod(bindings[i].source, jsonhash);
        if (nodhash == NULL) {
          RAVE_ERROR0("NO HASH, aborting");
          goto fail;
        }
        defaultZr = RaveValue_createDoubleArray(zrarr, 2);
        if (defaultZr == NULL || !RaveValueHash_put(nodhash, "zr", defaultZr)) {
          RAVE_ERROR1("Could not add default ZR to nod: %s", nod);
          RAVE_OBJECT_RELEASE(defaultZr);
          goto fail;
        }
        RAVE_OBJECT_RELEASE(nodhash);
        RAVE_OBJECT_RELEASE(defaultZr);        
      }
    }

    jsonstr = RaveValue_toJSON(jsonhash);
    if (jsonstr == NULL) {
      RAVE_ERROR0("Failed to create json hash");
      goto fail;
    }
    attr = RaveAttributeHelp_createString("how/product_parameters/json", jsonstr);
    if (attr == NULL || !CartesianParam_addAttribute(param, attr)) {
      RAVE_ERROR0("Failed to add json attribute to parameter");
      goto fail;
    }
  }
  result = 1;
fail:
  RAVE_OBJECT_RELEASE(param);
  RAVE_OBJECT_RELEASE(attr);
  RAVE_OBJECT_RELEASE(jsonhash);
  RAVE_OBJECT_RELEASE(nodhash);
  RAVE_FREE(jsonstr);

  return result;
#else
  RAVE_INFO0("Rave not built with json-c support. Will not update product parameters");
  return 1;
#endif  
}

double CompositeEngineFunction_convertDbzToRate(CompositeEngineObjectBinding_t* binding, RaveValueType valuetype, double value, double default_zr_a, double default_zr_b)
{
  double result = value;

  if (valuetype == RaveValueType_DATA) {
    double zr_a = default_zr_a, zr_b = default_zr_b;
    if (binding->value != NULL && RaveValue_isDoubleArray(binding->value)) {
      double* v = NULL;
      int len = 0;
      RaveValue_getDoubleArray(binding->value, &v, &len);
      if (len == 2) {
        zr_a = v[0];
        zr_b = v[1];
      }
      RAVE_FREE(v);
    }
    double rr = dBZ2R(value, zr_a, zr_b);
    result = rr;
  }
  return result;
}

int CompositeEngineFunctions_getRATEValueAtPosition(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue)
{
  
  int result = 0;
  /* We calculate RATE using DBZH */
  result = CompositeEngineUtility_getPolarValueAtPosition(engine, extradata, arguments, binding, "DBZH", navinfo, qiFieldName, otype, ovalue, qivalue);

  *ovalue = CompositeEngineFunction_convertDbzToRate(binding, *otype, *ovalue, DEFAULT_ZR_A, DEFAULT_ZR_B);

  return result;
}
