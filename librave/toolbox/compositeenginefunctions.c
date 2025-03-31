
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
      if (binding->source != NULL) {
        RAVE_OBJECT_RELEASE(binding->value);
        RaveValue_t* value = RaveProperties_get(properties, "rave.rate.zr.coefficients");
        if (value != NULL && RaveValue_type(value) == RaveValue_Type_Hashtable) {
          RaveObjectHashTable_t* rates = RaveValue_toHashTable(value);
          if (RaveObjectHashTable_exists(rates, OdimSource_getNod(binding->source))) {
            RaveValue_t* zr = (RaveValue_t*)RaveObjectHashTable_get(rates, OdimSource_getNod(binding->source));
            if (zr != NULL && RaveValue_type(zr) == RaveValue_Type_DoubleArray) {
              double* v = NULL;
              int len = 0;
              RaveValue_getDoubleArray(zr, &v, &len);
              if (len == 2) {
                binding->value = RAVE_OBJECT_COPY(zr);
              } else {
                RAVE_ERROR1("rave.rate.zr.coefficients coefficient for %s could not be read, should be (zr_a, zr_b)", OdimSource_getNod(binding->source));
              }
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

double CompositeEngineFunction_convertDbzToRate(CompositeEngineObjectBinding_t* binding, RaveValueType valuetype, double value, double default_zr_a, double default_zr_b)
{
  double result = value;

  if (valuetype == RaveValueType_DATA) {
    double zr_a = default_zr_a, zr_b = default_zr_b;
    if (binding->value != NULL && RaveValue_type(binding->value) == RaveValue_Type_DoubleArray) {
      double* v = NULL;
      int len = 0;
      RaveValue_getDoubleArray(binding->value, &v, &len);
      if (len == 2) {
        zr_a = v[0];
        zr_b = v[1];
      }
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
