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
 #ifndef COMPOSITE_ENGINE_FUNCTIONS_H
 #define COMPOSITE_ENGINE_FUNCTIONS_H
 #include "cartesian.h"
 #include "cartesianvolume.h"
 #include "compositearguments.h"
 #include "compositeengine.h"
 #include "rave_object.h"
 #include "rave_types.h"
 #include "rave_value.h"
 #include <strings.h>

/*@{ Composite engine functions for handling RATE */

#define DEFAULT_ZR_A 200.0

#define DEFAULT_ZR_B 1.6

/**
 * Prepares the binding with the RATE coefficients so that they can be used when generating the product.
 */
int CompositeEngineFunctions_prepareRATE(CompositeEngine_t* engine, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, int nbindings);

double CompositeEngineFunction_convertDbzToRate(CompositeEngineObjectBinding_t* binding, RaveValueType valuetype, double value, double default_zr_a, double default_zr_b);

int CompositeEngineFunctions_getRATEValueAtPosition(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue);

 /*@} End of Composite engine functions for handling RATE */

 #endif /* COMPOSITE_ENGINE_BASE_H */
 