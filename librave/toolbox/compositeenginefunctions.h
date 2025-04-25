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

/**
 * The Marshall - Palmer A in the ZR-relationship
 */
#define DEFAULT_ZR_A 200.0

/**
 * The Marshall - Palmer b in the ZR-relationship
 */
 #define DEFAULT_ZR_B 1.6

/**
 * Prepares the binding with the RATE coefficients so that they can be used when generating the product.
 * @param[in] engine - the engine associated with these bindings
 * @param[in] arguments - the compositing arguments
 * @param[in] bindings - the bindings
 * @param[in] nbindings - number of bindings
 * @return 1 on success otherwise 0
 */
int CompositeEngineFunctions_prepareRATE(CompositeEngine_t* engine, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* bindings, int nbindings);

/**
 * Updates the how/product_parameters - group for the cartesian parameter "RATE" if found.
 * @param[in] arguments - the compositing arguments
 * @param[in] cartesian - the cartesian product
 * @param[in] bindings - the bindings
 * @param[in] nbindings - number of bindings
 * @return 1 on success otherwise 0
 */
int CompositeEngineFunctions_updateRATECoefficients(CompositeArguments_t* arguments, Cartesian_t* cartesian, CompositeEngineObjectBinding_t* bindings, int nbindings);

/**
 * Each binding has got a member called value which is of type \ref RaveValue_t. This contains a zr-relationship for RATE products.
 * @param[in] binding - the binding where the value contains the individual ZR coefficients
 * @param[in] valuetype - the type of value (basically, if it is RaveValueType_DATA, the returned value will be the RR )
 * @param[in] value - the dbz
 * @param[in] default_zr_a  - the ZR A coefficient (will only be used if there is no binding value ZR coefficients)
 * @param[in] default_zr_b  - the ZR b coefficient (will only be used if there is no binding value ZR coefficients)
 * @return the rain rate or the original value
 */
double CompositeEngineFunction_convertDbzToRate(CompositeEngineObjectBinding_t* binding, RaveValueType valuetype, double value, double default_zr_a, double default_zr_b);

/**
 * Gets the DBZH (quantity) value at the position and converts it to rain rate if possible which is returned in ovalue
 * @param[in] engine - the engine
 * @param[in] extradata - NOT USED
 * @param[in] arguments - compositing arguments
 * @param[in] binding - the polar object binding
 * @param[in] quantity - NOT USED, it will always assume DBZH
 * @param[in] navinfo - the navigation for data from where to pick the value
 const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue);
 * @param[in] qiFieldName - The quality field value if a qi value also should be returned
 * @param[out] otype - the type of data at specified position
 * @param[out] ovalue - the rate
 * @param[out] qivalue - the qi value if requested
 * @return 1 on succeess otherwise 0
 */
int CompositeEngineFunctions_getRATEValueAtPosition(CompositeEngine_t* engine, void* extradata, CompositeArguments_t* arguments, CompositeEngineObjectBinding_t* binding, const char* quantity, PolarNavigationInfo* navinfo, const char* qiFieldName, RaveValueType* otype, double* ovalue, double* qivalue);

 /*@} End of Composite engine functions for handling RATE */

 #endif /* COMPOSITE_ENGINE_BASE_H */
 