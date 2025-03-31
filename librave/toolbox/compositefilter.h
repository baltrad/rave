/* --------------------------------------------------------------------
Copyright (C) 2024 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * A filter for matching composite arguments.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-27
 */
#ifndef COMPOSITE_FILTER_H
#define COMPOSITE_FILTER_H
#include "compositearguments.h"
#include "rave_proj.h"
#include "rave_object.h"
#include "rave_attribute.h"
#include "raveobject_list.h"

/**
 * Defines a Geographical Area
 */
typedef struct _CompositeFilter_t CompositeFilter_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType CompositeFilter_TYPE;

/**
 * Sets the products that this filter should match
 * @param[in] filter - self
 * @param[in] products - an array of product strings
 * @param[in] nrproducts - the number of products in array
 * @return 1 on success, otherwise 0
 */
int CompositeFilter_setProductsArray(CompositeFilter_t* filter, const char* products[], int nrproducts);

/**
 * Sets the products that this filter should match
 * @param[in] filter - self
 * @param[in] products - the list of strings
 * @return 1 on success, otherwise 0
 */
int CompositeFilter_setProductsList(CompositeFilter_t* filter, RaveList_t* products);

/**
 * Returns the number of products set in filter
 * @param[in] filter - self
 * @return the number of products
 */
int CompositeFilter_getProductCount(CompositeFilter_t* filter);

/**
 * Returns the product string at specified position
 * @param[in] filter - self
 * @param[in] index - the index
 * @return the string at specified index
 */
const char* CompositeFilter_getProduct(CompositeFilter_t* filter, int index);

/**
 * Sets the quantities that this filter should match
 * @param[in] filter - self
 * @param[in] quantities - an array of quantities
 * @param[in] nrquantities - the number of quantities in array
 * @return 1 on success, otherwise 0
 */
int CompositeFilter_setQuantitiesArray(CompositeFilter_t* filter, const char* quantities[], int nrquantities);

/**
 * Sets the quantities that this filter should match
 * @param[in] filter - self
 * @param[in] quantities - a list of quantities
 * @return 1 on success, otherwise 0
 */
int CompositeFilter_setQuantitiesList(CompositeFilter_t* filter, RaveList_t* quantities);

/**
 * Returns the number of quantities set in filter
 * @param[in] filter - self
 * @return the number of quantities
 */
int CompositeFilter_getQuantityCount(CompositeFilter_t* filter);

/**
 * Returns the quantity string at specified position
 * @param[in] filter - self
 * @param[in] index - the index
 * @return the string at specified index
 */
const char* CompositeFilter_getQuantity(CompositeFilter_t* filter, int index);

/**
 * Sets the interpolation methods that this filter should match
 * @param[in] filter - self
 * @param[in] methods - an array of interpolation methods
 * @param[in] nrmethods - the number of interpolation methods in array
 * @return 1 on success, otherwise 0
 */
int CompositeFilter_setInterpolationMethodsArray(CompositeFilter_t* filter, const char* methods[], int nrmethods);

/**
 * Sets the interpolation methods that this filter should match
 * @param[in] filter - self
 * @param[in] methods - a list of interpolation methods
 * @return 1 on success, otherwise 0
 */
int CompositeFilter_setInterpolationMethodsList(CompositeFilter_t* filter, RaveList_t* methods);

/**
 * Returns the number of interpolation methods set in filter
 * @param[in] filter - self
 * @return the number of interpolation methods
 */
int CompositeFilter_getInterpolationMethodCount(CompositeFilter_t* filter);

/**
 * Returns the interpolation method at specified position
 * @param[in] filter - self
 * @param[in] index - the index
 * @return the string at specified index
 */
const char* CompositeFilter_getInterpolationMethod(CompositeFilter_t* filter, int index);

/**
 * Matches the filter against the arguments.
 * @param[in] filter - self
 * @param[in] arguments - the arguments
 * @return 1 if it is matching, otherwise 0
 */
int CompositeFilter_match(CompositeFilter_t* filter, CompositeArguments_t* arguments);

#endif /* COMPOSITE_FILTER_H */
