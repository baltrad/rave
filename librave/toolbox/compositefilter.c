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
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-27
 */
#include "compositefilter.h"
#include "compositearguments.h"
#include "composite_utils.h"
#include "rave_attribute.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_list.h"
#include "rave_object.h"
#include "rave_types.h"
#include "raveobject_hashtable.h"
#include "raveobject_list.h"
#include <string.h>
#include <strings.h>
#include <stdio.h>

/**
 * Represents the area
 */
struct _CompositeFilter_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveList_t* products; /**< the products */
  RaveList_t* quantities; /**< the quantities */
  RaveList_t* interpolation_methods; /**< the interpolation methods */
};

/*@{ Private functions */

static void CompositeFilterInternal_clearRaveList(RaveList_t* inlist)
{
  while (RaveList_size(inlist) > 0) {
    char* tmps = (char*)RaveList_removeLast(inlist);
    RAVE_FREE(tmps);
  }
}

/**
 * Clears and sets the list with the content of the array.
 * @param[in] inlist - the list to be set with new array content
 * @param[in] arr - an array of strings
 * @param[in] narr - number of strings in array
 */
static int CompositeFilterInternal_setRaveListWithArray(RaveList_t* inlist, const char** arr, int narr)
{
  CompositeFilterInternal_clearRaveList(inlist);
  if (arr != NULL && narr > 0) {
    int i = 0;
    for (i = 0; i < narr; i++) {
      char* tmps = RAVE_STRDUP(arr[i]);
      if (tmps == NULL || !RaveList_add(inlist, tmps)) {
        RAVE_FREE(tmps);
        RAVE_ERROR0("Failed to add string");
        goto fail;
      }
    }
  }
  return 1;
fail:
  CompositeFilterInternal_clearRaveList(inlist);
  return 0;
}

static int CompositeFilterInternal_copyRaveListContent(RaveList_t* inlist, RaveList_t* srclist)
{
  int i = 0, nsrc = 0;
  CompositeFilterInternal_clearRaveList(inlist);
  nsrc = RaveList_size(srclist);
  for (i = 0; i < nsrc; i++) {
    char* src = (char*)RaveList_get(srclist, i);
    char* tmps = RAVE_STRDUP(src);
    if (tmps == NULL || !RaveList_add(inlist, tmps)) {
      RAVE_FREE(tmps);
      RAVE_ERROR0("Failed to duplicate string");
      goto fail;
    }
  }

  return 1;
fail:
  CompositeFilterInternal_clearRaveList(inlist);
  return 0;
}

static int CompositeFilterInternal_doesStringExistInList(const char* str, RaveList_t* inlist)
{
  int i = 0, nlen = RaveList_size(inlist);
  for (i = 0; i < nlen; i++) {
    if (strcasecmp(str, (const char*)RaveList_get(inlist, i))==0) {
      return 1;
    }
  }
  return 0;
}

/**
 * Constructor.
 */
static int CompositeFilter_constructor(RaveCoreObject* obj)
{
  CompositeFilter_t* this = (CompositeFilter_t*)obj;
  this->products = RAVE_OBJECT_NEW(&RaveList_TYPE);
  this->quantities = RAVE_OBJECT_NEW(&RaveList_TYPE);
  this->interpolation_methods = RAVE_OBJECT_NEW(&RaveList_TYPE);

  if (this->products == NULL || this->quantities == NULL || this->interpolation_methods == NULL) {
    goto fail;
  }
  return 1;
fail:
  if (this->products != NULL) {
    RaveList_freeAndDestroy(&this->products);
  }
  if (this->quantities != NULL) {
    RaveList_freeAndDestroy(&this->quantities);
  }
  if (this->interpolation_methods != NULL) {
    RaveList_freeAndDestroy(&this->interpolation_methods);
  }  return 0;
}

/**
 * Copy constructor
 */
static int CompositeFilter_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  CompositeFilter_t* this = (CompositeFilter_t*)obj;
  CompositeFilter_t* src = (CompositeFilter_t*)srcobj;
  this->products = CompositeUtils_cloneRaveListStrings(src->products);
  this->quantities = CompositeUtils_cloneRaveListStrings(src->quantities);
  this->interpolation_methods = CompositeUtils_cloneRaveListStrings(src->interpolation_methods);

  if (this->products == NULL || this->quantities == NULL || this->interpolation_methods == NULL) {
    RAVE_ERROR0("Failed to clone filter");
    goto fail;
  }

  return 1;
fail:
  if (this->products != NULL) {
    RaveList_freeAndDestroy(&this->products);
  }
  if (this->quantities != NULL) {
    RaveList_freeAndDestroy(&this->quantities);
  }
  if (this->interpolation_methods != NULL) {
    RaveList_freeAndDestroy(&this->interpolation_methods);
  }
  return 0;
}

/**
 * Destroys the argument class
 * @param[in] obj - the the CompositeFilter_t instance
 */
static void CompositeFilter_destructor(RaveCoreObject* obj)
{
  CompositeFilter_t* this = (CompositeFilter_t*)obj;
  if (this->products != NULL) {
    RaveList_freeAndDestroy(&this->products);
  }
  if (this->quantities != NULL) {
    RaveList_freeAndDestroy(&this->quantities);
  }
  if (this->interpolation_methods != NULL) {
    RaveList_freeAndDestroy(&this->interpolation_methods);
  }
}
/*@} End of Private functions */

/*@{ Interface functions */
int CompositeFilter_setProductsArray(CompositeFilter_t* filter, const char* products[], int nrproducts)
{
  RAVE_ASSERT((filter != NULL), "args == NULL");
  return CompositeFilterInternal_setRaveListWithArray(filter->products, products, nrproducts);
}

int CompositeFilter_setProductsList(CompositeFilter_t* filter, RaveList_t* products)
{
  RAVE_ASSERT((filter != NULL), "args == NULL");
  return CompositeFilterInternal_copyRaveListContent(filter->products, products);
}

int CompositeFilter_getProductCount(CompositeFilter_t* filter)
{
  RAVE_ASSERT((filter != NULL), "filter == NULL");
  return RaveList_size(filter->products);
}

const char* CompositeFilter_getProduct(CompositeFilter_t* filter, int index)
{
  RAVE_ASSERT((filter != NULL), "filter == NULL");
  return RaveList_get(filter->products, index);
}

int CompositeFilter_setQuantitiesArray(CompositeFilter_t* filter, const char* quantities[], int nrquantities)
{
  RAVE_ASSERT((filter != NULL), "args == NULL");
  return CompositeFilterInternal_setRaveListWithArray(filter->quantities, quantities, nrquantities);
}

int CompositeFilter_setQuantitiesList(CompositeFilter_t* filter, RaveList_t* quantities)
{
  RAVE_ASSERT((filter != NULL), "args == NULL");
  return CompositeFilterInternal_copyRaveListContent(filter->quantities, quantities);
}

int CompositeFilter_getQuantityCount(CompositeFilter_t* filter)
{
  RAVE_ASSERT((filter != NULL), "filter == NULL");
  return RaveList_size(filter->quantities);
}

const char* CompositeFilter_getQuantity(CompositeFilter_t* filter, int index)
{
  RAVE_ASSERT((filter != NULL), "filter == NULL");
  return RaveList_get(filter->quantities, index);
}

int CompositeFilter_setInterpolationMethodsArray(CompositeFilter_t* filter, const char* methods[], int nrmethods)
{
  RAVE_ASSERT((filter != NULL), "args == NULL");
  return CompositeFilterInternal_setRaveListWithArray(filter->interpolation_methods, methods, nrmethods);
}

int CompositeFilter_setInterpolationMethodsList(CompositeFilter_t* filter, RaveList_t* methods)
{
  RAVE_ASSERT((filter != NULL), "args == NULL");
  return CompositeFilterInternal_copyRaveListContent(filter->interpolation_methods, methods);
}

int CompositeFilter_getInterpolationMethodCount(CompositeFilter_t* filter)
{
  RAVE_ASSERT((filter != NULL), "filter == NULL");
  return RaveList_size(filter->interpolation_methods);
}

const char* CompositeFilter_getInterpolationMethod(CompositeFilter_t* filter, int index)
{
  RAVE_ASSERT((filter != NULL), "filter == NULL");
  return RaveList_get(filter->interpolation_methods, index);
}

int CompositeFilter_match(CompositeFilter_t* filter, CompositeArguments_t* arguments)
{
  const char* product;
  int i = 0, nlen = 0;
  int result = 0;
  RAVE_ASSERT((filter != NULL), "filter == NULL");
  if (arguments == NULL) {
    RAVE_ERROR0("Must provide arguments");
    return 0;
  }

  result = 1; /** Always assume that we have a match, then if set it to False on failure. */

  if (result && RaveList_size(filter->products) > 0) {
    product = CompositeArguments_getProduct(arguments);
    if (product != NULL) {
      if (!CompositeFilterInternal_doesStringExistInList(product, filter->products)) {
        result = 0;
      }
    } else {
      result = 0;
    }
  }

  if (result && RaveList_size(filter->quantities) > 0) {
    nlen = CompositeArguments_getParameterCount(arguments);
    for (i = 0; result && i < nlen; i++) {
      const char* param = CompositeArguments_getParameterAtIndex(arguments, i, NULL, NULL, NULL, NULL, NULL);
      if (!CompositeFilterInternal_doesStringExistInList(param, filter->quantities)) {
        result = 0;
      }
    }
  }

  if (result && RaveList_size(filter->interpolation_methods) > 0) {
    RaveAttribute_t* attr = CompositeArguments_getArgument(arguments, "interpolation_method");
    if (attr != NULL && RaveAttribute_getFormat(attr)==RaveAttribute_Format_String) {
      char* value = NULL;
      RaveAttribute_getString(attr, &value);
      if (value == NULL || !CompositeFilterInternal_doesStringExistInList((const char*)value, filter->interpolation_methods)) {
        result = 0;
      }
    } else {
      result = 0;
    }
    RAVE_OBJECT_RELEASE(attr);
  }

  return result;
}

/*@} End of Interface functions */

RaveCoreObjectType CompositeFilter_TYPE = {
    "CompositeFilter",
    sizeof(CompositeFilter_t),
    CompositeFilter_constructor,
    CompositeFilter_destructor,
    CompositeFilter_copyconstructor
};
