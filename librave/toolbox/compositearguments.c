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
 * The arguments that should be passed on to a composite generator
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-10-14
 */
#include "compositearguments.h"
#include "composite_utils.h"
#include "rave_attribute.h"
#include "rave_datetime.h"
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
struct _CompositeArguments_t {
  RAVE_OBJECT_HEAD /** Always on top */
  char* product; /**< the compositing product / method */
  Rave_CompositingProduct compositingProduct; /**< the compositing product, tightly connected with product */
  Area_t* area; /**< the area */
  RaveDateTime_t* datetime;  /**< the date and time */
  double height; /**< the height when generating pcapppi, cappi, pmax default 1000 */
  double elangle; /**< the elevation angle when generating ppi, default 0.0 */
  double range;  /*< the range when generating pmax, default = 500000 meters */
  char* strategy; /**< can be used to identify registered plugin / factories */
  RaveObjectHashTable_t* arguments; /**< the method specific arguments */
  RaveList_t* parameters; /**< the parameters to generate */
  RaveObjectList_t* objects; /**< the rave objects to be used in the compositing */
  RaveList_t* qualityflags; /**< the quality flags that should be generated */
  RaveObjectHashTable_t* radarIndexMapping; /**< the suggested radar index mapping */
  char* qualityFieldName; /**< the quality field name used in quality based compositing */
};

static const char* RAVE_COMPOSITE_PRODUCT_STRINGS[] =
{
  "PPI",
  "CAPPI",
  "PCAPPI",
  "ETOP",
  "MAX",
  "RR",
  "PMAX",
  "ACQVA"
  /*UNDEFINED*/
};

typedef struct ProductTypeMapping_t {
  const char* productString;
  Rave_ProductType productType;
} ProductTypeMapping_t;

static ProductTypeMapping_t PRODUCT_TYPE_MAPPING[] = {
  {"PPI", Rave_ProductType_PPI},
  {"CAPPI", Rave_ProductType_CAPPI},
  {"PCAPPI", Rave_ProductType_PCAPPI},
  {"ETOP", Rave_ProductType_ETOP},
  {"MAX", Rave_ProductType_MAX},
  {"RR", Rave_ProductType_RR},
  {"PMAX", Rave_ProductType_PMAX},
  {NULL, Rave_ProductType_UNDEFINED}
};

/**
 * Structure for keeping track on parameters that should be composited.
 */
typedef struct CompositeArgumentParameter_t {
  char* name;      /**< quantity */
  double gain;     /**< gain to be used in composite data*/
  double offset;   /**< offset to be used in composite data*/
  RaveDataType datatype; /**< the datatype */
  double nodata;   /**< the nodata value */
  double undetect; /**< the undetect value */
} CompositeArgumentParameter_t;

/*@{ Private CompositeArgumentObjectEntry class */

/**
 * The object entry that is stored inside the arguments.
 */
typedef struct CompositeArgumentObjectEntry_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveCoreObject* object; /**< the rave core object */
  int radarIndexValue; /**< the mapped index */
} CompositeArgumentObjectEntry_t;

static int CompositeArgumentObjectEntry_constructor(RaveCoreObject* obj)
{
  CompositeArgumentObjectEntry_t* this = (CompositeArgumentObjectEntry_t*)obj;
  this->object = NULL;
  this->radarIndexValue = 0;
  return 1;
}
static int CompositeArgumentObjectEntry_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  CompositeArgumentObjectEntry_t* this = (CompositeArgumentObjectEntry_t*)obj;
  CompositeArgumentObjectEntry_t* src = (CompositeArgumentObjectEntry_t*)srcobj;
  this->object = NULL;
  if (src->object != NULL) {
    this->object = RAVE_OBJECT_CLONE(src->object);
    if (this->object == NULL) {
      goto fail;
    }
  }
  this->radarIndexValue = src->radarIndexValue;
  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->object);
  return 0;
}
static void CompositeArgumentObjectEntry_destructor(RaveCoreObject* obj)
{
  CompositeArgumentObjectEntry_t* this = (CompositeArgumentObjectEntry_t*)obj;
  RAVE_OBJECT_RELEASE(this->object);
}

RaveCoreObjectType CompositeArgumentObjectEntry_TYPE = {
    "CompositeArgumentObjectEntry",
    sizeof(CompositeArgumentObjectEntry_t),
    CompositeArgumentObjectEntry_constructor,
    CompositeArgumentObjectEntry_destructor,
    CompositeArgumentObjectEntry_copyconstructor
};

/*@{ Private functions */
/**
 * Creates a parameter that should be composited
 * @param[in] name - quantity
 * @param[in] gain - gain
 * @param[in] offset - offset
 * @param[in] datatype - datatype
 * @param[in] nodata - nodata
 * @param[in] undetect - undetect
 * @return the parameter or NULL on failure
 */
static CompositeArgumentParameter_t* CompositeArgumentsInternal_createParameter(const char* name, double gain, double offset, RaveDataType datatype, double nodata, double undetect)
{
  CompositeArgumentParameter_t* result = NULL;
  if (name != NULL) {
    result = RAVE_MALLOC(sizeof(CompositeArgumentParameter_t));
    if (result != NULL) {
      result->name = RAVE_STRDUP(name);
      result->gain = gain;
      result->offset = offset;
      result->datatype = datatype;
      result->nodata = nodata;
      result->undetect = undetect;
      if (result->name == NULL) {
        RAVE_FREE(result);
        result = NULL;
      }
    }
  }
  return result;
}
/**
 * Frees the parameter including its members
 * @param[in] p - the parameter to release
 */
static void CompositeArgumentsInternal_freeParameter(CompositeArgumentParameter_t* p)
{
  if (p != NULL) {
    RAVE_FREE(p->name);
    RAVE_FREE(p);
  }
}

/**
 * Releases the complete parameter list including its items
 * @param[in] p - a pointer to the parameter list
 */
static void CompositeArgumentsInternal_freeParameterList(RaveList_t** p)
{
  if (p != NULL && *p != NULL) {
    CompositeArgumentParameter_t* cp = RaveList_removeLast(*p);
    while (cp != NULL) {
      CompositeArgumentsInternal_freeParameter(cp);
      cp = RaveList_removeLast(*p);
    }
    RAVE_OBJECT_RELEASE(*p);
  }
}


/**
 * Clones a parameter
 * @param[in] p - parameter to clone
 * @return the clone or NULL on failure
 */
static CompositeArgumentParameter_t* CompositeArgumentsInternal_cloneParameter(CompositeArgumentParameter_t* p)
{
  CompositeArgumentParameter_t* result = NULL;
  if (p != NULL) {
    result = RAVE_MALLOC(sizeof(CompositeArgumentParameter_t));
    if (result != NULL) {
      result->name = RAVE_STRDUP(p->name);
      result->gain = p->gain;
      result->offset = p->offset;
      result->datatype = p->datatype;
      result->nodata = p->nodata;
      result->undetect = p->undetect;
      if (result->name == NULL) {
        RAVE_FREE(result);
        result = NULL;
      }
    }
  }
  return result;
}

/**
 * Clones a parameter list
 * @param[in] p - the parameter list to clone
 * @return the clone or NULL on failure
 */
static RaveList_t* CompositeArgumentsInternal_cloneParameterList(RaveList_t* p)
{
  int len = 0, i = 0;
  RaveList_t *result = NULL, *clone = NULL;;
  if (p != NULL) {
    clone = RAVE_OBJECT_NEW(&RaveList_TYPE);
    if (clone != NULL) {
      len = RaveList_size(p);
      for (i = 0; i < len; i++) {
        CompositeArgumentParameter_t* cp = RaveList_get(p, i);
        CompositeArgumentParameter_t* cpclone = CompositeArgumentsInternal_cloneParameter(cp);
        if (cpclone == NULL || !RaveList_add(clone, cpclone)) {
          CompositeArgumentsInternal_freeParameter(cpclone);
          goto done;
        }
      }
    }
  }

  result = RAVE_OBJECT_COPY(clone);
done:
  RAVE_OBJECT_RELEASE(clone);
  return result;
}

/**
 * Returns a pointer to the internall stored parameter in the composite.
 * @param[in] composite - composite
 * @param[in] quantity - the parameter
 * @return the found parameter or NULL if not found
 */
static CompositeArgumentParameter_t* CompositeArgumentsInternal_getParameterByName(CompositeArguments_t* args, const char* quantity)
{
  int len = 0, i = 0;
  if (quantity != NULL) {
    len = RaveList_size(args->parameters);
    for (i = 0; i < len; i++) {
      CompositeArgumentParameter_t* cp = RaveList_get(args->parameters, i);
      if (strcmp(cp->name, quantity) == 0) {
        return cp;
      }
    }
  }
  return NULL;
}

static RaveList_t* CompositeArgumentsInternal_cloneQualityFlags(RaveList_t* qualityflags)
{
  RaveList_t *result = NULL, *oqualityflags = NULL;;

  if (qualityflags != NULL) {
    oqualityflags = RAVE_OBJECT_NEW(&RaveList_TYPE);
    if (oqualityflags != NULL) {
      int nlen = 0, i = 0;
      nlen = RaveList_size(qualityflags);
      for (i = 0; i < nlen; i++) {
        char* str = RAVE_STRDUP((const char*)RaveList_get(qualityflags, i));
        if (str == NULL) {
          RAVE_ERROR0("Could not duplicate string");
          goto fail;
        }
        if (!RaveList_add(oqualityflags, str)) {
          goto fail;
        }
      }
    }
  }

  result = oqualityflags;
  qualityflags = NULL;
fail:
  if (oqualityflags != NULL) {
    RaveList_freeAndDestroy(&oqualityflags);
  }
  return result;
}

/**
 * Constructor.
 */
static int CompositeArguments_constructor(RaveCoreObject* obj)
{
  CompositeArguments_t* this = (CompositeArguments_t*)obj;
  this->product = NULL;
  this->compositingProduct = Rave_CompositingProduct_UNDEFINED;
  this->area = NULL;
  this->datetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  this->arguments = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  this->parameters = RAVE_OBJECT_NEW(&RaveList_TYPE);
  this->objects = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  this->qualityflags = RAVE_OBJECT_NEW(&RaveList_TYPE);
  this->radarIndexMapping = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  this->qualityFieldName = NULL;

  if (this->datetime == NULL || this->arguments == NULL || this->parameters == NULL || this->objects == NULL || this->qualityflags == NULL || this->radarIndexMapping == NULL) {
    goto fail;
  }
  this->height = 1000.0;
  this->elangle = 0.0;
  this->range = 500000.0;
  this->strategy = NULL;
  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_OBJECT_RELEASE(this->arguments);
  CompositeArgumentsInternal_freeParameterList(&this->parameters);
  RAVE_OBJECT_RELEASE(this->objects);
  RaveList_freeAndDestroy(&this->qualityflags);
  RAVE_OBJECT_RELEASE(this->radarIndexMapping);
  return 0;
}

/**
 * Copy constructor
 */
static int CompositeArguments_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  CompositeArguments_t* this = (CompositeArguments_t*)obj;
  CompositeArguments_t* src = (CompositeArguments_t*)srcobj;
  this->product = NULL;
  this->compositingProduct = Rave_CompositingProduct_UNDEFINED;  
  this->area = NULL;
  this->datetime = NULL;
  this->strategy = NULL;
  this->arguments = NULL;
  this->qualityflags = NULL;
  this->radarIndexMapping = NULL;
  this->qualityFieldName = NULL;

  this->area = RAVE_OBJECT_CLONE(src->area);
  if (this->area == NULL) {
    goto fail;
  }
  if (src->product != NULL) {
    this->product = RAVE_STRDUP(src->product);
    if (this->product == NULL) {
      goto fail;
    }
  }
  this->compositingProduct = src->compositingProduct;
  this->datetime = RAVE_OBJECT_CLONE(src->datetime);
  if (this->datetime == NULL) {
    goto fail;
  }
  this->height = src->height;
  this->elangle = src->elangle;
  this->range = src->range;  
  if (src->strategy != NULL) {
    this->strategy = RAVE_STRDUP(src->strategy);
    if (this->strategy == NULL) {
      goto fail;
    }
  }
  this->arguments = RAVE_OBJECT_CLONE(src->arguments);
  if (this->arguments == NULL) {
    goto fail;
  }
  this->parameters = CompositeArgumentsInternal_cloneParameterList(src->parameters);
  if (this->parameters == NULL) {
    goto fail;
  }
  this->objects = RAVE_OBJECT_CLONE(src->objects);
  if (this->objects == NULL) {
    goto fail;
  }
  this->qualityflags = CompositeArgumentsInternal_cloneQualityFlags(src->qualityflags);
  if (this->qualityflags == NULL) {
    goto fail;
  }
  this->radarIndexMapping = RAVE_OBJECT_CLONE(this->radarIndexMapping);
  if (this->radarIndexMapping == NULL) {
    goto fail;
  }
  if (src->qualityFieldName != NULL) {
    this->qualityFieldName = RAVE_STRDUP(src->qualityFieldName);
    if (this->qualityFieldName == NULL) {
      goto fail;
    }
  }  
  return 1;
fail:
  RAVE_FREE(this->product);
  RAVE_OBJECT_RELEASE(this->area);
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_FREE(this->strategy);
  RAVE_OBJECT_RELEASE(this->arguments);
  CompositeArgumentsInternal_freeParameterList(&this->parameters);
  RAVE_OBJECT_RELEASE(this->objects);
  RaveList_freeAndDestroy(&this->qualityflags);
  RAVE_OBJECT_RELEASE(this->radarIndexMapping);
  RAVE_FREE(this->qualityFieldName);
  return 0;
}

/**
 * Destroys the argument class
 * @param[in] obj - the the CompositeArguments_t instance
 */
static void CompositeArguments_destructor(RaveCoreObject* obj)
{
  CompositeArguments_t* this = (CompositeArguments_t*)obj;
  RAVE_FREE(this->product);
  RAVE_OBJECT_RELEASE(this->area);
  RAVE_OBJECT_RELEASE(this->datetime);
  RAVE_FREE(this->strategy);
  RAVE_OBJECT_RELEASE(this->arguments);
  CompositeArgumentsInternal_freeParameterList(&this->parameters);
  RAVE_OBJECT_RELEASE(this->objects);
  RaveList_freeAndDestroy(&this->qualityflags);
  RAVE_OBJECT_RELEASE(this->radarIndexMapping);
  RAVE_FREE(this->qualityFieldName);  
}

/*@} End of Private functions */

/*@{ Interface functions */

const char* CompositeArguments_productToString(Rave_CompositingProduct product)
{
  if (product >= Rave_CompositingProduct_PPI && product < Rave_CompositingProduct_UNDEFINED) {
    return RAVE_COMPOSITE_PRODUCT_STRINGS[(int)product];
  }
  return NULL;
}

Rave_CompositingProduct CompositeArguments_stringToProduct(const char* product)
{
  if (product != NULL) {
    int i = 0, nlen = 0;
    nlen = strlen(product);
    for (i = 0; i < Rave_CompositingProduct_UNDEFINED; i++) {
      if (strncasecmp(product, RAVE_COMPOSITE_PRODUCT_STRINGS[i], nlen) == 0) {
        return (Rave_CompositingProduct)i;
      }
    }
  }
  return Rave_CompositingProduct_UNDEFINED;
}


int CompositeArguments_setProduct(CompositeArguments_t* args, const char* product)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  if (args->product != NULL) {
    RAVE_FREE(args->product);
    args->product = NULL;
  }
  args->compositingProduct = Rave_CompositingProduct_UNDEFINED;
  if (product != NULL) {
    args->product = RAVE_STRDUP(product);
    if (args->product == NULL) {
      RAVE_ERROR0("Failed to set product");
      return 0;
    }
    args->compositingProduct = CompositeArguments_stringToProduct(args->product);
  }
  return 1;
}

const char* CompositeArguments_getProduct(CompositeArguments_t* args)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return (const char*)args->product;
}

Rave_CompositingProduct CompositeArguments_getCompositingProduct(CompositeArguments_t* args)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return args->compositingProduct;
}

Rave_ProductType CompositeArguments_getProductType(CompositeArguments_t* args)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  if (args->product != NULL) {
    int ctr = 0;
    while (PRODUCT_TYPE_MAPPING[ctr].productType != Rave_ProductType_UNDEFINED) {
      if (strcmp(PRODUCT_TYPE_MAPPING[ctr].productString, args->product) == 0) {
        return PRODUCT_TYPE_MAPPING[ctr].productType;
      }
      ctr++;
    }
  }
  return Rave_ProductType_UNDEFINED;
}

int CompositeArguments_setArea(CompositeArguments_t* args, Area_t* area)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  RAVE_OBJECT_RELEASE(args->area);
  if (area != NULL) {
    args->area = RAVE_OBJECT_COPY(area);
  }
  return 1;
}

Area_t* CompositeArguments_getArea(CompositeArguments_t* args)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return RAVE_OBJECT_COPY(args->area);
}

int CompositeArguments_setTime(CompositeArguments_t* args, const char* value)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return RaveDateTime_setTime(args->datetime, value);
}

const char* CompositeArguments_getTime(CompositeArguments_t* args)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return RaveDateTime_getTime(args->datetime);
}

int CompositeArguments_setDate(CompositeArguments_t* args, const char* value)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return RaveDateTime_setDate(args->datetime, value);
}

const char* CompositeArguments_getDate(CompositeArguments_t* args)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return RaveDateTime_getDate(args->datetime);
}

void CompositeArguments_setHeight(CompositeArguments_t* args, double height)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  args->height = height;
}

double CompositeArguments_getHeight(CompositeArguments_t* args)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return args->height;
}

void CompositeArguments_setElevationAngle(CompositeArguments_t* args, double angle)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  args->elangle = angle;
}

double CompositeArguments_getElevationAngle(CompositeArguments_t* args)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return args->elangle;
}

void CompositeArguments_setRange(CompositeArguments_t* args, double range)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  args->range = range;
}

double CompositeArguments_getRange(CompositeArguments_t* args)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return args->range;
}

int CompositeArguments_setStrategy(CompositeArguments_t* args, const char* strategy)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  if (args->strategy != NULL) {
    RAVE_FREE(args->strategy);
    args->strategy = NULL;
  }
  if (strategy != NULL) {
    args->strategy = RAVE_STRDUP(strategy);
    if (args->strategy == NULL) {
      RAVE_ERROR0("Failed to set strategy");
      return 0;
    }
  }
  return 1;
}

const char* CompositeArguments_getStrategy(CompositeArguments_t* args)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return (const char*)args->strategy;
}


int CompositeArguments_addArgument(CompositeArguments_t* args, RaveAttribute_t* argument)
{
  int result = 0;
  RAVE_ASSERT((args != NULL), "args == NULL");
  if (argument != NULL) {
    const char* name = NULL;
    name = RaveAttribute_getName(argument);
    if (name != NULL) {
      if (RaveObjectHashTable_put(args->arguments, name, (RaveCoreObject*)argument)) {
        result = 1;
      }
    }
  }
  return result;
}

void CompositeArguments_removeArgument(CompositeArguments_t* args, const char* name)
{
  RaveCoreObject* obj = NULL;
  RAVE_ASSERT((args != NULL), "args == NULL");
  obj = RaveObjectHashTable_remove(args->arguments, name);
  RAVE_OBJECT_RELEASE(obj);
}

RaveAttribute_t* CompositeArguments_getArgument(CompositeArguments_t* args, const char* name)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return (RaveAttribute_t*)RaveObjectHashTable_get(args->arguments, name);
}

int CompositeArguments_hasArgument(CompositeArguments_t* args, const char* name)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return RaveObjectHashTable_exists(args->arguments, name);
}

int CompositeArguments_addParameter(CompositeArguments_t* args, const char* quantity, double gain, double offset, RaveDataType datatype, double nodata, double undetect)
{
  int result = 0;
  CompositeArgumentParameter_t* param = NULL;
  RAVE_ASSERT((args != NULL), "args == NULL");

  param = CompositeArgumentsInternal_getParameterByName(args, quantity);
  if (param != NULL) {
    param->gain = gain;
    param->offset = offset;
    result = 1;
  } else {
    param = CompositeArgumentsInternal_createParameter(quantity, gain, offset, datatype, nodata, undetect);
    if (param != NULL) {
      result = RaveList_add(args->parameters, param);
      if (!result) {
        CompositeArgumentsInternal_freeParameter(param);
      }
    }
  }
  return result;
}

int CompositeArguments_hasParameter(CompositeArguments_t* args, const char* parameter)
{
  int result = 0;
  RAVE_ASSERT((args != NULL), "args == NULL");
  if (parameter != NULL) {
    int i = 0;
    int len = RaveList_size(args->parameters);
    for (i = 0; result == 0 && i < len ; i++) {
      CompositeArgumentParameter_t* s = RaveList_get(args->parameters, i);
      if (s != NULL && s->name != NULL && strcmp(parameter, s->name) == 0) {
        result = 1;
      }
    }
  }
  return result;  
}

int CompositeArguments_getParameter(CompositeArguments_t* args, const char* parameter, double* gain, double* offset, RaveDataType* datatype, double* nodata, double* undetect)
{
  int result = 0;
  RAVE_ASSERT((args != NULL), "args == NULL");
  if (parameter != NULL) {
    int i = 0;
    int len = RaveList_size(args->parameters);
    for (i = 0; result == 0 && i < len ; i++) {
      CompositeArgumentParameter_t* s = RaveList_get(args->parameters, i);
      if (s != NULL && s->name != NULL && strcmp(parameter, s->name) == 0) {
        if (gain != NULL) {
          *gain = s->gain;
        }
        if (offset != NULL) {
          *offset = s->offset;
        }
        if (datatype != NULL) {
          *datatype = s->datatype;
        }
        if (nodata != NULL) {
          *nodata = s->nodata;
        }
        if (undetect != NULL) {
          *undetect = s->undetect;
        }
        result = 1;
      }
    }
  }
  return result;
}

int CompositeArguments_getParameterCount(CompositeArguments_t* args)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return RaveList_size(args->parameters);
}

const char* CompositeArguments_getParameterAtIndex(CompositeArguments_t* args, int index, double* gain, double* offset, RaveDataType* datatype, double* nodata, double* undetect)
{
  CompositeArgumentParameter_t* param = NULL;
  RAVE_ASSERT((args != NULL), "args == NULL");
  param = RaveList_get(args->parameters, index);
  if (param != NULL) {
    if (gain != NULL) {
      *gain = param->gain;
    }
    if (offset != NULL) {
      *offset = param->offset;
    }
    if (datatype != NULL) {
      *datatype = param->datatype;
    }
    if (nodata != NULL) {
      *nodata = param->nodata;
    }
    if (undetect != NULL) {
      *undetect = param->undetect;
    }    
    return (const char*)param->name;
  }
  return NULL;
}

const char* CompositeArguments_getParameterName(CompositeArguments_t* args, int index)
{
  CompositeArgumentParameter_t* param = NULL;
  RAVE_ASSERT((args != NULL), "args == NULL");
  param = RaveList_get(args->parameters, index);
  if (param != NULL) {
    return (const char*)param->name;
  }
  return NULL;
}

int CompositeArguments_addObject(CompositeArguments_t* args, RaveCoreObject* object)
{
  CompositeArgumentObjectEntry_t* entry = NULL;
  RAVE_ASSERT((args != NULL), "args == NULL");
  entry = RAVE_OBJECT_NEW(&CompositeArgumentObjectEntry_TYPE);
  if (entry != NULL) {
    entry->object = RAVE_OBJECT_COPY(object);
    if (!RaveObjectList_add(args->objects, (RaveCoreObject*)entry)) {
      RAVE_OBJECT_RELEASE(entry);
      return 0;
    }
    entry->radarIndexValue = RaveObjectList_size(args->objects);
  }
  RAVE_OBJECT_RELEASE(entry);
  return 1;
}

int CompositeArguments_getNumberOfObjects(CompositeArguments_t* args)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return RaveObjectList_size(args->objects);
}

RaveCoreObject* CompositeArguments_getObject(CompositeArguments_t* args, int index)
{
  CompositeArgumentObjectEntry_t* entry = NULL;
  RaveCoreObject* result = NULL;
  RAVE_ASSERT((args != NULL), "args == NULL");
  entry = (CompositeArgumentObjectEntry_t*)RaveObjectList_get(args->objects, index);
  if (entry != NULL) {
    result = RAVE_OBJECT_COPY(entry->object);
  }
  RAVE_OBJECT_RELEASE(entry);
  return result;
}

RaveObjectList_t* CompositeArguments_getObjects(CompositeArguments_t* args)
{
  RaveObjectList_t* result = NULL;
  int nlen = 0, i = 0;
  RAVE_ASSERT((args != NULL), "args == NULL");
  result = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (result == NULL) {
    RAVE_ERROR0("Failed to create object list");
  }
  nlen = RaveObjectList_size(args->objects);
  for (i = 0; i < nlen; i++) {
    CompositeArgumentObjectEntry_t* entry = (CompositeArgumentObjectEntry_t*)RaveObjectList_get(args->objects, i);
    if (!RaveObjectList_add(result, entry->object)) {
      RAVE_ERROR0("Failed to add entry to object list");
      RAVE_OBJECT_RELEASE(result);
      goto done;
    }
  }
done:
  return result;
}

int CompositeArguments_setQIFieldName(CompositeArguments_t* args, const char* fieldname)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  if (args->qualityFieldName != NULL) {
    RAVE_FREE(args->qualityFieldName);
    args->qualityFieldName = NULL;
  }
  if (fieldname != NULL) {
    args->qualityFieldName = RAVE_STRDUP(fieldname);
    if (args->qualityFieldName == NULL) {
      RAVE_ERROR0("Failed to set quality field name");
      return 0;
    }
  }
  return 1;
}

const char* CompositeArguments_getQIFieldName(CompositeArguments_t* args)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return (const char*)args->qualityFieldName;
}


int CompositeArguments_addQualityFlag(CompositeArguments_t* args, const char* flag)
{
  int result = 0;

  RAVE_ASSERT((args != NULL), "args == NULL");

  if (flag != NULL) {
    char* d = RAVE_STRDUP(flag);
    if (d != NULL) {
      if (!RaveList_add(args->qualityflags, d)) {
        RAVE_FREE(d);
        goto fail;
      }
      result = 1;
    }
  }

fail:
  return result;
}

int CompositeArguments_setQualityFlags(CompositeArguments_t* args, const char* flags[], int nrflags)
{
  int i = 0;

  RAVE_ASSERT((args != NULL), "args == NULL");

  while (RaveList_size(args->qualityflags) > 0) {
    char* str = RaveList_removeLast(args->qualityflags);
    RAVE_FREE(str);
  }

  for (i = 0; i < nrflags; i++) {
    if (!CompositeArguments_addQualityFlag(args, flags[i])) {
      RAVE_ERROR1("Failed to add quality flag: %s", flags[i]);
      goto fail;
    }
  }

  return 1;
fail:
  while (RaveList_size(args->qualityflags) > 0) {
    char* str = RaveList_removeLast(args->qualityflags);
    RAVE_FREE(str);
  }
  return 0;
}


int CompositeArguments_getNumberOfQualityFlags(CompositeArguments_t* args)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return RaveList_size(args->qualityflags);
}

const char* CompositeArguments_getQualityFlagAt(CompositeArguments_t* args, int index)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return (const char*)RaveList_get(args->qualityflags, index);
}

int CompositeArguments_removeQualityFlag(CompositeArguments_t* args, const char* flag)
{
  int i = 0, nlen = 0;
  RAVE_ASSERT((args != NULL), "args == NULL");
  if (flag == NULL) {
    return 0;
  }

  nlen = RaveList_size(args->qualityflags);
  for (i = 0; i < nlen; i++) {
    const char* name = (const char*)RaveList_get(args->qualityflags, i);
    if (strcmp(name, flag) == 0) {
      char* str = (char*)RaveList_remove(args->qualityflags, i);
      RAVE_FREE(str);
      break;
    }
  }
  return 1;
}

int CompositeArguments_removeQualityFlagAt(CompositeArguments_t* args, int index)
{
  int result = 0;
  char* str = NULL;

  RAVE_ASSERT((args != NULL), "args == NULL");

  str = (char*)RaveList_remove(args->qualityflags, index);
  if (str != NULL) {
    RAVE_FREE(str);
    result = 1;
  }
  return result;
}

RaveList_t* CompositeArguments_getQualityFlags(CompositeArguments_t* args)
{
  RaveList_t* result = NULL;

  RAVE_ASSERT((args != NULL), "args == NULL");
  result = RAVE_OBJECT_NEW(&RaveList_TYPE);
  if (result != NULL) {
    int i = 0;
    int nlen = RaveList_size(args->qualityflags);
    for (i = 0; i < nlen; i++) {
      const char* qflag = (const char*)RaveList_get(args->qualityflags, i);
      if (qflag != NULL) {
        char* t = RAVE_STRDUP(qflag);
        if (t == NULL || !RaveList_add(result, t)) {
          RAVE_ERROR0("Failed to add item to rave list");
          RAVE_FREE(t);
          goto fail;
        }
      }
    }
  }
  return result;
fail:
  RaveList_freeAndDestroy(&result);
  return NULL;
}

int CompositeArguments_hasQualityFlag(CompositeArguments_t* args, const char* name)
{
  int i = 0, nlen = 0;
  RAVE_ASSERT((args != NULL), "args == NULL");
  if (name == NULL) {
    return 0;
  }

  nlen = RaveList_size(args->qualityflags);
  for (i = 0; i < nlen; i++) {
    const char* qflag = (const char*)RaveList_get(args->qualityflags, i);
    if (qflag != NULL && strcasecmp(name, qflag) == 0) {
      return 1;
    }
  }
  return 0;
}

static char* CompositeArgumentsInternal_getAnyIdFromSource(const char* source)
{
  char* result = NULL;
  result = OdimSource_getIdFromOdimSourceInclusive(source, "NOD:");
  if (result == NULL) {
    result = OdimSource_getIdFromOdimSourceInclusive(source, "WMO:");
    if (result != NULL && strcmp("WMO:00000", result)==0) {
      RAVE_FREE(result);
    }
  }
  if (result == NULL) {
    result = OdimSource_getIdFromOdimSourceInclusive(source, "RAD:");
  }
  if (result == NULL) {
    result = OdimSource_getIdFromOdimSourceInclusive(source, "PLC:");
  }
  return result;
}

int CompositeArguments_createRadarIndexMapping(CompositeArguments_t* args, OdimSources_t* sources)
{
  int i = 0, nobjects = 0;
  int result = 1;

  RAVE_ASSERT((args != NULL), "args == NULL");
  nobjects = CompositeArguments_getNumberOfObjects(args);
  RaveObjectHashTable_clear(args->radarIndexMapping);
  if (sources != NULL) {
    int ctr = 1;
    for (i = 0; result && i < nobjects; i++) {
      CompositeArgumentObjectEntry_t* entry = (CompositeArgumentObjectEntry_t*)RaveObjectList_get(args->objects, i);
      char srcbuff[1024];
      if (CompositeUtils_getObjectSource(entry->object, srcbuff, 1024) > 0) {
        OdimSource_t* source = OdimSources_identify(sources, (const char*)srcbuff);
        if (source != NULL) {
          char nodbuff[16];
          snprintf(nodbuff, 16, "NOD:%s", OdimSource_getNod(source));
          if (!RaveObjectHashTable_exists(args->radarIndexMapping, nodbuff)) {
            RaveAttribute_t* attrctr = RaveAttributeHelp_createLong("index", (long)ctr);
            if (attrctr != NULL) {
              if (!RaveObjectHashTable_put(args->radarIndexMapping, nodbuff, (RaveCoreObject*)attrctr)) {
                RAVE_ERROR0("Failed to add radar index counter");
                result = 0;
              } else {
                entry->radarIndexValue = ctr++;
              }
            }
            RAVE_OBJECT_RELEASE(attrctr);
          } else {
            RaveAttribute_t* attrctr = (RaveAttribute_t*)RaveObjectHashTable_get(args->radarIndexMapping, nodbuff);
            if (attrctr != NULL) {
              long v = 0;
              RaveAttribute_getLong(attrctr, &v);
              entry->radarIndexValue = (int)v;
            }
            RAVE_OBJECT_RELEASE(attrctr);
          }
        }
        RAVE_OBJECT_RELEASE(source);
      }
      RAVE_OBJECT_RELEASE(entry);
    }
  } else {
    int ctr = 1;
    char source[1024];
    for (i = 0; result && i < nobjects; i++) {
      CompositeArgumentObjectEntry_t* entry = (CompositeArgumentObjectEntry_t*)RaveObjectList_get(args->objects, i);
      if (CompositeUtils_getObjectSource(entry->object, source, 1024)) {
        char* key = CompositeArgumentsInternal_getAnyIdFromSource(source);
        if (!RaveObjectHashTable_exists(args->radarIndexMapping, key)) {
          RaveAttribute_t* attrctr = RaveAttributeHelp_createLong("index", (long)ctr);
          if (attrctr != NULL) {
            if (!RaveObjectHashTable_put(args->radarIndexMapping, key, (RaveCoreObject*)attrctr)) {
              RAVE_ERROR0("Failed to add radar index counter");
              result = 0;
            } else {
              entry->radarIndexValue = ctr++;
            }
          }
          RAVE_OBJECT_RELEASE(attrctr);
        } else {
            RaveAttribute_t* attrctr = (RaveAttribute_t*)RaveObjectHashTable_get(args->radarIndexMapping, key);
            if (attrctr != NULL) {
              long v = 0;
              RaveAttribute_getLong(attrctr, &v);
              entry->radarIndexValue = (int)v;
            }
            RAVE_OBJECT_RELEASE(attrctr);
        }
        RAVE_FREE(key);
      }
      RAVE_OBJECT_RELEASE(entry);
    }
  }

  if (!result) {
    RaveObjectHashTable_clear(args->radarIndexMapping);
  }

  return result;
}

RaveList_t* CompositeArguments_getRadarIndexKeys(CompositeArguments_t* args)
{
  RAVE_ASSERT((args != NULL), "args == NULL");
  return RaveObjectHashTable_keys(args->radarIndexMapping);
}

int CompositeArguments_getRadarIndexValue(CompositeArguments_t* args, const char* key)
{
  int result = 0;
  RaveAttribute_t* attr = NULL;
  RAVE_ASSERT((args != NULL), "args == NULL");
  attr = (RaveAttribute_t*)RaveObjectHashTable_get(args->radarIndexMapping, key);
  if (attr != NULL) {
    long v = 0;
    RaveAttribute_getLong(attr, &v);
    result = (int)v;
  }
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

int CompositeArguments_createRadarIndex(CompositeArguments_t* args, const char* key)
{
  int result = 0;
  RaveAttribute_t* attr = NULL;
  RAVE_ASSERT((args != NULL), "args == NULL");

  attr = (RaveAttribute_t*)RaveObjectHashTable_get(args->radarIndexMapping, key);
  if (attr != NULL) {
    long v = 0;
    RaveAttribute_getLong(attr, &v);
    result = (int)v;
  } else {
    int sz = RaveObjectHashTable_size(args->radarIndexMapping);
    int ctr = sz + 1;
    RaveAttribute_t* attrctr = RaveAttributeHelp_createLong("index", (long)ctr);
    if (attrctr != NULL) {
      if (!RaveObjectHashTable_put(args->radarIndexMapping, key, (RaveCoreObject*)attrctr)) {
        RAVE_ERROR0("Failed to add radar index counter");
        result = 0;
      }
      result = ctr;
    } else {
      RAVE_ERROR0("Failed to create long attribute for radar index");
    }
    RAVE_OBJECT_RELEASE(attrctr);
  }
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

int CompositeArguments_getObjectRadarIndexValue(CompositeArguments_t* args, int index)
{
  CompositeArgumentObjectEntry_t* entry = NULL;
  int result = 0;
  RAVE_ASSERT((args != NULL), "args == NULL");
  entry = (CompositeArgumentObjectEntry_t*)RaveObjectList_get(args->objects, index);
  if (entry != NULL) {
    result = entry->radarIndexValue;
  }
  RAVE_OBJECT_RELEASE(entry);
  return result;
}

/*@} End of Interface functions */

RaveCoreObjectType CompositeArguments_TYPE = {
    "CompositeArguments",
    sizeof(CompositeArguments_t),
    CompositeArguments_constructor,
    CompositeArguments_destructor,
    CompositeArguments_copyconstructor
};
