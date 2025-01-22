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
 * Provides functionality for creating composites.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-10-10
 */
#include "legacycompositegeneratorfactory.h"
#include "compositearguments.h"
#include "polarvolume.h"
#include "rave_attribute.h"
#include "rave_object.h"
#include "raveobject_list.h"
#include "rave_types.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_utilities.h"
#include <strings.h>

static const char* SUPPORTED_PRODUCTS[]={
  "PPI",
  "CAPPI",
  "PCAPPI",
  "MAX",
  "PMAX",
  NULL
};

typedef struct _LegacyCompositeGeneratorFactory_t {
  RAVE_OBJECT_HEAD /**< Always on top */
  COMPOSITE_GENERATOR_FACTORY_HEAD /**< composite generator plugin specifics */
} LegacyCompositeGeneratorFactory_t;
/*@{ Private functions */
/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int LegacyCompositeGeneratorFactory_constructor(RaveCoreObject* obj)
{
  LegacyCompositeGeneratorFactory_t* this = (LegacyCompositeGeneratorFactory_t*)obj;
  this->getName = LegacyCompositeGeneratorFactory_getName;
  this->canHandle = LegacyCompositeGeneratorFactory_canHandle;
  this->generate = LegacyCompositeGeneratorFactory_generate;
  this->create = LegacyCompositeGeneratorFactory_create;
  return 1;
}

/**
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int LegacyCompositeGeneratorFactory_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  LegacyCompositeGeneratorFactory_t* this = (LegacyCompositeGeneratorFactory_t*)obj;
  //LegacyCompositeGeneratorFactory_t* src = (LegacyCompositeGeneratorFactory_t*)srcobj;
  this->getName = LegacyCompositeGeneratorFactory_getName;
  this->canHandle = LegacyCompositeGeneratorFactory_canHandle;
  this->generate = LegacyCompositeGeneratorFactory_generate;
  this->create = LegacyCompositeGeneratorFactory_create;

  return 1;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void LegacyCompositeGeneratorFactory_destructor(RaveCoreObject* obj)
{
  //LegacyCompositeGeneratorFactory_t* this = (LegacyCompositeGeneratorFactory_t*)obj;
}
/*@} End of Private functions */

/*@{ Interface functions */

/**
 * @returns the unique name for this composite generator plugin
 */
const char* LegacyCompositeGeneratorFactory_getName(CompositeGeneratorFactory_t* self)
{
  return "LegacyCompositeGenerator";
}

/**
 * @returns if this plugin can handle the generator request
 */
int LegacyCompositeGeneratorFactory_canHandle(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments)
{
  const char* productid;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (arguments == NULL) {
    return 0;
  }
  productid = CompositeArguments_getProduct(arguments);
  if (productid != NULL) {
    if (!RaveUtilities_arrayContains(SUPPORTED_PRODUCTS, productid, 0)) {
      return 0;
    }
    if (strcasecmp("PMAX", productid) == 0 || strcasecmp("MAX", productid) == 0) {
      RaveAttribute_t* attr = CompositeArguments_getArgument(arguments, "interpolation_method");
      if (attr != NULL) {
        char* value = NULL;
        RaveAttribute_getString(attr, &value);
        if (value == NULL || strcasecmp("NEAREST", value) != 0) {
          RAVE_OBJECT_RELEASE(attr);
          return 0;
        }
      }
      RAVE_OBJECT_RELEASE(attr);
    }
  }

  return 1;
}

/**
 * @returns the result from the generation
 */
Cartesian_t* LegacyCompositeGeneratorFactory_generate(CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments)
{
  return NULL;
}

/**
 * The initializing function so that we know what composite generator we are working with.
 * Note, this function will be called by the composite module and will override any previous calls.
 * @return 1 on success otherwise 0
 */
CompositeGeneratorFactory_t* LegacyCompositeGeneratorFactory_create(CompositeGeneratorFactory_t* self)
{
  LegacyCompositeGeneratorFactory_t* result = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (RAVE_OBJECT_CHECK_TYPE(self, &LegacyCompositeGeneratorFactory_TYPE)) {
    return RAVE_OBJECT_CLONE(self);
  }
  RAVE_ERROR0("Something is wrong in implementation. Should not arrive here unless type is a LegacyCompositeGeneratorFactory_TYPE");
  return NULL;
}

/*@} End of Interface functions */

RaveCoreObjectType LegacyCompositeGeneratorFactory_TYPE = {
    "LegacyCompositeGeneratorFactory",
    sizeof(LegacyCompositeGeneratorFactory_t),
    LegacyCompositeGeneratorFactory_constructor,
    LegacyCompositeGeneratorFactory_destructor,
    LegacyCompositeGeneratorFactory_copyconstructor
};
