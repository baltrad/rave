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
#include "compositegenerator.h"
#include "cartesian.h"
#include "compositearguments.h"
#include "compositefactorymanager.h"
#include "compositefilter.h"
#include "compositegeneratorfactory.h"
#include "polarvolume.h"
#include "rave_attribute.h"
#include "rave_list.h"
#include "rave_object.h"
#include "raveobject_hashtable.h"
#include "raveobject_list.h"
#include "rave_types.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_utilities.h"
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include "rave_simplexml.h"

#include "legacycompositegeneratorfactory.h"
#include "acqvacompositegeneratorfactory.h"

/**
 * Represents the cartesian product.
 */
struct _CompositeGenerator_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveObjectList_t* factories; /**< the factory filters */
  CompositeFactoryManager_t* manager; /**< the factory manager */
  RaveProperties_t* properties; /**< properties */
};

/*@{ Private CompositeFactoryEntry class */

/**
 * The object entry that is stored inside the arguments.
 */
typedef struct CompositeFactoryEntry_t {
  RAVE_OBJECT_HEAD /** Always on top */
  char* name;  /**< the name identifying this factory */
  char factory_class[256]; /**< the factory class name */
  CompositeGeneratorFactory_t* factory; /**< the factory */
  RaveObjectList_t* filters; /**< the filters */
} CompositeFactoryEntry_t;

static int CompositeFactoryEntry_constructor(RaveCoreObject* obj)
{
  CompositeFactoryEntry_t* this = (CompositeFactoryEntry_t*)obj;
  this->name = NULL;
  strcpy(this->factory_class, "");
  this->factory = NULL;
  this->filters = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (this->filters == NULL) {
    return 0;
  }
  return 1;
}

static int CompositeFactoryEntry_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  CompositeFactoryEntry_t* this = (CompositeFactoryEntry_t*)obj;
  CompositeFactoryEntry_t* src = (CompositeFactoryEntry_t*)srcobj;
  this->name = NULL;
  strcpy(this->factory_class, src->factory_class);
  this->factory = NULL;
  this->filters = NULL;


  if (src->name != NULL) {
    this->name = RAVE_STRDUP(src->name);
    if (this->name == NULL) {
      RAVE_ERROR0("Failed to clone name");
      goto fail;
    }
  }
  if (src->factory != NULL) {
    this->factory = RAVE_OBJECT_CLONE(src->factory);
    if (this->factory == NULL) {
      RAVE_ERROR0("Failed to clone factory");
      goto fail;
    }
  }

  this->filters = RAVE_OBJECT_CLONE(src->filters);
  if (this->filters == NULL) {
    goto fail;
  }

  return 1;
fail:
  RAVE_FREE(this->name);
  RAVE_OBJECT_RELEASE(this->factory);
  RAVE_OBJECT_RELEASE(this->filters);
  return 0;
}

static void CompositeFactoryEntry_destructor(RaveCoreObject* obj)
{
  CompositeFactoryEntry_t* this = (CompositeFactoryEntry_t*)obj;
  RAVE_FREE(this->name);
  RAVE_OBJECT_RELEASE(this->factory);
  RAVE_OBJECT_RELEASE(this->filters);
}

RaveCoreObjectType CompositeFactoryEntry_TYPE = {
    "CompositeFactoryEntry",
    sizeof(CompositeFactoryEntry_t),
    CompositeFactoryEntry_constructor,
    CompositeFactoryEntry_destructor,
    CompositeFactoryEntry_copyconstructor
};
/*@} End of Private functions */

/*@{ Private functions */

/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int CompositeGenerator_constructor(RaveCoreObject* obj)
{
  CompositeGenerator_t* this = (CompositeGenerator_t*)obj;
  this->factories = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  this->manager = NULL;
  this->properties = NULL;

  if (this->factories == NULL) {
    goto fail;
  }
  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->factories);
  return 0;
}

/**
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int CompositeGenerator_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  CompositeGenerator_t* this = (CompositeGenerator_t*)obj;
  CompositeGenerator_t* src = (CompositeGenerator_t*)srcobj;
  this->factories = RAVE_OBJECT_CLONE(src->factories);
  this->properties = NULL;
  if (this->factories == NULL) {
    goto fail;
  }

  if (src->manager != NULL) {
    this->manager = RAVE_OBJECT_CLONE(src->manager);
    if (this->manager == NULL) {
      RAVE_ERROR0("Failed to clone factory manager");
      goto fail;
    }
  }
  if (src->properties != NULL) {
    this->properties = RAVE_OBJECT_CLONE(src->properties);
    if (this->properties == NULL) {
      RAVE_ERROR0("Failed to clone properties");
      goto fail;
    }
  }
  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->factories);
  RAVE_OBJECT_RELEASE(this->manager);
  RAVE_OBJECT_RELEASE(this->properties);
  return 0;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void CompositeGenerator_destructor(RaveCoreObject* obj)
{
  CompositeGenerator_t* this = (CompositeGenerator_t*)obj;
  RAVE_OBJECT_RELEASE(this->factories);
  RAVE_OBJECT_RELEASE(this->manager);
  RAVE_OBJECT_RELEASE(this->properties);
}

static int CompositeGeneratorInternal_loadXml(CompositeGenerator_t* generator, const char* filename)
{
#ifdef RAVE_XML_SUPPORTED  
  SimpleXmlNode_t* node = NULL;
  SimpleXmlNode_t *factory = NULL, *filter = NULL, *attr = NULL;

  int result = 0, nfactories = 0, i = 0, nfilters = 0, j = 0, nattributes = 0, k = 0;
  RaveList_t* products = NULL;
  RaveList_t* quantities = NULL;
  RaveList_t* interpolation_methods = NULL;

  CompositeFactoryEntry_t* cfentry = NULL;
  CompositeFilter_t* cfilter = NULL;

  node = SimpleXmlNode_parseFile(filename);
  if (node == NULL) {
    RAVE_ERROR1("Could not parse file: %s", filename);
    goto done;
  }
  if (strcasecmp("rave-composite-generator", SimpleXmlNode_getName(node)) != 0) {
    RAVE_ERROR0("XML configuration must have root-element rave-composite-generator");
    goto done;
  }

  nfactories = SimpleXmlNode_getNumberOfChildren(node);
  for (i = 0; i < nfactories; i++) {
    factory = SimpleXmlNode_getChild(node, i);
    if (factory != NULL) {
      const char* name = SimpleXmlNode_getAttribute(factory, "name");
      const char* factory_class = SimpleXmlNode_getAttribute(factory, "factory_class");

      if (name == NULL) {
        RAVE_ERROR0("name must be set");
        goto done;
      }
      if (factory_class == NULL) {
        RAVE_ERROR0("factory_class must be set");
        goto done;
      }

      cfentry = RAVE_OBJECT_NEW(&CompositeFactoryEntry_TYPE);
      if (cfentry == NULL) {
        RAVE_ERROR0("Failed to allocate memory for factory");
        goto done;
      }
      cfentry->name = RAVE_STRDUP(name);
      if (cfentry->name == NULL) {
        RAVE_ERROR0("Failed to duplicate name");
        goto done;
      }
      strcpy(cfentry->factory_class, factory_class);

      nfilters = SimpleXmlNode_getNumberOfChildren(factory);

      for (j = 0; j < nfilters; j++) {
        filter = SimpleXmlNode_getChild(factory, j);
        if (filter != NULL) {
          cfilter = RAVE_OBJECT_NEW(&CompositeFilter_TYPE);
          if (cfilter == NULL) {
            RAVE_ERROR0("Failed to allocate cfilter");
            goto done;
          }

          nattributes = SimpleXmlNode_getNumberOfChildren(filter);
          for (k = 0; k < nattributes; k++) {
            attr = SimpleXmlNode_getChild(filter, k);
            if (attr != NULL) {
              if (strcasecmp("products", SimpleXmlNode_getName(attr)) == 0) {
                const char* text = SimpleXmlNode_getText(attr);
                if (text != NULL) {
                  products = RaveUtilities_getTrimmedTokens(text, ',');
                  if (products == NULL || !CompositeFilter_setProductsList(cfilter, products)) {
                    RAVE_ERROR0("Failed to read composite filter for products");
                    goto done;
                  }
                  RaveList_freeAndDestroy(&products);
                }
              } else if (strcasecmp("quantities", SimpleXmlNode_getName(attr)) == 0) {
                const char* text = SimpleXmlNode_getText(attr);
                if (text != NULL) {
                  quantities = RaveUtilities_getTrimmedTokens(text, ',');
                  if (quantities == NULL || !CompositeFilter_setQuantitiesList(cfilter, quantities)) {
                    RAVE_ERROR0("Failed to read composite filter for quantities");
                    goto done;
                  }
                  RaveList_freeAndDestroy(&quantities);
                }
              } else if (strcasecmp("interpolation_methods", SimpleXmlNode_getName(attr)) == 0) {
                const char* text = SimpleXmlNode_getText(attr);
                if (text != NULL) {
                  interpolation_methods = RaveUtilities_getTrimmedTokens(text, ',');
                  if (interpolation_methods == NULL|| !CompositeFilter_setInterpolationMethodsList(cfilter, interpolation_methods)) {
                    RAVE_ERROR0("Failed to read composite filter for interpolation_methods");
                    goto done;
                  }
                  RaveList_freeAndDestroy(&interpolation_methods);
                }
              }
            }
            RAVE_OBJECT_RELEASE(attr);
          }
          if (!RaveObjectList_add(cfentry->filters, (RaveCoreObject*)cfilter)) {
            RAVE_ERROR0("Failed to add filter to filter entries");
          }
          RAVE_OBJECT_RELEASE(cfilter);
        }
        RAVE_OBJECT_RELEASE(filter);
      }
      if (CompositeFactoryManager_isRegistered(generator->manager, cfentry->factory_class)) {
        cfentry->factory = CompositeFactoryManager_get(generator->manager, cfentry->factory_class);
        if (!RaveObjectList_add(generator->factories, (RaveCoreObject*)cfentry)) {
          RAVE_ERROR0("Failed to add factory entry to generator");
          goto done;
        }
      } else {
        RAVE_WARNING1("No factory class called %s. Check factory configuration.", cfentry->factory_class);
        goto done;
      }
      RAVE_OBJECT_RELEASE(cfentry);
    }
    RAVE_OBJECT_RELEASE(factory);
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(node);
  RAVE_OBJECT_RELEASE(attr);
  RAVE_OBJECT_RELEASE(filter);
  RAVE_OBJECT_RELEASE(factory);
  RAVE_OBJECT_RELEASE(cfentry);
  RAVE_OBJECT_RELEASE(cfilter);
  if (products != NULL) {
    RaveList_freeAndDestroy(&products);
  }
  if (quantities != NULL) {
    RaveList_freeAndDestroy(&quantities);
  }
  if (interpolation_methods != NULL) {
    RaveList_freeAndDestroy(&interpolation_methods);
  }
  return result;
#else
  return 0;
#endif
}

/*@} End of Private functions */

/*@{ Interface functions */
CompositeGenerator_t* CompositeGenerator_create(CompositeFactoryManager_t* manager, const char* filename)
{
  CompositeGenerator_t* result = NULL;
  result = RAVE_OBJECT_NEW(&CompositeGenerator_TYPE);
  if (result != NULL) {
    if (manager != NULL) {
      if (CompositeFactoryManager_size(manager) == 0) {
        RAVE_ERROR0("Can't create a generator with an empty factory manager");
        goto fail;
      }
      result->manager = RAVE_OBJECT_COPY(manager);
    } else {
      result->manager = RAVE_OBJECT_NEW(&CompositeFactoryManager_TYPE);
      if (result->manager == NULL) {
        RAVE_ERROR0("Failed to create factory manager manager");
        goto fail;
      }
    }

    if (filename != NULL) {
      if (!CompositeGeneratorInternal_loadXml(result, filename)) {
        RAVE_ERROR0("Failed to load filter definition file");
        goto fail;
      }
    } else {
      int i = 0, nlen = 0;
      RaveObjectList_t* factories = CompositeFactoryManager_getRegisteredFactories(result->manager);
      if (factories == NULL) {
        RAVE_ERROR0("No factories!!!");
        goto fail;
      }
      nlen = RaveObjectList_size(factories);
      for (i = 0; i < nlen; i++) {
        CompositeGeneratorFactory_t* factory = (CompositeGeneratorFactory_t*)RaveObjectList_get(factories, i);
        if (factory == NULL || !CompositeGenerator_register(result, CompositeGeneratorFactory_getDefaultId(factory), factory, NULL)) {
          RAVE_ERROR0("Failed to register default factory");
          goto fail;
        }
        RAVE_OBJECT_RELEASE(factory);
      }
      RAVE_OBJECT_RELEASE(factories);
    }
  }

  return result;
fail:
  RAVE_OBJECT_RELEASE(result);
  return NULL;
}

int CompositeGenerator_setProperties(CompositeGenerator_t* generator, RaveProperties_t* properties)
{
  RAVE_ASSERT((generator != NULL), "generator == NULL");
  RAVE_OBJECT_RELEASE(generator->properties);
  if (properties != NULL) {
    generator->properties = RAVE_OBJECT_COPY(properties);
  }
  return 1;
}

RaveProperties_t* CompositeGenerator_getProperties(CompositeGenerator_t* generator)
{
  RAVE_ASSERT((generator != NULL), "generator == NULL");
  return RAVE_OBJECT_COPY(generator->properties);
}

int CompositeGenerator_register(CompositeGenerator_t* generator, const char* id, CompositeGeneratorFactory_t* factory, RaveObjectList_t* filters)
{
  CompositeFactoryEntry_t* factoryEntry = NULL;
  int result = 0;
  RAVE_ASSERT((generator != NULL), "generator == NULL");
  if (id == NULL || factory == NULL) {
    RAVE_ERROR0("Must provide both id and factory");
    return 0;
  }

  factoryEntry = RAVE_OBJECT_NEW(&CompositeFactoryEntry_TYPE);
  if (factoryEntry == NULL) {
    RAVE_ERROR0("Failed to create factory entry");
    goto fail;
  }
  factoryEntry->name = RAVE_STRDUP(id);
  if (factoryEntry->name == NULL) {
    RAVE_ERROR0("Failed to duplicate name");
    goto fail;
  }
  strcpy(factoryEntry->name, id);

  factoryEntry->factory = RAVE_OBJECT_COPY(factory);

  if (filters != NULL) {
    RAVE_OBJECT_RELEASE(factoryEntry->filters);
    factoryEntry->filters = RAVE_OBJECT_COPY(filters);
  } else {
    RAVE_OBJECT_RELEASE(factoryEntry->filters);
    factoryEntry->filters = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
    if (factoryEntry->filters == NULL) {
      RAVE_ERROR0("Failed to create RaveObjectList");
      goto fail;
    }
  }
  result = RaveObjectList_add(generator->factories, (RaveCoreObject*)factoryEntry);
  RAVE_OBJECT_RELEASE(factoryEntry);
  return result;
fail:
  RAVE_OBJECT_RELEASE(factoryEntry);
  return 0;
}

RaveList_t* CompositeGenerator_getFactoryIDs(CompositeGenerator_t* generator)
{
  RaveList_t *result = NULL, *rlist = NULL;
  RAVE_ASSERT((generator != NULL), "generator == NULL");
  rlist = RAVE_OBJECT_NEW(&RaveList_TYPE);
  if (rlist != NULL) {
    int i = 0, nlen = RaveObjectList_size(generator->factories);
    for (i = 0; i < nlen; i++) {
      CompositeFactoryEntry_t* entry = (CompositeFactoryEntry_t*)RaveObjectList_get(generator->factories, i);
      char* tmps = RAVE_STRDUP(entry->name);
      if (tmps == NULL || !RaveList_add(rlist, tmps)) {
        RAVE_ERROR0("Failed to add string to list");
        RAVE_OBJECT_RELEASE(entry);
        RAVE_FREE(tmps);
        goto done;
      }
      RAVE_OBJECT_RELEASE(entry);
    }
  }

  result = RAVE_OBJECT_COPY(rlist);
done:
  RAVE_OBJECT_RELEASE(rlist);
  return result;
}

void CompositeGenerator_unregister(CompositeGenerator_t* generator, const char* id)
{
  int i = 0, nlen = 0;
  RAVE_ASSERT((generator != NULL), "generator == NULL");
  if (id == NULL) {
    return;
  }

  nlen = RaveObjectList_size(generator->factories);
  for (i = 0; i < nlen; i++) {
    CompositeFactoryEntry_t* entry = (CompositeFactoryEntry_t*)RaveObjectList_get(generator->factories, i);
    if (strcasecmp(entry->name, id) == 0) {
      CompositeFactoryEntry_t* entryToRemove = (CompositeFactoryEntry_t*)RaveObjectList_remove(generator->factories, i);
      RAVE_OBJECT_RELEASE(entryToRemove);
      return;
    }
  }
}

CompositeGeneratorFactory_t* CompositeGenerator_identify(CompositeGenerator_t* generator, CompositeArguments_t* arguments)
{
  CompositeGeneratorFactory_t* factory = NULL;

  RAVE_ASSERT((generator != NULL), "generator == NULL");
  if (arguments == NULL) {
    RAVE_ERROR0("Must provide arguments when generating a product");
    return NULL;
  }

  /* If strategy is set, this will override any filters. */
  if (CompositeArguments_getStrategy(arguments) != NULL) {
    int i = 0, leni = RaveObjectList_size(generator->factories);
    for (i = 0; factory == NULL && i < leni; i++) {
      CompositeFactoryEntry_t* cfentry = (CompositeFactoryEntry_t*)RaveObjectList_get(generator->factories, i);
      if (strcasecmp(cfentry->name, CompositeArguments_getStrategy(arguments))==0) {
        factory = RAVE_OBJECT_COPY(cfentry->factory);
      }
      RAVE_OBJECT_RELEASE(cfentry);
    }
  }

  /* If there is no factory with registered name, then we will check for any factory that can support provided arguments
     If no strategy. Check if factory filters has been set.
   */
  if (factory == NULL) {
    if (RaveObjectList_size(generator->factories) > 0) {
      int i = 0, leni = RaveObjectList_size(generator->factories);
      int useCanHandle = 1;
      /* If no factory has any filter we revert to using canHandle instead of filters. This means that if one
       * factory has no filter but the other factories have. The factories without a filter will not be targeted
       * at all.*/
      for (i = 0; useCanHandle && i < leni; i++) {
        CompositeFactoryEntry_t* cfentry = (CompositeFactoryEntry_t*)RaveObjectList_get(generator->factories, i);
        if (RaveObjectList_size(cfentry->filters) != 0) {
          useCanHandle = 0;
        }
        RAVE_OBJECT_RELEASE(cfentry);
      }
      if (!useCanHandle) {
        for (i = 0; factory == NULL && i < leni; i++) {
          CompositeFactoryEntry_t* cfentry = (CompositeFactoryEntry_t*)RaveObjectList_get(generator->factories, i);

          int j = 0, lenj = RaveObjectList_size(cfentry->filters);
          if (lenj == 0) {
            factory = RAVE_OBJECT_COPY(cfentry->factory);
          } else {
            for (j = 0; factory == NULL && j < lenj; j++) {
              CompositeFilter_t* filter = (CompositeFilter_t*)RaveObjectList_get(cfentry->filters, j);
              if (CompositeFilter_match(filter, arguments)) {
                factory = RAVE_OBJECT_COPY(cfentry->factory);
              }
              RAVE_OBJECT_RELEASE(filter);
            }
          }
          RAVE_OBJECT_RELEASE(cfentry);
        }
      } else {
        for (i = 0; factory == NULL && i < leni; i++) {
          CompositeFactoryEntry_t* cfentry = (CompositeFactoryEntry_t*)RaveObjectList_get(generator->factories, i);
          if (CompositeGeneratorFactory_canHandle(cfentry->factory, arguments)) {
            factory = RAVE_OBJECT_COPY(cfentry->factory);
          }
          RAVE_OBJECT_RELEASE(cfentry);
        }
      }
    }
  }

  return factory;
}

CompositeGeneratorFactory_t* CompositeGenerator_createFactory(CompositeGenerator_t* generator, CompositeArguments_t* arguments)
{
  CompositeGeneratorFactory_t *factory = NULL, *result = NULL;
  RAVE_ASSERT((generator != NULL), "generator == NULL");
  factory = CompositeGenerator_identify(generator, arguments);
  if (factory != NULL) {
    result = CompositeGeneratorFactory_create(factory);
    if (result != NULL) {
      if (!CompositeGeneratorFactory_setProperties(result, generator->properties)) {
        RAVE_OBJECT_RELEASE(result);
        RAVE_ERROR0("Failed to initialize generator factory with properties");
      }
    }
  }

  RAVE_OBJECT_RELEASE(factory);
  return result;
}


Cartesian_t* CompositeGenerator_generate(CompositeGenerator_t* generator, CompositeArguments_t* arguments)
{
  Cartesian_t* result = NULL;
  CompositeGeneratorFactory_t* factory = NULL;
  RAVE_ASSERT((generator != NULL), "generator == NULL");
  if (arguments == NULL) {
    RAVE_ERROR0("Must provide arguments when generating a product");
    return NULL;
  }
  factory = CompositeGenerator_identify(generator, arguments);
  if (factory != NULL) {
    CompositeGeneratorFactory_t* worker = CompositeGeneratorFactory_create(factory);
    if (worker != NULL) {
      if (CompositeGeneratorFactory_setProperties(worker, generator->properties)) {
        result = CompositeGeneratorFactory_generate(worker, arguments);
        if (result != NULL) {
          RaveAttribute_t* attr = RaveAttributeHelp_createStringFmt("how/product_parameters/factory", "%s", CompositeGeneratorFactory_getName(worker));
          if (attr != NULL) {
            Cartesian_addAttribute(result, attr);
          }
          RAVE_OBJECT_RELEASE(attr);
          attr = RaveAttributeHelp_createStringFmt("how/software", "BALTRAD");
          if (attr != NULL) {
            Cartesian_addAttribute(result, attr);
          }
          RAVE_OBJECT_RELEASE(attr);
        }
      } else {
        RAVE_ERROR0("Failed to initialize generator factory with properties");
      }
    }
    RAVE_OBJECT_RELEASE(worker);
  }
  RAVE_OBJECT_RELEASE(factory);
  return result;
}

/*@{ Interface functions */

/*@} End of Interface functions */

RaveCoreObjectType CompositeGenerator_TYPE = {
    "CompositeGenerator",
    sizeof(CompositeGenerator_t),
    CompositeGenerator_constructor,
    CompositeGenerator_destructor,
    CompositeGenerator_copyconstructor
};

