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
 * Interface for defining your own compositing factory. If CompositeGeneratorFactory_canHandle(...) returns true,
 * then CompositeGeneratorFactory_generate(....) will be called. 
 *
 * NOTE! It is essential that the factory implements the copy constructor since the factory will be set in the
 * manager and the manager keeps a list of instances.
 *
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-10-10
 */
#ifndef COMPOSITE_GENERATOR_FACTORY_H
#define COMPOSITE_GENERATOR_FACTORY_H

#include "compositearguments.h"
#include "cartesian.h"
#include "rave_object.h"
#include "rave_types.h"
#include "raveobject_list.h"
#include "rave_field.h"
#include "rave_properties.h"

/**
 * Forward declaration of struct
 */
struct _CompositeGeneratorFactory_t;

/**
 * @returns the unique name for this composite generator factory
 */
typedef const char*(*composite_generator_factory_getName_fun)(struct _CompositeGeneratorFactory_t* self);

/**
 * @returns the default id for this this composite generator factory
 */
typedef const char*(*composite_generator_factory_getDefaultId_fun)(struct _CompositeGeneratorFactory_t* self);

/**
 * @returns if this factory can handle the generator request
 */
typedef int(*composite_generator_factory_canHandle_fun)(struct _CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments);

/**
 * Sets properties in the factory
 */
typedef int(*composite_generator_factory_setProperties_fun)(struct _CompositeGeneratorFactory_t* self, RaveProperties_t* properties);

/**
 * @return properties from the factory
 */
 typedef RaveProperties_t*(*composite_generator_factory_getProperties_fun)(struct _CompositeGeneratorFactory_t* self);

/**
 * @returns the result from the generation
 */
typedef Cartesian_t*(*composite_generator_factory_generate_fun)(struct _CompositeGeneratorFactory_t* self, CompositeArguments_t* arguments);

/**
 * The factory creation method for the generator. Should return a (new) instance of the factory.
 * After a factory has been registered in the composite generator it will create a new factory of this same instance each time
 * a call to generate is called.
 * @return 1 on success otherwise 0
 */
typedef struct _CompositeGeneratorFactory_t*(*composite_generator_factory_create_fun)(struct _CompositeGeneratorFactory_t* self);

/**
 * The head part for a CompositeGeneratorFactory subclass. Should be placed directly under
 * RAVE_OBJECT_HEAD like in CompositeGeneratorFactory_t.
 */
#define COMPOSITE_GENERATOR_FACTORY_HEAD \
  composite_generator_factory_getName_fun getName; \
  composite_generator_factory_getDefaultId_fun getDefaultId; \
  composite_generator_factory_canHandle_fun canHandle; \
  composite_generator_factory_setProperties_fun setProperties; \
  composite_generator_factory_getProperties_fun getProperties; \
  composite_generator_factory_generate_fun generate; \
  composite_generator_factory_create_fun create;

/**
 * The basic composite algorithm that can be cast into a subclassed processor.
 */
typedef struct _CompositeGeneratorFactory_t {
  RAVE_OBJECT_HEAD /**< Always on top */
  COMPOSITE_GENERATOR_FACTORY_HEAD /**< composite specifics */
} CompositeGeneratorFactory_t;

/**
 * Macro expansion for calling the name function
 * @param[in] self - self
 * @returns the unique name for this algorithm
 */
#define CompositeGeneratorFactory_getName(self) \
  ((CompositeGeneratorFactory_t*)self)->getName((CompositeGeneratorFactory_t*)self)

/**
 * Macro expansion for calling the default id function
 * @param[in] self - self
 * @returns the default id for this factory
 */
#define CompositeGeneratorFactory_getDefaultId(self) \
  ((CompositeGeneratorFactory_t*)self)->getDefaultId((CompositeGeneratorFactory_t*)self)

/**
 * Macro expansion if this plugin supports generate or not
 */
#define CompositeGeneratorFactory_canHandle(self, args) \
    ((CompositeGeneratorFactory_t*)self)->canHandle((CompositeGeneratorFactory_t*)self, args)

/**
 * Macro expansion for calling the set properties function
 * @param[in] self - self
 * @param[in] properties - properties
 * @returns 1 on success otherwise 0
 */
#define CompositeGeneratorFactory_setProperties(self, properties) \
((CompositeGeneratorFactory_t*)self)->setProperties((CompositeGeneratorFactory_t*)self, properties)

/**
 * Macro expansion for calling the get properties function
 * @param[in] self - self
 * @returns the properties or NULL
 */
 #define CompositeGeneratorFactory_getProperties(self) \
 ((CompositeGeneratorFactory_t*)self)->getProperties((CompositeGeneratorFactory_t*)self)
 
/**
 * Macro expansion for calling the generate function
 * @param[in] self - self
 * @returns The cartesian product on success
 */
#define CompositeGeneratorFactory_generate(self, args) \
    ((CompositeGeneratorFactory_t*)self)->generate((CompositeGeneratorFactory_t*)self, args)

/**
 * Macro expansion for initializing the plugin
 * @param[in] self - self
 * @returns 1 on success otherwise 0
 */
#define CompositeGeneratorFactory_create(self) \
    ((CompositeGeneratorFactory_t*)self)->create((CompositeGeneratorFactory_t*)self)

#endif /* COMPOSITE_GENERATOR_FACTORY_H */
