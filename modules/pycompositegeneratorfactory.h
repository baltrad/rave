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
 * Python version of the Compositing Generator Factory API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-12-09
 */
#ifndef PYCOMPOSITEGENERATORFACTORY_
#define PYCOMPOSITEGENERATORFACTORY_
#include "Python.h"
#include "compositegeneratorfactory.h"

/**
 * The poo composite algorithm instance
 */
typedef struct {
  PyObject_HEAD
  CompositeGeneratorFactory_t* factory;  /**< the composite generator factory */
} PyCompositeGeneratorFactory;

#define PyCompositeGeneratorFactory_Type_NUM 0                                           /**< index of type */

#define PyCompositeGeneratorFactory_GetNative_NUM 1                                      /**< index of GetNative*/
#define PyCompositeGeneratorFactory_GetNative_RETURN CompositeGeneratorFactory_t*         /**< return type for GetNative */
#define PyCompositeGeneratorFactory_GetNative_PROTO (PyCompositeGeneratorFactory*)        /**< arguments for GetNative */

#define PyCompositeGeneratorFactory_New_NUM 2                                            /**< index of New */
#define PyCompositeGeneratorFactory_New_RETURN PyCompositeGeneratorFactory*               /**< return type for New */
#define PyCompositeGeneratorFactory_New_PROTO (CompositeGeneratorFactory_t*)              /**< arguments for New */

#define PyCompositeGeneratorFactory_API_pointers 3                                      /**< number of type and function pointers */

#define PyCompositeGeneratorFactory_CAPSULE_NAME "_compositegeneratorfactory._C_API"

#ifdef PYCOMPOSITEGENERATORFACTORY_MODULE
/** Forward declaration of type */
extern PyTypeObject PyCompositeGeneratorFactory_Type;

/** Checks if the object is a PyPolarVolume or not */
#define PyCompositeGeneratorFactory_Check(op) ((op)->ob_type == &PyCompositeGeneratorFactory_Type)

/** Forward declaration of PyCompositeGeneratorFactory_GetNative */
static PyCompositeGeneratorFactory_GetNative_RETURN PyCompositeGeneratorFactory_GetNative PyCompositeGeneratorFactory_GetNative_PROTO;

/** Forward declaration of PyCompositeGeneratorFactory_New */
static PyCompositeGeneratorFactory_New_RETURN PyCompositeGeneratorFactory_New PyCompositeGeneratorFactory_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyCompositeGeneratorFactory_API;

/**
 * Returns a pointer to the internal composite generator plugin, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyCompositeGeneratorFactory_GetNative \
  (*(PyCompositeGeneratorFactory_GetNative_RETURN (*)PyCompositeGeneratorFactory_GetNative_PROTO) PyCompositeGeneratorFactory_API[PyCompositeGeneratorFactory_GetNative_NUM])

/**
 * Creates a composite generator factory instance. Release this object with Py_DECREF. If a CompositeGeneratorFactory_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] volume - the CompositeGeneratorFactory_t intance.
 * @returns the PyCompositeGeneratorFactory instance.
 */
#define PyCompositeGeneratorFactory_New \
  (*(PyCompositeGeneratorFactory_New_RETURN (*)PyCompositeGeneratorFactory_New_PROTO) PyCompositeGeneratorFactory_API[PyCompositeGeneratorFactory_New_NUM])


/**
 * Checks if the object is a python composite generator plugin.
 */
#define PyCompositeGeneratorFactory_Check(op) \
   (Py_TYPE(op) == &PyCompositeGeneratorFactory_Type)

#define PyCompositeGeneratorFactory_Type (*(PyTypeObject*)PyCompositeGeneratorFactory_API[PyCompositeGeneratorFactory_Type_NUM])

/**
 * Imports the PyCompositeGeneratorFactory module (like import _compositegeneratorfactory in python).
 */
#define import_compositegeneratorfactory() \
    PyCompositeGeneratorFactory_API = (void **)PyCapsule_Import(PyCompositeGeneratorFactory_CAPSULE_NAME, 1);

#endif

#endif /* PYCOMPOSITEGENERATORFACTORY_H */
