/* --------------------------------------------------------------------
Copyright (C) 2009 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the Composite Factory Manager API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-30
 */
#ifndef PYCOMPOSITEFACTORYMANAGER_H
#define PYCOMPOSITEFACTORYMANAGER_H
#include "compositefactorymanager.h"

/**
 * A composite generator
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  CompositeFactoryManager_t* manager; /**< the composite generator */
} PyCompositeFactoryManager;

#define PyCompositeFactoryManager_Type_NUM 0                       /**< index of type */

#define PyCompositeFactoryManager_GetNative_NUM 1                  /**< index of GetNative */
#define PyCompositeFactoryManager_GetNative_RETURN CompositeFactoryManager_t*    /**< return type for GetNative */
#define PyCompositeFactoryManager_GetNative_PROTO (PyCompositeFactoryManager*)   /**< arguments for GetNative */

#define PyCompositeFactoryManager_New_NUM 2                        /**< index of New */
#define PyCompositeFactoryManager_New_RETURN PyCompositeFactoryManager*          /**< return type for New */
#define PyCompositeFactoryManager_New_PROTO (CompositeFactoryManager_t*)         /**< arguments for New */

#define PyCompositeFactoryManager_API_pointers 3                   /**< number of api pointers */

#define PyCompositeFactoryManager_CAPSULE_NAME "_compositefactorymanager._C_API"


#ifdef PYCOMPOSITEFACTORYMANAGER_MODULE
/** Forward declaration of type*/
extern PyTypeObject PyCompositeFactoryManager_Type;

/** Checks if the object is a PyComposite or not */
#define PyCompositeFactoryManager_Check(op) ((op)->ob_type == &PyCompositeFactoryManager_Type)

/** Forward declaration of PyCompositeFactoryManager_GetNative */
static PyCompositeFactoryManager_GetNative_RETURN PyCompositeFactoryManager_GetNative PyCompositeFactoryManager_GetNative_PROTO;

/** Forward declaration of PyCompositeFactoryManager_New */
static PyCompositeFactoryManager_New_RETURN PyCompositeFactoryManager_New PyCompositeFactoryManager_New_PROTO;

#else
/** pointers to types and functions */
static void **PyCompositeFactoryManager_API;

/**
 * Returns a pointer to the internal composite, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyCompositeFactoryManager_GetNative \
  (*(PyCompositeFactoryManager_GetNative_RETURN (*)PyCompositeFactoryManager_GetNative_PROTO) PyCompositeFactoryManager_API[PyCompositeFactoryManager_GetNative_NUM])

/**
 * Creates a new composite generator instance. Release this object with Py_DECREF. If a CompositeFactoryManager_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] generator - the CompositeFactoryManager_t intance.
 * @returns the PyCompositeFactoryManager instance.
 */
#define PyCompositeFactoryManager_New \
  (*(PyCompositeFactoryManager_New_RETURN (*)PyCompositeFactoryManager_New_PROTO) PyCompositeFactoryManager_API[PyCompositeFactoryManager_New_NUM])

/**
 * Checks if the object is a python composite.
 */
#define PyCompositeFactoryManager_Check(op) \
   (Py_TYPE(op) == &PyCompositeFactoryManager_Type)

#define PyCompositeFactoryManager_Type (*(PyTypeObject*)PyCompositeFactoryManager_API[PyCompositeFactoryManager_Type_NUM])

/**
 * Imports the PyCompositeFactoryManager module (like import _compositefactorymanager in python).
 */
#define import_compositefactorymanager() \
    PyCompositeFactoryManager_API = (void **)PyCapsule_Import(PyCompositeFactoryManager_CAPSULE_NAME, 0);

#endif



#endif /* PYCOMPOSITEFACTORYMANAGER_H */
