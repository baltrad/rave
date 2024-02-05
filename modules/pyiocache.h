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
 * Python version of the RaveIOCache API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-02-05
 */
#ifndef PYRAVEIOCACHE_H
#define PYRAVEIOCACHE_H
#include "rave_iocache.h"

/**
 * The RaveIOCache
 */
typedef struct {
  PyObject_HEAD /* Always has to be on top */
  RaveIOCache_t* iocache;  /**< the iocache instance */
} PyRaveIOCache;

#define PyRaveIOCache_Type_NUM 0                          /**< index for Type */

#define PyRaveIOCache_GetNative_NUM 1                     /**< index for GetNative */
#define PyRaveIOCache_GetNative_RETURN RaveIOCache_t*     /**< return type for GetNative */
#define PyRaveIOCache_GetNative_PROTO (PyRaveIOCache*)    /**< argument prototype for GetNative */

#define PyRaveIOCache_New_NUM 2                           /**< index for New */
#define PyRaveIOCache_New_RETURN PyRaveIOCache*           /**< return type for New */
#define PyRaveIOCache_New_PROTO (RaveIOCache_t*)          /**< argument prototype for New */

#define PyRaveIOCache_API_pointers 3                      /**< Total number of C API pointers */

#define PyRaveIOCache_CAPSULE_NAME "_iocache._C_API"

#ifdef PYRAVEIOCACHE_MODULE
/** declared in pyraveio module */
extern PyTypeObject PyRaveIOCache_Type;

/** checks if the object is a PyRaveIOCache type or not */
#define PyRaveIOCache_Check(op) ((op)->ob_type == &PyRaveIOCache_Type)

/** Prototype for PyRaveIOCache modules GetNative function */
static PyRaveIOCache_GetNative_RETURN PyRaveIOCache_GetNative PyRaveIOCache_GetNative_PROTO;

/** Prototype for PyRaveIOCache modules New function */
static PyRaveIOCache_New_RETURN PyRaveIOCache_New PyRaveIOCache_New_PROTO;

#else
/** static pointer containing the pointers to function pointers and other definitions */
static void **PyRaveIOCache_API;

/**
 * Returns a pointer to the internal rave io, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyRaveIOCache_GetNative \
  (*(PyRaveIOCache_GetNative_RETURN (*)PyRaveIOCache_GetNative_PROTO) PyRaveIOCache_API[PyRaveIOCache_GetNative_NUM])

/**
 * Creates a new rave io cache instance. Release this object with Py_DECREF. If a RaveIOCache_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] raveio - the RaveIOCache_t intance.
 * @returns the PyRaveIOCache instance.
 */
#define PyRaveIOCache_New \
  (*(PyRaveIOCache_New_RETURN (*)PyRaveIOCache_New_PROTO) PyRaveIOCache_API[PyRaveIOCache_New_NUM])

/**
 * Checks if the object is a python rave io cache.
 */
#define PyRaveIOCache_Check(op) \
   (Py_TYPE(op) == &PyRaveIOCache_Type)

#define PyRaveIOCache_Type (*(PyTypeObject*)PyRaveIOCache_API[PyRaveIOCache_Type_NUM])

/**
 * Imports the PyRaveIOCache module (like import _iocache in python).
 */
#define import_iocache() \
    PyRaveIOCache_API = (void **)PyCapsule_Import(PyRaveIOCache_CAPSULE_NAME, 1);

#endif

#endif /* PYRAVEIOCACHE_H */
