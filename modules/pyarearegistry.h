/* --------------------------------------------------------------------
Copyright (C) 2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the Area registry API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#ifndef PYAREAREGISTRY_H
#define PYAREAREGISTRY_H
#include "arearegistry.h"
#include "pyprojectionregistry.h"

/**
 * A cartesian product
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  AreaRegistry_t* registry; /**< the area registry */
} PyAreaRegistry;

#define PyAreaRegistry_Type_NUM 0                   /**< index of type */

#define PyAreaRegistry_GetNative_NUM 1              /**< index of GetNative */
#define PyAreaRegistry_GetNative_RETURN AreaRegistry_t*     /**< return type for GetNative */
#define PyAreaRegistry_GetNative_PROTO (PyAreaRegistry*)    /**< arguments for GetNative */

#define PyAreaRegistry_New_NUM 2                    /**< index of New */
#define PyAreaRegistry_New_RETURN PyAreaRegistry*           /**< return type for New */
#define PyAreaRegistry_New_PROTO (AreaRegistry_t*)          /**< arguments for New */

#define PyAreaRegistry_Load_NUM 3                     /**< index for Load */
#define PyAreaRegistry_Load_RETURN PyAreaRegistry*          /**< return type for Load */
#define PyAreaRegistry_Load_PROTO (const char* filename, PyProjectionRegistry* pyprojregistry) /**< argument prototype for Open */

#define PyAreaRegistry_API_pointers 4               /**< number of API pointers */

#define PyAreaRegistry_CAPSULE_NAME "_area._C_API"

#ifdef PYAREAREGISTRY_MODULE
/** Forward declaration of type */
extern PyTypeObject PyAreaRegistry_Type;

/** Checks if the object is a PyArea or not */
#define PyAreaRegistry_Check(op) ((op)->ob_type == &PyAreaRegistry_Type)

/** Forward declaration of PyArea_GetNative */
static PyAreaRegistry_GetNative_RETURN PyAreaRegistry_GetNative PyAreaRegistry_GetNative_PROTO;

/** Forward declaration of PyArea_New */
static PyAreaRegistry_New_RETURN PyAreaRegistry_New PyAreaRegistry_New_PROTO;

/** Prototype for PyProjectionRegistry modules Load function */
static PyAreaRegistry_Load_RETURN PyAreaRegistry_Load PyAreaRegistry_Load_PROTO;

#else
/** Pointers to types and functions */
static void **PyAreaRegistry_API;

/**
 * Returns a pointer to the internal area, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyAreaRegistry_GetNative \
  (*(PyAreaRegistry_GetNative_RETURN (*)PyAreaRegistry_GetNative_PROTO) PyAreaRegistry_API[PyAreaRegistry_GetNative_NUM])

/**
 * Creates a area registry instance. Release this object with Py_DECREF.  If a AreaRegistry_t area is
 * provided and this registry already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] area - the Area_t intance.
 * @returns the PyArea instance.
 */
#define PyAreaRegistry_New \
  (*(PyAreaRegistry_New_RETURN (*)PyAreaRegistry_New_PROTO) PyAreaRegistry_API[PyAreaRegistry_New_NUM])

/**
 * Loads a area registry instance. Release this object with Py_DECREF.
 * @param[in] filename - the filename.
 * @param[in] pyprojregistry - the py projection registry
 * @returns the PyAreaRegistry instance.
 */
#define PyAreaRegistry_Load \
  (*(PyAreaRegistry_Load_RETURN (*)PyAreaRegistry_Load_PROTO) PyAreaRegistry_API[PyAreaRegistry_Load_NUM])

/**
 * Checks if the object is a python area registry.
 */
#define PyAreaRegistry_Check(op) \
   (Py_TYPE(op) == &PyAreaRegistry_Type)

#define PyAreaRegistry_Type (*(PyTypeObject*)PyAreaRegistry_API[PyAreaRegistry_Type_NUM])

/**
 * Imports the PyAreaRegistry module (like import _arearegistry in python).
 */
#define import_pyarearegistry() \
    PyAreaRegistry_API = (void **)PyCapsule_Import(PyAreaRegistry_CAPSULE_NAME, 1);

#endif

#endif /* PYAREAREGISTRY_H */
