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
 * Python version of the Projection registry API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-15
 */
#ifndef PYPROJECTIONREGISTRY_H
#define PYPROJECTIONREGISTRY_H
#include "projectionregistry.h"

/**
 * A projection registry
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  ProjectionRegistry_t* registry; /**< the projection registry */
} PyProjectionRegistry;

#define PyProjectionRegistry_Type_NUM 0                   /**< index of type */

#define PyProjectionRegistry_GetNative_NUM 1              /**< index of GetNative */
#define PyProjectionRegistry_GetNative_RETURN ProjectionRegistry_t*     /**< return type for GetNative */
#define PyProjectionRegistry_GetNative_PROTO (PyProjectionRegistry*)    /**< arguments for GetNative */

#define PyProjectionRegistry_New_NUM 2                    /**< index of New */
#define PyProjectionRegistry_New_RETURN PyProjectionRegistry*           /**< return type for New */
#define PyProjectionRegistry_New_PROTO (ProjectionRegistry_t*)          /**< arguments for New */

#define PyProjectionRegistry_Load_NUM 3                     /**< index for Load */
#define PyProjectionRegistry_Load_RETURN PyProjectionRegistry*          /**< return type for Load */
#define PyProjectionRegistry_Load_PROTO (const char* filename) /**< argument prototype for Open */

#define PyProjectionRegistry_API_pointers 4               /**< number of API pointers */

#ifdef PYPROJECTIONREGISTRY_MODULE
/** Forward declaration of type */
extern PyTypeObject PyProjectionRegistry_Type;

/** Checks if the object is a PyArea or not */
#define PyProjectionRegistry_Check(op) ((op)->ob_type == &PyProjectionRegistry_Type)

/** Forward declaration of PyArea_GetNative */
static PyProjectionRegistry_GetNative_RETURN PyProjectionRegistry_GetNative PyProjectionRegistry_GetNative_PROTO;

/** Forward declaration of PyArea_New */
static PyProjectionRegistry_New_RETURN PyProjectionRegistry_New PyProjectionRegistry_New_PROTO;

/** Prototype for PyProjectionRegistry modules Load function */
static PyProjectionRegistry_Load_RETURN PyProjectionRegistry_Load PyProjectionRegistry_Load_PROTO;

#else
/** Pointers to types and functions */
static void **PyProjectionRegistry_API;

/**
 * Returns a pointer to the internal registry, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyProjectionRegistry_GetNative \
  (*(PyProjectionRegistry_GetNative_RETURN (*)PyProjectionRegistry_GetNative_PROTO) PyProjectionRegistry_API[PyProjectionRegistry_GetNative_NUM])

/**
 * Creates a projection registry instance. Release this object with Py_DECREF.  If a ProjectionRegistry_t registry is
 * provided and this registry already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] registry - the ProjectionRegistry_t intance.
 * @returns the PyProjectionRegistry instance.
 */
#define PyProjectionRegistry_New \
  (*(PyProjectionRegistry_New_RETURN (*)PyProjectionRegistry_New_PROTO) PyProjectionRegistry_API[PyProjectionRegistry_New_NUM])

/**
 * Loads a projection registry instance. Release this object with Py_DECREF.
 * @param[in] filename - the filename.
 * @returns the PyProjectionRegistry instance.
 */
#define PyProjectionRegistry_Load \
  (*(PyProjectionRegistry_Load_RETURN (*)PyProjectionRegistry_Load_PROTO) PyProjectionRegistry_API[ProjectionRegistry_Load_NUM])

/**
 * Checks if the object is a python projection registry.
 */
#define PyProjectionRegistry_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyProjectionRegistry_API[PyProjectionRegistry_Type_NUM])

/**
 * Imports the PyProjectionRegistry module (like import _projectionregistry in python).
 */
static int
import_pyprojectionregistry(void)
{
  PyObject *module;
  PyObject *c_api_object;

  module = PyImport_ImportModule("_projectionregistry");
  if (module == NULL) {
    return -1;
  }

  c_api_object = PyObject_GetAttrString(module, "_C_API");
  if (c_api_object == NULL) {
    Py_DECREF(module);
    return -1;
  }
  if (PyCObject_Check(c_api_object)) {
    PyProjectionRegistry_API = (void **)PyCObject_AsVoidPtr(c_api_object);
  }
  Py_DECREF(c_api_object);
  Py_DECREF(module);
  return 0;
}

#endif

#endif /* PYPROJECTIONREGISTRY_H */
