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
 * Python version of the projection API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-07
 */
#ifndef PYPROJECTION_H
#define PYPROJECTION_H
#include "projection.h"

/**
 * A projection
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  Projection_t* projection; /**< The projection definition */
} PyProjection;

#define PyProjection_Type_NUM 0                        /**< index for Type */

#define PyProjection_GetNative_NUM 1                   /**< index for GetNative */
#define PyProjection_GetNative_RETURN Projection_t*    /**< return type for GetNative */
#define PyProjection_GetNative_PROTO (PyProjection*)   /**< argument prototype for GetNative */

#define PyProjection_New_NUM 2                         /**< index for New */
#define PyProjection_New_RETURN PyProjection*          /**< return type for New */
#define PyProjection_New_PROTO (Projection_t*)         /**< argument prototype for New */

#define PyProjection_NewFromDef_NUM 3                  /**< index for NewFromDef */
#define PyProjection_NewFromDef_RETURN PyProjection*   /**< return type for NewFromDef */
#define PyProjection_NewFromDef_PROTO (const char* id, const char* definition, const char* description) /**< argument prototype for NewFromDef */

#define PyProjection_API_pointers 4                    /**< number of function and variable pointers */

#ifdef PYPROJECTION_MODULE
/** To be used within the PyProjection-Module */
extern PyTypeObject PyProjection_Type;

/** Checks if the object is a PyProjection or not */
#define PyProjection_Check(op) ((op)->ob_type == &PyProjection_Type)

/**
 * forward declaration of PyProjection_GetNative.
 */
static PyProjection_GetNative_RETURN PyProjection_GetNative PyProjection_GetNative_PROTO;

/**
 * forward declaration of PyProjection_New.
 */
static PyProjection_New_RETURN PyProjection_New PyProjection_New_PROTO;

/**
 * forward declaration of PyProjection_NewFromDef.
 */
static PyProjection_NewFromDef_RETURN PyProjection_NewFromDef PyProjection_NewFromDef_PROTO;

#else
/** Pointers to the functions and variables */
static void **PyProjection_API;

/**
 * Returns a pointer to the internal projection, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyProjection_GetNative \
  (*(PyProjection_GetNative_RETURN (*)PyProjection_GetNative_PROTO) PyProjection_API[PyProjection_GetNative_NUM])

/**
 * Creates a new projection instance. Release this object with Py_DECREF. If the passed Projection_t instance is
 * bound to a python instance, this instance will be increfed and returned.
 * @param[in] proj - the Projection_t intance.
 * @returns the PyProjection instance.
 */
#define PyProjection_New \
  (*(PyProjection_New_RETURN (*)PyProjection_New_PROTO) PyProjection_API[PyProjection_New_NUM])

/**
 * Creates a new projection instance from a definition. Release this object with Py_DECREF.
 * @param[in] id - the id for this projection
 * @param[in] description - the description of this projection
 * @param[in] definition - the proj.4 definition
 * @returns the PyProjection instance.
 */
#define PyProjection_NewFromDef \
  (*(PyProjection_NewFromDef_RETURN (*)PyProjection_NewFromDef_PROTO) PyProjection_API[PyProjection_NewFromDef_NUM])

/**
 * Checks if the object is a python projection.
 */
#define PyProjection_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyProjection_API[PyProjection_Type_NUM])

/**
 * Imports the pyprojection module (like import _projection in python).
 */
static int
import_pyprojection(void)
{
  PyObject *module;
  PyObject *c_api_object;

  module = PyImport_ImportModule("_projection");
  if (module == NULL) {
    return -1;
  }

  c_api_object = PyObject_GetAttrString(module, "_C_API");
  if (c_api_object == NULL) {
    Py_DECREF(module);
    return -1;
  }
  if (PyCObject_Check(c_api_object)) {
    PyProjection_API = (void **)PyCObject_AsVoidPtr(c_api_object);
  }
  Py_DECREF(c_api_object);
  Py_DECREF(module);
  return 0;
}

#endif

#endif /* PYPROJECTION_H */
