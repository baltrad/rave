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
 * Python version of the Transform API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#ifndef PYBITMAP_GENERATOR_H
#define PYBITMAP_GENERATOR_H
#include "bitmap_generator.h"

/**
 * The definition
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  BitmapGenerator_t* generator;  /**< the c-api bitmap generator */
} PyBitmapGenerator;

#define PyBitmapGenerator_Type_NUM 0                     /**< index for Type */

#define PyBitmapGenerator_GetNative_NUM 1                /**< index for GetNative fp */
#define PyBitmapGenerator_GetNative_RETURN BitmapGenerator_t*  /**< Return type for GetNative */
#define PyBitmapGenerator_GetNative_PROTO (PyBitmapGenerator*) /**< Argument prototype for GetNative */

#define PyBitmapGenerator_New_NUM 2                      /**< index for New fp */
#define PyBitmapGenerator_New_RETURN PyBitmapGenerator*        /**< Return type for New */
#define PyBitmapGenerator_New_PROTO (BitmapGenerator_t*)       /**< Argument prototype for New */

#define PyBitmapGenerator_API_pointers 3                 /**< total number of C API pointers */

#ifdef PYBITMAP_GENERATOR_MODULE
/** declared in pybitmapgenerator module */
extern PyTypeObject PyBitmapGenerator_Type;

/** checks if the object is a PyBitmapGenerator type or not */
#define PyBitmapGenerator_Check(op) ((op)->ob_type == &PyBitmapGenerator_Type)

/** Prototype for PyBitmapGenerator modules GetNative function */
static PyBitmapGenerator_GetNative_RETURN PyBitmapGenerator_GetNative PyBitmapGenerator_GetNative_PROTO;

/** Prototype for PyBitmapGenerator modules New function */
static PyBitmapGenerator_New_RETURN PyBitmapGenerator_New PyBitmapGenerator_New_PROTO;

#else
/** static pointer containing the pointers to function pointers and other definitions */
static void **PyBitmapGenerator_API;

/**
 * Returns a pointer to the internal polar scan, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyBitmapGenerator_GetNative \
  (*(PyBitmapGenerator_GetNative_RETURN (*)PyBitmapGenerator_GetNative_PROTO) PyBitmapGenerator_API[PyBitmapGenerator_GetNative_NUM])

/**
 * Creates a new polar scan instance. Release this object with Py_DECREF.
 * @param[in] scan - the PolarScan_t intance.
 * @returns the PyBitmapGenerator instance.
 */
#define PyBitmapGenerator_New \
  (*(PyBitmapGenerator_New_RETURN (*)PyBitmapGenerator_New_PROTO) PyBitmapGenerator_API[PyBitmapGenerator_New_NUM])

/**
 * Checks if the object is a python polar scan.
 */
#define PyBitmapGenerator_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyBitmapGenerator_API[PyBitmapGenerator_Type_NUM])

/**
 * Imports the PyBitmapGenerator module (like import _polarscan in python).
 */
static int
import_pybitmapgenerator(void)
{
  PyObject *module;
  PyObject *c_api_object;

  module = PyImport_ImportModule("_bitmapgenerator");
  if (module == NULL) {
    return -1;
  }

  c_api_object = PyObject_GetAttrString(module, "_C_API");
  if (c_api_object == NULL) {
    Py_DECREF(module);
    return -1;
  }
  if (PyCObject_Check(c_api_object)) {
    PyBitmapGenerator_API = (void **)PyCObject_AsVoidPtr(c_api_object);
  }
  Py_DECREF(c_api_object);
  Py_DECREF(module);
  return 0;
}

#endif



#endif /* PYBITMAP_GENERATOR_H */
