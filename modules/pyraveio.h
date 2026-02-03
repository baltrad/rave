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
 * Python version of the RaveIO API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#ifndef PYRAVEIO_H
#define PYRAVEIO_H
#include "rave_io.h"

/**
 * The RaveIO
 */
typedef struct {
  PyObject_HEAD /* Always has to be on top */
  RaveIO_t* raveio;  /**< the raveio instance */
} PyRaveIO;

#define PyRaveIO_Type_NUM 0                     /**< index for Type */

#define PyRaveIO_GetNative_NUM 1                /**< index for GetNative */
#define PyRaveIO_GetNative_RETURN RaveIO_t*     /**< return type for GetNative */
#define PyRaveIO_GetNative_PROTO (PyRaveIO*)    /**< argument prototype for GetNative */

#define PyRaveIO_New_NUM 2                      /**< index for New */
#define PyRaveIO_New_RETURN PyRaveIO*           /**< return type for New */
#define PyRaveIO_New_PROTO (RaveIO_t*)          /**< argument prototype for New */

#define PyRaveIO_Open_NUM 3                     /**< index for Open */
#define PyRaveIO_Open_RETURN PyRaveIO*          /**< return type for Open */
#define PyRaveIO_Open_PROTO (const char* filename, int lazyLoading, const char* preloadQuantities) /**< argument prototype for Open */

#define PyRaveIO_OpenFileObject_NUM 4                     /**< index for Open */
#define PyRaveIO_OpenFileObject_RETURN PyRaveIO*          /**< return type for Open */
#define PyRaveIO_OpenFileObject_PROTO (const char* filename, int lazyLoading, const char* preloadQuantities) /**< argument prototype for Open */


#define PyRaveIO_API_pointers 5                 /**< Total number of C API pointers */

#define PyRaveIO_CAPSULE_NAME "_raveio._C_API"

#ifdef PYRAVEIO_MODULE
/** declared in pyraveio module */
extern PyTypeObject PyRaveIO_Type;

/** checks if the object is a PyRaveIO type or not */
#define PyRaveIO_Check(op) ((op)->ob_type == &PyRaveIO_Type)

/** Prototype for PyRaveIO modules GetNative function */
static PyRaveIO_GetNative_RETURN PyRaveIO_GetNative PyRaveIO_GetNative_PROTO;

/** Prototype for PyRaveIO modules New function */
static PyRaveIO_New_RETURN PyRaveIO_New PyRaveIO_New_PROTO;

/** Prototype for PyRaveIO modules Open function */
static PyRaveIO_Open_RETURN PyRaveIO_Open PyRaveIO_Open_PROTO;

/** Prototype for PyRaveIO modules Open function */
static PyRaveIO_OpenFileObject_RETURN PyRaveIO_OpenFileObject PyRaveIO_OpenFileObject_PROTO;

#else
/** static pointer containing the pointers to function pointers and other definitions */
static void **PyRaveIO_API;

/**
 * Returns a pointer to the internal rave io, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyRaveIO_GetNative \
  (*(PyRaveIO_GetNative_RETURN (*)PyRaveIO_GetNative_PROTO) PyRaveIO_API[PyRaveIO_GetNative_NUM])

/**
 * Creates a new rave io instance. Release this object with Py_DECREF. If a RaveIO_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] raveio - the RaveIO_t intance.
 * @returns the PyRaveIO instance.
 */
#define PyRaveIO_New \
  (*(PyRaveIO_New_RETURN (*)PyRaveIO_New_PROTO) PyRaveIO_API[PyRaveIO_New_NUM])

/**
 * Opens a rave io instance. Release this object with Py_DECREF.
 * @param[in] filename - the filename.
 * @returns the PyRaveIO instance.
 */
#define PyRaveIO_Open \
  (*(PyRaveIO_Open_RETURN (*)PyRaveIO_Open_PROTO) PyRaveIO_API[PyRaveIO_Open_NUM])

/**
 * Opens a rave io instance. Release this object with Py_DECREF.
 * @param[in] filename - the filename.
 * @returns the PyRaveIO instance.
 */
#define PyRaveIO_OpenFileObject \
  (*(PyRaveIO_OpenFileObject_RETURN (*)PyRaveIO_OpenFileObject_PROTO) PyRaveIO_API[PyRaveIO_OpenFileObject_NUM])

/**
 * Checks if the object is a python rave io.
 */
#define PyRaveIO_Check(op) \
   (Py_TYPE(op) == &PyRaveIO_Type)

#define PyRaveIO_Type (*(PyTypeObject*)PyRaveIO_API[PyRaveIO_Type_NUM])

/**
 * Imports the PyRaveIO module (like import _raveio in python).
 */
#define import_pyraveio() \
    PyRaveIO_API = (void **)PyCapsule_Import(PyRaveIO_CAPSULE_NAME, 1);


#ifdef KALLE
/**
 * Checks if the object is a python rave io.
 */
#define PyRaveIO_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyRaveIO_API[PyRaveIO_Type_NUM])

/**
 * Imports the PyRaveIO module (like import _polarscan in python).
 */
static int
import_pyraveio(void)
{
  PyObject *module;
  PyObject *c_api_object;

  module = PyImport_ImportModule("_raveio");
  if (module == NULL) {
    return -1;
  }

  c_api_object = PyObject_GetAttrString(module, "_C_API");
  if (c_api_object == NULL) {
    Py_DECREF(module);
    return -1;
  }
  if (PyCObject_Check(c_api_object)) {
    PyRaveIO_API = (void **)PyCObject_AsVoidPtr(c_api_object);
  }
  Py_DECREF(c_api_object);
  Py_DECREF(module);
  return 0;
}
#endif

#endif

#endif /* PYRAVEIO_H */
