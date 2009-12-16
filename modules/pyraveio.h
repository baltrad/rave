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

/* C API functions */
#define PyRaveIO_Type_NUM 0

#define PyRaveIO_GetNative_NUM 1
#define PyRaveIO_GetNative_RETURN RaveIO_t*
#define PyRaveIO_GetNative_PROTO (PyRaveIO*)

#define PyRaveIO_New_NUM 2
#define PyRaveIO_New_RETURN PyRaveIO*
#define PyRaveIO_New_PROTO (RaveIO_t*)

#define PyRaveIO_Open_NUM 3
#define PyRaveIO_Open_RETURN PyRaveIO*
#define PyRaveIO_Open_PROTO (const char* filename)

/* Total number of C API pointers */
#define PyRaveIO_API_pointers 4

#ifdef PYRAVEIO_MODULE
/* To be used within the PyRaveIO-Module */
extern PyTypeObject PyRaveIO_Type;

#define PyRaveIO_Check(op) ((op)->ob_type == &PyRaveIO_Type)

static PyRaveIO_GetNative_RETURN PyRaveIO_GetNative PyRaveIO_GetNative_PROTO;

static PyRaveIO_New_RETURN PyRaveIO_New PyRaveIO_New_PROTO;

static PyRaveIO_Open_RETURN PyRaveIO_Open PyRaveIO_Open_PROTO;

#else
/* This section is for clients using the PyRaveIO API */
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

#endif /* PYRAVEIO_H */
