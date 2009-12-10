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
 * Python version of the Cartesian API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#ifndef PYCARTESIAN_H
#define PYCARTESIAN_H
#include "cartesian.h"

/**
 * A cartesian product
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  Cartesian_t* cartesian; /**< the cartesian product */
} PyCartesian;

/* C API functions */
#define PyCartesian_Type_NUM 0

#define PyCartesian_GetNative_NUM 1
#define PyCartesian_GetNative_RETURN Cartesian_t*
#define PyCartesian_GetNative_PROTO (PyCartesian*)

#define PyCartesian_New_NUM 2
#define PyCartesian_New_RETURN PyCartesian*
#define PyCartesian_New_PROTO (Cartesian_t*)

/* Total number of C API pointers */
#define PyCartesian_API_pointers 3

#ifdef PYCARTESIAN_MODULE
/* To be used within the PyCartesian-Module */
extern PyTypeObject PyCartesian_Type;

#define PyCartesian_Check(op) ((op)->ob_type == &PyCartesian_Type)

static PyCartesian_GetNative_RETURN PyCartesian_GetNative PyCartesian_GetNative_PROTO;

static PyCartesian_New_RETURN PyCartesian_New PyCartesian_New_PROTO;

#else
/* This section is for clients using the PyCartesian API */
static void **PyCartesian_API;

/**
 * Returns a pointer to the internal polar scan, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyCartesian_GetNative \
  (*(PyCartesian_GetNative_RETURN (*)PyCartesian_GetNative_PROTO) PyCartesian_API[PyCartesian_GetNative_NUM])

/**
 * Creates a new polar scan instance. Release this object with Py_DECREF.
 * @param[in] scan - the PolarScan_t intance.
 * @returns the PyCartesian instance.
 */
#define PyCartesian_New \
  (*(PyCartesian_New_RETURN (*)PyCartesian_New_PROTO) PyCartesian_API[PyCartesian_New_NUM])

/**
 * Checks if the object is a python polar scan.
 */
#define PyCartesian_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyCartesian_API[PyCartesian_Type_NUM])

/**
 * Imports the PyCartesian module (like import _polarscan in python).
 */
static int
import_pycartesian(void)
{
  PyObject *module;
  PyObject *c_api_object;

  module = PyImport_ImportModule("_cartesian");
  if (module == NULL) {
    return -1;
  }

  c_api_object = PyObject_GetAttrString(module, "_C_API");
  if (c_api_object == NULL) {
    Py_DECREF(module);
    return -1;
  }
  if (PyCObject_Check(c_api_object)) {
    PyCartesian_API = (void **)PyCObject_AsVoidPtr(c_api_object);
  }
  Py_DECREF(c_api_object);
  Py_DECREF(module);
  return 0;
}

#endif

#endif /* PYCARTESIAN_H */
