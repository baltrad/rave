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
 * Python module for performing basic polar navigation.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-11
 */
#ifndef PYPOLARNAV_H
#define PYPOLARNAV_H
#include "polarnav.h"

/**
 * A polar navigator
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  PolarNavigator_t* navigator;
} PyPolarNavigator;

/* C API functions */
#define PyPolarNavigator_Type_NUM 0

#define PyPolarNavigator_GetNative_NUM 1
#define PyPolarNavigator_GetNative_RETURN PolarNavigator_t*
#define PyPolarNavigator_GetNative_PROTO (PyPolarNavigator*)

#define PyPolarNavigator_New_NUM 2
#define PyPolarNavigator_New_RETURN PyPolarNavigator*
#define PyPolarNavigator_New_PROTO (PolarNavigator_t*)

/* Total number of C API pointers */
#define PyPolarNavigator_API_pointers 3

#ifdef PYPOLARNAV_MODULE
/* To be used within the PyPolarNavigator-Module */
extern PyTypeObject PyPolarNavigator_Type;

#define PyPolarNavigator_Check(op) ((op)->ob_type == &PyPolarNavigator_Type)

static PyPolarNavigator_GetNative_RETURN PyPolarNavigator_GetNative PyPolarNavigator_GetNative_PROTO;

static PyPolarNavigator_New_RETURN PyPolarNavigator_New PyPolarNavigator_New_PROTO;

#else
/* This section is for clients using the PyPolarNavigator API */
static void **PyPolarNavigator_API;

/**
 * Returns a pointer to the internal polar scan, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyPolarNavigator_GetNative \
  (*(PyPolarNavigator_GetNative_RETURN (*)PyPolarNavigator_GetNative_PROTO) PyPolarNavigator_API[PyPolarNavigator_GetNative_NUM])

/**
 * Creates a new polar scan instance. Release this object with Py_DECREF.
 * @param[in] scan - the PolarScan_t intance.
 * @returns the PyPolarNavigator instance.
 */
#define PyPolarNavigator_New \
  (*(PyPolarNavigator_New_RETURN (*)PyPolarNavigator_New_PROTO) PyPolarNavigator_API[PyPolarNavigator_New_NUM])

/**
 * Checks if the object is a python polar scan.
 */
#define PyPolarNavigator_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyPolarNavigator_API[PyPolarNavigator_Type_NUM])

/**
 * Imports the PyPolarNavigator module (like import _polarscan in python).
 */
static int
import_pypolarnav(void)
{
  PyObject *module;
  PyObject *c_api_object;

  module = PyImport_ImportModule("_polarnav");
  if (module == NULL) {
    return -1;
  }

  c_api_object = PyObject_GetAttrString(module, "_C_API");
  if (c_api_object == NULL) {
    Py_DECREF(module);
    return -1;
  }
  if (PyCObject_Check(c_api_object)) {
    PyPolarNavigator_API = (void **)PyCObject_AsVoidPtr(c_api_object);
  }
  Py_DECREF(c_api_object);
  Py_DECREF(module);
  return 0;
}

#endif


#endif /* PYPOLARNAV_H */
