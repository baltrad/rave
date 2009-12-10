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
 * Python version of the PolarScan API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-08
 */
#ifndef PYPOLARSCAN_H
#define PYPOLARSCAN_H
#include "polarscan.h"

/**
 * A polar scan
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  PolarScan_t* scan; /**< the scan type */
} PyPolarScan;

/* C API functions */
#define PyPolarScan_Type_NUM 0

#define PyPolarScan_GetNative_NUM 1
#define PyPolarScan_GetNative_RETURN PolarScan_t*
#define PyPolarScan_GetNative_PROTO (PyPolarScan*)

#define PyPolarScan_New_NUM 2
#define PyPolarScan_New_RETURN PyPolarScan*
#define PyPolarScan_New_PROTO (PolarScan_t*)

/* Total number of C API pointers */
#define PyPolarScan_API_pointers 3

#ifdef PYPOLARSCAN_MODULE
/* To be used within the PyPolarScan-Module */
extern PyTypeObject PyPolarScan_Type;

#define PyPolarScan_Check(op) ((op)->ob_type == &PyPolarScan_Type)

static PyPolarScan_GetNative_RETURN PyPolarScan_GetNative PyPolarScan_GetNative_PROTO;

static PyPolarScan_New_RETURN PyPolarScan_New PyPolarScan_New_PROTO;

#else
/* This section is for clients using the pypolarscan API */
static void **PyPolarScan_API;

/**
 * Returns a pointer to the internal polar scan, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyPolarScan_GetNative \
  (*(PyPolarScan_GetNative_RETURN (*)PyPolarScan_GetNative_PROTO) PyPolarScan_API[PyPolarScan_GetNative_NUM])

/**
 * Creates a new polar scan instance. Release this object with Py_DECREF.
 * @param[in] scan - the PolarScan_t intance.
 * @returns the PyPolarScan instance.
 */
#define PyPolarScan_New \
  (*(PyPolarScan_New_RETURN (*)PyPolarScan_New_PROTO) PyPolarScan_API[PyPolarScan_New_NUM])

/**
 * Checks if the object is a python polar scan.
 */
#define PyPolarScan_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyPolarScan_API[PyPolarScan_Type_NUM])

/**
 * Imports the pypolarscan module (like import _polarscan in python).
 */
static int
import_pypolarscan(void)
{
  PyObject *module;
  PyObject *c_api_object;

  module = PyImport_ImportModule("_polarscan");
  if (module == NULL) {
    return -1;
  }

  c_api_object = PyObject_GetAttrString(module, "_C_API");
  if (c_api_object == NULL) {
    Py_DECREF(module);
    return -1;
  }
  if (PyCObject_Check(c_api_object)) {
    PyPolarScan_API = (void **)PyCObject_AsVoidPtr(c_api_object);
  }
  Py_DECREF(c_api_object);
  Py_DECREF(module);
  return 0;
}

#endif


#endif /* PYPOLARSCAN_H */
