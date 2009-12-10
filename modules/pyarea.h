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
 * Python version of the Area API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#ifndef PYAREA_H
#define PYAREA_H
#include "area.h"

/**
 * A cartesian product
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  Area_t* area; /**< the area */
} PyArea;

/* C API functions */
#define PyArea_Type_NUM 0

#define PyArea_GetNative_NUM 1
#define PyArea_GetNative_RETURN Area_t*
#define PyArea_GetNative_PROTO (PyArea*)

#define PyArea_New_NUM 2
#define PyArea_New_RETURN PyArea*
#define PyArea_New_PROTO (Area_t*)

/* Total number of C API pointers */
#define PyArea_API_pointers 3

#ifdef PYAREA_MODULE
/* To be used within the PyArea-Module */
extern PyTypeObject PyArea_Type;

#define PyArea_Check(op) ((op)->ob_type == &PyArea_Type)

static PyArea_GetNative_RETURN PyArea_GetNative PyArea_GetNative_PROTO;

static PyArea_New_RETURN PyArea_New PyArea_New_PROTO;

#else
/* This section is for clients using the PyArea API */
static void **PyArea_API;

/**
 * Returns a pointer to the internal polar scan, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyArea_GetNative \
  (*(PyArea_GetNative_RETURN (*)PyArea_GetNative_PROTO) PyArea_API[PyArea_GetNative_NUM])

/**
 * Creates a new polar scan instance. Release this object with Py_DECREF.
 * @param[in] scan - the PolarScan_t intance.
 * @returns the PyArea instance.
 */
#define PyArea_New \
  (*(PyArea_New_RETURN (*)PyArea_New_PROTO) PyArea_API[PyArea_New_NUM])

/**
 * Checks if the object is a python polar scan.
 */
#define PyArea_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyArea_API[PyArea_Type_NUM])

/**
 * Imports the PyArea module (like import _area in python).
 */
static int
import_pyarea(void)
{
  PyObject *module;
  PyObject *c_api_object;

  module = PyImport_ImportModule("_area");
  if (module == NULL) {
    return -1;
  }

  c_api_object = PyObject_GetAttrString(module, "_C_API");
  if (c_api_object == NULL) {
    Py_DECREF(module);
    return -1;
  }
  if (PyCObject_Check(c_api_object)) {
    PyArea_API = (void **)PyCObject_AsVoidPtr(c_api_object);
  }
  Py_DECREF(c_api_object);
  Py_DECREF(module);
  return 0;
}

#endif

#endif /* PYAREA_H */
