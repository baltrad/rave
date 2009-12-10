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
#ifndef PYTRANSFORM_H
#define PYTRANSFORM_H
#include "transform.h"

/**
 * The transformator
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  Transform_t* transform;
} PyTransform;

/* C API functions */
#define PyTransform_Type_NUM 0

#define PyTransform_GetNative_NUM 1
#define PyTransform_GetNative_RETURN Transform_t*
#define PyTransform_GetNative_PROTO (PyTransform*)

#define PyTransform_New_NUM 2
#define PyTransform_New_RETURN PyTransform*
#define PyTransform_New_PROTO (Transform_t*)

/* Total number of C API pointers */
#define PyTransform_API_pointers 3

#ifdef PYTRANSFORM_MODULE
/* To be used within the PyTransform-Module */
extern PyTypeObject PyTransform_Type;

#define PyTransform_Check(op) ((op)->ob_type == &PyTransform_Type)

static PyTransform_GetNative_RETURN PyTransform_GetNative PyTransform_GetNative_PROTO;

static PyTransform_New_RETURN PyTransform_New PyTransform_New_PROTO;

#else
/* This section is for clients using the PyTransform API */
static void **PyTransform_API;

/**
 * Returns a pointer to the internal polar scan, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyTransform_GetNative \
  (*(PyTransform_GetNative_RETURN (*)PyTransform_GetNative_PROTO) PyTransform_API[PyTransform_GetNative_NUM])

/**
 * Creates a new polar scan instance. Release this object with Py_DECREF.
 * @param[in] scan - the PolarScan_t intance.
 * @returns the PyTransform instance.
 */
#define PyTransform_New \
  (*(PyTransform_New_RETURN (*)PyTransform_New_PROTO) PyTransform_API[PyTransform_New_NUM])

/**
 * Checks if the object is a python polar scan.
 */
#define PyTransform_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyTransform_API[PyTransform_Type_NUM])

/**
 * Imports the PyTransform module (like import _polarscan in python).
 */
static int
import_pytransform(void)
{
  PyObject *module;
  PyObject *c_api_object;

  module = PyImport_ImportModule("_transform");
  if (module == NULL) {
    return -1;
  }

  c_api_object = PyObject_GetAttrString(module, "_C_API");
  if (c_api_object == NULL) {
    Py_DECREF(module);
    return -1;
  }
  if (PyCObject_Check(c_api_object)) {
    PyTransform_API = (void **)PyCObject_AsVoidPtr(c_api_object);
  }
  Py_DECREF(c_api_object);
  Py_DECREF(module);
  return 0;
}

#endif



#endif /* PYTRANSFORM_H */
