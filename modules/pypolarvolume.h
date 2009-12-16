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
 * Python version of the PolarVolume API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-08
 */
#ifndef PYPOLARVOLUME_H
#define PYPOLARVOLUME_H
#include "polarvolume.h"

/**
 * The polar volume object
 */
typedef struct {
   PyObject_HEAD /*Always have to be on top*/
   PolarVolume_t* pvol;  /**< the polar volume */
} PyPolarVolume;

/* C API functions */
#define PyPolarVolume_Type_NUM 0

#define PyPolarVolume_GetNative_NUM 1
#define PyPolarVolume_GetNative_RETURN PolarVolume_t*
#define PyPolarVolume_GetNative_PROTO (PyPolarVolume*)

#define PyPolarVolume_New_NUM 2
#define PyPolarVolume_New_RETURN PyPolarVolume*
#define PyPolarVolume_New_PROTO (PolarVolume_t*)

/* Total number of C API pointers */
#define PyPolarVolume_API_pointers 3

#ifdef PYPOLARVOLUME_MODULE
/* To be used within the PyPolarVolume-Module */
extern PyTypeObject PyPolarVolume_Type;

#define PyPolarVolume_Check(op) ((op)->ob_type == &PyPolarVolume_Type)

static PyPolarVolume_GetNative_RETURN PyPolarVolume_GetNative PyPolarVolume_GetNative_PROTO;

static PyPolarVolume_New_RETURN PyPolarVolume_New PyPolarVolume_New_PROTO;

#else
/* This section is for clients using the PyPolarVolume API */
static void **PyPolarVolume_API;

/**
 * Returns a pointer to the internal polar volume, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyPolarVolume_GetNative \
  (*(PyPolarVolume_GetNative_RETURN (*)PyPolarVolume_GetNative_PROTO) PyPolarVolume_API[PyPolarVolume_GetNative_NUM])

/**
 * Creates a new polar volume instance. Release this object with Py_DECREF. If a PolarVolume_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] volume - the PolarVolume_t intance.
 * @returns the PyPolarVolume instance.
 */
#define PyPolarVolume_New \
  (*(PyPolarVolume_New_RETURN (*)PyPolarVolume_New_PROTO) PyPolarVolume_API[PyPolarVolume_New_NUM])

/**
 * Checks if the object is a python polar volume.
 */
#define PyPolarVolume_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyPolarVolume_API[PyPolarVolume_Type_NUM])

/**
 * Imports the PyPolarVolume module (like import _polarscan in python).
 */
static int
import_pypolarvolume(void)
{
  PyObject *module;
  PyObject *c_api_object;

  module = PyImport_ImportModule("_polarvolume");
  if (module == NULL) {
    return -1;
  }

  c_api_object = PyObject_GetAttrString(module, "_C_API");
  if (c_api_object == NULL) {
    Py_DECREF(module);
    return -1;
  }
  if (PyCObject_Check(c_api_object)) {
    PyPolarVolume_API = (void **)PyCObject_AsVoidPtr(c_api_object);
  }
  Py_DECREF(c_api_object);
  Py_DECREF(module);
  return 0;
}

#endif

#endif /* PYPOLARVOLUME_H */
