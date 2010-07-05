/* --------------------------------------------------------------------
Copyright (C) 2009-2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the RaveField API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-07-05
 */
#ifndef PYRAVEFIELD_H
#define PYRAVEFIELD_H
#include "rave_field.h"

/**
 * The rave field object
 */
typedef struct {
   PyObject_HEAD /*Always have to be on top*/
   RaveField_t* field;  /**< the rave field */
} PyRaveField;

#define PyRaveField_Type_NUM 0                              /**< index of type */

#define PyRaveField_GetNative_NUM 1                         /**< index of GetNative*/
#define PyRaveField_GetNative_RETURN RaveField_t*         /**< return type for GetNative */
#define PyRaveField_GetNative_PROTO (PyRaveField*)        /**< arguments for GetNative */

#define PyRaveField_New_NUM 2                               /**< index of New */
#define PyRaveField_New_RETURN PyRaveField*               /**< return type for New */
#define PyRaveField_New_PROTO (RaveField_t*)              /**< arguments for New */

#define PyRaveField_API_pointers 3                          /**< number of type and function pointers */

#ifdef PYRAVEFIELD_MODULE
/** Forward declaration of type */
extern PyTypeObject PyRaveField_Type;

/** Checks if the object is a PyRaveField or not */
#define PyRaveField_Check(op) ((op)->ob_type == &PyRaveField_Type)

/** Forward declaration of PyRaveField_GetNative */
static PyRaveField_GetNative_RETURN PyRaveField_GetNative PyRaveField_GetNative_PROTO;

/** Forward declaration of PyRaveField_New */
static PyRaveField_New_RETURN PyRaveField_New PyRaveField_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyRaveField_API;

/**
 * Returns a pointer to the internal field, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyRaveField_GetNative \
  (*(PyRaveField_GetNative_RETURN (*)PyRaveField_GetNative_PROTO) PyRaveField_API[PyRaveField_GetNative_NUM])

/**
 * Creates a new rave field instance. Release this object with Py_DECREF. If a RaveField_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] field - the RaveField_t intance.
 * @returns the PyRaveField instance.
 */
#define PyRaveField_New \
  (*(PyRaveField_New_RETURN (*)PyRaveField_New_PROTO) PyRaveField_API[PyRaveField_New_NUM])

/**
 * Checks if the object is a python rave field.
 */
#define PyRaveField_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyRaveField_API[PyRaveField_Type_NUM])

/**
 * Imports the PyRaveField module (like import _ravefield in python).
 */
static int
import_pyravefield(void)
{
  PyObject *module;
  PyObject *c_api_object;

  module = PyImport_ImportModule("_ravefield");
  if (module == NULL) {
    return -1;
  }

  c_api_object = PyObject_GetAttrString(module, "_C_API");
  if (c_api_object == NULL) {
    Py_DECREF(module);
    return -1;
  }
  if (PyCObject_Check(c_api_object)) {
    PyRaveField_API = (void **)PyCObject_AsVoidPtr(c_api_object);
  }
  Py_DECREF(c_api_object);
  Py_DECREF(module);
  return 0;
}

#endif


#endif /* PYRAVEFIELD_H */
