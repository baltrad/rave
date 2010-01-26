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
 * Python version of the PolarScanParam API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-01-22
 */
#ifndef PYPOLARSCANPARAM_H
#define PYPOLARSCANPARAM_H
#include "polarscanparam.h"

/**
 * A polar scan param
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  PolarScanParam_t* scanparam; /**< the scan type */
} PyPolarScanParam;

#define PyPolarScanParam_Type_NUM 0                      /**< index of the type */

#define PyPolarScanParam_GetNative_NUM 1                 /**< index of GetNative */
#define PyPolarScanParam_GetNative_RETURN PolarScanParam_t*   /**< return type for GetNative */
#define PyPolarScanParam_GetNative_PROTO (PyPolarScanParam*)  /**< arguments for GetNative */

#define PyPolarScanParam_New_NUM 2                       /**< index of New */
#define PyPolarScanParam_New_RETURN PyPolarScanParam*         /**< return type for New */
#define PyPolarScanParam_New_PROTO (PolarScanParam_t*)        /**< arguments for New */

#define PyPolarScanParam_API_pointers 3                  /**< number of pointers */

#ifdef PYPOLARSCANPARAM_MODULE
/** Forward declaration of the type */
extern PyTypeObject PyPolarScanParam_Type;

/** Checks if the object is a PyPolarScanParam or not */
#define PyPolarScanParam_Check(op) ((op)->ob_type == &PyPolarScanParam_Type)

/** Forward declaration of PyPolarScanParam_GetNative */
static PyPolarScanParam_GetNative_RETURN PyPolarScanParam_GetNative PyPolarScanParam_GetNative_PROTO;

/** Forward declaration of PyPolarScanParam_New */
static PyPolarScanParam_New_RETURN PyPolarScanParam_New PyPolarScanParam_New_PROTO;

#else
/**Forward declaration of the pointers */
static void **PyPolarScanParam_API;

/**
 * Returns a pointer to the internal polar scan param, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyPolarScanParam_GetNative \
  (*(PyPolarScanParam_GetNative_RETURN (*)PyPolarScanParam_GetNative_PROTO) PyPolarScanParam_API[PyPolarScanParam_GetNative_NUM])

/**
 * Creates a new polar scan param instance. Release this object with Py_DECREF. If a PolarScanParam_t scan is
 * provided and this scan param already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] scan - the PolarScanParam_t intance.
 * @returns the PyPolarScanParam instance.
 */
#define PyPolarScanParam_New \
  (*(PyPolarScanParam_New_RETURN (*)PyPolarScanParam_New_PROTO) PyPolarScanParam_API[PyPolarScanParam_New_NUM])

/**
 * Checks if the object is a python polar scan param.
 */
#define PyPolarScanParam_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyPolarScanParam_API[PyPolarScanParam_Type_NUM])

/**
 * Imports the pypolarscanparam module (like import _polarscanparam in python).
 */
static int
import_pypolarscanparam(void)
{
  PyObject *module;
  PyObject *c_api_object;

  module = PyImport_ImportModule("_polarscanparam");
  if (module == NULL) {
    return -1;
  }

  c_api_object = PyObject_GetAttrString(module, "_C_API");
  if (c_api_object == NULL) {
    Py_DECREF(module);
    return -1;
  }
  if (PyCObject_Check(c_api_object)) {
    PyPolarScanParam_API = (void **)PyCObject_AsVoidPtr(c_api_object);
  }
  Py_DECREF(c_api_object);
  Py_DECREF(module);
  return 0;
}

#endif


#endif /* PYPOLARSCANPARAM_H */
