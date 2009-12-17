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

#define PyPolarScan_Type_NUM 0                      /**< index of the type */

#define PyPolarScan_GetNative_NUM 1                 /**< index of GetNative */
#define PyPolarScan_GetNative_RETURN PolarScan_t*   /**< return type for GetNative */
#define PyPolarScan_GetNative_PROTO (PyPolarScan*)  /**< arguments for GetNative */

#define PyPolarScan_New_NUM 2                       /**< index of New */
#define PyPolarScan_New_RETURN PyPolarScan*         /**< return type for New */
#define PyPolarScan_New_PROTO (PolarScan_t*)        /**< arguments for New */

#define PyPolarScan_API_pointers 3                  /**< number of pointers */

#ifdef PYPOLARSCAN_MODULE
/** Forward declaration of the type */
extern PyTypeObject PyPolarScan_Type;

/** Checks if the object is a PyPolarScan or not */
#define PyPolarScan_Check(op) ((op)->ob_type == &PyPolarScan_Type)

/** Forward declaration of PyPolarScan_GetNative */
static PyPolarScan_GetNative_RETURN PyPolarScan_GetNative PyPolarScan_GetNative_PROTO;

/** Forward declaration of PyPolarScan_New */
static PyPolarScan_New_RETURN PyPolarScan_New PyPolarScan_New_PROTO;

#else
/**Forward declaration of the pointers */
static void **PyPolarScan_API;

/**
 * Returns a pointer to the internal polar scan, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyPolarScan_GetNative \
  (*(PyPolarScan_GetNative_RETURN (*)PyPolarScan_GetNative_PROTO) PyPolarScan_API[PyPolarScan_GetNative_NUM])

/**
 * Creates a new polar scan instance. Release this object with Py_DECREF. If a PolarScan_t scan is
 * provided and this scan already is bound to a python instance, this instance will be increfed and
 * returned.
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
