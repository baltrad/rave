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
 * Python version of the CartesianComposite API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2013-10-09
 */
#ifndef PYCARTESIANCOMPOSITE_H
#define PYCARTESIANCOMPOSITE_H
#include "cartesiancomposite.h"

/**
 * A cartesian product
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  CartesianComposite_t* generator; /**< the cartesian composite generator*/
} PyCartesianComposite;

#define PyCartesianComposite_Type_NUM 0                       /**< index of type */

#define PyCartesianComposite_GetNative_NUM 1                  /**< index of GetNative */
#define PyCartesianComposite_GetNative_RETURN CartesianComposite_t*    /**< return type for GetNative */
#define PyCartesianComposite_GetNative_PROTO (PyCartesianComposite*)   /**< arguments for GetNative */

#define PyCartesianComposite_New_NUM 2                        /**< index of New */
#define PyCartesianComposite_New_RETURN PyCartesianComposite*          /**< return type for New */
#define PyCartesianComposite_New_PROTO (CartesianComposite_t*)         /**< arguments for New */

#define PyCartesianComposite_API_pointers 3                   /**< number of api pointers */

#ifdef PYCARTESIANCOMPOSITE_MODULE
/** Forward declaration of type*/
extern PyTypeObject PyCartesianComposite_Type;

/** Checks if the object is a PyCartesianComposite or not */
#define PyCartesianComposite_Check(op) ((op)->ob_type == &PyCartesianComposite_Type)

/** Forward declaration of PyCartesianComposite_GetNative */
static PyCartesianComposite_GetNative_RETURN PyCartesianComposite_GetNative PyCartesianComposite_GetNative_PROTO;

/** Forward declaration of PyCartesianComposite_New */
static PyCartesianComposite_New_RETURN PyCartesianComposite_New PyCartesianComposite_New_PROTO;

#else
/** pointers to types and functions */
static void **PyCartesianComposite_API;

/**
 * Returns a pointer to the internal cartesian, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyCartesianComposite_GetNative \
  (*(PyCartesianComposite_GetNative_RETURN (*)PyCartesianComposite_GetNative_PROTO) PyCartesianComposite_API[PyCartesianComposite_GetNative_NUM])

/**
 * Creates a new cartesian instance. Release this object with Py_DECREF. If a CartesianComposite_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] generator - the CartesianComposite_t intance.
 * @returns the PyCartesianComposite instance.
 */
#define PyCartesianComposite_New \
  (*(PyCartesianComposite_New_RETURN (*)PyCartesianComposite_New_PROTO) PyCartesianComposite_API[PyCartesianComposite_New_NUM])

/**
 * Checks if the object is a python cartesian.
 */
#define PyCartesianComposite_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyCartesianComposite_API[PyCartesianComposite_Type_NUM])

/**
 * Imports the PyCartesianComposite module (like import _polarscan in python).
 */
static int
import_pycartesiancomposite(void)
{
  PyObject *module;
  PyObject *c_api_object;

  module = PyImport_ImportModule("_cartesiancomposite");
  if (module == NULL) {
    return -1;
  }

  c_api_object = PyObject_GetAttrString(module, "_C_API");
  if (c_api_object == NULL) {
    Py_DECREF(module);
    return -1;
  }
  if (PyCObject_Check(c_api_object)) {
    PyCartesianComposite_API = (void **)PyCObject_AsVoidPtr(c_api_object);
  }
  Py_DECREF(c_api_object);
  Py_DECREF(module);
  return 0;
}

#endif



#endif /* PYCARTESIANCOMPOSITE_H_ */
