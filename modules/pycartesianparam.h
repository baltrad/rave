/* --------------------------------------------------------------------
Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI,

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
#ifndef PYCARTESIANPARAM_H
#define PYCARTESIANPARAM_H
#include "cartesianparam.h"

/**
 * A cartesian product
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  CartesianParam_t* param; /**< the cartesian parameter */
} PyCartesianParam;

#define PyCartesianParam_Type_NUM 0                       /**< index of type */

#define PyCartesianParam_GetNative_NUM 1                  /**< index of GetNative */
#define PyCartesianParam_GetNative_RETURN CartesianParam_t*    /**< return type for GetNative */
#define PyCartesianParam_GetNative_PROTO (PyCartesianParam*)   /**< arguments for GetNative */

#define PyCartesianParam_New_NUM 2                        /**< index of New */
#define PyCartesianParam_New_RETURN PyCartesianParam*          /**< return type for New */
#define PyCartesianParam_New_PROTO (CartesianParam_t*)         /**< arguments for New */

#define PyCartesianParam_API_pointers 3                   /**< number of api pointers */

#define PyCartesianParam_CAPSULE_NAME "_cartesianparam._C_API"

#ifdef PYCARTESIANPARAM_MODULE
/** Forward declaration of type*/
extern PyTypeObject PyCartesianParam_Type;

/** Checks if the object is a PyCartesian or not */
#define PyCartesianParam_Check(op) ((op)->ob_type == &PyCartesianParam_Type)

/** Forward declaration of PyCartesianParam_GetNative */
static PyCartesianParam_GetNative_RETURN PyCartesianParam_GetNative PyCartesianParam_GetNative_PROTO;

/** Forward declaration of PyCartesianParam_New */
static PyCartesianParam_New_RETURN PyCartesianParam_New PyCartesianParam_New_PROTO;

#else
/** pointers to types and functions */
static void **PyCartesianParam_API;

/**
 * Returns a pointer to the internal cartesian, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyCartesianParam_GetNative \
  (*(PyCartesianParam_GetNative_RETURN (*)PyCartesianParam_GetNative_PROTO) PyCartesianParam_API[PyCartesianParam_GetNative_NUM])

/**
 * Creates a new cartesian instance. Release this object with Py_DECREF. If a Cartesian_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] cartesian - the Cartesian_t intance.
 * @returns the PyCartesian instance.
 */
#define PyCartesianParam_New \
  (*(PyCartesianParam_New_RETURN (*)PyCartesianParam_New_PROTO) PyCartesianParam_API[PyCartesianParam_New_NUM])


/**
 * Checks if the object is a python cartesian param.
 */
#define PyCartesianParam_Check(op) \
   (Py_TYPE(op) == &PyCartesianParam_Type)

#define PyCartesianParam_Type (*(PyTypeObject*)PyCartesianParam_API[PyCartesianParam_Type_NUM])

/**
 * Imports the PyCartesianParam module (like import _area in python).
 */
#define import_pycartesianparam() \
    PyCartesianParam_API = (void **)PyCapsule_Import(PyCartesianParam_CAPSULE_NAME, 1);

#ifdef KALLE
/**
 * Checks if the object is a python cartesian.
 */
#define PyCartesianParam_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyCartesianParam_API[PyCartesianParam_Type_NUM])

/**
 * Imports the PyCartesian module (like import _polarscan in python).
 */
static int
import_pycartesianparam(void)
{
  PyObject *module;
  PyObject *c_api_object;

  module = PyImport_ImportModule("_cartesianparam");
  if (module == NULL) {
    return -1;
  }

  c_api_object = PyObject_GetAttrString(module, "_C_API");
  if (c_api_object == NULL) {
    Py_DECREF(module);
    return -1;
  }
  if (PyCObject_Check(c_api_object)) {
    PyCartesianParam_API = (void **)PyCObject_AsVoidPtr(c_api_object);
  }
  Py_DECREF(c_api_object);
  Py_DECREF(module);
  return 0;
}
#endif



#endif

#endif /* PYCARTESIANPARAM_H */
