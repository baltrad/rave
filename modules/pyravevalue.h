/* --------------------------------------------------------------------
Copyright (C) 2025 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the Rave Value API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-02-14
 */
#ifndef PYRAVE_VALUE_H
#define PYRAVE_VALUE_H
#include "rave_value.h"

/**
 * A cartesian product
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  RaveValue_t* value; /**< the value */
} PyRaveValue;

#define PyRaveValue_Type_NUM 0                   /**< index of type */

#define PyRaveValue_GetNative_NUM 1              /**< index of GetNative */
#define PyRaveValue_GetNative_RETURN RaveValue_t*     /**< return type for GetNative */
#define PyRaveValue_GetNative_PROTO (PyRaveValue*)    /**< arguments for GetNative */

#define PyRaveValue_New_NUM 2                    /**< index of New */
#define PyRaveValue_New_RETURN PyRaveValue*           /**< return type for New */
#define PyRaveValue_New_PROTO (RaveValue_t*)          /**< arguments for New */

#define PyRaveValue_API_pointers 3               /**< number of API pointers */

#define PyRaveValue_CAPSULE_NAME "_ravevalue._C_API"

#ifdef PYRAVEVALUE_MODULE
/** Forward declaration of type */
extern PyTypeObject PyRaveValue_Type;

/** Checks if the object is a PyRaveValue or not */
#define PyRaveValue_Check(op) ((op)->ob_type == &PyRaveValue_Type)

/** Forward declaration of PyRaveValue_GetNative */
static PyRaveValue_GetNative_RETURN PyRaveValue_GetNative PyRaveValue_GetNative_PROTO;

/** Forward declaration of PyRaveValue_New */
static PyRaveValue_New_RETURN PyRaveValue_New PyRaveValue_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyRaveValue_API;

/**
 * Returns a pointer to the internal area, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyRaveValue_GetNative \
  (*(PyRaveValue_GetNative_RETURN (*)PyRaveValue_GetNative_PROTO) PyRaveValue_API[PyRaveValue_GetNative_NUM])

/**
 * Creates a new rave value instance. Release this object with Py_DECREF.  If a RaveValue_t is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] value - the RaveValue_t intance.
 * @returns the PyRaveValue instance.
 */
#define PyRaveValue_New \
  (*(PyRaveValue_New_RETURN (*)PyRaveValue_New_PROTO) PyRaveValue_API[PyRaveValue_New_NUM])

/**
 * Checks if the object is a python area.
 */
#define PyRaveValue_Check(op) \
   (Py_TYPE(op) == &PyRaveValue_Type)

#define PyRaveValue_Type (*(PyTypeObject*)PyRaveValue_API[PyRaveValue_Type_NUM])

/**
 * Imports the PyRaveValue module (like import _area in python).
 */
#define import_ravevalue() \
    PyRaveValue_API = (void **)PyCapsule_Import(PyRaveValue_CAPSULE_NAME, 1);

#endif

#endif /* PYRAVEVALUE_H */
