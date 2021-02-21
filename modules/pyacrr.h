/* --------------------------------------------------------------------
Copyright (C) 2012 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the Acrr API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2012-06-01
 */
#ifndef PYACRR_H
#define PYACRR_H
#include "Python.h"
#include "rave_acrr.h"

/**
 * A cartesian product
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  RaveAcrr_t* acrr; /**< the acrr */
} PyAcrr;

#define PyAcrr_Type_NUM 0                   /**< index of type */

#define PyAcrr_GetNative_NUM 1              /**< index of GetNative */
#define PyAcrr_GetNative_RETURN RaveAcrr_t* /**< return type for GetNative */
#define PyAcrr_GetNative_PROTO (PyAcrr*)    /**< arguments for GetNative */

#define PyAcrr_New_NUM 2                    /**< index of New */
#define PyAcrr_New_RETURN PyAcrr*           /**< return type for New */
#define PyAcrr_New_PROTO (RaveAcrr_t*)      /**< arguments for New */

#define PyAcrr_API_pointers 3               /**< number of API pointers */

#define PyAcrr_CAPSULE_NAME "_acrr._C_API"

#ifdef PYACRR_MODULE
/** Forward declaration of type */
extern PyTypeObject PyAcrr_Type;

/** Checks if the object is a PyAcrr or not */
#define PyAcrr_Check(op) ((op)->ob_type == &PyAcrr_Type)

/** Forward declaration of PyAcrr_GetNative */
static PyAcrr_GetNative_RETURN PyAcrr_GetNative PyAcrr_GetNative_PROTO;

/** Forward declaration of PyAcrr_New */
static PyAcrr_New_RETURN PyAcrr_New PyAcrr_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyAcrr_API;

/**
 * Returns a pointer to the internal acrr, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyAcrr_GetNative \
  (*(PyAcrr_GetNative_RETURN (*)PyAcrr_GetNative_PROTO) PyAcrr_API[PyAcrr_GetNative_NUM])

/**
 * Creates a new polar scan instance. Release this object with Py_DECREF.  If a RaveAcrr_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] area - the RaveAcrr_t intance.
 * @returns the PyAcrr instance.
 */
#define PyAcrr_New \
  (*(PyAcrr_New_RETURN (*)PyAcrr_New_PROTO) PyAcrr_API[PyAcrr_New_NUM])

/**
 * Checks if the object is a python acrr instance.
 */
#define PyAcrr_Check(op) \
   (Py_TYPE(op) == &PyAcrr_Type)

#define PyAcrr_Type (*(PyTypeObject*)PyAcrr_API[PyAcrr_Type_NUM])

/**
 * Imports the PyAcrr module (like import _acrr in python).
 */
#define import_pyacrr() \
    PyAcrr_API = (void **)PyCapsule_Import(PyAcrr_CAPSULE_NAME, 1);


#endif

#endif /* PYACRR_H */
