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
 * Python module for performing basic polar navigation.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-11
 */
#ifndef PYPOLARNAV_H
#define PYPOLARNAV_H
#include "polarnav.h"

/**
 * A polar navigator
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  PolarNavigator_t* navigator; /**< the polar navigator */
} PyPolarNavigator;

#define PyPolarNavigator_Type_NUM 0                             /**< index of type */

#define PyPolarNavigator_GetNative_NUM 1                        /**< index of GetNative */
#define PyPolarNavigator_GetNative_RETURN PolarNavigator_t*     /**< return type for GetNative */
#define PyPolarNavigator_GetNative_PROTO (PyPolarNavigator*)    /**< arguments for GetNative */

#define PyPolarNavigator_New_NUM 2                              /**< index of New */
#define PyPolarNavigator_New_RETURN PyPolarNavigator*           /**< return type for New */
#define PyPolarNavigator_New_PROTO (PolarNavigator_t*)          /**< arguments for New */

#define PyPolarNavigator_API_pointers 3                         /**< number of API pointers */

#define PyPolarNavigator_CAPSULE_NAME  "_polarnav._C_API"

#ifdef PYPOLARNAV_MODULE
/** Forward declaration of type */
extern PyTypeObject PyPolarNavigator_Type;

/** Checks if the object is a PyPolarNavigator or not */
#define PyPolarNavigator_Check(op) ((op)->ob_type == &PyPolarNavigator_Type)

/** Forward declaration of PyPolarNavigator_GetNative */
static PyPolarNavigator_GetNative_RETURN PyPolarNavigator_GetNative PyPolarNavigator_GetNative_PROTO;

/** Forward declaration of PyPolarNavigator_New */
static PyPolarNavigator_New_RETURN PyPolarNavigator_New PyPolarNavigator_New_PROTO;

#else
/** Pointers to the type and functions */
static void **PyPolarNavigator_API;

/**
 * Returns a pointer to the internal polar navigator, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyPolarNavigator_GetNative \
  (*(PyPolarNavigator_GetNative_RETURN (*)PyPolarNavigator_GetNative_PROTO) PyPolarNavigator_API[PyPolarNavigator_GetNative_NUM])

/**
 * Creates a new polar scan instance. Release this object with Py_DECREF. If a PolarNavigator_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] polarnav - the PolarNavigator_t instance.
 * @returns the PyPolarNavigator instance.
 */
#define PyPolarNavigator_New \
  (*(PyPolarNavigator_New_RETURN (*)PyPolarNavigator_New_PROTO) PyPolarNavigator_API[PyPolarNavigator_New_NUM])


/**
 * Checks if the object is a python polar navigator.
 */
#define PyPolarNavigator_Check(op) \
   (Py_TYPE(op) == &PyPolarNavigator_Type)

#define PyPolarNavigator_Type (*(PyTypeObject*)PyPolarNavigator_API[PyPolarNavigator_Type_NUM])

/**
 * Imports the PyArea module (like import _area in python).
 */
#define import_pypolarnav() \
    PyPolarNavigator_API = (void **)PyCapsule_Import(PyPolarNavigator_CAPSULE_NAME, 1);

#endif


#endif /* PYPOLARNAV_H */
