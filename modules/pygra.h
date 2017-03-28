/* --------------------------------------------------------------------
Copyright (C) 2014 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the Gra API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2014-03-28
 */
#ifndef PYGRA_H
#define PYGRA_H
#include "Python.h"
#include "rave_gra.h"

/**
 * A gra coefficient applier
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  RaveGra_t* gra; /**< the gra applier*/
} PyGra;

#define PyGra_Type_NUM 0                   /**< index of type */

#define PyGra_GetNative_NUM 1              /**< index of GetNative */
#define PyGra_GetNative_RETURN RaveGra_t* /**< return type for GetNative */
#define PyGra_GetNative_PROTO (PyGra*)    /**< arguments for GetNative */

#define PyGra_New_NUM 2                    /**< index of New */
#define PyGra_New_RETURN PyGra*           /**< return type for New */
#define PyGra_New_PROTO (RaveGra_t*)      /**< arguments for New */

#define PyGra_API_pointers 3               /**< number of API pointers */

#define PyGra_CAPSULE_NAME "_gra._C_API"


#ifdef PYGRA_MODULE
/** Forward declaration of type */
extern PyTypeObject PyGra_Type;

/** Checks if the object is a PyGra or not */
#define PyGra_Check(op) ((op)->ob_type == &PyGra_Type)

/** Forward declaration of PyGra_GetNative */
static PyGra_GetNative_RETURN PyGra_GetNative PyGra_GetNative_PROTO;

/** Forward declaration of PyGra_New */
static PyGra_New_RETURN PyGra_New PyGra_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyGra_API;

/**
 * Returns a pointer to the internal acrr, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyGra_GetNative \
  (*(PyGra_GetNative_RETURN (*)PyGra_GetNative_PROTO) PyGra_API[PyGra_GetNative_NUM])

/**
 * Creates a new polar scan instance. Release this object with Py_DECREF.  If a RaveAcrr_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] area - the RaveAcrr_t intance.
 * @returns the PyGra instance.
 */
#define PyGra_New \
  (*(PyGra_New_RETURN (*)PyGra_New_PROTO) PyGra_API[PyGra_New_NUM])

/**
 * Checks if the object is a python gra instance.
 */
#define PyGra_Check(op) \
   (Py_TYPE(op) == &PyGra_Type)

#define PyGra_Type (*(PyTypeObject*)PyGra_API[PyGra_Type_NUM])

/**
 * Imports the PyGra module (like import _gra in python).
 */
#define import_gra() \
    PyGra_API = (void **)PyCapsule_Import(PyGra_CAPSULE_NAME, 1);

#endif

#endif /* PYGRA_H */
