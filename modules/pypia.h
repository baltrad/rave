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
 * Python version of the PIA API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-12-04
 */
#ifndef PYPIA_H
#define PYPIA_H
#include "Python.h"
#include "ravepia.h"

/**
 * A pia 
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  RavePIA_t* pia; /**< the pia */
} PyPia;

#define PyPia_Type_NUM 0                   /**< index of type */

#define PyPia_GetNative_NUM 1              /**< index of GetNative */
#define PyPia_GetNative_RETURN RavePIA_t* /**< return type for GetNative */
#define PyPia_GetNative_PROTO (PyPia*)    /**< arguments for GetNative */

#define PyPia_New_NUM 2                    /**< index of New */
#define PyPia_New_RETURN PyPia*           /**< return type for New */
#define PyPia_New_PROTO (RavePIA_t*)      /**< arguments for New */

#define PyPia_API_pointers 3               /**< number of API pointers */

#define PyPia_CAPSULE_NAME "_pia._C_API"


#ifdef PYPIA_MODULE
/** Forward declaration of type */
extern PyTypeObject PyPia_Type;

/** Checks if the object is a PyPia or not */
#define PyPia_Check(op) ((op)->ob_type == &PyPia_Type)

/** Forward declaration of PyPia_GetNative */
static PyPia_GetNative_RETURN PyPia_GetNative PyPia_GetNative_PROTO;

/** Forward declaration of PyPia_New */
static PyPia_New_RETURN PyPia_New PyPia_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyPia_API;

/**
 * Returns a pointer to the internal pia, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyPia_GetNative \
  (*(PyPia_GetNative_RETURN (*)PyPia_GetNative_PROTO) PyPia_API[PyPia_GetNative_NUM])

/**
 * Creates a new pia instance. Release this object with Py_DECREF.  If a RavePIA_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] pia - the RavePIA_t intance.
 * @returns the PyPia instance.
 */
#define PyPia_New \
  (*(PyPia_New_RETURN (*)PyPia_New_PROTO) PyPia_API[PyPia_New_NUM])

/**
 * Checks if the object is a python gra instance.
 */
#define PyPia_Check(op) \
   (Py_TYPE(op) == &PyPia_Type)

#define PyPia_Type (*(PyTypeObject*)PyPia_API[PyPia_Type_NUM])

/**
 * Imports the PyPia module (like import _pia in python).
 */
#define import_pia() \
    PyPia_API = (void **)PyCapsule_Import(PyPia_CAPSULE_NAME, 1);

#endif

#endif /* PYPIA_H */
