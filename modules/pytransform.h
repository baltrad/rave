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
 * Python version of the Transform API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#ifndef PYTRANSFORM_H
#define PYTRANSFORM_H
#include "transform.h"

/**
 * The transformator
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  Transform_t* transform;  /**< the c-api transformator */
} PyTransform;

#define PyTransform_Type_NUM 0                     /**< index for Type */

#define PyTransform_GetNative_NUM 1                /**< index for GetNative fp */
#define PyTransform_GetNative_RETURN Transform_t*  /**< Return type for GetNative */
#define PyTransform_GetNative_PROTO (PyTransform*) /**< Argument prototype for GetNative */

#define PyTransform_New_NUM 2                      /**< index for New fp */
#define PyTransform_New_RETURN PyTransform*        /**< Return type for New */
#define PyTransform_New_PROTO (Transform_t*)       /**< Argument prototype for New */

#define PyTransform_API_pointers 3                 /**< total number of C API pointers */

#define PyTransform_CAPSULE_NAME "_transform._C_API"

#ifdef PYTRANSFORM_MODULE
/** declared in pytransform module */
extern PyTypeObject PyTransform_Type;

/** checks if the object is a PyTransform type or not */
#define PyTransform_Check(op) ((op)->ob_type == &PyTransform_Type)

/** Prototype for PyTransform modules GetNative function */
static PyTransform_GetNative_RETURN PyTransform_GetNative PyTransform_GetNative_PROTO;

/** Prototype for PyTransform modules New function */
static PyTransform_New_RETURN PyTransform_New PyTransform_New_PROTO;

#else
/** static pointer containing the pointers to function pointers and other definitions */
static void **PyTransform_API;

/**
 * Returns a pointer to the internal polar scan, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyTransform_GetNative \
  (*(PyTransform_GetNative_RETURN (*)PyTransform_GetNative_PROTO) PyTransform_API[PyTransform_GetNative_NUM])

/**
 * Creates a new polar scan instance. Release this object with Py_DECREF.
 * @param[in] scan - the PolarScan_t intance.
 * @returns the PyTransform instance.
 */
#define PyTransform_New \
  (*(PyTransform_New_RETURN (*)PyTransform_New_PROTO) PyTransform_API[PyTransform_New_NUM])

/**
 * Checks if the object is a python transform instance.
 */
#define PyTransform_Check(op) \
   (Py_TYPE(op) == &PyTransform_Type)

#define PyTransform_Type (*(PyTypeObject*)PyTransform_API[PyTransform_Type_NUM])

/**
 * Imports the PyArea module (like import _transform in python).
 */
#define import_pytransform() \
    PyTransform_API = (void **)PyCapsule_Import(PyTransform_CAPSULE_NAME, 1);

#endif



#endif /* PYTRANSFORM_H */
