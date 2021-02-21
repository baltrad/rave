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
 * Python version of the VerticalProfile API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2012-08-24
 */

#ifndef PYVERTICALPROFILE_H
#define PYVERTICALPROFILE_H
#include "vertical_profile.h"
#include <Python.h>

/**
 * A generic vertical profile
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  VerticalProfile_t* vp; /**< the vertical profile type */
} PyVerticalProfile;

#define PyVerticalProfile_Type_NUM 0                      /**< index of the type */

#define PyVerticalProfile_GetNative_NUM 1                 /**< index of GetNative */
#define PyVerticalProfile_GetNative_RETURN VerticalProfile_t*   /**< return type for GetNative */
#define PyVerticalProfile_GetNative_PROTO (PyVerticalProfile*)  /**< arguments for GetNative */

#define PyVerticalProfile_New_NUM 2                       /**< index of New */
#define PyVerticalProfile_New_RETURN PyVerticalProfile*         /**< return type for New */
#define PyVerticalProfile_New_PROTO (VerticalProfile_t*)        /**< arguments for New */

#define PyVerticalProfile_API_pointers 3                  /**< number of pointers */

#define PyVerticalProfile_CAPSULE_NAME "_verticalprofile._C_API"

#ifdef PYVERTICALPROFILE_MODULE
/** Forward declaration of the type */
extern PyTypeObject PyVerticalProfile_Type;

/** Checks if the object is a PyVerticalProfile or not */
#define PyVerticalProfile_Check(op) ((op)->ob_type == &PyVerticalProfile_Type)

/** Forward declaration of PyVerticalProfile_GetNative */
static PyVerticalProfile_GetNative_RETURN PyVerticalProfile_GetNative PyVerticalProfile_GetNative_PROTO;

/** Forward declaration of PyVerticalProfile_New */
static PyVerticalProfile_New_RETURN PyVerticalProfile_New PyVerticalProfile_New_PROTO;

#else
/**Forward declaration of the pointers */
static void **PyVerticalProfile_API;

/**
 * Returns a pointer to the internal vertical profile, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyVerticalProfile_GetNative \
  (*(PyVerticalProfile_GetNative_RETURN (*)PyVerticalProfile_GetNative_PROTO) PyVerticalProfile_API[PyVerticalProfile_GetNative_NUM])

/**
 * Creates a new vertical profile instance. Release this object with Py_DECREF. If a VerticalProfile_t is
 * provided and this object is already bound to a Python instance, this instance will be increfed and
 * returned.
 * @param[in] vertical profile - the VerticalProfile_t intance.
 * @returns the PyVerticalProfile instance.
 */
#define PyVerticalProfile_New \
  (*(PyVerticalProfile_New_RETURN (*)PyVerticalProfile_New_PROTO) PyVerticalProfile_API[PyVerticalProfile_New_NUM])

/**
 * Checks if the object is a python polar scan .
 */
#define PyVerticalProfile_Check(op) \
   (Py_TYPE(op) == &PyVerticalProfile_Type)

#define PyVerticalProfile_Type (*(PyTypeObject*)PyVerticalProfile_API[PyVerticalProfile_Type_NUM])

/**
 * Imports the PyVerticalProfile module (like import _verticalprofile in python).
 */
#define import_pyverticalprofile() \
    PyVerticalProfile_API = (void **)PyCapsule_Import(PyVerticalProfile_CAPSULE_NAME, 1);

#endif




#endif /* PYVERTICALPROFILE_H_ */
