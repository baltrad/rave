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
 * Python version of the properties API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-02-14
 */
#ifndef PYRAVEPROPERTIES_H
#define PYRAVEPROPERTIES_H
#include "rave_properties.h"

/**
 * A cartesian product
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  RaveProperties_t* properties; /**< the instance */
} PyRaveProperties;

#define PyRaveProperties_Type_NUM 0                   /**< index of type */

#define PyRaveProperties_GetNative_NUM 1              /**< index of GetNative */
#define PyRaveProperties_GetNative_RETURN RaveProperties_t*     /**< return type for GetNative */
#define PyRaveProperties_GetNative_PROTO (PyRaveProperties*)    /**< arguments for GetNative */

#define PyRaveProperties_New_NUM 2                    /**< index of New */
#define PyRaveProperties_New_RETURN PyRaveProperties*           /**< return type for New */
#define PyRaveProperties_New_PROTO (RaveProperties_t*)          /**< arguments for New */

//#define PyRaveProperties_Load_NUM 3                     /**< index for Load */
//#define PyRaveProperties_Load_RETURN PyRaveProperties*          /**< return type for Load */
//#define PyRaveProperties_Load_PROTO (const char* filename, PyProjectionRegistry* pyprojregistry) /**< argument prototype for Open */

#define PyRaveProperties_API_pointers 3               /**< number of API pointers */

#define PyRaveProperties_CAPSULE_NAME "_raveproperties._C_API"

#ifdef PYRAVEPROPERTIES_MODULE
/** Forward declaration of type */
extern PyTypeObject PyRaveProperties_Type;

/** Checks if the object is a PyArea or not */
#define PyRaveProperties_Check(op) ((op)->ob_type == &PyRaveProperties_Type)

/** Forward declaration of PyArea_GetNative */
static PyRaveProperties_GetNative_RETURN PyRaveProperties_GetNative PyRaveProperties_GetNative_PROTO;

/** Forward declaration of PyArea_New */
static PyRaveProperties_New_RETURN PyRaveProperties_New PyRaveProperties_New_PROTO;

/** Prototype for PyProjectionRegistry modules Load function
static PyRaveProperties_Load_RETURN PyRaveProperties_Load PyRaveProperties_Load_PROTO;
*/
#else
/** Pointers to types and functions */
static void **PyRaveProperties_API;

/**
 * Returns a pointer to the internal area, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyRaveProperties_GetNative \
  (*(PyRaveProperties_GetNative_RETURN (*)PyRaveProperties_GetNative_PROTO) PyRaveProperties_API[PyRaveProperties_GetNative_NUM])

/**
 * Creates a area registry instance. Release this object with Py_DECREF.  If a AreaRegistry_t area is
 * provided and this registry already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] area - the Area_t intance.
 * @returns the PyArea instance.
 */
#define PyRaveProperties_New \
  (*(PyRaveProperties_New_RETURN (*)PyRaveProperties_New_PROTO) PyRaveProperties_API[PyRaveProperties_New_NUM])

/**
 * Loads a area registry instance. Release this object with Py_DECREF.
 * @param[in] filename - the filename.
 * @param[in] pyprojregistry - the py projection registry
 * @returns the PyRaveProperties instance.
#define PyRaveProperties_Load \
   (*(PyRaveProperties_Load_RETURN (*)PyRaveProperties_Load_PROTO) PyRaveProperties_API[PyRaveProperties_Load_NUM])
 */

/**
 * Checks if the object is a python area registry.
 */
#define PyRaveProperties_Check(op) \
   (Py_TYPE(op) == &PyRaveProperties_Type)

#define PyRaveProperties_Type (*(PyTypeObject*)PyRaveProperties_API[PyRaveProperties_Type_NUM])

/**
 * Imports the PyRaveProperties module (like import _arearegistry in python).
 */
#define import_raveproperties() \
    PyRaveProperties_API = (void **)PyCapsule_Import(PyRaveProperties_CAPSULE_NAME, 1);

#endif

#endif /* PYRAVEPROPERTIES_H */
