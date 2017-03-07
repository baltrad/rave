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
 * Python version of the Area API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#ifndef PYAREA_H
#define PYAREA_H
#include "area.h"

/**
 * A cartesian product
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  Area_t* area; /**< the area */
} PyArea;

#define PyArea_Type_NUM 0                   /**< index of type */

#define PyArea_GetNative_NUM 1              /**< index of GetNative */
#define PyArea_GetNative_RETURN Area_t*     /**< return type for GetNative */
#define PyArea_GetNative_PROTO (PyArea*)    /**< arguments for GetNative */

#define PyArea_New_NUM 2                    /**< index of New */
#define PyArea_New_RETURN PyArea*           /**< return type for New */
#define PyArea_New_PROTO (Area_t*)          /**< arguments for New */

#define PyArea_API_pointers 3               /**< number of API pointers */

#define PyArea_CAPSULE_NAME "_area._C_API"

#ifdef PYAREA_MODULE
/** Forward declaration of type */
extern PyTypeObject PyArea_Type;

/** Checks if the object is a PyArea or not */
#define PyArea_Check(op) ((op)->ob_type == &PyArea_Type)

/** Forward declaration of PyArea_GetNative */
static PyArea_GetNative_RETURN PyArea_GetNative PyArea_GetNative_PROTO;

/** Forward declaration of PyArea_New */
static PyArea_New_RETURN PyArea_New PyArea_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyArea_API;

/**
 * Returns a pointer to the internal area, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyArea_GetNative \
  (*(PyArea_GetNative_RETURN (*)PyArea_GetNative_PROTO) PyArea_API[PyArea_GetNative_NUM])

/**
 * Creates a new polar scan instance. Release this object with Py_DECREF.  If a Area_t area is
 * provided and this area already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] area - the Area_t intance.
 * @returns the PyArea instance.
 */
#define PyArea_New \
  (*(PyArea_New_RETURN (*)PyArea_New_PROTO) PyArea_API[PyArea_New_NUM])

/**
 * Checks if the object is a python area.
 */
#define PyArea_Check(op) \
   (Py_TYPE(op) == &PyArea_Type)

#define PyArea_Type (*(PyTypeObject*)PyArea_API[PyArea_Type_NUM])

/**
 * Imports the PyArea module (like import _area in python).
 */
#define import_pyarea() \
    PyArea_API = (void **)PyCapsule_Import(PyArea_CAPSULE_NAME, 1);

#endif

#endif /* PYAREA_H */
