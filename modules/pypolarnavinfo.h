/* --------------------------------------------------------------------
Copyright (C) 2024 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the PolarNavigationInfo.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-05-21
 */
#ifndef PYAREA_H
#define PYAREA_H
#include "rave_types.h"

/**
 * A cartesian product
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  PolarNavigationInfo info; /**< the navigation info */
} PyPolarNavigationInfo;

#define PyPolarNavigationInfo_Type_NUM 0                   /**< index of type */

#define PyPolarNavigationInfo_GetNative_NUM 1              /**< index of GetNative */
#define PyPolarNavigationInfo_GetNative_RETURN PolarNavigationInfo     /**< return type for GetNative */
#define PyPolarNavigationInfo_GetNative_PROTO (PyPolarNavigationInfo*)    /**< arguments for GetNative */

#define PyPolarNavigationInfo_New_NUM 2                    /**< index of New */
#define PyPolarNavigationInfo_New_RETURN PyPolarNavigationInfo* /**< return type for New */
#define PyPolarNavigationInfo_New_PROTO (PolarNavigationInfo) /**< arguments for New */

#define PyPolarNavigationInfo_API_pointers 3               /**< number of API pointers */

#define PyPolarNavigationInfo_CAPSULE_NAME "_pypolarnavinfo._C_API"

#ifdef PYPOLARNAVIGATIONINFO_MODULE
/** Forward declaration of type */
extern PyTypeObject PyPolarNavigationInfo_Type;

/** Checks if the object is a PyPolarNavigationInfo or not */
#define PyPolarNavigationInfo_Check(op) ((op)->ob_type == &PyPolarNavigationInfo_Type)

/** Forward declaration of PyPolarNavigationInfo_GetNative */
static PyPolarNavigationInfo_GetNative_RETURN PyPolarNavigationInfo_GetNative PyPolarNavigationInfo_GetNative_PROTO;

/** Forward declaration of PyPolarNavigationInfo_New */
static PyPolarNavigationInfo_New_RETURN PyPolarNavigationInfo_New PyPolarNavigationInfo_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyPolarNavigationInfo_API;

/**
 * Returns a pointer to the internal area, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyPolarNavigationInfo_GetNative \
  (*(PyPolarNavigationInfo_GetNative_RETURN (*)PyPolarNavigationInfo_GetNative_PROTO) PyPolarNavigationInfo_API[PyPolarNavigationInfo_GetNative_NUM])

/**
 * Creates a new polar scan instance. Release this object with Py_DECREF.  If a Area_t area is
 * provided and this area already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] navinfo - the PolarNavigationInfo intance.
 * @returns the PyPolarNavigationInfo instance.
 */
#define PyPolarNavigationInfo_New \
  (*(PyPolarNavigationInfo_New_RETURN (*)PyPolarNavigationInfo_New_PROTO) PyPolarNavigationInfo_API[PyPolarNavigationInfo_New_NUM])

/**
 * Checks if the object is a python area.
 */
#define PyPolarNavigationInfo_Check(op) \
   (Py_TYPE(op) == &PyPolarNavigationInfo_Type)

#define PyPolarNavigationInfo_Type (*(PyTypeObject*)PyPolarNavigationInfo_API[PyPolarNavigationInfo_Type_NUM])

/**
 * Imports the PyArea module (like import _pypolarnavinfo in python).
 */
#define import_pypolarnavinfo() \
    PyPolarNavigationInfo_API = (void **)PyCapsule_Import(PyPolarNavigationInfo_CAPSULE_NAME, 1);
#endif

#endif /* PYPOLARNAVIGATIONINFO_H */
