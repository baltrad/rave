/* --------------------------------------------------------------------
Copyright (C) 2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the CartesianVolume API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-06-23
 */
#ifndef PYCARTESIANVOLUME_H
#define PYCARTESIANVOLUME_H
#include "cartesianvolume.h"

/**
 * The cartesian volume object
 */
typedef struct {
   PyObject_HEAD /*Always have to be on top*/
   CartesianVolume_t* cvol;  /**< the cartesian volume */
} PyCartesianVolume;

#define PyCartesianVolume_Type_NUM 0                              /**< index of type */

#define PyCartesianVolume_GetNative_NUM 1                         /**< index of GetNative*/
#define PyCartesianVolume_GetNative_RETURN CartesianVolume_t*         /**< return type for GetNative */
#define PyCartesianVolume_GetNative_PROTO (PyCartesianVolume*)        /**< arguments for GetNative */

#define PyCartesianVolume_New_NUM 2                               /**< index of New */
#define PyCartesianVolume_New_RETURN PyCartesianVolume*               /**< return type for New */
#define PyCartesianVolume_New_PROTO (CartesianVolume_t*)              /**< arguments for New */

#define PyCartesianVolume_API_pointers 3                          /**< number of type and function pointers */

#define PyCartesianVolume_CAPSULE_NAME "_cartesianvolume._C_API"

#ifdef PYCARTESIANVOLUME_MODULE
/** Forward declaration of type */
extern PyTypeObject PyCartesianVolume_Type;

/** Checks if the object is a PyCartesianVolume or not */
#define PyCartesianVolume_Check(op) ((op)->ob_type == &PyCartesianVolume_Type)

/** Forward declaration of PyCartesianVolume_GetNative */
static PyCartesianVolume_GetNative_RETURN PyCartesianVolume_GetNative PyCartesianVolume_GetNative_PROTO;

/** Forward declaration of PyCartesianVolume_New */
static PyCartesianVolume_New_RETURN PyCartesianVolume_New PyCartesianVolume_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyCartesianVolume_API;

/**
 * Returns a pointer to the internal cartesian volume, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyCartesianVolume_GetNative \
  (*(PyCartesianVolume_GetNative_RETURN (*)PyCartesianVolume_GetNative_PROTO) PyCartesianVolume_API[PyCartesianVolume_GetNative_NUM])

/**
 * Creates a new cartesian volume instance. Release this object with Py_DECREF. If a CartesianVolume_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] volume - the CartesianVolume_t intance.
 * @returns the PyCartesianVolume instance.
 */
#define PyCartesianVolume_New \
  (*(PyCartesianVolume_New_RETURN (*)PyCartesianVolume_New_PROTO) PyCartesianVolume_API[PyCartesianVolume_New_NUM])


/**
 * Checks if the object is a python area.
 */
#define PyCartesianVolume_Check(op) \
   (Py_TYPE(op) == &PyCartesianVolume_Type)

#define PyCartesianVolume_Type (*(PyTypeObject*)PyCartesianVolume_API[PyCartesianVolume_Type_NUM])

/**
 * Imports the PyCartesianVolume module (like import _cartesianvolume in python).
 */
#define import_pycartesianvolume() \
    PyCartesianVolume_API = (void **)PyCapsule_Import(PyCartesianVolume_CAPSULE_NAME, 1);

#endif

#endif /* PYCARTESIANVOLUME_H */
