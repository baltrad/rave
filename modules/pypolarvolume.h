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
 * Python version of the PolarVolume API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-08
 */
#ifndef PYPOLARVOLUME_H
#define PYPOLARVOLUME_H
#include "polarvolume.h"

/**
 * The polar volume object
 */
typedef struct {
   PyObject_HEAD /*Always have to be on top*/
   PolarVolume_t* pvol;  /**< the polar volume */
} PyPolarVolume;

#define PyPolarVolume_Type_NUM 0                              /**< index of type */

#define PyPolarVolume_GetNative_NUM 1                         /**< index of GetNative*/
#define PyPolarVolume_GetNative_RETURN PolarVolume_t*         /**< return type for GetNative */
#define PyPolarVolume_GetNative_PROTO (PyPolarVolume*)        /**< arguments for GetNative */

#define PyPolarVolume_New_NUM 2                               /**< index of New */
#define PyPolarVolume_New_RETURN PyPolarVolume*               /**< return type for New */
#define PyPolarVolume_New_PROTO (PolarVolume_t*)              /**< arguments for New */

#define PyPolarVolume_API_pointers 3                          /**< number of type and function pointers */

#define PyPolarVolume_CAPSULE_NAME "_polarvolume._C_API"

#ifdef PYPOLARVOLUME_MODULE
/** Forward declaration of type */
extern PyTypeObject PyPolarVolume_Type;

/** Checks if the object is a PyPolarVolume or not */
#define PyPolarVolume_Check(op) ((op)->ob_type == &PyPolarVolume_Type)

/** Forward declaration of PyPolarVolume_GetNative */
static PyPolarVolume_GetNative_RETURN PyPolarVolume_GetNative PyPolarVolume_GetNative_PROTO;

/** Forward declaration of PyPolarVolume_New */
static PyPolarVolume_New_RETURN PyPolarVolume_New PyPolarVolume_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyPolarVolume_API;

/**
 * Returns a pointer to the internal polar volume, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyPolarVolume_GetNative \
  (*(PyPolarVolume_GetNative_RETURN (*)PyPolarVolume_GetNative_PROTO) PyPolarVolume_API[PyPolarVolume_GetNative_NUM])

/**
 * Creates a new polar volume instance. Release this object with Py_DECREF. If a PolarVolume_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] volume - the PolarVolume_t intance.
 * @returns the PyPolarVolume instance.
 */
#define PyPolarVolume_New \
  (*(PyPolarVolume_New_RETURN (*)PyPolarVolume_New_PROTO) PyPolarVolume_API[PyPolarVolume_New_NUM])

/**
 * Checks if the object is a python polar volume.
 */
#define PyPolarVolume_Check(op) \
   (Py_TYPE(op) == &PyPolarVolume_Type)

#define PyPolarVolume_Type (*(PyTypeObject*)PyPolarVolume_API[PyPolarVolume_Type_NUM])

/**
 * Imports the PyPolarVolume module (like import _polarvolume in python).
 */
#define import_pypolarvolume() \
    PyPolarVolume_API = (void **)PyCapsule_Import(PyPolarVolume_CAPSULE_NAME, 1);

#endif

#endif /* PYPOLARVOLUME_H */
