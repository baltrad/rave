/* --------------------------------------------------------------------
Copyright (C) 2009-2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the RaveData2D API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2019-02-18
 */
#ifndef PYRAVEDATA2D_H
#define PYRAVEDATA2D_H
#include "rave_data2d.h"

/**
 * The rave field object
 */
typedef struct {
   PyObject_HEAD /*Always have to be on top*/
   RaveData2D_t* field;  /**< the rave data 2d array */
} PyRaveData2D;

#define PyRaveData2D_Type_NUM 0                              /**< index of type */

#define PyRaveData2D_GetNative_NUM 1                         /**< index of GetNative*/
#define PyRaveData2D_GetNative_RETURN RaveData2D_t*         /**< return type for GetNative */
#define PyRaveData2D_GetNative_PROTO (PyRaveData2D*)        /**< arguments for GetNative */

#define PyRaveData2D_New_NUM 2                               /**< index of New */
#define PyRaveData2D_New_RETURN PyRaveData2D*               /**< return type for New */
#define PyRaveData2D_New_PROTO (RaveData2D_t*)              /**< arguments for New */

#define PyRaveData2D_API_pointers 3                          /**< number of type and function pointers */

#define PyRaveData2D_CAPSULE_NAME "_ravedata2d._C_API"

#ifdef PYRAVEDATA2D_MODULE
/** Forward declaration of type */
extern PyTypeObject PyRaveData2D_Type;

/** Checks if the object is a PyRaveData2D or not */
#define PyRaveData2D_Check(op) ((op)->ob_type == &PyRaveData2D_Type)

/** Forward declaration of PyRaveData2D_GetNative */
static PyRaveData2D_GetNative_RETURN PyRaveData2D_GetNative PyRaveData2D_GetNative_PROTO;

/** Forward declaration of PyRaveData2D_New */
static PyRaveData2D_New_RETURN PyRaveData2D_New PyRaveData2D_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyRaveData2D_API;

/**
 * Returns a pointer to the internal field, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyRaveData2D_GetNative \
  (*(PyRaveData2D_GetNative_RETURN (*)PyRaveData2D_GetNative_PROTO) PyRaveData2D_API[PyRaveData2D_GetNative_NUM])

/**
 * Creates a new rave data 2d instance. Release this object with Py_DECREF. If a RaveData2D_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] field - the RaveData2D_t intance.
 * @returns the PyRaveData2D instance.
 */
#define PyRaveData2D_New \
  (*(PyRaveData2D_New_RETURN (*)PyRaveData2D_New_PROTO) PyRaveData2D_API[PyRaveData2D_New_NUM])

/**
 * Checks if the object is a python rave data 2d.
 */
#define PyRaveData2D_Check(op) \
   (Py_TYPE(op) == &PyRaveData2D_Type)

#define PyRaveData2D_Type (*(PyTypeObject*)PyRaveData2D_API[PyRaveData2D_Type_NUM])

/**
 * Imports the PyArea module (like import _area in python).
 */
#define import_ravedata2d() \
    PyRaveData2D_API = (void **)PyCapsule_Import(PyRaveData2D_CAPSULE_NAME, 1);

#endif


#endif /* PYRAVEDATA2D_H */
