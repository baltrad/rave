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
 * Python version of the RaveField API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-02-10
 */
#ifndef PYRAVELEGEND_H
#define PYRAVELEGEND_H
#include "rave_legend.h"

/**
 * The rave field object
 */
typedef struct {
   PyObject_HEAD /*Always have to be on top*/
   RaveLegend_t* legend;  /**< the rave legend */
} PyRaveLegend;

#define PyRaveLegend_Type_NUM 0                              /**< index of type */

#define PyRaveLegend_GetNative_NUM 1                         /**< index of GetNative*/
#define PyRaveLegend_GetNative_RETURN RaveLegend_t*          /**< return type for GetNative */
#define PyRaveLegend_GetNative_PROTO (PyRaveLegend*)         /**< arguments for GetNative */

#define PyRaveLegend_New_NUM 2                               /**< index of New */
#define PyRaveLegend_New_RETURN PyRaveLegend*                /**< return type for New */
#define PyRaveLegend_New_PROTO (RaveLegend_t*)               /**< arguments for New */

#define PyRaveLegend_API_pointers 3                          /**< number of type and function pointers */

#define PyRaveLegend_CAPSULE_NAME "_ravelegend._C_API"

#ifdef PYRAVELEGEND_MODULE
/** Forward declaration of type */
extern PyTypeObject PyRaveLegend_Type;

/** Checks if the object is a PyRaveLegend or not */
#define PyRaveLegend_Check(op) ((op)->ob_type == &PyRaveLegend_Type)

/** Forward declaration of PyRaveLegend_GetNative */
static PyRaveLegend_GetNative_RETURN PyRaveLegend_GetNative PyRaveLegend_GetNative_PROTO;

/** Forward declaration of PyRaveLegend_New */
static PyRaveLegend_New_RETURN PyRaveLegend_New PyRaveLegend_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyRaveLegend_API;

/**
 * Returns a pointer to the internal legend, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyRaveLegend_GetNative \
  (*(PyRaveLegend_GetNative_RETURN (*)PyRaveLegend_GetNative_PROTO) PyRaveLegend_API[PyRaveLegend_GetNative_NUM])

/**
 * Creates a new rave legend instance. Release this object with Py_DECREF. If a RaveLegend_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] field - the RaveLegend_t intance.
 * @returns the PyRaveLegend instance.
 */
#define PyRaveLegend_New \
  (*(PyRaveLegend_New_RETURN (*)PyRaveLegend_New_PROTO) PyRaveLegend_API[PyRaveLegend_New_NUM])

/**
 * Checks if the object is a python rave legend.
 */
#define PyRaveLegend_Check(op) \
   (Py_TYPE(op) == &PyRaveLegend_Type)

#define PyRaveLegend_Type (*(PyTypeObject*)PyRaveLegend_API[PyRaveLegend_Type_NUM])

/**
 * Imports the PyRaveLegend module (like import _ravelegend in python).
 */
#define import_pyravelegend() \
    PyRaveLegend_API = (void **)PyCapsule_Import(PyRaveLegend_CAPSULE_NAME, 1);

#endif


#endif /* PYRAVELEGEND_H */
