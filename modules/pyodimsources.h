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
 * Python version of the OdimSources API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-16
 */
#ifndef PYODIMSOURCES_H
#define PYODIMSOURCES_H
#include "odim_sources.h"

/**
 * A cartesian product
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  OdimSources_t* sources; /**< the odim sources */
} PyOdimSources;

#define PyOdimSources_Type_NUM 0                   /**< index of type */

#define PyOdimSources_GetNative_NUM 1              /**< index of GetNative */
#define PyOdimSources_GetNative_RETURN OdimSources_t*     /**< return type for GetNative */
#define PyOdimSources_GetNative_PROTO (PyOdimSources*)    /**< arguments for GetNative */

#define PyOdimSources_New_NUM 2                    /**< index of New */
#define PyOdimSources_New_RETURN PyOdimSources*           /**< return type for New */
#define PyOdimSources_New_PROTO (OdimSources_t*)          /**< arguments for New */

#define PyOdimSources_Load_NUM 3                     /**< index for Load */
#define PyOdimSources_Load_RETURN PyOdimSources*          /**< return type for Load */
#define PyOdimSources_Load_PROTO (const char* filename) /**< argument prototype for Open */

#define PyOdimSources_API_pointers 4               /**< number of API pointers */

#define PyOdimSources_CAPSULE_NAME "_odimsources._C_API"

#ifdef PYODIMSOURCES_MODULE
/** Forward declaration of type */
extern PyTypeObject PyOdimSources_Type;

/** Checks if the object is a PyOdimSources or not */
#define PyOdimSources_Check(op) ((op)->ob_type == &PyOdimSources_Type)

/** Forward declaration of PyOdimSources_GetNative */
static PyOdimSources_GetNative_RETURN PyOdimSources_GetNative PyOdimSources_GetNative_PROTO;

/** Forward declaration of PyOdimSources_New */
static PyOdimSources_New_RETURN PyOdimSources_New PyOdimSources_New_PROTO;

/** Prototype for PyOdimSources modules Load function */
static PyOdimSources_Load_RETURN PyOdimSources_Load PyOdimSources_Load_PROTO;

#else
/** Pointers to types and functions */
static void **PyOdimSources_API;

/**
 * Returns a pointer to the internal odim sources, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyOdimSources_GetNative \
  (*(PyOdimSources_GetNative_RETURN (*)PyOdimSources_GetNative_PROTO) PyOdimSources_API[PyOdimSources_GetNative_NUM])

/**
 * Creates a odim sources instance. Release this object with Py_DECREF.  If a OdimSources_t sources is
 * provided and this registry already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] area - the OdimSources_t intance.
 * @returns the PyOdimSources instance.
 */
#define PyOdimSources_New \
  (*(PyOdimSources_New_RETURN (*)PyOdimSources_New_PROTO) PyOdimSources_API[PyOdimSources_New_NUM])

/**
 * Loads a odim sources instance. Release this object with Py_DECREF.
 * @param[in] filename - the filename.
 * @returns the PyOdimSources instance.
 */
#define PyOdimSources_Load \
  (*(PyOdimSources_Load_RETURN (*)PyOdimSources_Load_PROTO) PyOdimSources_API[PyOdimSources_Load_NUM])

/**
 * Checks if the object is a python odim sources.
 */
#define PyOdimSources_Check(op) \
   (Py_TYPE(op) == &PyOdimSources_Type)

#define PyOdimSources_Type (*(PyTypeObject*)PyOdimSources_API[PyOdimSources_Type_NUM])

/**
 * Imports the PyOdimSources module (like import _odimsources in python).
 */
#define import_odimsources() \
    PyOdimSources_API = (void **)PyCapsule_Import(PyOdimSources_CAPSULE_NAME, 1);

#endif

#endif /* PYODIMSOURCES_H */
