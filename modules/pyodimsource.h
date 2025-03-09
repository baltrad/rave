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
 * Python version of the OdimSource API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-16
 */
#ifndef PYODIMSOURCE_H
#define PYODIMSOURCE_H
#include "odim_source.h"

/**
 * A odim source
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  OdimSource_t* source; /**< the source */
} PyOdimSource;

#define PyOdimSource_Type_NUM 0                   /**< index of type */

#define PyOdimSource_GetNative_NUM 1              /**< index of GetNative */
#define PyOdimSource_GetNative_RETURN OdimSource_t*     /**< return type for GetNative */
#define PyOdimSource_GetNative_PROTO (PyOdimSource*)    /**< arguments for GetNative */

#define PyOdimSource_New_NUM 2                    /**< index of New */
#define PyOdimSource_New_RETURN PyOdimSource*           /**< return type for New */
#define PyOdimSource_New_PROTO (OdimSource_t*)          /**< arguments for New */

#define PyOdimSource_API_pointers 3               /**< number of API pointers */

#define PyOdimSource_CAPSULE_NAME "_odimsource._C_API"

#ifdef PYODIMSOURCE_MODULE
/** Forward declaration of type */
extern PyTypeObject PyOdimSource_Type;

/** Checks if the object is a PyOdimSource or not */
#define PyOdimSource_Check(op) ((op)->ob_type == &PyOdimSource_Type)

/** Forward declaration of PyOdimSource_GetNative */
static PyOdimSource_GetNative_RETURN PyOdimSource_GetNative PyOdimSource_GetNative_PROTO;

/** Forward declaration of PyOdimSource_New */
static PyOdimSource_New_RETURN PyOdimSource_New PyOdimSource_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyOdimSource_API;

/**
 * Returns a pointer to the internal area, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyOdimSource_GetNative \
  (*(PyOdimSource_GetNative_RETURN (*)PyOdimSource_GetNative_PROTO) PyOdimSource_API[PyOdimSource_GetNative_NUM])

/**
 * Creates a new odim source instance. Release this object with Py_DECREF.  If a OdimSource_t source is
 * provided and this source already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] source - the OdimSource_t intance.
 * @returns the PyOdimSource instance.
 */
#define PyOdimSource_New \
  (*(PyOdimSource_New_RETURN (*)PyOdimSource_New_PROTO) PyOdimSource_API[PyOdimSource_New_NUM])

/**
 * Checks if the object is a python area.
 */
#define PyOdimSource_Check(op) \
   (Py_TYPE(op) == &PyOdimSource_Type)

#define PyOdimSource_Type (*(PyTypeObject*)PyOdimSource_API[PyOdimSource_Type_NUM])

/**
 * Imports the PyOdimSource module (like import _odimsource in python).
 */
#define import_odimsource() \
    PyOdimSource_API = (void **)PyCapsule_Import(PyOdimSource_CAPSULE_NAME, 1);

#endif

#endif /* PYODIMSOURCE_H */
