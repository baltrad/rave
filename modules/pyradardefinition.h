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
 * Python version of the RadarDefinition API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-08-31
 */
#ifndef PYRADARDEFINITION_H
#define PYRADARDEFINITION_H
#include "radardefinition.h"

/**
 * A radar definition
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  RadarDefinition_t* def; /**< the radar definition */
} PyRadarDefinition;

#define PyRadarDefinition_Type_NUM 0                   /**< index of type */

#define PyRadarDefinition_GetNative_NUM 1              /**< index of GetNative */
#define PyRadarDefinition_GetNative_RETURN RadarDefinition_t*     /**< return type for GetNative */
#define PyRadarDefinition_GetNative_PROTO (PyRadarDefinition*)    /**< arguments for GetNative */

#define PyRadarDefinition_New_NUM 2                       /**< index of New */
#define PyRadarDefinition_New_RETURN PyRadarDefinition*   /**< return type for New */
#define PyRadarDefinition_New_PROTO (RadarDefinition_t*)  /**< arguments for New */

#define PyRadarDefinition_API_pointers 3               /**< number of API pointers */

#define PyRadarDefinition_CAPSULE_NAME "_radardef._C_API"

#ifdef PYRADARDEFINITION_MODULE
/** Forward declaration of type */
extern PyTypeObject PyRadarDefinition_Type;

/** Checks if the object is a PyRadarDefinition or not */
#define PyRadarDefinition_Check(op) ((op)->ob_type == &PyRadarDefinition_Type)

/** Forward declaration of PyRadarDefinition_GetNative */
static PyRadarDefinition_GetNative_RETURN PyRadarDefinition_GetNative PyRadarDefinition_GetNative_PROTO;

/** Forward declaration of PyRadarDefinition_New */
static PyRadarDefinition_New_RETURN PyRadarDefinition_New PyRadarDefinition_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyRadarDefinition_API;

/**
 * Returns a pointer to the internal definition, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyRadarDefinition_GetNative \
  (*(PyRadarDefinition_GetNative_RETURN (*)PyRadarDefinition_GetNative_PROTO) PyRadarDefinition_API[PyRadarDefinition_GetNative_NUM])

/**
 * Creates a new radar definition instance. Release this object with Py_DECREF.  If a RadarDefinition_t def is
 * provided and this definition already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] radar - the RadarDefinition_t intance.
 * @returns the PyRadarDefinition instance.
 */
#define PyRadarDefinition_New \
  (*(PyRadarDefinition_New_RETURN (*)PyRadarDefinition_New_PROTO) PyRadarDefinition_API[PyRadarDefinition_New_NUM])

/**
 * Checks if the object is a python radar definition.
 */
#define PyRadarDefinition_Check(op) \
   (Py_TYPE(op) == &PyRadarDefinition_Type)

#define PyRadarDefinition_Type (*(PyTypeObject*)PyRadarDefinition_API[PyRadarDefinition_Type_NUM])

/**
 * Imports the PyRadarDefinition module (like import _radardef in python).
 */
#define import_pyradardefinition() \
    PyRadarDefinition_API = (void **)PyCapsule_Import(PyRadarDefinition_CAPSULE_NAME, 1);

#endif



#endif /* PYRADARDEFINITION_H */
