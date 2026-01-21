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
 * Python version of the AcqvaFeatureMap API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-11-11
 */
#ifndef PYACQVAFEATUREMAP_H
#define PYACQVAFEATUREMAP_H
#include "acqvafeaturemap.h"

/**
 * A feature map
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  AcqvaFeatureMap_t* featuremap; /**< the feature map */
} PyAcqvaFeatureMap;

typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  AcqvaFeatureMapElevation_t* elevation; /**< the elevation group */
} PyAcqvaFeatureMapElevation;

typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  AcqvaFeatureMapField_t* field; /**< the field */
} PyAcqvaFeatureMapField;

#define PyAcqvaFeatureMap_Type_NUM 0                              /**< index of type */

#define PyAcqvaFeatureMap_GetNative_NUM 1                         /**< index of GetNative */
#define PyAcqvaFeatureMap_GetNative_RETURN AcqvaFeatureMap_t*     /**< return type for GetNative */
#define PyAcqvaFeatureMap_GetNative_PROTO (PyAcqvaFeatureMap*)    /**< arguments for GetNative */

#define PyAcqvaFeatureMap_New_NUM 2                               /**< index of New */
#define PyAcqvaFeatureMap_New_RETURN PyAcqvaFeatureMap*           /**< return type for New */
#define PyAcqvaFeatureMap_New_PROTO (AcqvaFeatureMap_t*)          /**< arguments for New */

#define PyAcqvaFeatureMapElevation_Type_NUM 3                     /**< index of elevation type */

#define PyAcqvaFeatureMapElevation_GetNative_NUM 4                         /**< index of GetNative */
#define PyAcqvaFeatureMapElevation_GetNative_RETURN AcqvaFeatureMapElevation_t*     /**< return type for GetNative */
#define PyAcqvaFeatureMapElevation_GetNative_PROTO (PyAcqvaFeatureMapElevation*)    /**< arguments for GetNative */

#define PyAcqvaFeatureMapElevation_New_NUM 5                     /**< index of elevation type */
#define PyAcqvaFeatureMapElevation_New_RETURN PyAcqvaFeatureMapElevation*     /**< return type for New */
#define PyAcqvaFeatureMapElevation_New_PROTO (AcqvaFeatureMapElevation_t*)    /**< arguments for New */

#define PyAcqvaFeatureMapField_Type_NUM 6                         /**< index of field type */

#define PyAcqvaFeatureMapField_GetNative_NUM 7                         /**< index of GetNative */
#define PyAcqvaFeatureMapField_GetNative_RETURN AcqvaFeatureMapField_t*     /**< return type for GetNative */
#define PyAcqvaFeatureMapField_GetNative_PROTO (PyAcqvaFeatureMapField*)    /**< arguments for GetNative */

#define PyAcqvaFeatureMapField_New_NUM 8                     /**< index of elevation type */
#define PyAcqvaFeatureMapField_New_RETURN PyAcqvaFeatureMapField*     /**< return type for GetNative */
#define PyAcqvaFeatureMapField_New_PROTO (AcqvaFeatureMapField_t*)    /**< arguments for GetNative */

#define PyAcqvaFeatureMap_API_pointers 9                          /**< number of API pointers */

#define PyAcqvaFeatureMap_CAPSULE_NAME "_acqvafeaturemap._C_API"


#ifdef PYACQVAFEATUREMAP_MODULE
/** Forward declaration of type */
extern PyTypeObject PyAcqvaFeatureMap_Type;

extern PyTypeObject PyAcqvaFeatureMapElevation_Type;

extern PyTypeObject PyAcqvaFeatureMapField_Type;

/** Checks if the object is a PyAcqvaFeatureMap or not */
#define PyAcqvaFeatureMap_Check(op) ((op)->ob_type == &PyAcqvaFeatureMap_Type)

/** Checks if the object is a PyAcqvaFeatureMapElevation or not */
#define PyAcqvaFeatureMapElevation_Check(op) ((op)->ob_type == &PyAcqvaFeatureMapElevation_Type)

/** Checks if the object is a PyAcqvaFeatureMapField or not */
#define PyAcqvaFeatureMapField_Check(op) ((op)->ob_type == &PyAcqvaFeatureMapField_Type)

/** Forward declaration of PyAcqvaFeatureMap_GetNative */
static PyAcqvaFeatureMap_GetNative_RETURN PyAcqvaFeatureMap_GetNative PyAcqvaFeatureMap_GetNative_PROTO;

/** Forward declaration of PyAcqvaFeatureMap_New */
static PyAcqvaFeatureMap_New_RETURN PyAcqvaFeatureMap_New PyAcqvaFeatureMap_New_PROTO;

/** Forward declaration of PyAcqvaFeatureMapElevation_GetNative */
static PyAcqvaFeatureMapElevation_GetNative_RETURN PyAcqvaFeatureMapElevation_GetNative PyAcqvaFeatureMapElevation_GetNative_PROTO;

/** Forward declaration of PyAcqvaFeatureMapElevation_New */
static PyAcqvaFeatureMapElevation_New_RETURN PyAcqvaFeatureMapElevation_New PyAcqvaFeatureMapElevation_New_PROTO;

/** Forward declaration of PyAcqvaFeatureMapField_GetNative */
static PyAcqvaFeatureMapField_GetNative_RETURN PyAcqvaFeatureMapField_GetNative PyAcqvaFeatureMapField_GetNative_PROTO;

/** Forward declaration of PyAcqvaFeatureMapElevation_New */
static PyAcqvaFeatureMapField_New_RETURN PyAcqvaFeatureMapField_New PyAcqvaFeatureMapField_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyAcqvaFeatureMap_API;

/**
 * Returns a pointer to the internal acqva feature map, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyAcqvaFeatureMap_GetNative \
  (*(PyAcqvaFeatureMap_GetNative_RETURN (*)PyAcqvaFeatureMap_GetNative_PROTO) PyAcqvaFeatureMap_API[PyAcqvaFeatureMap_GetNative_NUM])

/**
 * Creates a acqva feature map instance. Release this object with Py_DECREF.  If a AcqvaFeatureMap_t is
 * provided and this object already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] obj - the AcqvaFeatureMap_t intance.
 * @returns the PyAcqvaFeatureMap instance.
 */
#define PyAcqvaFeatureMap_New \
  (*(PyAcqvaFeatureMap_New_RETURN (*)PyAcqvaFeatureMap_New_PROTO) PyAcqvaFeatureMap_API[PyAcqvaFeatureMap_New_NUM])

/**
 * Returns a pointer to the internal acqva feature map elevation, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyAcqvaFeatureMapElevation_GetNative \
  (*(PyAcqvaFeatureMapElevation_GetNative_RETURN (*)PyAcqvaFeatureMapElevation_GetNative_PROTO) PyAcqvaFeatureMapElevation_API[PyAcqvaFeatureMapElevation_GetNative_NUM])

/**
 * Creates a new acqva feature map elevation. Release this object with Py_DECREF.  If a AcqvaFeatureMapElevation_t is
 * provided and this object already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] obj - the AcqvaFeatureMapElevation_t intance.
 * @returns the PyAcqvaFeatureMapElevation instance.
 */
#define PyAcqvaFeatureMapElevation_New \
  (*(PyAcqvaFeatureMapElevation_New_RETURN (*)PyAcqvaFeatureMapElevation_New_PROTO) PyAcqvaFeatureMapElevation_API[PyAcqvaFeatureMapElevation_New_NUM])

/**
 * Returns a pointer to the internal acqva feature map field, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyAcqvaFeatureMapField_GetNative \
  (*(PyAcqvaFeatureMapField_GetNative_RETURN (*)PyAcqvaFeatureMapField_GetNative_PROTO) PyAcqvaFeatureMapField_API[PyAcqvaFeatureMapField_GetNative_NUM])

/**
 * Creates a new acqva feature map field. Release this object with Py_DECREF.  If a AcqvaFeatureMapField_t is
 * provided and this object already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] obj - the AcqvaFeatureMapField_t intance.
 * @returns the PyAcqvaFeatureMapField instance.
 */
#define PyAcqvaFeatureMapField_New \
  (*(PyAcqvaFeatureMapField_New_RETURN (*)PyAcqvaFeatureMapField_New_PROTO) PyAcqvaFeatureMapField_API[PyAcqvaFeatureMapField_New_NUM])

/**
 * Checks if the object is a python area.
 */
#define PyAcqvaFeatureMap_Check(op) \
   (Py_TYPE(op) == &PyAcqvaFeatureMap_Type)

#define PyAcqvaFeatureMapElevation_Check(op) \
   (Py_TYPE(op) == &PyAcqvaFeatureMapElevation_Type)

#define PyAcqvaFeatureMapField_Check(op) \
   (Py_TYPE(op) == &PyAcqvaFeatureMapField_Type)

#define PyAcqvaFeatureMap_Type (*(PyTypeObject*)PyAcqvaFeatureMap_API[PyAcqvaFeatureMap_Type_NUM])

#define PyAcqvaFeatureMapElevation_Type (*(PyTypeObject*)PyAcqvaFeatureMap_API[PyAcqvaFeatureMapElevation_Type_NUM])

#define PyAcqvaFeatureMapField_Type (*(PyTypeObject*)PyAcqvaFeatureMap_API[PyAcqvaFeatureMapElevation_Type_NUM])

/**
 * Imports the PyArea module (like import _area in python).
 */
#define import_acqvafeaturemap() \
    PyAcqvaFeatureMap_API = (void **)PyCapsule_Import(PyAcqvaFeatureMap_CAPSULE_NAME, 1);

#endif

#endif /* PYACQVAFEATUREMAP_H */
