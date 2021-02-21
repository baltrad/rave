/* --------------------------------------------------------------------
Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the Detection range API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2011-02-18
 */
#ifndef PYDETECTIONRANGE_H
#define PYDETECTIONRANGE_H
#include "detection_range.h"

/**
 * A detection range generator
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  DetectionRange_t* dr; /**< the detection range generator */
} PyDetectionRange;

#define PyDetectionRange_Type_NUM 0                       /**< index of type */

#define PyDetectionRange_GetNative_NUM 1                  /**< index of GetNative */
#define PyDetectionRange_GetNative_RETURN DetectionRange_t*    /**< return type for GetNative */
#define PyDetectionRange_GetNative_PROTO (PyDetectionRange*)   /**< arguments for GetNative */

#define PyDetectionRange_New_NUM 2                        /**< index of New */
#define PyDetectionRange_New_RETURN PyDetectionRange*          /**< return type for New */
#define PyDetectionRange_New_PROTO (DetectionRange_t*)         /**< arguments for New */

#define PyDetectionRange_API_pointers 3                   /**< number of api pointers */

#define PyDetectionRange_CAPSULE_NAME "_detectionrange._C_API"

#ifdef PYDETECTIONRANGE_MODULE
/** Forward declaration of type*/
extern PyTypeObject PyDetectionRange_Type;

/** Checks if the object is a PyComposite or not */
#define PyDetectionRange_Check(op) ((op)->ob_type == &PyDetectionRange_Type)

/** Forward declaration of PyComposite_GetNative */
static PyDetectionRange_GetNative_RETURN PyDetectionRange_GetNative PyDetectionRange_GetNative_PROTO;

/** Forward declaration of PyComposite_New */
static PyDetectionRange_New_RETURN PyDetectionRange_New PyDetectionRange_New_PROTO;

#else
/** pointers to types and functions */
static void **PyDetectionRange_API;

/**
 * Returns a pointer to the internal detection range, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyDetectionRange_GetNative \
  (*(PyDetectionRange_GetNative_RETURN (*)PyDetectionRange_GetNative_PROTO) PyDetectionRange_API[PyDetectionRange_GetNative_NUM])

/**
 * Creates a new detection range instance. Release this object with Py_DECREF. If a DetectionRange_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] dr - the DetectopmRamge_t intance.
 * @returns the PyDetectionRange instance.
 */
#define PyDetectionRange_New \
  (*(PyDetectionRange_New_RETURN (*)PyDetectionRange_New_PROTO) PyDetectionRange_API[PyDetectionRange_New_NUM])

/**
 * Checks if the object is a python detection range.
 */
#define PyDetectionRange_Check(op) \
   (Py_TYPE(op) == &PyDetectionRange_Type)

#define PyDetectionRange_Type (*(PyTypeObject*)PyDetectionRange_API[PyDetectionRange_Type_NUM])

/**
 * Imports the PyDetectionRange module (like import _pydetectionrange in python).
 */
#define import_detectionrange() \
    PyArea_API = (void **)PyCapsule_Import(PyDetectionRange_CAPSULE_NAME, 1);


#define PyDetectionRange_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyDetectionRange_API[PyDetectionRange_Type_NUM])


#endif



#endif /* PYCOMPOSITE_H */
