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
 * Python version of the projection pipeline API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2021-10-14
 */
#ifndef PYPROJECTIONPIPELINE_H
#define PYPROJECTIONPIPELINE_H
#include "projection_pipeline.h"

/**
 * A projection
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  ProjectionPipeline_t* pipeline; /**< The pipeline definition */
} PyProjectionPipeline;

#define PyProjectionPipeline_Type_NUM 0                                /**< index for Type */

#define PyProjectionPipeline_GetNative_NUM 1                           /**< index for GetNative */
#define PyProjectionPipeline_GetNative_RETURN ProjectionPipeline_t*    /**< return type for GetNative */
#define PyProjectionPipeline_GetNative_PROTO (PyProjectionPipeline*)   /**< argument prototype for GetNative */

#define PyProjectionPipeline_New_NUM 2                                 /**< index for New */
#define PyProjectionPipeline_New_RETURN PyProjectionPipeline*          /**< return type for New */
#define PyProjectionPipeline_New_PROTO (ProjectionPipeline_t*)         /**< argument prototype for New */

#define PyProjectionPipeline_API_pointers 3                                    /**< number of function and variable pointers */

#define PyProjectionPipeline_CAPSULE_NAME "_projectionpipeline._C_API"

#ifdef PYPROJECTIONPIPELINE_MODULE
/** To be used within the PyProjectionPipeline-Module */
extern PyTypeObject PyProjectionPipeline_Type;

/** Checks if the object is a PyProjection or not */
#define PyProjectionPipeline_Check(op) ((op)->ob_type == &PyProjectionPipeline_Type)

/**
 * forward declaration of PyProjectionPipeline_GetNative.
 */
static PyProjectionPipeline_GetNative_RETURN PyProjectionPipeline_GetNative PyProjectionPipeline_GetNative_PROTO;

/**
 * forward declaration of PyProjectionPipeline_New.
 */
static PyProjectionPipeline_New_RETURN PyProjectionPipeline_New PyProjectionPipeline_New_PROTO;

#else
/** Pointers to the functions and variables */
static void **PyProjectionPipeline_API;

/**
 * Returns a pointer to the internal projection pipeline, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyProjectionPipeline_GetNative \
  (*(PyProjectionPipeline_GetNative_RETURN (*)PyProjectionPipeline_GetNative_PROTO) PyProjectionPipeline_API[PyProjectionPipeline_GetNative_NUM])

/**
 * Creates a new projection pipeline instance. Release this object with Py_DECREF. If the passed ProjectionPipeline_t instance is
 * bound to a python instance, this instance will be increfed and returned.
 * @param[in] proj - the ProjectionPipeline_t intance.
 * @returns the PyProjectionPipeline instance.
 */
#define PyProjectionPipeline_New \
  (*(PyProjectionPipeline_New_RETURN (*)PyProjectionPipeline_New_PROTO) PyProjectionPipeline_API[PyProjectionPipeline_New_NUM])

/**
 * Checks if the object is a python projection pipeline.
 */
#define PyProjectionPipeline_Check(op) \
    (Py_TYPE(op) == &PyProjectionPipeline_Type)

#define PyProjectionPipeline_Type (*(PyTypeObject *)PyProjectionPipeline_API[PyProjectionPipeline_Type_NUM])

/**
 * Imports the pyprojectionpipeline module (like import _projectionpipeline in python).
 */
#define import_pyprojectionpipeline() \
    PyProjectionPipeline_API = (void **)PyCapsule_Import(PyProjectionPipeline_CAPSULE_NAME, 1);

#endif

#endif /* PYPROJECTIONPIPELINE_H */
