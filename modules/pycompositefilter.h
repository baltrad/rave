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
 * Python version of the Composite Filter API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-27
 */
#ifndef PYCOMPOSITEFILTER_H
#define PYCOMPOSITEFILTER_H
#include "compositefilter.h"
#include <Python.h>

/**
 * A composite argument structure
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  CompositeFilter_t* filter; /**< the composite filter */
} PyCompositeFilter;

#define PyCompositeFilter_Type_NUM 0                       /**< index of type */

#define PyCompositeFilter_GetNative_NUM 1                  /**< index of GetNative */
#define PyCompositeFilter_GetNative_RETURN CompositeFilter_t*    /**< return type for GetNative */
#define PyCompositeFilter_GetNative_PROTO (PyCompositeFilter*)   /**< arguments for GetNative */

#define PyCompositeFilter_New_NUM 2                        /**< index of New */
#define PyCompositeFilter_New_RETURN PyCompositeFilter*          /**< return type for New */
#define PyCompositeFilter_New_PROTO (CompositeFilter_t*)         /**< arguments for New */

#define PyCompositeFilter_API_pointers 3                   /**< number of api pointers */

#define PyCompositeFilter_CAPSULE_NAME "_compositefilter._C_API"


#ifdef PYCOMPOSITEFILTER_MODULE
/** Forward declaration of type*/
extern PyTypeObject PyCompositeFilter_Type;

/** Checks if the object is a PyComposite or not */
#define PyCompositeFilter_Check(op) ((op)->ob_type == &PyCompositeFilter_Type)

/** Forward declaration of PyCompositeFilter_GetNative */
static PyCompositeFilter_GetNative_RETURN PyCompositeFilter_GetNative PyCompositeFilter_GetNative_PROTO;

/** Forward declaration of PyCompositeFilter_New */
static PyCompositeFilter_New_RETURN PyCompositeFilter_New PyCompositeFilter_New_PROTO;

#else
/** pointers to types and functions */
static void **PyCompositeFilter_API;

/**
 * Returns a pointer to the internal composite, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyCompositeFilter_GetNative \
  (*(PyCompositeFilter_GetNative_RETURN (*)PyCompositeFilter_GetNative_PROTO) PyCompositeFilter_API[PyCompositeFilter_GetNative_NUM])

/**
 * Creates a new composite generator instance. Release this object with Py_DECREF. If a CompositeArguments_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] generator - the CompositeArguments_t intance.
 * @returns the PyCompositeFilter instance.
 */
#define PyCompositeFilter_New \
  (*(PyCompositeFilter_New_RETURN (*)PyCompositeFilter_New_PROTO) PyCompositeFilter_API[PyCompositeFilter_New_NUM])

/**
 * Checks if the object is a python composite.
 */
#define PyCompositeFilter_Check(op) \
   (Py_TYPE(op) == &PyCompositeFilter_Type)

#define PyCompositeFilter_Type (*(PyTypeObject*)PyCompositeFilter_API[PyCompositeFilter_Type_NUM])

/**
 * Imports the PyCompositeFilter module (like import _compositefilter in python).
 */
#define import_compositefilter() \
    PyCompositeFilter_API = (void **)PyCapsule_Import(PyCompositeFilter_CAPSULE_NAME, 1);

#endif



#endif /* PYCOMPOSITEFILTER_H */
