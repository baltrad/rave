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
 * Python version of the Composite Generator API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-12-05
 */
#ifndef PYCOMPOSITEGENERATOR_H
#define PYCOMPOSITEGENERATOR_H
#include "compositegenerator.h"

/**
 * A composite generator
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  CompositeGenerator_t* generator; /**< the composite generator */
} PyCompositeGenerator;

#define PyCompositeGenerator_Type_NUM 0                       /**< index of type */

#define PyCompositeGenerator_GetNative_NUM 1                  /**< index of GetNative */
#define PyCompositeGenerator_GetNative_RETURN CompositeGenerator_t*    /**< return type for GetNative */
#define PyCompositeGenerator_GetNative_PROTO (PyCompositeGenerator*)   /**< arguments for GetNative */

#define PyCompositeGenerator_New_NUM 2                        /**< index of New */
#define PyCompositeGenerator_New_RETURN PyCompositeGenerator*          /**< return type for New */
#define PyCompositeGenerator_New_PROTO (CompositeGenerator_t*)         /**< arguments for New */

#define PyCompositeGenerator_API_pointers 3                   /**< number of api pointers */

#define PyCompositeGenerator_CAPSULE_NAME "_compositegenerator._C_API"


#ifdef PYCOMPOSITEGENERATOR_MODULE
/** Forward declaration of type*/
extern PyTypeObject PyCompositeGenerator_Type;

/** Checks if the object is a PyComposite or not */
#define PyCompositeGenerator_Check(op) ((op)->ob_type == &PyCompositeGenerator_Type)

/** Forward declaration of PyCompositeGenerator_GetNative */
static PyCompositeGenerator_GetNative_RETURN PyCompositeGenerator_GetNative PyCompositeGenerator_GetNative_PROTO;

/** Forward declaration of PyCompositeGenerator_New */
static PyCompositeGenerator_New_RETURN PyCompositeGenerator_New PyCompositeGenerator_New_PROTO;

#else
/** pointers to types and functions */
static void **PyCompositeGenerator_API;

/**
 * Returns a pointer to the internal composite, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyCompositeGenerator_GetNative \
  (*(PyCompositeGenerator_GetNative_RETURN (*)PyCompositeGenerator_GetNative_PROTO) PyCompositeGenerator_API[PyCompositeGenerator_GetNative_NUM])

/**
 * Creates a new composite generator instance. Release this object with Py_DECREF. If a CompositeGenerator_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] generator - the CompositeGenerator_t intance.
 * @returns the PyCompositeGenerator instance.
 */
#define PyCompositeGenerator_New \
  (*(PyCompositeGenerator_New_RETURN (*)PyCompositeGenerator_New_PROTO) PyCompositeGenerator_API[PyCompositeGenerator_New_NUM])

/**
 * Checks if the object is a python composite.
 */
#define PyCompositeGenerator_Check(op) \
   (Py_TYPE(op) == &PyCompositeGenerator_Type)

#define PyCompositeGenerator_Type (*(PyTypeObject*)PyCompositeGenerator_API[PyCompositeGenerator_Type_NUM])

/**
 * Imports the PyCompositeGenerator module (like import _compositegenerator in python).
 */
#define import_compositegenerator() \
    PyCompositeGenerator_API = (void **)PyCapsule_Import(PyCompositeGenerator_CAPSULE_NAME, 0);

#endif



#endif /* PYCOMPOSITEGENERATOR_H */
