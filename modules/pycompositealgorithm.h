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
 * Python version of the Compositing Algorithm API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2011-10-28
 */
#ifndef PYCOMPOSITEALGORITHM_H
#define PYCOMPOSITEALGORITHM_H
#include "Python.h"
#include "composite_algorithm.h"

/**
 * The poo composite algorithm instance
 */
typedef struct {
  PyObject_HEAD
  CompositeAlgorithm_t* algorithm;  /**< the composite algorithm */
} PyCompositeAlgorithm;

#define PyCompositeAlgorithm_Type_NUM 0                                     /**< index of type */

#define PyCompositeAlgorithm_GetNative_NUM 1                                /**< index of GetNative*/
#define PyCompositeAlgorithm_GetNative_RETURN CompositeAlgorithm_t*         /**< return type for GetNative */
#define PyCompositeAlgorithm_GetNative_PROTO (PyCompositeAlgorithm*)        /**< arguments for GetNative */

#define PyCompositeAlgorithm_New_NUM 2                                      /**< index of New */
#define PyCompositeAlgorithm_New_RETURN PyCompositeAlgorithm*               /**< return type for New */
#define PyCompositeAlgorithm_New_PROTO (CompositeAlgorithm_t*)              /**< arguments for New */

#define PyCompositeAlgorithm_API_pointers 3                                 /**< number of type and function pointers */

#define PyCompositeAlgorithm_CAPSULE_NAME "_compositealgorithm._C_API"

#ifdef PYCOMPOSITEALGORITHM_MODULE
/** Forward declaration of type */
extern PyTypeObject PyCompositeAlgorithm_Type;

/** Checks if the object is a PyPolarVolume or not */
#define PyCompositeAlgorithm_Check(op) ((op)->ob_type == &PyCompositeAlgorithm_Type)

/** Forward declaration of PyPolarVolume_GetNative */
static PyCompositeAlgorithm_GetNative_RETURN PyCompositeAlgorithm_GetNative PyCompositeAlgorithm_GetNative_PROTO;

/** Forward declaration of PyPolarVolume_New */
static PyCompositeAlgorithm_New_RETURN PyCompositeAlgorithm_New PyCompositeAlgorithm_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyCompositeAlgorithm_API;

/**
 * Returns a pointer to the internal composite algorithm, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyCompositeAlgorithm_GetNative \
  (*(PyCompositeAlgorithm_GetNative_RETURN (*)PyCompositeAlgorithm_GetNative_PROTO) PyCompositeAlgorithm_API[PyCompositeAlgorithm_GetNative_NUM])

/**
 * Creates a composite algorithm instance. Release this object with Py_DECREF. If a CompositeAlgorithm_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] volume - the CompositeAlgorithm_t intance.
 * @returns the PyCompositeAlgorithm instance.
 */
#define PyCompositeAlgorithm_New \
  (*(PyCompositeAlgorithm_New_RETURN (*)PyCompositeAlgorithm_New_PROTO) PyCompositeAlgorithm_API[PyCompositeAlgorithm_New_NUM])


/**
 * Checks if the object is a python composite algorithm.
 */
#define PyCompositeAlgorithm_Check(op) \
   (Py_TYPE(op) == &PyCompositeAlgorithm_Type)

#define PyCompositeAlgorithm_Type (*(PyTypeObject*)PyCompositeAlgorithm_API[PyCompositeAlgorithm_Type_NUM])

/**
 * Imports the PyCompositeAlgorithm module (like import _compositealgorithm in python).
 */
#define import_compositealgorithm() \
    PyCompositeAlgorithm_API = (void **)PyCapsule_Import(PyCompositeAlgorithm_CAPSULE_NAME, 1);

#endif

#endif /* PYPOOCOMPOSITEALGORITHM_H */
