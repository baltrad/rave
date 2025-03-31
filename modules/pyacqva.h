/* --------------------------------------------------------------------
Copyright (C) 2024 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the Acqva API.
 * @file
 * @deprecated 
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-01-18
 */
#ifndef PYACQVA_H
#define PYACQVA_H
#include "acqva.h"

/**
 * A composite generator
 * @deprecated {Will be removed in the future in favor of PyAcqvaCompositeGeneratorFactory}
 */
typedef struct {
  PyObject_HEAD; /*Always has to be on top*/
  Acqva_t* acqva; /**< the composite generator */
} PyAcqva;

#define PyAcqva_Type_NUM 0                       /**< index of type */

#define PyAcqva_GetNative_NUM 1                  /**< index of GetNative */
#define PyAcqva_GetNative_RETURN Acqva_t*    /**< return type for GetNative */
#define PyAcqva_GetNative_PROTO (PyAcqva*)   /**< arguments for GetNative */

#define PyAcqva_New_NUM 2                        /**< index of New */
#define PyAcqva_New_RETURN PyAcqva*          /**< return type for New */
#define PyAcqva_New_PROTO (Acqva_t*)         /**< arguments for New */

#define PyAcqva_API_pointers 3                   /**< number of api pointers */

#define PyAcqva_CAPSULE_NAME "_pyacqva._C_API"


#ifdef PYACQVA_MODULE
/** Forward declaration of type*/
extern PyTypeObject PyAcqva_Type;

/** Checks if the object is a PyAcqva or not */
#define PyAcqva_Check(op) ((op)->ob_type == &PyAcqva_Type)

/** Forward declaration of PyAcqva_GetNative */
static PyAcqva_GetNative_RETURN PyAcqva_GetNative PyAcqva_GetNative_PROTO;

/** Forward declaration of PyAcqva_New */
static PyAcqva_New_RETURN PyAcqva_New PyAcqva_New_PROTO;

#else
/** pointers to types and functions */
static void **PyAcqva_API;

/**
 * Returns a pointer to the internal composite, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyAcqva_GetNative \
  (*(PyAcqva_GetNative_RETURN (*)PyAcqva_GetNative_PROTO) PyAcqva_API[PyAcqva_GetNative_NUM])

/**
 * Creates a new acqva instance. Release this object with Py_DECREF. If a Acqva_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] acqva - the Acqva_t intance.
 * @returns the PyAcqva instance.
 */
#define PyAcqva_New \
  (*(PyAcqva_New_RETURN (*)PyAcqva_New_PROTO) PyAcqva_API[PyAcqva_New_NUM])

/**
 * Checks if the object is a python acqva.
 */
#define PyAcqva_Check(op) \
   (Py_TYPE(op) == &PyAcqva_Type)

#define PyAcqva_Type (*(PyTypeObject*)PyAcqva_API[PyAcqva_Type_NUM])

/**
 * Imports the PyAcqva module (like import _pyacqva in python).
 */
#define import_pyacqva() \
    PyAcqva_API = (void **)PyCapsule_Import(PyAcqva_CAPSULE_NAME, 1);

#endif



#endif /* PyAcqva_H */
