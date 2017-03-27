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
 * Python version of the Composite API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-01-29
 */
#ifndef PYCOMPOSITE_H
#define PYCOMPOSITE_H
#include "composite.h"

/**
 * A composite generator
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  Composite_t* composite; /**< the composite generator */
} PyComposite;

#define PyComposite_Type_NUM 0                       /**< index of type */

#define PyComposite_GetNative_NUM 1                  /**< index of GetNative */
#define PyComposite_GetNative_RETURN Composite_t*    /**< return type for GetNative */
#define PyComposite_GetNative_PROTO (PyComposite*)   /**< arguments for GetNative */

#define PyComposite_New_NUM 2                        /**< index of New */
#define PyComposite_New_RETURN PyComposite*          /**< return type for New */
#define PyComposite_New_PROTO (Composite_t*)         /**< arguments for New */

#define PyComposite_API_pointers 3                   /**< number of api pointers */

#define PyComposite_CAPSULE_NAME "_pycomposite._C_API"


#ifdef PYCOMPOSITE_MODULE
/** Forward declaration of type*/
extern PyTypeObject PyComposite_Type;

/** Checks if the object is a PyComposite or not */
#define PyComposite_Check(op) ((op)->ob_type == &PyComposite_Type)

/** Forward declaration of PyComposite_GetNative */
static PyComposite_GetNative_RETURN PyComposite_GetNative PyComposite_GetNative_PROTO;

/** Forward declaration of PyComposite_New */
static PyComposite_New_RETURN PyComposite_New PyComposite_New_PROTO;

#else
/** pointers to types and functions */
static void **PyComposite_API;

/**
 * Returns a pointer to the internal composite, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyComposite_GetNative \
  (*(PyComposite_GetNative_RETURN (*)PyComposite_GetNative_PROTO) PyComposite_API[PyComposite_GetNative_NUM])

/**
 * Creates a new composite instance. Release this object with Py_DECREF. If a Composite_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] composite - the Composite_t intance.
 * @returns the PyComposite instance.
 */
#define PyComposite_New \
  (*(PyComposite_New_RETURN (*)PyComposite_New_PROTO) PyComposite_API[PyComposite_New_NUM])

/**
 * Checks if the object is a python composite.
 */
#define PyComposite_Check(op) \
   (Py_TYPE(op) == &PyComposite_Type)

#define PyComposite_Type (*(PyTypeObject*)PyComposite_API[PyComposite_Type_NUM])

/**
 * Imports the PyArea module (like import _area in python).
 */
#define import_pycomposite() \
    PyComposite_API = (void **)PyCapsule_Import(PyComposite_CAPSULE_NAME, 1);

#endif



#endif /* PYCOMPOSITE_H */
