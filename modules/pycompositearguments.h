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
 * Python version of the Composite Arguments API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-12-13
 */
#ifndef PYCOMPOSITEARGUMENTS_H
#define PYCOMPOSITEARGUMENTS_H
#include "compositearguments.h"

/**
 * A composite argument structure
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  CompositeArguments_t* args; /**< the composite arguments */
} PyCompositeArguments;

#define PyCompositeArguments_Type_NUM 0                       /**< index of type */

#define PyCompositeArguments_GetNative_NUM 1                  /**< index of GetNative */
#define PyCompositeArguments_GetNative_RETURN CompositeArguments_t*    /**< return type for GetNative */
#define PyCompositeArguments_GetNative_PROTO (PyCompositeArguments*)   /**< arguments for GetNative */

#define PyCompositeArguments_New_NUM 2                        /**< index of New */
#define PyCompositeArguments_New_RETURN PyCompositeArguments*          /**< return type for New */
#define PyCompositeArguments_New_PROTO (CompositeArguments_t*)         /**< arguments for New */

#define PyCompositeArguments_API_pointers 3                   /**< number of api pointers */

#define PyCompositeArguments_CAPSULE_NAME "_compositearguments._C_API"


#ifdef PYCOMPOSITEARGUMENTS_MODULE
/** Forward declaration of type*/
extern PyTypeObject PyCompositeArguments_Type;

/** Checks if the object is a PyComposite or not */
#define PyCompositeArguments_Check(op) ((op)->ob_type == &PyCompositeArguments_Type)

/** Forward declaration of PyCompositeArguments_GetNative */
static PyCompositeArguments_GetNative_RETURN PyCompositeArguments_GetNative PyCompositeArguments_GetNative_PROTO;

/** Forward declaration of PyCompositeArguments_New */
static PyCompositeArguments_New_RETURN PyCompositeArguments_New PyCompositeArguments_New_PROTO;

#else
/** pointers to types and functions */
static void **PyCompositeArguments_API;

/**
 * Returns a pointer to the internal composite, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyCompositeArguments_GetNative \
  (*(PyCompositeArguments_GetNative_RETURN (*)PyCompositeArguments_GetNative_PROTO) PyCompositeArguments_API[PyCompositeArguments_GetNative_NUM])

/**
 * Creates a new composite generator instance. Release this object with Py_DECREF. If a CompositeArguments_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] generator - the CompositeArguments_t intance.
 * @returns the PyCompositeArguments instance.
 */
#define PyCompositeArguments_New \
  (*(PyCompositeArguments_New_RETURN (*)PyCompositeArguments_New_PROTO) PyCompositeArguments_API[PyCompositeArguments_New_NUM])

/**
 * Checks if the object is a python composite.
 */
#define PyCompositeArguments_Check(op) \
   (Py_TYPE(op) == &PyCompositeArguments_Type)

#define PyCompositeArguments_Type (*(PyTypeObject*)PyCompositeArguments_API[PyCompositeArguments_Type_NUM])

/**
 * Imports the PyCompositeArguments module (like import _compositearguments in python).
 */
#define import_compositearguments() \
    PyCompositeArguments_API = (void **)PyCapsule_Import(PyCompositeArguments_CAPSULE_NAME, 1);

#endif



#endif /* PYCOMPOSITEARGUMENTS_H */
