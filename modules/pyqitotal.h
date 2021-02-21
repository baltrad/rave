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
 * Python version of the QI total algorithm
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2014-02-27
 */
#ifndef PYQITOTAL_H
#define PYQITOTAL_H
#include "rave_qitotal.h"

/**
 * A qi total object
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  RaveQITotal_t* qitotal; /**< The qi total algorithm */
} PyQITotal;

#define PyQITotal_Type_NUM 0                         /**< index for Type */

#define PyQITotal_GetNative_NUM 1                    /**< index for GetNative */
#define PyQITotal_GetNative_RETURN RaveQITotal_t*    /**< return type for GetNative */
#define PyQITotal_GetNative_PROTO (PyQITotal*)       /**< argument prototype for GetNative */

#define PyQITotal_New_NUM 2                          /**< index for New */
#define PyQITotal_New_RETURN PyQITotal*              /**< return type for New */
#define PyQITotal_New_PROTO (RaveQITotal_t*)         /**< argument prototype for New */

#define PyQITotal_API_pointers 3                     /**< number of function and variable pointers */

#define PyQITotal_CAPSULE_NAME "_qitotal._C_API"

#ifdef PYQITOTAL_MODULE
/** To be used within the PyQITotal-Module */
extern PyTypeObject PyQITotal_Type;

/** Checks if the object is a PyQITotal or not */
#define PyQITotal_Check(op) ((op)->ob_type == &PyQITotal_Type)

/**
 * forward declaration of PyQITotal_GetNative.
 */
static PyQITotal_GetNative_RETURN PyQITotal_GetNative PyQITotal_GetNative_PROTO;

/**
 * forward declaration of PyQITotal_New.
 */
static PyQITotal_New_RETURN PyQITotal_New PyQITotal_New_PROTO;

#else
/** Pointers to the functions and variables */
static void **PyQITotal_API;

/**
 * Returns a pointer to the internal qi total, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyQITotal_GetNative \
  (*(PyQITotal_GetNative_RETURN (*)PyQITotal_GetNative_PROTO) PyQITotal_API[PyQITotal_GetNative_NUM])

/**
 * Creates a new qi total instance. Release this object with Py_DECREF. If the passed RaveQITotal_t instance is
 * bound to a python instance, this instance will be increfed and returned.
 * @param[in] qitotal - the RaveQITotal_t instance.
 * @returns the PyQITotal instance.
 */
#define PyQITotal_New \
  (*(PyQITotal_New_RETURN (*)PyQITotal_New_PROTO) PyQITotal_API[PyQITotal_New_NUM])

/**
 * Checks if the object is a python qi total instance.
 */
#define PyQITotal_Check(op) \
   (Py_TYPE(op) == &PyQITotal_Type)

#define PyQITotal_Type (*(PyTypeObject*)PyQITotal_API[PyQITotal_Type_NUM])

/**
 * Imports the PyArea module (like import _area in python).
 */
#define import_qitotal() \
    PyQITotal_API = (void **)PyCapsule_Import(PyQITotal_CAPSULE_NAME, 1);


#endif

#endif /* PYQITOTAL_H */
