/* --------------------------------------------------------------------
Copyright (C) 2015 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the VPR Correction API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2015-03-23
 */
#ifndef PYVPRCORRECTION_H
#define PYVPRCORRECTION_H
#include "rave_vpr_correction.h"

/**
 * The transformator
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  RaveVprCorrection_t* vpr;  /**< the c-api vpr correction api */
} PyVprCorrection;

#define PyVprCorrection_Type_NUM 0                     /**< index for Type */

#define PyVprCorrection_GetNative_NUM 1                /**< index for GetNative fp */
#define PyVprCorrection_GetNative_RETURN RaveVprCorrection_t*  /**< Return type for GetNative */
#define PyVprCorrection_GetNative_PROTO (PyVprCorrection*) /**< Argument prototype for GetNative */

#define PyVprCorrection_New_NUM 2                      /**< index for New fp */
#define PyVprCorrection_New_RETURN PyVprCorrection*        /**< Return type for New */
#define PyVprCorrection_New_PROTO (RaveVprCorrection_t*)       /**< Argument prototype for New */

#define PyVprCorrection_API_pointers 3                 /**< total number of C API pointers */

#define PyVprCorrection_CAPSULE_NAME "_vprcorrection._C_API"

#ifdef PYVPRCORRECTION_MODULE
/** declared in pyvprcorrection module */
extern PyTypeObject PyVprCorrection_Type;

/** checks if the object is a PyVprCorrection type or not */
#define PyVprCorrection_Check(op) ((op)->ob_type == &PyVprCorrection_Type)

/** Prototype for PyVprCorrection modules GetNative function */
static PyVprCorrection_GetNative_RETURN PyVprCorrection_GetNative PyVprCorrection_GetNative_PROTO;

/** Prototype for PyVprCorrection modules New function */
static PyVprCorrection_New_RETURN PyVprCorrection_New PyVprCorrection_New_PROTO;

#else
/** static pointer containing the pointers to function pointers and other definitions */
static void **PyVprCorrection_API;

/**
 * Returns a pointer to the internal polar scan, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyVprCorrection_GetNative \
  (*(PyVprCorrection_GetNative_RETURN (*)PyVprCorrection_GetNative_PROTO) PyVprCorrection_API[PyVprCorrection_GetNative_NUM])

/**
 * Creates a new polar scan instance. Release this object with Py_DECREF.
 * @param[in] scan - the PolarScan_t intance.
 * @returns the PyVprCorrection instance.
 */
#define PyVprCorrection_New \
  (*(PyVprCorrection_New_RETURN (*)PyVprCorrection_New_PROTO) PyVprCorrection_API[PyVprCorrection_New_NUM])

/**
 * Checks if the object is a python vpr correction generator.
 */
#define PyVprCorrection_Check(op) \
   (Py_TYPE(op) == &PyVprCorrection_Type)

#define PyVprCorrection_Type (*(PyTypeObject*)PyVprCorrection_API[PyVprCorrection_Type_NUM])

/**
  * Imports the PyVprCorrection module (like import _vprcorrection in python).
 */
#define import_pyvprcorrection() \
    PyVprCorrection_API = (void **)PyCapsule_Import(PyQITotal_CAPSULE_NAME, 1);


#endif

#endif /* PYVPRCORRECTION_H */
