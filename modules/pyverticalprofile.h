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
 * Python version of the VerticalProfile API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2012-08-24
 */

#ifndef PYVERTICALPROFILE_H
#define PYVERTICALPROFILE_H
#include "vertical_profile.h"
#include <Python.h>

/**
 * A polar scan
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  VerticalProfile_t* vp; /**< the vertical profile type */
} PyVerticalProfile;

#define PyVerticalProfile_Type_NUM 0                      /**< index of the type */

#define PyVerticalProfile_GetNative_NUM 1                 /**< index of GetNative */
#define PyVerticalProfile_GetNative_RETURN VerticalProfile_t*   /**< return type for GetNative */
#define PyVerticalProfile_GetNative_PROTO (PyVerticalProfile*)  /**< arguments for GetNative */

#define PyVerticalProfile_New_NUM 2                       /**< index of New */
#define PyVerticalProfile_New_RETURN PyVerticalProfile*         /**< return type for New */
#define PyVerticalProfile_New_PROTO (VerticalProfile_t*)        /**< arguments for New */

#define PyVerticalProfile_API_pointers 3                  /**< number of pointers */

#ifdef PYVERTICALPROFILE_MODULE
/** Forward declaration of the type */
extern PyTypeObject PyVerticalProfile_Type;

/** Checks if the object is a PyVerticalProfile or not */
#define PyVerticalProfile_Check(op) ((op)->ob_type == &PyVerticalProfile_Type)

/** Forward declaration of PyVerticalProfile_GetNative */
static PyVerticalProfile_GetNative_RETURN PyVerticalProfile_GetNative PyVerticalProfile_GetNative_PROTO;

/** Forward declaration of PyVerticalProfile_New */
static PyVerticalProfile_New_RETURN PyVerticalProfile_New PyVerticalProfile_New_PROTO;

#else
/**Forward declaration of the pointers */
static void **PyVerticalProfile_API;

/**
 * Returns a pointer to the internal polar scan, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyVerticalProfile_GetNative \
  (*(PyVerticalProfile_GetNative_RETURN (*)PyVerticalProfile_GetNative_PROTO) PyVerticalProfile_API[PyVerticalProfile_GetNative_NUM])

/**
 * Creates a new polar scan instance. Release this object with Py_DECREF. If a VerticalProfile_t scan is
 * provided and this scan already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] scan - the VerticalProfile_t intance.
 * @returns the PyVerticalProfile instance.
 */
#define PyVerticalProfile_New \
  (*(PyVerticalProfile_New_RETURN (*)PyVerticalProfile_New_PROTO) PyVerticalProfile_API[PyVerticalProfile_New_NUM])

/**
 * Checks if the object is a python polar scan.
 */
#define PyVerticalProfile_Check(op) \
   ((op)->ob_type == (PyTypeObject *)PyVerticalProfile_API[PyVerticalProfile_Type_NUM])

/**
 * Imports the pypolarscan module (like import _polarscan in python).
 */
static int
import_pyverticalprofile(void)
{
  PyObject *module;
  PyObject *c_api_object;

  module = PyImport_ImportModule("_verticalprofile");
  if (module == NULL) {
    return -1;
  }

  c_api_object = PyObject_GetAttrString(module, "_C_API");
  if (c_api_object == NULL) {
    Py_DECREF(module);
    return -1;
  }
  if (PyCObject_Check(c_api_object)) {
    PyVerticalProfile_API = (void **)PyCObject_AsVoidPtr(c_api_object);
  }
  Py_DECREF(c_api_object);
  Py_DECREF(module);
  return 0;
}

#endif




#endif /* PYVERTICALPROFILE_H_ */
