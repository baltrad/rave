/* --------------------------------------------------------------------
Copyright (C) 2020- Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the LazyNodeListReader API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2020-11-06
 */
#ifndef PYLAZYNODELISTIO_H
#define PYLAZYNODELISTIO_H
#include <lazy_nodelist_reader.h>

/**
 * The LazyNodeListReader
 */
typedef struct {
  PyObject_HEAD /* Always has to be on top */
  LazyNodeListReader_t* reader;  /**< the lazy nodelist instance instance */
} PyLazyNodeListReader;

#define PyLazyNodeListReader_Type_NUM 0                     /**< index for Type */

#define PyLazyNodeListReader_GetNative_NUM 1                /**< index for GetNative */
#define PyLazyNodeListReader_GetNative_RETURN LazyNodeListReader_t*     /**< return type for GetNative */
#define PyLazyNodeListReader_GetNative_PROTO (PyLazyNodeListReader*)    /**< argument prototype for GetNative */

#define PyLazyNodeListReader_New_NUM 2                      /**< index for New */
#define PyLazyNodeListReader_New_RETURN PyLazyNodeListReader*   /**< return type for New */
#define PyLazyNodeListReader_New_PROTO (LazyNodeListReader_t*)  /**< argument prototype for New */

#define PyLazyNodeListReader_Read_NUM 3                     /**< index for Read */
#define PyLazyNodeListReader_Read_RETURN PyLazyNodeListReader*          /**< return type for Read */
#define PyLazyNodeListReader_Read_PROTO (const char* filename) /**< argument prototype for Read */


#define PyLazyNodeListReader_API_pointers 4                 /**< Total number of C API pointers */

#define PyLazyNodeListReader_CAPSULE_NAME "_lazynodelistreader._C_API"

#ifdef PYLAZYNODELISTREADER_MODULE
/** declared in pylazynodelistreader module */
extern PyTypeObject PyLazyNodeListReader_Type;

/** checks if the object is a PyLazyNodeListReader type or not */
#define PyLazyNodeListReader_Check(op) ((op)->ob_type == &PyLazyNodeListReader_Type)

/** Prototype for PyLazyNodeListReader modules GetNative function */
static PyLazyNodeListReader_GetNative_RETURN PyLazyNodeListReader_GetNative PyLazyNodeListReader_GetNative_PROTO;

/** Prototype for PyLazyNodeListReader modules New function */
static PyLazyNodeListReader_New_RETURN PyLazyNodeListReader_New PyLazyNodeListReader_New_PROTO;

/** Prototype for PyLazyNodeListReader modules Open function */
static PyLazyNodeListReader_Read_RETURN PyLazyNodeListReader_Read PyLazyNodeListReader_Read_PROTO;

#else
/** static pointer containing the pointers to function pointers and other definitions */
static void **PyLazyNodeListReader_API;

/**
 * Returns a pointer to the internal lazy nodelist io, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyLazyNodeListReader_GetNative \
  (*(PyLazyNodeListReader_GetNative_RETURN (*)PyLazyNodeListReader_GetNative_PROTO) PyLazyNodeListReader_API[PyLazyNodeListReader_GetNative_NUM])

/**
 * Creates a new lazy nodelist io instance. Release this object with Py_DECREF. If a LazyNodeListReader_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] raveio - the LazyNodeListReader_t intance.
 * @returns the PyLazyNodeListReader instance.
 */
#define PyLazyNodeListReader_New \
  (*(PyLazyNodeListReader_New_RETURN (*)PyLazyNodeListReader_New_PROTO) PyLazyNodeListReader_API[PyLazyNodeListReader_New_NUM])

/**
 * Read a file into a lazynodelistio instance. Release this object with Py_DECREF.
 * @param[in] filename - the filename.
 * @returns the PyLazyNodeListReader instance.
 */
#define PyLazyNodeListReader_Read \
  (*(PyLazyNodeListReader_Read_RETURN (*)PyLazyNodeListReader_Read_PROTO) PyLazyNodeListReader_API[PyLazyNodeListReader_Read_NUM])

/**
 * Checks if the object is a python rave io.
 */
#define PyLazyNodeListReader_Check(op) \
   (Py_TYPE(op) == &PyLazyNodeListReader_Type)

#define PyLazyNodeListReader_Type (*(PyTypeObject*)PyLazyNodeListReader_API[PyLazyNodeListReader_Type_NUM])

/**
 * Imports the PyLazyNodeListReader module (like import _lazynodelistio in python).
 */
#define import_pylazynodelistio() \
    PyLazyNodeListReader_API = (void **)PyCapsule_Import(PyLazyNodeListReader_CAPSULE_NAME, 1);

#endif

#endif /* PYLAZYNODELISTIO_H */
