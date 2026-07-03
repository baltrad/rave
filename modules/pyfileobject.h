/* --------------------------------------------------------------------
Copyright (C) 2025 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the FileObject API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-12-19
 */
#ifndef PYFILEOBJECT_H
#define PYFILEOBJECT_H
#include "Python.h"
#include "file_object.h"

/**
 * A pia 
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  FileObject_t* fobj; /**< the file object */
} PyFileObject;

#define PyFileObject_Type_NUM 0                   /**< index of type */

#define PyFileObject_GetNative_NUM 1              /**< index of GetNative */
#define PyFileObject_GetNative_RETURN FileObject_t* /**< return type for GetNative */
#define PyFileObject_GetNative_PROTO (PyFileObject*)    /**< arguments for GetNative */

#define PyFileObject_New_NUM 2                    /**< index of New */
#define PyFileObject_New_RETURN PyFileObject*           /**< return type for New */
#define PyFileObject_New_PROTO (FileObject_t*)      /**< arguments for New */

#define PyFileObject_API_pointers 3               /**< number of API pointers */

#define PyFileObject_CAPSULE_NAME "_fileobject._C_API"


#ifdef PYFILEOBJECT_MODULE
/** Forward declaration of type */
extern PyTypeObject PyFileObject_Type;

/** Checks if the object is a PyFileObject or not */
#define PyFileObject_Check(op) ((op)->ob_type == &PyFileObject_Type)

/** Forward declaration of PyFileObject_GetNative */
static PyFileObject_GetNative_RETURN PyFileObject_GetNative PyFileObject_GetNative_PROTO;

/** Forward declaration of PyFileObject_New */
static PyFileObject_New_RETURN PyFileObject_New PyFileObject_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyFileObject_API;

/**
 * Returns a pointer to the internal file object, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyFileObject_GetNative \
  (*(PyFileObject_GetNative_RETURN (*)PyFileObject_GetNative_PROTO) PyFileObject_API[PyFileObject_GetNative_NUM])

/**
 * Creates a new file object instance. Release this object with Py_DECREF.  If a FileObject_t instance is
 * provided and this instance already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] pia - the FileObject_t intance.
 * @returns the PyFileObject instance.
 */
#define PyFileObject_New \
  (*(PyFileObject_New_RETURN (*)PyFileObject_New_PROTO) PyFileObject_API[PyFileObject_New_NUM])

/**
 * Checks if the object is a python gra instance.
 */
#define PyFileObject_Check(op) \
   (Py_TYPE(op) == &PyFileObject_Type)

#define PyFileObject_Type (*(PyTypeObject*)PyFileObject_API[PyFileObject_Type_NUM])

/**
 * Imports the PyFileObject module (like import _pia in python).
 */
#define import_fileobject() \
    PyFileObject_API = (void **)PyCapsule_Import(PyFileObject_CAPSULE_NAME, 1);

#endif

#endif /* PYFILEOBJECT_H */
