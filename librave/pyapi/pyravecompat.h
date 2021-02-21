/* --------------------------------------------------------------------
Copyright (C) 2016 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python compability file for making it possible to compile rave for both python >2.6 and python3
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2016-04-19
 */

/* Define this to ensure that we are not using old API:s from Numpy. However, for now we are more interested in getting the
 * code to build on both 2.7 and 3.x
 */
/*
 * #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
 */
#include "Python.h"

#ifndef PYRAVECOMPAT_H
#define PYRAVECOMPAT_H


/**
 * Tests if a python object string (or unicode) is equal to name.
 * @param[in] ptr - the python string or unicode
 * @param[in] name - the string to compare with
 * @return 0, -1 or 1
 */
int PyRaveAPI_CompareWithASCIIString(PyObject* ptr, const char* name);

/**
 * Creates a string compatible with the current python interpreeter.
 */
PyObject* PyRaveAPI_StringOrUnicode_FromASCII(const char *buffer);

#ifndef PyInt_Check
#define PyInt_Check             PyLong_Check
#define PyInt_FromLong          PyLong_FromLong
#define PyInt_AsLong            PyLong_AsLong
#define PyInt_Type              PyLong_Type
#endif

#ifndef PyString_Check
#define PyString_Check          PyUnicode_Check
#define PyString_AsString       PyUnicode_AsUTF8
#define PyString_FromString     PyUnicode_FromString
#define PyString_FromFormat     PyUnicode_FromFormat
#endif

#if PY_MAJOR_VERSION >= 3
#define PY_RAVE_ATTRO_NAME_TO_STRING PyUnicode_AsUTF8

#else
#define PY_RAVE_ATTRO_NAME_TO_STRING PyString_AsString

#endif

#define PY_COMPARE_ATTRO_NAME_WITH_STRING(ptr, name) PyRaveAPI_CompareWithASCIIString(ptr, name)

#define PY_COMPARE_STRING_WITH_ATTRO_NAME(name, ptr) PyRaveAPI_CompareWithASCIIString(ptr, name)

#if PY_MAJOR_VERSION >= 3
#define MOD_INIT_ERROR        NULL
#define MOD_INIT_SUCCESS(val) val
#define MOD_INIT(name)        PyMODINIT_FUNC PyInit_##name(void)
#define MOD_INIT_DEF(ob, name, doc, methods) \
  static struct PyModuleDef moduledef = { \
    PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
    ob = PyModule_Create(&moduledef);

#define MOD_INIT_CREATE_CAPI(ptr, name) PyCapsule_New(ptr, name, NULL)
#define MOD_INIT_IS_CAPI(ptr) PyCapsule_CheckExact(ptr)
#define MOD_INIT_GET_CAPI(ptr, name) PyCapsule_GetPointer(ptr, name)
#define MOD_INIT_SETUP_TYPE(itype, otype) Py_TYPE(&itype) = otype
#define MOD_INIT_VERIFY_TYPE_READY(type) if (PyType_Ready(type) < 0) return MOD_INIT_ERROR

#else
#define MOD_INIT_ERROR
#define MOD_INIT_SUCCESS(val)
#define MOD_INIT(name) void init##name(void)
#define MOD_INIT_DEF(ob, name, doc, methods) \
  ob = Py_InitModule3(name, methods, doc);
#define MOD_INIT_CREATE_CAPI(ptr, name) PyCObject_FromVoidPtr(ptr, name)
#define MOD_INIT_IS_CAPI(ptr) PyCObject_Check(ptr)
#define MOD_INIT_GET_CAPI(ptr, name) PyCObject_AsVoidPtr(ptr)
#define MOD_INIT_SETUP_TYPE(itype, otype) itype.ob_type = otype
#define MOD_INIT_VERIFY_TYPE_READY(type)
#endif

/**
 * Macros that can be used to simplify way of generating __dir__ content that is necessary for executing dir() on an object
 */
#define MOD_DIR_FORWARD_DECLARE(name) static PyObject* _##name##__dir__(name *self)

#define MOD_DIR_REFERENCE(name) _##name##__dir__

#define MOD_DIR_APPEND(list, str)                               \
  do {                                            \
    PyObject *o = PyUnicode_FromString(str);        \
    if (o != NULL)                          \
      PyList_Append(list, o);         \
    Py_XDECREF(o);                          \
  } while (0)

#define MOD_DIR_FUNCTION(name, method_listing) static PyObject * _##name##__dir__(name *self) { \
  int i=0; \
  PyObject* rc = PyList_New(0); \
  if (!rc) \
    return NULL; \
\
  while (method_listing[i].ml_name != NULL) { \
    MOD_DIR_APPEND(rc, method_listing[i++].ml_name); \
  } \
\
  return rc; \
}


#endif
