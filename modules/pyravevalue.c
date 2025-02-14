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
 * Python version of the RaveValue API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-02-14
 */
#include "pyravecompat.h"
#include "pyraveapi.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"
#include "rave_value.h"

#define PYRAVEVALUE_MODULE    /**< to get correct part in pyravevalue.h */
#include "pyravevalue.h"
#include "rave_alloc.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_ravevalue");

/**
 * Sets a python exception and goto tag
 */
#define raiseException_gotoTag(tag, type, msg) \
{PyErr_SetString(type, msg); goto tag;}

/**
 * Sets a python exception and return NULL
 */
#define raiseException_returnNULL(type, msg) \
{PyErr_SetString(type, msg); return NULL;}

/**
 * Error object for reporting errors to the python interpreeter
 */
static PyObject *ErrorObject;

/*@{ RaveValue */
/**
 * Returns the native RaveValue_t instance.
 * @param[in] pyobj - the python instance
 * @returns the native instance.
 */
static RaveValue_t*
PyRaveValue_GetNative(PyRaveValue* pyobj)
{
  RAVE_ASSERT((pyobj != NULL), "pyobj == NULL");
  return RAVE_OBJECT_COPY(pyobj->value);
}

/**
 * Creates a python instance from a native instance or will create an
 * initial native instance if p is NULL.
 * @param[in] p - the native instance (or NULL)
 * @returns the python instance product.
 */
static PyRaveValue*
PyRaveValue_New(RaveValue_t* p)
{
  PyRaveValue* result = NULL;
  RaveValue_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RaveValue_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for value.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for value.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyRaveValue, &PyRaveValue_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->value = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->value, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyRaveValue instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyRaveValue.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the value
 * @param[in] obj the object to deallocate.
 */
static void _pyravevalue_dealloc(PyRaveValue* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->value, obj);
  RAVE_OBJECT_RELEASE(obj->value);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the area.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyravevalue_new(PyObject* self, PyObject* args)
{
  PyRaveValue* result = PyRaveValue_New(NULL);
  return (PyObject*)result;
}

/**
 * All methods a area can have
 */
static struct PyMethodDef _pyravevalue_methods[] =
{
  {"value", NULL, METH_VARARGS},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the rave field
 * @param[in] self - the rave field
 */
static PyObject* _pyravevalue_getattro(PyRaveValue* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "value") == 0) {
    return PyRaveApi_RaveValueToObject(self->value);
    // if (RaveValue_type(self->value) == RaveValue_Type_Undefined) {
    //   Py_RETURN_NONE;
    // } else if (RaveValue_type(self->value) == RaveValue_Type_String) {
    //   return PyString_FromString(RaveValue_toString(self->value));
    // } else if (RaveValue_type(self->value) == RaveValue_Type_Long) {
    //   return PyLong_FromLong(RaveValue_toLong(self->value));
    // } else if (RaveValue_type(self->value) == RaveValue_Type_Double) {
    //   return PyFloat_FromDouble(RaveValue_toDouble(self->value));
    // } else if (RaveValue_type(self->value) == RaveValue_Type_StringArray || RaveValue_type(self->value) == RaveValue_Type_LongArray || RaveValue_type(self->value) == RaveValue_Type_DoubleArray) {
    //   PyObject* result = NULL;
    //   int i = 0, arraylen = 0;
    //   char** sarray = NULL;
    //   double* darray = NULL;
    //   long* larray = NULL;

    //   if (RaveValue_type(self->value) == RaveValue_Type_StringArray) {
    //     RaveValue_getStringArray(self->value, &sarray, &arraylen);
    //   } else if (RaveValue_type(self->value) == RaveValue_Type_LongArray) {
    //     RaveValue_getLongArray(self->value, &larray, &arraylen);
    //   } else {
    //     RaveValue_getDoubleArray(self->value, &darray, &arraylen);
    //   }

    //   result = PyList_New(0);
    //   if (result == NULL) {
    //     raiseException_returnNULL(PyExc_MemoryError, "Could not allocate memory for list");
    //   }
    //   for (i = 0; i < arraylen; i++) {
    //     PyObject* obj = NULL;
    //     if (sarray != NULL) {
    //       obj = PyString_FromString(sarray[i]);
    //     } else if (larray != NULL) {
    //       obj = PyInt_FromLong(larray[i]);
    //     } else {
    //       obj = PyFloat_FromDouble(darray[i]);
    //     }
    //     if (obj == NULL) {
    //       Py_XDECREF(result);
    //       raiseException_returnNULL(PyExc_MemoryError, "failed to create value object");
    //     }
    //     if (PyList_Append(result, obj) != 0) {
    //       Py_XDECREF(result);
    //       Py_XDECREF(obj);
    //       raiseException_returnNULL(PyExc_MemoryError, "failed to create value object");
    //     }
    //     Py_DECREF(obj);        
    //   }
    //   return result;
    // }
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the attribute value
 */
static int _pyravevalue_setattro(PyRaveValue *self, PyObject *name, PyObject *val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "value") == 0) {
    if (!PyRaveApi_UpdateRaveValue(val, self->value)) {
      goto done;
    }
    // if (val == Py_None) {
    //   RaveValue_reset(self->value);
    // } else if (PyString_Check(val)) {
    //   if (!RaveValue_setString(self->value, PyString_AsString(val))) {
    //     raiseException_gotoTag(done, PyExc_MemoryError, "Failed to set string");
    //   }
    // } else if (PyInt_Check(val)) {
    //   RaveValue_setLong(self->value, PyInt_AsLong(val));
    // } else if (PyFloat_Check(val)) {
    //   RaveValue_setDouble(self->value, PyFloat_AsDouble(val));
    // } else if (PyList_Check(val)) {
    //   Py_ssize_t nvalues = PyObject_Length(val);
    //   int i = 0;
    //   RaveValue_Type vtype = RaveValue_Type_Undefined;
    //   if (nvalues > 0) {
    //     for (i = 0; i < nvalues; i++) {
    //       PyObject* pyobj = PyList_GetItem(val, i);  /* We don't need to release this since it is internal pointer.*/
    //       if (PyFloat_Check(pyobj)) {
    //         if (vtype == RaveValue_Type_Undefined || vtype == RaveValue_Type_LongArray) {
    //           vtype = RaveValue_Type_DoubleArray;
    //         } else if (vtype == RaveValue_Type_DoubleArray) {
    //           // NO OP
    //         } else {
    //           raiseException_gotoTag(done, PyExc_ValueError, "List can only contain floats, doubles and strings and all items in list must be of same type")
    //         }
    //       } else if (PyInt_Check(pyobj)) {
    //         if (vtype == RaveValue_Type_Undefined ) {
    //           vtype = RaveValue_Type_LongArray;
    //         } else if (vtype == RaveValue_Type_DoubleArray || vtype == RaveValue_Type_LongArray) {
    //           // NO OP
    //         } else {
    //           raiseException_gotoTag(done, PyExc_ValueError, "List can only contain floats, doubles and strings and all items in list must be of same type")
    //         }
    //       } else if (PyString_Check(pyobj)) {
    //         if (vtype == RaveValue_Type_Undefined) {
    //           vtype = RaveValue_Type_StringArray;
    //         } else if (vtype == RaveValue_Type_StringArray) {
    //           // NO OP
    //         } else {
    //           raiseException_gotoTag(done, PyExc_ValueError, "List can only contain floats, doubles and strings and all items in list must be of same type")
    //         }
    //       } else {
    //         raiseException_gotoTag(done, PyExc_ValueError, "List can only contain floats, doubles and strings and all items in list must be of same type")
    //       }
    //     }

    //     if (vtype == RaveValue_Type_LongArray) {
    //       long* larray = RAVE_MALLOC(sizeof(long)*nvalues);
    //       for (i = 0; i < nvalues; i++) {
    //         PyObject* pyobj = PyList_GetItem(val, i);  /* We don't need to release this since it is internal pointer.*/
    //         larray[i] = PyInt_AsLong(pyobj);
    //       }
    //       if (!RaveValue_setLongArray(self->value, larray, nvalues)) {
    //         RAVE_FREE(larray);
    //         raiseException_gotoTag(done, PyExc_MemoryError, "Failed to set list");
    //       }
    //       RAVE_FREE(larray);
    //     } else if (vtype == RaveValue_Type_DoubleArray) {
    //       double* darray = RAVE_MALLOC(sizeof(double)*nvalues);
    //       for (i = 0; i < nvalues; i++) {
    //         PyObject* pyobj = PyList_GetItem(val, i);  /* We don't need to release this since it is internal pointer.*/
    //         if (PyFloat_Check(pyobj)) {
    //           darray[i] = PyFloat_AsDouble(pyobj);
    //         } else {
    //           darray[i] = (double)PyInt_AsLong(pyobj);
    //         }
    //       }
    //       if (!RaveValue_setDoubleArray(self->value, darray, nvalues)) {
    //         RAVE_FREE(darray);
    //         raiseException_gotoTag(done, PyExc_MemoryError, "Failed to set list");
    //       }
    //       RAVE_FREE(darray);
    //     } else if (vtype == RaveValue_Type_StringArray) {
    //       char** sarray = RAVE_MALLOC(sizeof(char*)*nvalues);
    //       for (i = 0; i < nvalues; i++) {
    //         PyObject* pyobj = PyList_GetItem(val, i);  /* We don't need to release this since it is internal pointer.*/
    //         sarray[i] = (char*)PyString_AsString(pyobj);
    //       }
    //       if (!RaveValue_setStringArray(self->value, (const char**)sarray, nvalues)) {
    //         RAVE_FREE(sarray);
    //         raiseException_gotoTag(done, PyExc_MemoryError, "Failed to set list");
    //       }
    //       RAVE_FREE(sarray);
    //     }
    //   }
    // }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError,
        PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }
  result = 0;
done:
  return result;
}

static PyObject* _pyravevalue_isRaveValue(PyObject* self, PyObject* args)
{
  PyObject* inobj = NULL;
  if (!PyArg_ParseTuple(args,"O", &inobj)) {
    return NULL;
  }
  if (PyRaveValue_Check(inobj)) {
    return PyBool_FromLong(1);
  }
  return PyBool_FromLong(0);
}
/*@} End of RaveValue */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyravevalue_doc,
    "This class provides generic value object.\n"
    "\n"
    "Only member is value\n\n"
    "\n"
    );
/*@} End of Documentation about the type */


/*@{ Type definitions */
PyTypeObject PyRaveValue_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "RaveValueCore", /*tp_name*/
  sizeof(PyRaveValue), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyravevalue_dealloc,  /*tp_dealloc*/
  0,                            /*tp_print*/
  (getattrfunc)0,               /*tp_getattr*/
  (setattrfunc)0,               /*tp_setattr*/
  0,                            /*tp_compare*/
  0,                            /*tp_repr*/
  0,                            /*tp_as_number */
  0,
  0,                            /*tp_as_mapping */
  0,                            /*tp_hash*/
  (ternaryfunc)0,               /*tp_call*/
  (reprfunc)0,                  /*tp_str*/
  (getattrofunc)_pyravevalue_getattro, /*tp_getattro*/
  (setattrofunc)_pyravevalue_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyravevalue_doc,                  /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyravevalue_methods,              /*tp_methods*/
  0,                            /*tp_members*/
  0,                            /*tp_getset*/
  0,                            /*tp_base*/
  0,                            /*tp_dict*/
  0,                            /*tp_descr_get*/
  0,                            /*tp_descr_set*/
  0,                            /*tp_dictoffset*/
  0,                            /*tp_init*/
  0,                            /*tp_alloc*/
  0,                            /*tp_new*/
  0,                            /*tp_free*/
  0,                            /*tp_is_gc*/
};
/*@} End of Type definitions */

/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pyravevalue_new, 1,
      "new() -> new instance of the RaveValueCore object\n\n"
      "Creates a new instance of the RaveValueCore object"},
  {"isRaveValue", (PyCFunction)_pyravevalue_isRaveValue, 1,
      "isRaveValue(obj) -> True if object is a rave value, otherwise False\n\n"
      "Checks if the provided object is a python rave value object or not.\n\n"
      "obj - the object to check."},
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_ravevalue)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyRaveValue_API[PyRaveValue_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyRaveValue_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyRaveValue_Type);

  MOD_INIT_DEF(module, "_ravevalue", _pyravevalue_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyRaveValue_API[PyRaveValue_Type_NUM] = (void*)&PyRaveValue_Type;
  PyRaveValue_API[PyRaveValue_GetNative_NUM] = (void *)PyRaveValue_GetNative;
  PyRaveValue_API[PyRaveValue_New_NUM] = (void*)PyRaveValue_New;

  c_api_object = PyCapsule_New(PyRaveValue_API, PyRaveValue_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_ravevalue.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _ravevalue.error");
    return MOD_INIT_ERROR;
  }

  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
