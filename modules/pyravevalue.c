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
  PyObject* pyobj = NULL;
  RaveValue_t* ravevalue = NULL;

  PyRaveValue* result = NULL;
  if (!PyArg_ParseTuple(args, "|O", &pyobj)) {
    return NULL;
  }

  if (pyobj != NULL) {
    ravevalue = PyRaveApi_RaveValueFromObject(pyobj);
    if (ravevalue == NULL) {
      return NULL;
    }
  }

  result = PyRaveValue_New(ravevalue);
  
  RAVE_OBJECT_RELEASE(ravevalue);

  return (PyObject*)result;
}

static PyObject* _pyravevalue_isStringArray(PyRaveValue* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyBool_FromLong(RaveValue_isStringArray(self->value));
}

static PyObject* _pyravevalue_isLongArray(PyRaveValue* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyBool_FromLong(RaveValue_isLongArray(self->value));
}

static PyObject* _pyravevalue_isDoubleArray(PyRaveValue* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyBool_FromLong(RaveValue_isDoubleArray(self->value));
}

static PyObject* _pyravevalue_toJSON(PyRaveValue* self, PyObject* args)
{
  char* jsonStr = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  if (RaveValue_type(self->value) == RaveValue_Type_Undefined) {
    Py_RETURN_NONE;
  }

  jsonStr = RaveValue_toJSON(self->value);
  if (jsonStr == NULL) {
    raiseException_returnNULL(PyExc_RuntimeError, "Could not create json string");
  }
  result = PyString_FromString(jsonStr);
  RAVE_FREE(jsonStr);
  return result;
}

/**
 * All methods a area can have
 */
static struct PyMethodDef _pyravevalue_methods[] =
{
  {"value", NULL, METH_VARARGS},
  {"isStringArray", (PyCFunction) _pyravevalue_isStringArray, 1,
    "isStringArray()\n\n"
    "Returns if this instance can be represented as a string array or not.\n\n"},
  {"isLongArray", (PyCFunction) _pyravevalue_isLongArray, 1,
    "isLongArray()\n\n"
    "Returns if this instance can be represented as a long array or not.\n\n"},
  {"isDoubleArray", (PyCFunction) _pyravevalue_isDoubleArray, 1,
    "isDoubleArray()\n\n"
    "Returns if this instance can be represented as a double array or not.\n\n"},
  {"toJSON", (PyCFunction) _pyravevalue_toJSON, 1,
    "toJSON()\n\n"
    "Returns the JSON representation of self.\n\n"},
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

  } else {
    raiseException_gotoTag(done, PyExc_AttributeError,
        PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }
  result = 0;
done:
  return result;
}

static PyObject* _pyravevalue_fromJSON(PyObject* self, PyObject* args)
{
  char* str = NULL;
  PyObject* result = NULL;

  RaveValue_t* rvalue = NULL;
  if (!PyArg_ParseTuple(args,"s", &str)) {
    return NULL;
  }
  rvalue = RaveValue_fromJSON(str);
  if (rvalue != NULL) {
    result = (PyObject*)PyRaveValue_New(rvalue);
  }
  RAVE_OBJECT_RELEASE(rvalue);
  return result;
}

static PyObject* _pyravevalue_loadJSON(PyObject* self, PyObject* args)
{
  char* filename = NULL;
  PyObject* result = NULL;

  RaveValue_t* rvalue = NULL;
  if (!PyArg_ParseTuple(args,"s", &filename)) {
    return NULL;
  }
  rvalue = RaveValue_loadJSON(filename);
  if (rvalue != NULL) {
    result = (PyObject*)PyRaveValue_New(rvalue);
  } else {
    raiseException_returnNULL(PyExc_IOError, "Could not load file");
  }
  RAVE_OBJECT_RELEASE(rvalue);
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
  {"fromJSON", (PyCFunction)_pyravevalue_fromJSON, 1,
    "fromJSON(str) -> new instance of the RaveValueCore object\n\n"
    "Creates a new instance of the RaveValueCore object assuming provided string is json formatted"},
  {"loadJSON", (PyCFunction)_pyravevalue_loadJSON, 1,
    "loadJSON(filename) -> read instance of the RaveValueCore object\n\n"
    "Loads a JSON object from a file and creates a RaveValueCore object is possible"},
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
