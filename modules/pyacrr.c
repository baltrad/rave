/* --------------------------------------------------------------------
Copyright (C) 2012 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the Acrr API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2012-06-01
 */
#include "Python.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYACRR_MODULE    /**< to get correct part in pyacrr.h */
#include "pyacrr.h"
#include "rave_alloc.h"
#include "pycartesianparam.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_acrr");

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

/*@{ Acrr */
/**
 * Returns the native RaveAcrr_t instance.
 * @param[in] pyacrr - the python acrr instance
 * @returns the native acrr instance.
 */
static RaveAcrr_t*
PyAcrr_GetNative(PyAcrr* pyacrr)
{
  RAVE_ASSERT((pyacrr != NULL), "pyacrr == NULL");
  return RAVE_OBJECT_COPY(pyacrr->acrr);
}

/**
 * Creates a python acrr from a native acrr or will create an
 * initial native acrr if p is NULL.
 * @param[in] p - the native acrr (or NULL)
 * @returns the python acrr product.
 */
static PyAcrr*
PyAcrr_New(RaveAcrr_t* p)
{
  PyAcrr* result = NULL;
  RaveAcrr_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RaveAcrr_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for acrr.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for acrr.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyAcrr, &PyAcrr_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->acrr = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->acrr, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyAcrr instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyAcrr.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the acrr
 * @param[in] obj the object to deallocate.
 */
static void _pyacrr_dealloc(PyAcrr* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->acrr, obj);
  RAVE_OBJECT_RELEASE(obj->acrr);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the acrr.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyacrr_new(PyObject* self, PyObject* args)
{
  PyAcrr* result = PyAcrr_New(NULL);
  return (PyObject*)result;
}

/**
 * Returns if the acrr has been initialized which occurs after the
 * first call to sum.
 * @param[in] self - self
 * @param[in] args - N/A
 * @return a boolean
 */
static PyObject* _pyacrr_isInitialized(PyAcrr* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyBool_FromLong(RaveAcrr_isInitialized(self->acrr));
}

static PyObject* _pyacrr_getQuantity(PyAcrr* self, PyObject* args)
{
  if(!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  if (RaveAcrr_getQuantity(self->acrr) == NULL) {
    Py_RETURN_NONE;
  }
  return PyString_FromString(RaveAcrr_getQuantity(self->acrr));
}

/**
 * Sums a parameter with the previously calculated values.
 * @param[in] self - self
 * @param[in] args - param (cartesian parameter), zr_a (double), zr_b (double)
 * @return None on success otherwise an exception will be thrown
 */
static PyObject* _pyacrr_sum(PyAcrr* self, PyObject* args)
{
  PyObject* pyo = NULL;
  double zr_a = 0.0, zr_b = 0.0;
  if (!PyArg_ParseTuple(args, "Odd", &pyo, &zr_a, &zr_b)) {
    return NULL;
  }

  if (!PyCartesianParam_Check(pyo)) {
    raiseException_returnNULL(PyExc_ValueError, "First parameter must be a cartesian parameter");
  }

  if (!RaveAcrr_sum(self->acrr, ((PyCartesianParam*)pyo)->param, zr_a, zr_b)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to process parameter");
  }

  Py_RETURN_NONE;
}

/**
 * Generates the result
 * @param[in] self - self
 * @return the cartesian parameter with quantity ACRR and the associated quality field on success otherwise NULL
 */
static PyObject* _pyacrr_accumulate(PyAcrr* self, PyObject* args)
{
  double accept = 0.0;
  long N = 0;
  double hours = 0.0;
  CartesianParam_t* param = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "dld", &accept, &N, &hours)) {
    return NULL;
  }

  param = RaveAcrr_accumulate(self->acrr, accept, N, hours);
  if (param != NULL) {
    result = PyCartesianParam_New(param);
  } else {
    raiseException_returnNULL(PyExc_IOError, "Failure when accumulating result");
  }

  RAVE_OBJECT_RELEASE(param);
  return result;
}

/**
 * All methods a acrr can have
 */
static struct PyMethodDef _pyacrr_methods[] =
{
  {"nodata", NULL},
  {"undetect", NULL},
  {"quality_field_name", NULL},
  {"isInitialized", (PyCFunction) _pyacrr_isInitialized, 1},
  {"getQuantity", (PyCFunction) _pyacrr_getQuantity, 1},
  {"sum", (PyCFunction) _pyacrr_sum, 1},
  {"accumulate", (PyCFunction) _pyacrr_accumulate, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the acrr
 * @param[in] self - the acrr
 */
static PyObject* _pyacrr_getattr(PyAcrr* self, char* name)
{
  PyObject* res = NULL;

  if (strcmp("nodata", name) == 0) {
    return PyFloat_FromDouble(RaveAcrr_getNodata(self->acrr));
  } else if (strcmp("undetect", name) == 0) {
    return PyFloat_FromDouble(RaveAcrr_getUndetect(self->acrr));
  } else if (strcmp("quality_field_name", name) == 0) {
    return PyString_FromString(RaveAcrr_getQualityFieldName(self->acrr));
  }

  res = Py_FindMethod(_pyacrr_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the acrr
 */
static int _pyacrr_setattr(PyAcrr* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("nodata", name) == 0) {
    if (PyInt_Check(val)) {
      RaveAcrr_setNodata(self->acrr, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveAcrr_setNodata(self->acrr, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveAcrr_setNodata(self->acrr, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "nodata must be a number");
    }
  } else if (strcmp("undetect", name) == 0) {
    if (PyInt_Check(val)) {
      RaveAcrr_setUndetect(self->acrr, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveAcrr_setUndetect(self->acrr, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveAcrr_setUndetect(self->acrr, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "undetect must be a number");
    }
  } else if (strcmp("quality_field_name", name) == 0) {
    if (PyString_Check(val)) {
      if (!RaveAcrr_setQualityFieldName(self->acrr, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_MemoryError, "failure to set quality field name");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "quality_field_name must be a string");
    }
  }

  result = 0;
done:
  return result;
}

/*@} End of Acrr */

/*@{ Type definitions */
PyTypeObject PyAcrr_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "AcrrCore", /*tp_name*/
  sizeof(PyAcrr), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyacrr_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pyacrr_getattr, /*tp_getattr*/
  (setattrfunc)_pyacrr_setattr, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0,
  0, /*tp_as_mapping */
  0 /*tp_hash*/
};
/*@} End of Type definitions */

/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pyacrr_new, 1},
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_acrr(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyAcrr_API[PyAcrr_API_pointers];
  PyObject *c_api_object = NULL;
  PyAcrr_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_acrr", functions);
  if (module == NULL) {
    return;
  }
  PyAcrr_API[PyAcrr_Type_NUM] = (void*)&PyAcrr_Type;
  PyAcrr_API[PyAcrr_GetNative_NUM] = (void *)PyAcrr_GetNative;
  PyAcrr_API[PyAcrr_New_NUM] = (void*)PyAcrr_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyAcrr_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_acrr.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _acrr.error");
  }

  import_pycartesianparam();
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
