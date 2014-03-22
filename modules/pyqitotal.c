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
 * Python version of the QI total API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2014-02-27
 */
#include "Python.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"
#include "pyravefield.h"

#define PYQITOTAL_MODULE    /**< to get correct part in pyqitotal.h */
#include "pyqitotal.h"
#include "rave_alloc.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_qitotal");

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

/**@{ QI total */
/**
 * Returns the native RaveQITotal_t instance.
 * @param[in] qitotal - the python QI total instance
 * @returns the native qi total instance.
 */
static RaveQITotal_t*
PyQITotal_GetNative(PyQITotal* qitotal)
{
  RAVE_ASSERT((qitotal != NULL), "qitotal == NULL");
  return RAVE_OBJECT_COPY(qitotal->qitotal);
}

/**
 * Creates a python qi total from a native qi total or will create an
 * initial native qi total instance if p is NULL.
 * @param[in] p - the native qi total instance (or NULL)
 * @returns the python qi total product.
 */
static PyQITotal*
PyQITotal_New(RaveQITotal_t* p)
{
  PyQITotal* result = NULL;
  RaveQITotal_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RaveQITotal_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for qi total instance.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for qitotal.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyQITotal, &PyQITotal_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->qitotal = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->qitotal, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyQITotal instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyQITotal.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the qi total instance
 * @param[in] obj the object to deallocate.
 */
static void _pyqitotal_dealloc(PyQITotal* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->qitotal, obj);
  RAVE_OBJECT_RELEASE(obj->qitotal);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the qi total instance.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyqitotal_new(PyObject* self, PyObject* args)
{
  PyQITotal* result = PyQITotal_New(NULL);
  return (PyObject*)result;
}

/**
 * Since multiplicative, additive and minimum requires same boilercode with the exception of th emethod called in rave_qitotal so
 * this can cope with all of them.
 */
static PyObject* _pyqitotal_operator_func(PyQITotal* self, PyObject* args, RaveField_t*(*opfunc)(RaveQITotal_t*, RaveObjectList_t*))
{
  PyObject* pyfields = NULL;
  PyObject* pyresult = NULL;
  RaveField_t* result = NULL;
  Py_ssize_t n = 0, i = 0;
  RaveObjectList_t* fields = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyfields)) {
    return NULL;
  }
  if (!PySequence_Check(pyfields)) {
    raiseException_returnNULL(PyExc_AttributeError, "Input should be a list of rave fields");
  }
  fields = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (!fields) {
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory");
  }
  n = PySequence_Size(pyfields);
  for (i = 0; i < n; i++) {
    PyObject* v = PySequence_GetItem(pyfields, i);
    if (!PyRaveField_Check(v)) {
      Py_XDECREF(v);
      raiseException_gotoTag(done, PyExc_AttributeError, "Input should be a list of rave fields");
    }
    if (!RaveObjectList_add(fields, (RaveCoreObject*)((PyRaveField*)v)->field)) {
      Py_XDECREF(v);
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to add item to list");
    }
    Py_XDECREF(v);
  }

  result = opfunc(self->qitotal, fields);
  if (!result) {
    raiseException_gotoTag(done, PyExc_AttributeError, "Failed to generate qi total");
  }

  pyresult = (PyObject*)PyRaveField_New(result);

done:
  RAVE_OBJECT_RELEASE(fields);
  RAVE_OBJECT_RELEASE(result);
  return pyresult;
}

/**
 * Sets the weight for the specified how/task
 * @param[in] self - self
 * @param[in] args - the arguments
 * @return Py_None on success
 * @throws RuntimeError on error
 */
static PyObject* _pyqitotal_setWeight(PyQITotal* self, PyObject* args)
{
  char* howtask = NULL;
  double w = 0.0;

  if (!PyArg_ParseTuple(args, "sd", &howtask, &w)) {
    return NULL;
  }

  if (!RaveQITotal_setWeight(self->qitotal, howtask, w)) {
    raiseException_returnNULL(PyExc_RuntimeError, "Failed to set weight for how/task");
  }

  Py_RETURN_NONE;
}

/**
 * Returns the weight for specified how/task.
 * @param[in] self - self
 * @param[in] args - the how/task name
 * @return the weight if found
 * @throws AttributeError if attribute not could be found
 */
static PyObject* _pyqitotal_getWeight(PyQITotal* self, PyObject* args)
{
  char* howtask = NULL;
  double w = 0.0;

  if (!PyArg_ParseTuple(args, "s", &howtask)) {
    return NULL;
  }

  if (!RaveQITotal_getWeight(self->qitotal, howtask, &w)) {
    raiseException_returnNULL(PyExc_AttributeError, "Failed to get weight for how/task");
  }

  return PyFloat_FromDouble(w);
}

/**
 * Removes the weight for the specified how/task
 * @param[in] self - self
 * @param[in] args - the how/task name
 * @return None
 */
static PyObject* _pyqitotal_removeWeight(PyQITotal* self, PyObject* args)
{
  char* howtask = NULL;

  if (!PyArg_ParseTuple(args, "s", &howtask)) {
    return NULL;
  }
  RaveQITotal_removeWeight(self->qitotal, howtask);

  Py_RETURN_NONE;
}

/**
 * Generates the multiplicative QI total
 * @param[in] self - self
 * @return the resulting quality field
 */
static PyObject* _pyqitotal_multiplicative(PyQITotal* self, PyObject* args)
{
  return _pyqitotal_operator_func(self, args, RaveQITotal_multiplicative);
}

/**
 * Generates the additive QI total
 * @param[in] self - self
 * @return the resulting quality field
 */
static PyObject* _pyqitotal_additive(PyQITotal* self, PyObject* args)
{
  return _pyqitotal_operator_func(self, args, RaveQITotal_additive);
}

/**
 * Generates the minimum QI total
 * @param[in] self - self
 * @return the resulting quality field
 */
static PyObject* _pyqitotal_minimum(PyQITotal* self, PyObject* args)
{
  return _pyqitotal_operator_func(self, args, RaveQITotal_minimum);
}

/**
 * All methods the qi total generator can have
 */
static struct PyMethodDef _pyqitotal_methods[] =
{
  {"gain", NULL},
  {"offset", NULL},
  {"datatype", NULL},
  {"setWeight", (PyCFunction) _pyqitotal_setWeight, 1},
  {"getWeight", (PyCFunction) _pyqitotal_getWeight, 1},
  {"removeWeight", (PyCFunction) _pyqitotal_removeWeight, 1},
  {"multiplicative", (PyCFunction) _pyqitotal_multiplicative, 1},
  {"additive", (PyCFunction) _pyqitotal_additive, 1},
  {"minimum", (PyCFunction) _pyqitotal_minimum, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the qi total generator
 * @param[in] self - the acrr
 */
static PyObject* _pyqitotal_getattr(PyQITotal* self, char* name)
{
  PyObject* res = NULL;

  if (strcmp("gain", name) == 0) {
    return PyFloat_FromDouble(RaveQITotal_getGain(self->qitotal));
  } else if (strcmp("offset", name) == 0) {
    return PyFloat_FromDouble(RaveQITotal_getOffset(self->qitotal));
  } else if (strcmp("datatype", name) == 0) {
    return PyInt_FromLong(RaveQITotal_getDatatype(self->qitotal));
  }

  res = Py_FindMethod(_pyqitotal_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the qi total generator
 */
static int _pyqitotal_setattr(PyQITotal* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("gain", name) == 0) {
    if (PyInt_Check(val)) {
      if (!RaveQITotal_setGain(self->qitotal, (double)PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_AttributeError, "gain must not be 0.0");
      }
    } else if (PyLong_Check(val)) {
      if (!RaveQITotal_setGain(self->qitotal, PyLong_AsDouble(val))) {
        raiseException_gotoTag(done, PyExc_AttributeError, "gain must not be 0.0");
      }
    } else if (PyFloat_Check(val)) {
      if (!RaveQITotal_setGain(self->qitotal, PyFloat_AsDouble(val))) {
        raiseException_gotoTag(done, PyExc_AttributeError, "gain must not be 0.0");
      }
    } else {
      raiseException_gotoTag(done, PyExc_AttributeError, "gain must be a number");
    }
  } else if (strcmp("offset", name) == 0) {
    if (PyInt_Check(val)) {
      RaveQITotal_setOffset(self->qitotal, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveQITotal_setOffset(self->qitotal, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveQITotal_setOffset(self->qitotal, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_AttributeError, "offset must be a number");
    }
  } else if (strcmp("datatype", name) == 0) {
    if (PyInt_Check(val)) {
      RaveQITotal_setDatatype(self->qitotal, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_AttributeError, "datatype must be a valid rave data type");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, "Unknown attribute");
  }

  result = 0;
done:
  return result;
}

/*@} End of QI total */

/*@{ Type definitions */
PyTypeObject PyQITotal_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "QITotalCore", /*tp_name*/
  sizeof(PyQITotal), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyqitotal_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pyqitotal_getattr, /*tp_getattr*/
  (setattrfunc)_pyqitotal_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_pyqitotal_new, 1},
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_qitotal(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyQITotal_API[PyQITotal_API_pointers];
  PyObject *c_api_object = NULL;
  PyQITotal_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_qitotal", functions);
  if (module == NULL) {
    return;
  }
  PyQITotal_API[PyQITotal_Type_NUM] = (void*)&PyQITotal_Type;
  PyQITotal_API[PyQITotal_GetNative_NUM] = (void *)PyQITotal_GetNative;
  PyQITotal_API[PyQITotal_New_NUM] = (void*)PyQITotal_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyQITotal_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_qitotal.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _qitotal.error");
  }

  import_pyravefield();
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
