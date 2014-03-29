/* --------------------------------------------------------------------
Copyright (C) 2014 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the Gra API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2014-03-28
 */
#include "Python.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYGRA_MODULE    /**< to get correct part in pygra.h */
#include "pygra.h"
#include "pyravefield.h"
#include "pycartesianparam.h"
#include "rave_alloc.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_gra");

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

/*@{ Gra */
/**
 * Returns the native GraAcrr_t instance.
 * @param[in] pygra - the python gra instance
 * @returns the native gra instance.
 */
static RaveGra_t*
PyGra_GetNative(PyGra* pygra)
{
  RAVE_ASSERT((pygra != NULL), "pygra == NULL");
  return RAVE_OBJECT_COPY(pygra->gra);
}

/**
 * Creates a python gra from a native gra or will create an
 * initial native gra if p is NULL.
 * @param[in] p - the native gra (or NULL)
 * @returns the python gra product.
 */
static PyGra*
PyGra_New(RaveGra_t* p)
{
  PyGra* result = NULL;
  RaveGra_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RaveGra_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for gra.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for gra.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyGra, &PyGra_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->gra = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->gra, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyGra instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyGra.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the gra
 * @param[in] obj the object to deallocate.
 */
static void _pygra_dealloc(PyGra* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->gra, obj);
  RAVE_OBJECT_RELEASE(obj->gra);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the gra.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pygra_new(PyObject* self, PyObject* args)
{
  PyGra* result = PyGra_New(NULL);
  return (PyObject*)result;
}

/**
 * Generates the result
 * @param[in] self - self
 * @param[in] args - a rave field containing distances, a cartesian parameter containing the data
 * @return the cartesian parameter with the gra coefficients applied
 */
static PyObject* _pygra_apply(PyGra* self, PyObject* args)
{
  PyObject* pyfield = NULL;
  PyObject* pyparameter = NULL;
  CartesianParam_t* graparam = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyfield, &pyparameter)) {
    return NULL;
  }

  if (!PyRaveField_Check(pyfield) || !PyCartesianParam_Check(pyparameter)) {
    raiseException_returnNULL(PyExc_AttributeError, "Must provide apply with <rave field with distance>, <cartesian parameter with data>");
  }

  graparam = RaveGra_apply(self->gra, ((PyRaveField*)pyfield)->field, ((PyCartesianParam*)pyparameter)->param);

  if (graparam != NULL) {
    result = (PyObject*)PyCartesianParam_New(graparam);
  } else {
    raiseException_returnNULL(PyExc_IOError, "Failure when applying gra coefficients");
  }

  RAVE_OBJECT_RELEASE(graparam);
  return result;
}

/**
 * All methods a acrr can have
 */
static struct PyMethodDef _pygra_methods[] =
{
  {"A", NULL},
  {"B", NULL},
  {"C", NULL},
  {"upperThreshold", NULL},
  {"lowerThreshold", NULL},
  {"zrA", NULL},
  {"zrb", NULL},
  {"apply", (PyCFunction) _pygra_apply, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the gra
 * @param[in] self - the gra
 */
static PyObject* _pygra_getattr(PyGra* self, char* name)
{
  PyObject* res = NULL;

  if (strcmp("A", name) == 0) {
    return PyFloat_FromDouble(RaveGra_getA(self->gra));
  } else if (strcmp("B", name) == 0) {
    return PyFloat_FromDouble(RaveGra_getB(self->gra));
  } else if (strcmp("C", name) == 0) {
    return PyFloat_FromDouble(RaveGra_getC(self->gra));
  } else if (strcmp("upperThreshold", name) == 0) {
    return PyFloat_FromDouble(RaveGra_getUpperThreshold(self->gra));
  } else if (strcmp("lowerThreshold", name) == 0) {
    return PyFloat_FromDouble(RaveGra_getLowerThreshold(self->gra));
  } else if (strcmp("zrA", name) == 0) {
    return PyFloat_FromDouble(RaveGra_getZRA(self->gra));
  } else if (strcmp("zrb", name) == 0) {
    return PyFloat_FromDouble(RaveGra_getZRB(self->gra));
  }

  res = Py_FindMethod(_pygra_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the gra
 */
static int _pygra_setattr(PyGra* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("A", name) == 0) {
    if (PyInt_Check(val)) {
      RaveGra_setA(self->gra, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveGra_setA(self->gra, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveGra_setA(self->gra, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "A must be a number");
    }
  } else if (strcmp("B", name) == 0) {
    if (PyInt_Check(val)) {
      RaveGra_setB(self->gra, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveGra_setB(self->gra, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveGra_setB(self->gra, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "B must be a number");
    }
  } else if (strcmp("C", name) == 0) {
    if (PyInt_Check(val)) {
      RaveGra_setC(self->gra, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveGra_setC(self->gra, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveGra_setC(self->gra, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "C must be a number");
    }
  } else if (strcmp("upperThreshold", name) == 0) {
    if (PyInt_Check(val)) {
      RaveGra_setUpperThreshold(self->gra, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveGra_setUpperThreshold(self->gra, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveGra_setUpperThreshold(self->gra, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "upperThreshold must be a number");
    }
  } else if (strcmp("lowerThreshold", name) == 0) {
    if (PyInt_Check(val)) {
      RaveGra_setLowerThreshold(self->gra, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveGra_setLowerThreshold(self->gra, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveGra_setLowerThreshold(self->gra, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "lowerThreshold must be a number");
    }
  } else if (strcmp("zrA", name) == 0) {
    if (PyInt_Check(val)) {
      RaveGra_setZRA(self->gra, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveGra_setZRA(self->gra, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveGra_setZRA(self->gra, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "zrA must be a number");
    }
  } else if (strcmp("zrb", name) == 0) {
    if (PyInt_Check(val)) {
      RaveGra_setZRB(self->gra, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveGra_setZRB(self->gra, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveGra_setZRB(self->gra, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "zrb must be a number");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, "Unknown attribute");
  }

  result = 0;
done:
  return result;
}

/*@} End of Gra */

/*@{ Type definitions */
PyTypeObject PyGra_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "GraCore", /*tp_name*/
  sizeof(PyGra), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pygra_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pygra_getattr, /*tp_getattr*/
  (setattrfunc)_pygra_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_pygra_new, 1},
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_gra(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyGra_API[PyGra_API_pointers];
  PyObject *c_api_object = NULL;
  PyGra_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_gra", functions);
  if (module == NULL) {
    return;
  }
  PyGra_API[PyGra_Type_NUM] = (void*)&PyGra_Type;
  PyGra_API[PyGra_GetNative_NUM] = (void *)PyGra_GetNative;
  PyGra_API[PyGra_New_NUM] = (void*)PyGra_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyGra_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_gra.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _gra.error");
  }

  import_pyravefield();
  import_pycartesianparam();
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
