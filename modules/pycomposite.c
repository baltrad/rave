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
 * Python version of the Composite API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-01-29
 */
#include "Python.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYCOMPOSITE_MODULE        /**< to get correct part of pycomposite.h */
#include "pycomposite.h"

#include "pypolarvolume.h"
#include "pycartesian.h"
#include "pyarea.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_pycomposite");

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

/*@{ Composite generator */
/**
 * Returns the native Cartesian_t instance.
 * @param[in] pycartesian - the python cartesian instance
 * @returns the native cartesian instance.
 */
static Composite_t*
PyComposite_GetNative(PyComposite* pycomposite)
{
  RAVE_ASSERT((pycomposite != NULL), "pycomposite == NULL");
  return RAVE_OBJECT_COPY(pycomposite->composite);
}

/**
 * Creates a python composite from a native composite or will create an
 * initial native Composite if p is NULL.
 * @param[in] p - the native composite (or NULL)
 * @returns the python composite product generator.
 */
static PyComposite*
PyComposite_New(Composite_t* p)
{
  PyComposite* result = NULL;
  Composite_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&Composite_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for composite.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for composite.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyComposite, &PyComposite_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->composite = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->composite, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyComposite instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for composite.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the cartesian product
 * @param[in] obj the object to deallocate.
 */
static void _pycomposite_dealloc(PyComposite* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->composite, obj);
  RAVE_OBJECT_RELEASE(obj->composite);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the composite.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pycomposite_new(PyObject* self, PyObject* args)
{
  PyComposite* result = PyComposite_New(NULL);
  return (PyObject*)result;
}

/**
 * Adds a transformable rave object to the composite generator. Currently,
 * only volumes are supported.
 * @param[in] self - self
 * @param[in] args - a rave object (currently only polar volumes)
 * @returns None on success, otherwise NULL
 */
static PyObject* _pycomposite_add(PyComposite* self, PyObject* args)
{
  PyObject* obj = NULL;

  if(!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }

  if (!PyPolarVolume_Check(obj)) {
    raiseException_returnNULL(PyExc_AttributeError, "only supported objects are volumes");
  }

  if (!Composite_add(self->composite, (RaveCoreObject*)((PyPolarVolume*)obj)->pvol)) {
    raiseException_returnNULL(PyExc_MemoryError, "failed to add volume to composite generator");
  }

  Py_RETURN_NONE;
}

/**
 * Generates a composite according to nearest principle.
 * @param[in] self - self
 * @param[in] args - an area object followed by a height
 * @returns a cartesian product on success, otherwise NULL
 */
static PyObject* _pycomposite_nearest(PyComposite* self, PyObject* args)
{
  PyObject* obj = NULL;
  double height = 0.0L;
  Cartesian_t* result = NULL;
  PyObject* pyresult = NULL;

  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }
  if (!PyArea_Check(obj)) {
    raiseException_returnNULL(PyExc_AttributeError, "argument should be an area");
  }

  result = Composite_nearest(self->composite, ((PyArea*)obj)->area);
  if (result == NULL) {
    raiseException_returnNULL(PyExc_AttributeError, "failed to generate composite");
  }

  pyresult = (PyObject*)PyCartesian_New(result);
  RAVE_OBJECT_RELEASE(result);
  return pyresult;
}

/**
 * All methods a cartesian product can have
 */
static struct PyMethodDef _pycomposite_methods[] =
{
  {"height", NULL},
  {"product", NULL},
  {"quantity", NULL},
  {"date", NULL},
  {"time", NULL},
  {"add", (PyCFunction) _pycomposite_add, 1},
  {"nearest", (PyCFunction) _pycomposite_nearest, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the cartesian
 * @param[in] self - the cartesian product
 */
static PyObject* _pycomposite_getattr(PyComposite* self, char* name)
{
  PyObject* res = NULL;

  if (strcmp("height", name) == 0) {
    return PyFloat_FromDouble(Composite_getHeight(self->composite));
  } else if (strcmp("product", name) == 0) {
    return PyInt_FromLong(Composite_getProduct(self->composite));
  } else if (strcmp("quantity", name) == 0) {
    if (Composite_getQuantity(self->composite) != NULL) {
      return PyString_FromString(Composite_getQuantity(self->composite));
    } else {
      Py_RETURN_NONE;
    }
  }

  res = Py_FindMethod(_pycomposite_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pycomposite_setattr(PyComposite* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("height", name) == 0) {
    if (PyFloat_Check(val)) {
      Composite_setHeight(self->composite, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"height must be of type float");
    }
  } else if (strcmp("product", name) == 0) {
    if (PyInt_Check(val)) {
      Composite_setProduct(self->composite, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "product must be a valid product type")
    }
  } else if (strcmp("quantity", name) == 0) {
    if (PyString_Check(val)) {
      if (!Composite_setQuantity(self->composite, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "quantity could not be set");
      }
    } else if (val == Py_None) {
      Composite_setQuantity(self->composite, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"quantity must be of type string");
    }
  } else if (strcmp("time", name) == 0) {
    if (PyString_Check(val)) {
      if (!Composite_setTime(self->composite, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "time must be in the format HHmmss");
      }
    } else if (val == Py_None) {
      Composite_setTime(self->composite, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"time must be of type string");
    }
  } else if (strcmp("date", name) == 0) {
    if (PyString_Check(val)) {
      if (!Composite_setDate(self->composite, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "date must be in the format YYYYMMSS");
      }
    } else if (val == Py_None) {
      Composite_setDate(self->composite, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"date must be of type string");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, name);
  }

  result = 0;
done:
  return result;
}

/*@} End of Composite product generator */

/*@{ Type definitions */
PyTypeObject PyComposite_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "CompositeCore", /*tp_name*/
  sizeof(PyComposite), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pycomposite_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pycomposite_getattr, /*tp_getattr*/
  (setattrfunc)_pycomposite_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_pycomposite_new, 1},
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_pycomposite(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyComposite_API[PyComposite_API_pointers];
  PyObject *c_api_object = NULL;
  PyComposite_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_pycomposite", functions);
  if (module == NULL) {
    return;
  }
  PyComposite_API[PyComposite_Type_NUM] = (void*)&PyComposite_Type;
  PyComposite_API[PyComposite_GetNative_NUM] = (void *)PyComposite_GetNative;
  PyComposite_API[PyComposite_New_NUM] = (void*)PyComposite_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyComposite_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_pycomposite.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _pycomposite.error");
  }

  import_pypolarvolume();
  import_pycartesian();
  import_pyarea();
  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
