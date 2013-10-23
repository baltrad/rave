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
 * Python version of the Cartesian Composite generator API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2013-10-09
 */
#include "Python.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYCARTESIANCOMPOSITE_MODULE        /**< to get correct part of pycartesiancomposite.h */
#include "pycartesiancomposite.h"

#include "pyarea.h"
#include "pycartesian.h"
#include "pycartesianparam.h"
#include <arrayobject.h>
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_cartesiancomposite");

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

/*@{ Cartesian composite generator */
/**
 * Returns the native CartesianComposite_t instance.
 * @param[in] pygenerator - the python cartesian generator instance
 * @returns the native generator instance.
 */
static CartesianComposite_t*
PyCartesianComposite_GetNative(PyCartesianComposite* pygenerator)
{
  RAVE_ASSERT((pygenerator != NULL), "pygenerator == NULL");
  return RAVE_OBJECT_COPY(pygenerator->generator);
}

/**
 * Creates a python cartesian composite generator from a native one or will create an
 * initial native Cartesian composite generator if p is NULL.
 * @param[in] p - the native cartesian composite generator (or NULL)
 * @returns the python cartesian composite generator.
 */
static PyCartesianComposite*
PyCartesianComposite_New(CartesianComposite_t* p)
{
  PyCartesianComposite* result = NULL;
  CartesianComposite_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&CartesianComposite_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for cartesian composite generator.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for cartesian composite generator.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyCartesianComposite, &PyCartesianComposite_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->generator = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->generator, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyCartesianComposite instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for cartesian composite generator.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the cartesian composite generator
 * @param[in] obj the object to deallocate.
 */
static void _pycartesiancomposite_dealloc(PyCartesianComposite* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->generator, obj);
  RAVE_OBJECT_RELEASE(obj->generator);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the cartesian composite generator.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pycartesiancomposite_new(PyObject* self, PyObject* args)
{
  PyCartesianComposite* result = PyCartesianComposite_New(NULL);
  return (PyObject*)result;
}

/**
 * Adds a cartesian object to the composite generator.
 * @param[in] self - self
 * @param[in] args - a cartesian object
 * @returns None on success, otherwise NULL
 */
static PyObject* _pycartesiancomposite_add(PyCartesianComposite* self, PyObject* args)
{
  PyObject* obj = NULL;
  Cartesian_t* ct = NULL;

  if(!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }

  if (PyCartesian_Check(obj)) {
    ct = ((PyCartesian*)obj)->cartesian;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "only supported objects are cartesian products");
  }

  if (!CartesianComposite_add(self->generator, ct)) {
    raiseException_returnNULL(PyExc_MemoryError, "failed to add object to composite generator");
  }

  Py_RETURN_NONE;
}

/**
 * Returns the number of cartesian products that has been added to the composite generator
 * @param[in] self - self
 * @param[in] args - N/A
 * @returns Number of objects
 */
static PyObject* _pycartesiancomposite_getNumberOfObjects(PyCartesianComposite* self, PyObject* args)
{
  return PyInt_FromLong(CartesianComposite_getNumberOfObjects(self->generator));
}

/**
 * Returns the cartesian object at specified position
 * @param[in] self - self
 * @param[in] args - an index as integer
 * @returns the object at specified index
 * @throws IndexError if the object not could be found
 */
static PyObject* _pycartesiancomposite_get(PyCartesianComposite* self, PyObject* args)
{
  int idx = 0;
  Cartesian_t* cobj = NULL;
  PyObject* pyresult = NULL;

  if (!PyArg_ParseTuple(args, "i", &idx)) {
    return NULL;
  }
  cobj = CartesianComposite_get(self->generator, idx);
  if (cobj == NULL) {
    raiseException_gotoTag(done, PyExc_IndexError, "no cartesian object at index");
  }
  pyresult = (PyObject*)PyCartesian_New(cobj);

done:
  RAVE_OBJECT_RELEASE(cobj);
  return pyresult;
}

static PyObject* _pycartesiancomposite_nearest(PyCartesianComposite* self, PyObject* args)
{
  PyObject* pyarea = NULL;
  Cartesian_t* result = NULL;
  PyObject* pyresult = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyarea)) {
    return NULL;
  }
  if (!PyArea_Check(pyarea)) {
    raiseException_returnNULL(PyExc_AttributeError, "argument should be an area");
  }

  result = CartesianComposite_nearest(self->generator, ((PyArea*)pyarea)->area);
  if (result == NULL) {
    raiseException_gotoTag(done, PyExc_AttributeError, "failed to generate composite");
  }

  pyresult = (PyObject*)PyCartesian_New(result);
done:
  RAVE_OBJECT_RELEASE(result);
  return pyresult;
}

/**
 * All methods a cartesian product can have
 */
static struct PyMethodDef _pycartesiancomposite_methods[] =
{
  {"date", NULL},
  {"time", NULL},
  {"quantity", NULL},
  {"offset", NULL},
  {"gain", NULL},
  {"nodata", NULL},
  {"undetect", NULL},
  {"method", NULL},
  {"distance_field", NULL},
  {"add", (PyCFunction)_pycartesiancomposite_add, 1},
  {"getNumberOfObjects", (PyCFunction)_pycartesiancomposite_getNumberOfObjects, 1},
  {"get", (PyCFunction)_pycartesiancomposite_get, 1},
  {"nearest", (PyCFunction)_pycartesiancomposite_nearest, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the cartesian
 * @param[in] self - the cartesian product
 */
static PyObject* _pycartesiancomposite_getattr(PyCartesianComposite* self, char* name)
{
  PyObject* res = NULL;
  if (strcmp("time", name) == 0) {
    if (CartesianComposite_getTime(self->generator) != NULL) {
      return PyString_FromString(CartesianComposite_getTime(self->generator));
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("date", name) == 0) {
    if (CartesianComposite_getDate(self->generator) != NULL) {
      return PyString_FromString(CartesianComposite_getDate(self->generator));
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("quantity", name) == 0) {
    if (CartesianComposite_getQuantity(self->generator) != NULL) {
      return PyString_FromString(CartesianComposite_getQuantity(self->generator));
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("offset", name) == 0) {
    return PyFloat_FromDouble(CartesianComposite_getOffset(self->generator));
  } else if (strcmp("gain", name) == 0) {
    return PyFloat_FromDouble(CartesianComposite_getGain(self->generator));
  } else if (strcmp("nodata", name) == 0) {
    return PyFloat_FromDouble(CartesianComposite_getNodata(self->generator));
  } else if (strcmp("undetect", name) == 0) {
    return PyFloat_FromDouble(CartesianComposite_getUndetect(self->generator));
  } else if (strcmp("method", name) == 0) {
    return PyInt_FromLong(CartesianComposite_getMethod(self->generator));
  } else if (strcmp("distance_field", name) == 0) {
    if (CartesianComposite_getDistanceField(self->generator) != NULL) {
      return PyString_FromString(CartesianComposite_getDistanceField(self->generator));
    } else {
      Py_RETURN_NONE;
    }
  }

  res = Py_FindMethod(_pycartesiancomposite_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pycartesiancomposite_setattr(PyCartesianComposite* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("time", name) == 0) {
    if (PyString_Check(val)) {
      if (!CartesianComposite_setTime(self->generator, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "time must be in the format HHmmss");
      }
    } else if (val == Py_None) {
      CartesianComposite_setTime(self->generator, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"time must be of type string");
    }
  } else if (strcmp("date", name) == 0) {
    if (PyString_Check(val)) {
      if (!CartesianComposite_setDate(self->generator, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "date must be in the format YYYYMMSS");
      }
    } else if (val == Py_None) {
      CartesianComposite_setDate(self->generator, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"date must be of type string");
    }
  } else if (strcmp("quantity", name) == 0) {
    if (PyString_Check(val)) {
      if (!CartesianComposite_setQuantity(self->generator,
                                          PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError,
                               "Could not set quantity");
      }
    } else if (val == Py_None) {
      if (!CartesianComposite_setQuantity(self->generator, NULL)) {
        raiseException_gotoTag(done, PyExc_ValueError,
                               "Could not set quantity to nothing");
      }
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,
                             "quantity must be of type string");
    }
  } else if (strcmp("offset", name) == 0) {
    if (PyFloat_Check(val)) {
      CartesianComposite_setOffset(self->generator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
                             "offset must be of type float");
    }
  } else if (strcmp("gain", name) == 0) {
    if (PyFloat_Check(val)) {
      CartesianComposite_setGain(self->generator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
                             "gain must be of type float");
    }
  } else if (strcmp("nodata", name) == 0) {
    if (PyFloat_Check(val)) {
      CartesianComposite_setNodata(self->generator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
                             "nodata must be of type float");
    }
  } else if (strcmp("undetect", name) == 0) {
    if (PyFloat_Check(val)) {
      CartesianComposite_setUndetect(self->generator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
                             "undetect must be of type float");
    }
  } else if (strcmp("method", name) == 0) {
    if (!PyInt_Check(val)
        || !CartesianComposite_setMethod(self->generator, PyInt_AsLong(val))) {
      raiseException_gotoTag(done, PyExc_ValueError,
                             "not a valid selection method");
    }
  } else if (strcmp("distance_field", name) == 0) {
    if (PyString_Check(val)) {
      if (!CartesianComposite_setDistanceField(self->generator,
                                               PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError,
                               "Could not set distance_field");
      }
    } else if (val == Py_None) {
      if (!CartesianComposite_setDistanceField(self->generator, NULL)) {
        raiseException_gotoTag(done, PyExc_ValueError,
                               "Could not set distance_field to nothing");
      }
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,
                             "distance_field must be of type string");
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
PyTypeObject PyCartesianComposite_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "CartesianCompositeCore", /*tp_name*/
  sizeof(PyCartesianComposite), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pycartesiancomposite_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pycartesiancomposite_getattr, /*tp_getattr*/
  (setattrfunc)_pycartesiancomposite_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_pycartesiancomposite_new, 1},
  {NULL,NULL} /*Sentinel*/
};

/**
 * Adds constants to the dictionary (probably the modules dictionary).
 * @param[in] dictionary - the dictionary the long should be added to
 * @param[in] name - the name of the constant
 * @param[in] value - the value
 */
static void add_long_constant(PyObject* dictionary, const char* name, long value)
{
  PyObject* tmp = NULL;
  tmp = PyInt_FromLong(value);
  if (tmp != NULL) {
    PyDict_SetItemString(dictionary, name, tmp);
  }
  Py_XDECREF(tmp);
}

PyMODINIT_FUNC
init_cartesiancomposite(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyCartesianComposite_API[PyCartesianComposite_API_pointers];
  PyObject *c_api_object = NULL;
  PyCartesianComposite_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_cartesiancomposite", functions);
  if (module == NULL) {
    return;
  }
  PyCartesianComposite_API[PyCartesianComposite_Type_NUM] = (void*)&PyCartesianComposite_Type;
  PyCartesianComposite_API[PyCartesianComposite_GetNative_NUM] = (void *)PyCartesianComposite_GetNative;
  PyCartesianComposite_API[PyCartesianComposite_New_NUM] = (void*)PyCartesianComposite_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyCartesianComposite_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_cartesiancomposite.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _cartesiancomposite.error");
  }

  add_long_constant(dictionary, "SelectionMethod_FIRST", CartesianCompositeSelectionMethod_FIRST);
  add_long_constant(dictionary, "SelectionMethod_MINVALUE", CartesianCompositeSelectionMethod_MINVALUE);
  add_long_constant(dictionary, "SelectionMethod_MAXVALUE", CartesianCompositeSelectionMethod_MAXVALUE);
  add_long_constant(dictionary, "SelectionMethod_AVGVALUE", CartesianCompositeSelectionMethod_AVGVALUE);
  add_long_constant(dictionary, "SelectionMethod_DISTANCE", CartesianCompositeSelectionMethod_DISTANCE);

  import_pycartesian();
  import_pycartesianparam();
  import_pyarea();
  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
