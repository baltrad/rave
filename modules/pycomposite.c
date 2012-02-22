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
#include "pypolarscan.h"
#include "pycartesian.h"
#include "pyarea.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"
#include "pycompositealgorithm.h"

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
 * Adds a parameter to the composite generator
 * @param[in] self - self
 * @param[in] args - <quantity as a string>, <gain as double>, <offset as double>
 * @return None on success otherwise NULL
 */
static PyObject* _pycomposite_addParameter(PyComposite* self, PyObject* args)
{
  char* quantity = NULL;
  double gain = 1.0, offset = 0.0;

  if (!PyArg_ParseTuple(args, "sdd", &quantity, &gain, &offset)) {
    return NULL;
  }
  if (!Composite_addParameter(self->composite, quantity, gain, offset)) {
    raiseException_returnNULL(PyExc_AttributeError, "Could not add parameter");
  }

  Py_RETURN_NONE;
}

/**
 * Returns if the composite generator will composite specified parameter
 * @param[in] self - self
 * @param[in] args - <quantity as a string>
 * @return True or False
 */
static PyObject* _pycomposite_hasParameter(PyComposite* self, PyObject* args)
{
  char* quantity = NULL;
  if (!PyArg_ParseTuple(args, "s", &quantity)) {
    return NULL;
  }
  return PyBool_FromLong(Composite_hasParameter(self->composite, quantity));
}

/**
 * Returns the number of parameters this generator will process
 * @param[in] self - self
 * @param[in] args - N/A
 * @return The number of parameters
 */
static PyObject* _pycomposite_getParameterCount(PyComposite* self, PyObject* args)
{
  return PyLong_FromLong(Composite_getParameterCount(self->composite));
}

/**
 * Returns the parameter at specified index.
 * @param[in] self - self
 * @param[in] args - <index as int>
 * @return A tuple containing (<quantity as string>,<gain as double>,<offset as double>)
 */
static PyObject* _pycomposite_getParameter(PyComposite* self, PyObject* args)
{
  int i = 0;
  const char* quantity;
  double gain = 1.0, offset = 0.0;

  if (!PyArg_ParseTuple(args, "i", &i)) {
    return NULL;
  }
  quantity = Composite_getParameter(self->composite, i, &gain, &offset);
  if (quantity == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "No parameter at specified index");
  }

  return Py_BuildValue("(sdd)", quantity, gain, offset);
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
  RaveCoreObject* rcobject = NULL;

  if(!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }

  if (PyPolarVolume_Check(obj)) {
    rcobject = (RaveCoreObject*)((PyPolarVolume*)obj)->pvol;
  } else if (PyPolarScan_Check(obj)) {
    rcobject = (RaveCoreObject*)((PyPolarScan*)obj)->scan;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "only supported objects are volumes and scans");
  }

  if (!Composite_add(self->composite, rcobject)) {
    raiseException_returnNULL(PyExc_MemoryError, "failed to add object to composite generator");
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
  Cartesian_t* result = NULL;
  PyObject* pyresult = NULL;
  PyObject* pyqualitynames = NULL;
  RaveList_t* qualitynames = NULL;

  if (!PyArg_ParseTuple(args, "O|O", &obj, &pyqualitynames)) {
    return NULL;
  }
  if (!PyArea_Check(obj)) {
    raiseException_returnNULL(PyExc_AttributeError, "argument should be an area");
  }
  if (pyqualitynames != NULL && !PyList_Check(pyqualitynames)) {
    raiseException_returnNULL(PyExc_AttributeError, "second argument should be a list of quality (how/task) names");
  }
  if (pyqualitynames != NULL && PyObject_Length(pyqualitynames) > 0) {
    Py_ssize_t nnames = PyObject_Length(pyqualitynames);
    Py_ssize_t i = 0;
    qualitynames = RAVE_OBJECT_NEW(&RaveList_TYPE);
    if (qualitynames == NULL) {
      raiseException_gotoTag(done, PyExc_MemoryError, "Could not allocate memory");
    }
    for (i = 0; i < nnames; i++) {
      PyObject* pystr = PyList_GetItem(pyqualitynames, i);
      char* dupstr = NULL;
      if (pystr == NULL || !PyString_Check(pystr)) {
        raiseException_gotoTag(done, PyExc_AttributeError, "second argument should be a list of quality (how/task) names (strings)");
      }
      dupstr = RAVE_STRDUP(PyString_AsString(pystr));
      if (dupstr != NULL) {
        if (!RaveList_add(qualitynames, dupstr)) {
          raiseException_gotoTag(done, PyExc_MemoryError, "Could not allocate memory");
          RAVE_FREE(dupstr);
        }
      } else {
        raiseException_gotoTag(done, PyExc_MemoryError, "Could not allocate memory");
      }
      dupstr = NULL; // We have handed it over to the rave list.
    }
  }

  result = Composite_nearest(self->composite, ((PyArea*)obj)->area, qualitynames);
  if (result == NULL) {
    raiseException_gotoTag(done, PyExc_AttributeError, "failed to generate composite");
  }

  pyresult = (PyObject*)PyCartesian_New(result);
done:
  RAVE_OBJECT_RELEASE(result);
  RaveList_freeAndDestroy(&qualitynames);
  return pyresult;
}

/**
 * All methods a cartesian product can have
 */
static struct PyMethodDef _pycomposite_methods[] =
{
  {"height", NULL},
  {"elangle", NULL},
  {"product", NULL},
  {"selection_method", NULL},
  {"date", NULL},
  {"time", NULL},
  {"addParameter", (PyCFunction)_pycomposite_addParameter, 1},
  {"hasParameter", (PyCFunction)_pycomposite_hasParameter, 1},
  {"getParameterCount", (PyCFunction)_pycomposite_getParameterCount, 1},
  {"getParameter", (PyCFunction)_pycomposite_getParameter, 1},
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
  } else if (strcmp("elangle", name) == 0) {
    return PyFloat_FromDouble(Composite_getElevationAngle(self->composite));
  } else if (strcmp("product", name) == 0) {
    return PyInt_FromLong(Composite_getProduct(self->composite));
  } else if (strcmp("selection_method", name) == 0) {
    return PyInt_FromLong(Composite_getSelectionMethod(self->composite));
  } else if (strcmp("algorithm", name) == 0) {
    CompositeAlgorithm_t* algorithm = Composite_getAlgorithm(self->composite);
    if (algorithm != NULL) {
      res = (PyObject*)PyCompositeAlgorithm_New(algorithm);
      RAVE_OBJECT_RELEASE(algorithm);
      return res;
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
    } else if (PyLong_Check(val)) {
      Composite_setHeight(self->composite, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      Composite_setHeight(self->composite, (double)PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"height must be of type float");
    }
  } else if (strcmp("elangle", name) == 0) {
    if (PyFloat_Check(val)) {
      Composite_setElevationAngle(self->composite, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      Composite_setElevationAngle(self->composite, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      Composite_setElevationAngle(self->composite, (double)PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "elangle must be a float or decimal value")
    }
  } else if (strcmp("product", name) == 0) {
    if (PyInt_Check(val)) {
      Composite_setProduct(self->composite, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "product must be a valid product type")
    }
  } else if (strcmp("selection_method", name) == 0) {
    if (!PyInt_Check(val) || !Composite_setSelectionMethod(self->composite, PyInt_AsLong(val))) {
      raiseException_gotoTag(done, PyExc_ValueError, "not a valid selection method");
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
  } else if (strcmp("algorithm", name) == 0) {
    if (val == Py_None) {
      Composite_setAlgorithm(self->composite, NULL);
    } else if (PyCompositeAlgorithm_Check(val)) {
      Composite_setAlgorithm(self->composite, ((PyCompositeAlgorithm*)val)->algorithm);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "algorithm must either be None or a CompositeAlgorithm");
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

  add_long_constant(dictionary, "SelectionMethod_NEAREST", CompositeSelectionMethod_NEAREST);
  add_long_constant(dictionary, "SelectionMethod_HEIGHT", CompositeSelectionMethod_HEIGHT);

  import_pypolarvolume();
  import_pypolarscan();
  import_pycartesian();
  import_pyarea();
  import_array(); /*To make sure I get access to Numeric*/
  import_compositealgorithm();
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
