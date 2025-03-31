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
 * Python version of the Composite arguments API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024"-12-13
 */
#include "rave_types.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "compositefilter.h"
#include "pyravecompat.h"
#include "pycompositearguments.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYCOMPOSITEFILTER_MODULE        /**< to get correct part of pycompositefilter.h */
#include "pycompositefilter.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_compositefilter");

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
 * Returns the native CartesianFilter_t instance.
 * @param[in] pyargs - the python composite filter instance
 * @returns the native cartesian instance.
 */
static CompositeFilter_t*
PyCompositeFilter_GetNative(PyCompositeFilter* pyfilter)
{
  RAVE_ASSERT((pyfilter != NULL), "pyfilter == NULL");
  return RAVE_OBJECT_COPY(pyfilter->filter);
}

/**
 * Creates a python composite arguments from a native composite arguments or will create an
 * initial native CompositeArguments if p is NULL.
 * @param[in] p - the native composite arguments (or NULL)
 * @returns the python composite product arguments.
 */
static PyCompositeFilter*
PyCompositeFilter_New(CompositeFilter_t* p)
{
  PyCompositeFilter* result = NULL;
  CompositeFilter_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&CompositeFilter_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for composite filter.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for composite filter.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyCompositeFilter, &PyCompositeFilter_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->filter = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->filter, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyCompositeFilter instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for composite filter.");
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
static void _pycompositefilter_dealloc(PyCompositeFilter* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->filter, obj);
  RAVE_OBJECT_RELEASE(obj->filter);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the composite filter.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pycompositefilter_new(PyObject* self, PyObject* args)
{
  PyCompositeFilter* result = PyCompositeFilter_New(NULL);
  return (PyObject*)result;
}

/**
 * Validates the filter against the composite arguments
 * @param[in] self - self
 * @param[in] args - object
 * @return True if matching, otherwise False
 */
static PyObject* _pycompositefilter_match(PyCompositeFilter* self, PyObject* args)
{
  PyObject* pyargs;
  if(!PyArg_ParseTuple(args, "O", &pyargs)) {
    return NULL;
  }
  if (!PyCompositeArguments_Check(pyargs)) {
    raiseException_returnNULL(PyExc_AttributeError, "Must provide a composite arguments object");
  }
  return PyBool_FromLong(CompositeFilter_match(self->filter, ((PyCompositeArguments*)pyargs)->args));
}

/**
 * All methods a cartesian product can have
 */
static struct PyMethodDef _pycompositefilter_methods[] =
{
  {"products", NULL, METH_VARARGS},
  {"quantities", NULL, METH_VARARGS},
  {"interpolation_methods", NULL, METH_VARARGS},
  {"match", (PyCFunction) _pycompositefilter_match, 1,
    "match(args) \n\n"
    "Matches arguments against filter.\n"
    "args  - The composite arguments instance.\n"
  },

  {NULL, NULL } /* sentinel */
}; 

/**
 * Returns the specified attribute in the cartesian
 * @param[in] self - the cartesian product
 */
static PyObject* _pycompositefilter_getattro(PyCompositeFilter* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "products") == 0) {
    PyObject* result = PyList_New(0);
    if (result != NULL) {
      int i = 0, nlen = CompositeFilter_getProductCount(self->filter);
      for (i = 0; i < nlen; i++) {
        const char* prodname = CompositeFilter_getProduct(self->filter, i);
        PyObject* pynamestr = PyString_FromString(prodname);
        if (pynamestr == NULL) {
          Py_XDECREF(result);
          raiseException_returnNULL(PyExc_MemoryError, "failed to create string");
        }
        if (PyList_Append(result, pynamestr) != 0) {
          Py_XDECREF(result);
          Py_XDECREF(pynamestr);
          raiseException_returnNULL(PyExc_MemoryError, "failed to create string");
        }
        Py_DECREF(pynamestr);
      }
    }
    return result;  
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "quantities") == 0) {
    PyObject* result = PyList_New(0);
    if (result != NULL) {
      int i = 0, nlen = CompositeFilter_getQuantityCount(self->filter);
      for (i = 0; i < nlen; i++) {
        const char* quantity = CompositeFilter_getQuantity(self->filter, i);
        PyObject* pynamestr = PyString_FromString(quantity);
        if (pynamestr == NULL) {
          Py_XDECREF(result);
          raiseException_returnNULL(PyExc_MemoryError, "failed to create string");
        }
        if (PyList_Append(result, pynamestr) != 0) {
          Py_XDECREF(result);
          Py_XDECREF(pynamestr);
          raiseException_returnNULL(PyExc_MemoryError, "failed to create string");
        }
        Py_DECREF(pynamestr);
      }
    }
    return result;
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "interpolation_methods") == 0) {
    PyObject* result = PyList_New(0);
    if (result != NULL) {
      int i = 0, nlen = CompositeFilter_getInterpolationMethodCount(self->filter);
      for (i = 0; i < nlen; i++) {
        const char* quantity = CompositeFilter_getInterpolationMethod(self->filter, i);
        PyObject* pynamestr = PyString_FromString(quantity);
        if (pynamestr == NULL) {
          Py_XDECREF(result);
          raiseException_returnNULL(PyExc_MemoryError, "failed to create string");
        }
        if (PyList_Append(result, pynamestr) != 0) {
          Py_XDECREF(result);
          Py_XDECREF(pynamestr);
          raiseException_returnNULL(PyExc_MemoryError, "failed to create string");
        }
        Py_DECREF(pynamestr);
      }
    }
    return result; 
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pycompositefilter_setattro(PyCompositeFilter* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "products") == 0) {
    if (PyList_Check(val)) {
      Py_ssize_t nproducts = PyObject_Length(val);
      int i = 0;
      if (nproducts > 0) {
        char** products = RAVE_MALLOC(sizeof(char*)*nproducts);
        if (products == NULL) {
          raiseException_gotoTag(done, PyExc_MemoryError, "Could not create product array");
        }
        for (i = 0; i < nproducts; i++) {
          PyObject* pystr = PyList_GetItem(val, i);  /* We don't need to release this since it is internal pointer.*/
          products[i] = (char*)PyString_AsString(pystr);
        }
        if (!CompositeFilter_setProductsArray(self->filter, (const char**)products, (int)nproducts)) {
          RAVE_FREE(products);
          raiseException_gotoTag(done, PyExc_MemoryError, "Could not allocate products");
        }
        RAVE_FREE(products);
      } else {
        CompositeFilter_setProductsArray(self->filter, NULL, 0);  
      }
    } else if (val == Py_None) {
      CompositeFilter_setProductsArray(self->filter, NULL, 0);
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "quantities") == 0) {
    if (PyList_Check(val)) {
      Py_ssize_t nquantities = PyObject_Length(val);
      int i = 0;
      if (nquantities > 0) {
        char** quantities = RAVE_MALLOC(sizeof(char*)*nquantities);
        if (quantities == NULL) {
          raiseException_gotoTag(done, PyExc_MemoryError, "Could not create quantities array");
        }
        for (i = 0; i < nquantities; i++) {
          PyObject* pystr = PyList_GetItem(val, i);  /* We don't need to release this since it is internal pointer.*/
          quantities[i] = (char*)PyString_AsString(pystr);
        }
        if (!CompositeFilter_setQuantitiesArray(self->filter, (const char**)quantities, (int)nquantities)) {
          RAVE_FREE(quantities);
          raiseException_gotoTag(done, PyExc_MemoryError, "Could not allocate quantities");
        }
        RAVE_FREE(quantities);
      } else {
        CompositeFilter_setQuantitiesArray(self->filter, NULL, 0);
      }
    } else {
      CompositeFilter_setQuantitiesArray(self->filter, NULL, 0);
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "interpolation_methods") == 0) {
    if (PyList_Check(val)) {
      Py_ssize_t nmethods = PyObject_Length(val);
      int i = 0;
      if (nmethods > 0) {
        char** methods = RAVE_MALLOC(sizeof(char*)*nmethods);
        if (methods == NULL) {
          raiseException_gotoTag(done, PyExc_MemoryError, "Could not create methods array");
        }
        for (i = 0; i < nmethods; i++) {
          PyObject* pystr = PyList_GetItem(val, i);  /* We don't need to release this since it is internal pointer.*/
          methods[i] = (char*)PyString_AsString(pystr);
        }
        if (!CompositeFilter_setInterpolationMethodsArray(self->filter, (const char**)methods, (int)nmethods)) {
          RAVE_FREE(methods);
          raiseException_gotoTag(done, PyExc_MemoryError, "Could not allocate methods");
        }
        RAVE_FREE(methods);
      } else {
        CompositeFilter_setInterpolationMethodsArray(self->filter, NULL, 0);
      }    
    } else if (val == Py_None) {
      CompositeFilter_setInterpolationMethodsArray(self->filter, NULL, 0);
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }

  result = 0;
done:
  return result;
}

static PyObject* _pycompositefilter_isCompositeFilter(PyObject* self, PyObject* args)
{
  PyObject* inobj = NULL;
  if (!PyArg_ParseTuple(args,"O", &inobj)) {
    return NULL;
  }
  if (PyCompositeFilter_Check(inobj)) {
    return PyBool_FromLong(1);
  }
  return PyBool_FromLong(0);
}
/*@} End of Composite product generator */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pycompositefilter_type_doc,
    "Used to validate a composite argument to know if the arguments should be passed on to a factory instance or not.\n"

    "\n"
    "Usage:\n"
    " import _compositefilter\n"
    " filter = _compositefilter.new()\n"
    " filter.products = [\"PCAPPI\",\"CAPPI\"]\n"
    " result = filter.match(arguments)\n"
    );
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyCompositeFilter_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "CompositeFilterCore", /*tp_name*/
  sizeof(PyCompositeFilter), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pycompositefilter_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
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
  (getattrofunc)_pycompositefilter_getattro, /*tp_getattro*/
  (setattrofunc)_pycompositefilter_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pycompositefilter_type_doc,        /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pycompositefilter_methods,              /*tp_methods*/
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
  {"new", (PyCFunction)_pycompositefilter_new, 1,
    "new() -> new instance of the CompositeFilterCore object\n\n"
    "Creates a new instance of the CompositeFilterCore object"
  },
  {"isCompositeFilter", (PyCFunction)_pycompositefilter_isCompositeFilter, 1,
      "isCompositeFilter(obj) -> True if object is an composite filter, otherwise False\n\n"
      "Checks if the provided object is a python composite filter object or not.\n\n"
      "obj - the object to check."},  
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_compositefilter)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyCompositeFilter_API[PyCompositeFilter_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyCompositeFilter_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyCompositeFilter_Type);

  MOD_INIT_DEF(module, "_compositefilter", _pycompositefilter_type_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyCompositeFilter_API[PyCompositeFilter_Type_NUM] = (void*)&PyCompositeFilter_Type;
  PyCompositeFilter_API[PyCompositeFilter_GetNative_NUM] = (void *)&PyCompositeFilter_GetNative;
  PyCompositeFilter_API[PyCompositeFilter_New_NUM] = (void*)&PyCompositeFilter_New;


  c_api_object = PyCapsule_New(PyCompositeFilter_API, PyCompositeFilter_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_compositefilter.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _compositefilter.error");
    return MOD_INIT_ERROR;
  }

  import_compositearguments();
  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
