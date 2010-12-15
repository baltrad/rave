/* --------------------------------------------------------------------
Copyright (C) 2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the Area registry
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-12-15
 */
#include "Python.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"
#include "pyprojectionregistry.h"
#include "pyarea.h"

#define PYAREAREGISTRY_MODULE    /**< to get correct part in pyarearegistry.h */
#include "pyarearegistry.h"
#include "rave_alloc.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_arearegistry");

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

/*@{ AreaRegistry */
/**
 * Returns the native AreaRegistry_t instance.
 * @param[in] pyregistry - the python registry instance
 * @returns the native registry instance.
 */
static AreaRegistry_t*
PyAreaRegistry_GetNative(PyAreaRegistry* pyregistry)
{
  RAVE_ASSERT((pyregistry != NULL), "pyregistry == NULL");
  return RAVE_OBJECT_COPY(pyregistry->registry);
}

/**
 * Creates a python registry from a native registry or will create an
 * initial native area registry if p is NULL.
 * @param[in] p - the native area registry (or NULL)
 * @returns the python area registry.
 */
static PyAreaRegistry*
PyAreaRegistry_New(AreaRegistry_t* p)
{
  PyAreaRegistry* result = NULL;
  AreaRegistry_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&AreaRegistry_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for area registry.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for area registry.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyAreaRegistry, &PyAreaRegistry_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->registry = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->registry, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyAreaRegistry instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyAreaRegistry.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the registry
 * @param[in] obj the object to deallocate.
 */
static void _pyarearegistry_dealloc(PyAreaRegistry* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->registry, obj);
  RAVE_OBJECT_RELEASE(obj->registry);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the registry.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyarearegistry_new(PyObject* self, PyObject* args)
{
  PyAreaRegistry* result = PyAreaRegistry_New(NULL);
  return (PyObject*)result;
}

/**
 * All methods a registry can have
 */
static struct PyMethodDef _pyarearegistry_methods[] =
{
  {"pregistry", NULL},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the registry
 * @param[in] self - the registry
 */
static PyObject* _pyarearegistry_getattr(PyAreaRegistry* self, char* name)
{
  PyObject* res = NULL;

  if (strcmp("pregistry", name) == 0) {
    ProjectionRegistry_t* pregistry = AreaRegistry_getProjectionRegistry(self->registry);
    if (pregistry != NULL) {
      PyProjectionRegistry* result = PyProjectionRegistry_New(pregistry);
      RAVE_OBJECT_RELEASE(pregistry);
      return (PyObject*)result;
    } else {
      Py_RETURN_NONE;
    }
  }

  res = Py_FindMethod(_pyarearegistry_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Sets the specified attribute in the registry
 */
static int _pyarearegistry_setattr(PyAreaRegistry* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("projection", name)==0) {
    if (PyProjectionRegistry_Check(val)) {
      AreaRegistry_setProjectionRegistry(self->registry, ((PyProjectionRegistry*)val)->registry);
    } else if (val == Py_None) {
      AreaRegistry_setProjectionRegistry(self->registry, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"projection registry must be of ProjectionRegistryCore type");
    }
  }

  result = 0;
done:
  return result;
}

/*@} End of AreaRegistry */

/*@{ Type definitions */
PyTypeObject PyAreaRegistry_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "AreaRegistryCore", /*tp_name*/
  sizeof(PyAreaRegistry), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyarearegistry_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pyarearegistry_getattr, /*tp_getattr*/
  (setattrfunc)_pyarearegistry_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_pyarearegistry_new, 1},
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_arearegistry(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyAreaRegistry_API[PyAreaRegistry_API_pointers];
  PyObject *c_api_object = NULL;
  PyAreaRegistry_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_arearegistry", functions);
  if (module == NULL) {
    return;
  }
  PyAreaRegistry_API[PyAreaRegistry_Type_NUM] = (void*)&PyAreaRegistry_Type;
  PyAreaRegistry_API[PyAreaRegistry_GetNative_NUM] = (void *)PyAreaRegistry_GetNative;
  PyAreaRegistry_API[PyAreaRegistry_New_NUM] = (void*)PyAreaRegistry_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyAreaRegistry_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_arearegistry.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _arearegistry.error");
  }

  import_pyprojectionregistry();
  import_pyarea();
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
