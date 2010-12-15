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
 * Python version of the Projection registry
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
#include "pyprojection.h"

#define PYPROJECTIONREGISTRY_MODULE    /**< to get correct part in pyprojectionregistry.h */
#include "pyprojectionregistry.h"
#include "rave_alloc.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_projectionregistry");

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

/*@{ ProjectionRegistry */
/**
 * Returns the native ProjectionRegistry_t instance.
 * @param[in] pyregistry - the python registry instance
 * @returns the native registry instance.
 */
static ProjectionRegistry_t*
PyProjectionRegistry_GetNative(PyProjectionRegistry* pyregistry)
{
  RAVE_ASSERT((pyregistry != NULL), "pyregistry == NULL");
  return RAVE_OBJECT_COPY(pyregistry->registry);
}

/**
 * Creates a python registry from a native registry or will create an
 * initial native projection registry if p is NULL.
 * @param[in] p - the native projection registry (or NULL)
 * @returns the python projection registry.
 */
static PyProjectionRegistry*
PyProjectionRegistry_New(ProjectionRegistry_t* p)
{
  PyProjectionRegistry* result = NULL;
  ProjectionRegistry_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&ProjectionRegistry_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for projection registry.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for projection registry.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyProjectionRegistry, &PyProjectionRegistry_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->registry = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->registry, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyProjectionRegistry instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyProjectionRegistry.");
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
static void _pyprojectionregistry_dealloc(PyProjectionRegistry* obj)
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
static PyObject* _pyprojectionregistry_new(PyObject* self, PyObject* args)
{
  PyProjectionRegistry* result = PyProjectionRegistry_New(NULL);
  return (PyObject*)result;
}

/**
 * All methods a registry can have
 */
static struct PyMethodDef _pyprojectionregistry_methods[] =
{
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the registry
 * @param[in] self - the registry
 */
static PyObject* _pyprojectionregistry_getattr(PyProjectionRegistry* self, char* name)
{
  PyObject* res = NULL;

  res = Py_FindMethod(_pyprojectionregistry_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Sets the specified attribute in the registry
 */
static int _pyprojectionregistry_setattr(PyProjectionRegistry* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }

  result = 0;
done:
  return result;
}

/*@} End of ProjectionRegistry */

/*@{ Type definitions */
PyTypeObject PyProjectionRegistry_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "ProjectionRegistryCore", /*tp_name*/
  sizeof(PyProjectionRegistry), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyprojectionregistry_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pyprojectionregistry_getattr, /*tp_getattr*/
  (setattrfunc)_pyprojectionregistry_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_pyprojectionregistry_new, 1},
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_projectionregistry(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyProjectionRegistry_API[PyProjectionRegistry_API_pointers];
  PyObject *c_api_object = NULL;
  PyProjectionRegistry_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_projectionregistry", functions);
  if (module == NULL) {
    return;
  }
  PyProjectionRegistry_API[PyProjectionRegistry_Type_NUM] = (void*)&PyProjectionRegistry_Type;
  PyProjectionRegistry_API[PyProjectionRegistry_GetNative_NUM] = (void *)PyProjectionRegistry_GetNative;
  PyProjectionRegistry_API[PyProjectionRegistry_New_NUM] = (void*)PyProjectionRegistry_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyProjectionRegistry_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_projectionregistry.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _projectionregistry.error");
  }

  import_pyprojection();
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
