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
 * Python version of the Composite factory manager API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-30
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYCOMPOSITEFACTORYMANAGER_MODULE
#include "pycompositefactorymanager.h"
#include "pycompositegeneratorfactory.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_compositefactorymanager");

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
 * Returns the native CartesianFactoryManager_t instance.
 * @param[in] pymanager - the python composite factory manager instance
 * @returns the native cartesian instance.
 */
static CompositeFactoryManager_t*
PyCompositeFactoryManager_GetNative(PyCompositeFactoryManager* pymanager)
{
  RAVE_ASSERT((pymanager != NULL), "pymanager == NULL");
  return RAVE_OBJECT_COPY(pymanager->manager);
}

/**
 * Creates a python composite generator from a native composite generator or will create an
 * initial native CompositeFactoryManager if p is NULL.
 * @param[in] p - the native composite generator (or NULL)
 * @returns the python composite product generator.
 */
static PyCompositeFactoryManager*
PyCompositeFactoryManager_New(CompositeFactoryManager_t* p)
{
  PyCompositeFactoryManager* result = NULL;
  CompositeFactoryManager_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&CompositeFactoryManager_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for composite generator.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for composite generator.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyCompositeFactoryManager, &PyCompositeFactoryManager_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->manager = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->manager, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyCompositeFactoryManager instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for composite generator.");
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
static void _pycompositefactorymanager_dealloc(PyCompositeFactoryManager* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->manager, obj);
  RAVE_OBJECT_RELEASE(obj->manager);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the composite.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pycompositefactorymanager_new(PyObject* self, PyObject* args)
{
  PyCompositeFactoryManager* result = PyCompositeFactoryManager_New(NULL);
  return (PyObject*)result;
}

static PyObject* _pycompositefactorymanager_add(PyCompositeFactoryManager* self, PyObject* args)
{
  PyObject* pyfactory = NULL;
  if (!PyArg_ParseTuple(args, "O", &pyfactory)) {
    return NULL;
  }

  if (!PyCompositeGeneratorFactory_Check(pyfactory)) {
    raiseException_returnNULL(PyExc_TypeError, "object must be a CompositeGeneratorFactory");
  }

  if (!CompositeFactoryManager_add(self->manager, ((PyCompositeGeneratorFactory*)pyfactory)->factory)) {
    raiseException_returnNULL(PyExc_AttributeError, "Could not add factory to manager");
  }

  Py_RETURN_NONE;
}

static PyObject* _pycompositefactorymanager_getRegisteredFactoryNames(PyCompositeFactoryManager* self, PyObject* args)
{
  RaveList_t* ids = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }

  ids = CompositeFactoryManager_getRegisteredFactoryNames(self->manager);
  if (ids != NULL) {
    int i = 0, n = 0;
    n = RaveList_size(ids);
    result = PyList_New(0);
    for (i = 0; result != NULL && i < n; i++) {
      char* name = RaveList_get(ids, i);
      if (name != NULL) {
        PyObject* pynamestr = PyString_FromString(name);
        if (pynamestr == NULL) {
          goto fail;
        }
        if (PyList_Append(result, pynamestr) != 0) {
          Py_DECREF(pynamestr);
          goto fail;
        }
        Py_DECREF(pynamestr);
      }
    }
  }

  RaveList_freeAndDestroy(&ids);
  return result;
fail:
  RaveList_freeAndDestroy(&ids);
  Py_XDECREF(result);
  return NULL;  
}

static PyObject* _pycompositefactorymanager_remove(PyCompositeFactoryManager* self, PyObject* args)
{
  char* id = NULL;
  if (!PyArg_ParseTuple(args, "s", &id)) {
    return NULL;
  }
  CompositeFactoryManager_remove(self->manager, id);

  Py_RETURN_NONE;
}


static PyObject* _pycompositefactorymanager_get(PyCompositeFactoryManager* self, PyObject* args)
{
  char* name = NULL;
  PyObject* result = NULL;
  CompositeGeneratorFactory_t* factory = NULL;

  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }

  factory = CompositeFactoryManager_get(self->manager, name);
  if (factory != NULL) {
    result = (PyObject*)PyCompositeGeneratorFactory_New(factory);
  } else {
    PyErr_SetString(PyExc_KeyError, "No such factory");
  }
  RAVE_OBJECT_RELEASE(factory);
  return result;
}

static PyObject* _pycompositefactorymanager_isRegistered(PyCompositeFactoryManager* self, PyObject* args)
{
  char* id = NULL;
  if (!PyArg_ParseTuple(args, "s", &id)) {
    return NULL;
  }
  return PyBool_FromLong(CompositeFactoryManager_isRegistered(self->manager, id));
}

static PyObject* _pycompositefactorymanager_size(PyCompositeFactoryManager* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyLong_FromLong(CompositeFactoryManager_size(self->manager));
}

/**
 * All methods a cartesian product can have
 */
static struct PyMethodDef _pycompositefactorymanager_methods[] =
{
  {"add", (PyCFunction)_pycompositefactorymanager_add, 1,
    "add(factory)\n\n" //
    "Add a factory to the manager.\n\n"
    "plugin     - the factory instance\n"
  },
  {"getRegisteredFactoryNames", (PyCFunction)_pycompositefactorymanager_getRegisteredFactoryNames, 1,
    "getRegisteredFactoryNames()\n\n" //
    "Return the names of the registered factories.\n"
  },
  {"remove", (PyCFunction)_pycompositefactorymanager_remove, 1,
    "remove(name)\n\n" //
    "Removes a factory from the generator.\n\n"
    "name         - the name of the factory\n"
  },
  {"get", (PyCFunction)_pycompositefactorymanager_get, 1,
    "get(name)\n\n" // "sddd", &
    "Returns the factory with provided name. If not found, KeyError will be raised.\n\n"
    "name - the name of the factory"
  },
  {"isRegistered", (PyCFunction)_pycompositefactorymanager_isRegistered, 1,
    "isRegistered(name)\n\n" //
    "Returns if there is a factory with provided class name registered or not.\n\n"
    "name         - the name of the factory\n"
  },
  {"size", (PyCFunction)_pycompositefactorymanager_size, 1,
    "size()\n\n" //
    "Returns the number of registered factories.\n"
  },
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the cartesian
 * @param[in] self - the cartesian product
 */

static PyObject* _pycompositefactorymanager_getattro(PyCompositeFactoryManager* self, PyObject* name)
{
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pycompositefactorymanager_setattro(PyCompositeFactoryManager* self, PyObject* name, PyObject* val)
{
  int result = -1;
  return result;
}

/*@} End of Composite product generator */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pycompositefactorymanager_type_doc,
    "The composite factory manager is the registry for all available factories.\n"
    "When created it will contain the default provided factories that are available by the rave toolbox.\n"
    "It is possible to add new factories to the registry using the provided API methods.\n"
    "This factory can then be used to generate the composite. An alternative approach is to create\n"
    "a compositegenerator with the factory set. In that case, the filtering within the composite generator\n"
    "will determine what composite factory to use.\n"
    "\n"
    "Usage:\n"
    " import _compositefactorymanager\n"
    " manager = _compositefactorymanager.new()\n"
    " factory = manager.get(\"LegacyCompositeFactory\")\n"
    );
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyCompositeFactoryManager_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "CompositeFactoryManagerCore", /*tp_name*/
  sizeof(PyCompositeFactoryManager), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pycompositefactorymanager_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pycompositefactorymanager_getattro, /*tp_getattro*/
  (setattrofunc)_pycompositefactorymanager_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pycompositefactorymanager_type_doc,        /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pycompositefactorymanager_methods,              /*tp_methods*/
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
  {"new", (PyCFunction)_pycompositefactorymanager_new, 1,
    "new() -> new instance of the CompositeFactoryManagerCore object\n\n"
    "Creates a new instance of the CompositeFactoryManagerCore object"
  },
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_compositefactorymanager)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyCompositeFactoryManager_API[PyCompositeFactoryManager_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyCompositeFactoryManager_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyCompositeFactoryManager_Type);

  MOD_INIT_DEF(module, "_compositefactorymanagear", _pycompositefactorymanager_type_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyCompositeFactoryManager_API[PyCompositeFactoryManager_Type_NUM] = (void*)&PyCompositeFactoryManager_Type;
  PyCompositeFactoryManager_API[PyCompositeFactoryManager_GetNative_NUM] = (void *)&PyCompositeFactoryManager_GetNative;
  PyCompositeFactoryManager_API[PyCompositeFactoryManager_New_NUM] = (void*)&PyCompositeFactoryManager_New;

  c_api_object = PyCapsule_New(PyCompositeFactoryManager_API, PyCompositeFactoryManager_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_compositefactorymanager.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _compositefactorymanager.error");
    return MOD_INIT_ERROR;
  }

  import_compositegeneratorfactory();
  import_array();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
