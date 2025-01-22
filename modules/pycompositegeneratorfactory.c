/* --------------------------------------------------------------------
Copyright (C) 2024 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the Compositing Generator Plugin API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-12-09
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYCOMPOSITEGENERATORFACTORY_MODULE    /**< to get correct part in pycompositegeneratorfactory.h */
#include "pycompositegeneratorfactory.h"
#include "rave_alloc.h"
#include "pycompositearguments.h"
#include "pycartesian.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_compositegeneratorfactory");

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

/*@{ CompositeGeneratorFactory */
/**
 * Returns the native CompositeGeneratorPlugin_t instance.
 * @param[in] pyplugin - the python composite generator plugin
 * @returns the native plugin instance.
 */
static CompositeGeneratorFactory_t*
PyCompositeGeneratorFactory_GetNative(PyCompositeGeneratorFactory* pyfactory)
{
  RAVE_ASSERT((pyfactory != NULL), "pyfactory == NULL");
  return RAVE_OBJECT_COPY(pyfactory->factory);
}

/**
 * Creates a python composite algorithm from a native one.
 * @param[in] p - the native algorithm (MAY NOT BE NULL)
 * @returns the python algorithm.
 */
static PyCompositeGeneratorFactory*
PyCompositeGeneratorFactory_New(CompositeGeneratorFactory_t* p)
{
  PyCompositeGeneratorFactory* result = NULL;
  CompositeGeneratorFactory_t* cp = NULL;

  if (p == NULL) {
    RAVE_CRITICAL0("You can not create a composite generator plugin without the native plugin");
    raiseException_returnNULL(PyExc_MemoryError, "You can not create a composite generator plugin without the native plugin.");
  }
  cp = RAVE_OBJECT_COPY(p);
  result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
  if (result != NULL) {
    Py_INCREF(result);
  }

  if (result == NULL) {
    result = PyObject_NEW(PyCompositeGeneratorFactory, &PyCompositeGeneratorFactory_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->factory = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->factory, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyCompositeGeneratorFactory instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyCompositeGeneratorFactory.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the algorithm
 * @param[in] obj the object to deallocate.
 */
static void _pycompositegeneratorfactory_dealloc(PyCompositeGeneratorFactory* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->factory, obj);
  RAVE_OBJECT_RELEASE(obj->factory);
  PyObject_Del(obj);
}

/**
 * Returns the name of this composite generator plugin
 * @param[in] self - self
 * @param[in] args - N/A
 * @return the name of this algorithm
 */
static PyObject* _pycompositegeneratorfactory_getName(PyCompositeGeneratorFactory* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyString_FromString(CompositeGeneratorFactory_getName(self->factory));
}

/**
 * Returns if this factory can handle provided arguments
 * @param[in] self - self
 * @param[in] args - N/A
 * @return the name of this algorithm
 */
static PyObject* _pycompositegeneratorfactory_canHandle(PyCompositeGeneratorFactory* self, PyObject* args)
{
  PyObject* pyargs = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyargs)) {
    return NULL;
  }

  if (!PyCompositeArguments_Check(pyargs)) {
    raiseException_returnNULL(PyExc_AttributeError, "Expects a CompositeArgument object as argument");
  }
  return PyBool_FromLong(CompositeGeneratorFactory_canHandle(self->factory, ((PyCompositeArguments*)pyargs)->args));
}

static PyObject* _pycompositegeneratorfactory_create(PyCompositeGeneratorFactory* self, PyObject* args)
{
  PyObject* result = NULL;
  CompositeGeneratorFactory_t* factory = NULL;

  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }

  factory = CompositeGeneratorFactory_create(self->factory);
  if (factory == NULL) {
    raiseException_returnNULL(PyExc_RuntimeError, "Could not create factory object");
  }
  result = (PyObject*)PyCompositeGeneratorFactory_New(factory);

  RAVE_OBJECT_RELEASE(factory);

  return result;
}

static PyObject* _pycompositegeneratorfactory_generate(PyCompositeGeneratorFactory* self, PyObject* args)
{
  PyObject* result = NULL;
  PyObject* pyargs = NULL;
  Cartesian_t* cartesian = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyargs)) {
    return NULL;
  }

  if (!PyCompositeArguments_Check(pyargs)) {
    raiseException_returnNULL(PyExc_ValueError, "Must provide a composite argument instance");
  }
  cartesian = CompositeGeneratorFactory_generate(self->factory, ((PyCompositeArguments*)pyargs)->args);
  if (cartesian != NULL) {
    result = (PyObject*)PyCartesian_New(cartesian);
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Could not generate cartesian product");
  }
  RAVE_OBJECT_RELEASE(cartesian);
  return result;
}

/**
 * All methods a area can have
 */
static struct PyMethodDef _pycompositegeneratorfactory_methods[] =
{
  {"getName", (PyCFunction) _pycompositegeneratorfactory_getName, 1},
  {"canHandle", (PyCFunction) _pycompositegeneratorfactory_canHandle, 1},
  {"create", (PyCFunction) _pycompositegeneratorfactory_create, 1},
  {"generate", (PyCFunction) _pycompositegeneratorfactory_generate, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the area
 * @param[in] self - the area
 */
static PyObject* _pycompositegeneratorfactory_getattro(PyCompositeGeneratorFactory* self, PyObject* name)
{
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the area
 */
static int _pycompositegeneratorfactory_setattro(PyCompositeGeneratorFactory* self, PyObject* name, PyObject* val)
{
  return -1;
}

/*@} End of PooCompositeGeneratorFactory */

/*@{ Type definitions */
PyTypeObject PyCompositeGeneratorFactory_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "CompositeGeneratorFactoryCore", /*tp_name*/
  sizeof(PyCompositeGeneratorFactory), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pycompositegeneratorfactory_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pycompositegeneratorfactory_getattro, /*tp_getattro*/
  (setattrofunc)_pycompositegeneratorfactory_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  0,                            /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pycompositegeneratorfactory_methods,/*tp_methods*/
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
  0,                            /*tp_is_gc*/};
/*@} End of Type definitions */

/*@{ Module setup */
static PyMethodDef functions[] = {
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_compositegeneratorfactory)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyCompositeGeneratorFactory_API[PyCompositeGeneratorFactory_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyCompositeGeneratorFactory_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyCompositeGeneratorFactory_Type);

  MOD_INIT_DEF(module, "_compositegeneratorfactory", NULL/*doc*/, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyCompositeGeneratorFactory_API[PyCompositeGeneratorFactory_Type_NUM] = (void*)&PyCompositeGeneratorFactory_Type;
  PyCompositeGeneratorFactory_API[PyCompositeGeneratorFactory_GetNative_NUM] = (void *)PyCompositeGeneratorFactory_GetNative;
  PyCompositeGeneratorFactory_API[PyCompositeGeneratorFactory_New_NUM] = (void*)PyCompositeGeneratorFactory_New;

  c_api_object = PyCapsule_New(PyCompositeGeneratorFactory_API, PyCompositeGeneratorFactory_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_compositegeneratorfactory.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _compositegeneratorfactory.error");
    return MOD_INIT_ERROR;
  }

  PYRAVE_DEBUG_INITIALIZE;
  import_compositearguments();
  import_pycartesian();
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
