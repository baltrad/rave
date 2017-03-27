/* --------------------------------------------------------------------
Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the Compositing Algorithm API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2011-10-28
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYCOMPOSITEALGORITHM_MODULE    /**< to get correct part in pypoocompositealgorithm.h */
#include "pycompositealgorithm.h"
#include "rave_alloc.h"


/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_compositealgorithm");

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

/*@{ CompositeAlgorithm */
/**
 * Returns the native CompositeAlgorithm_t instance.
 * @param[in] pyalgorithm - the python poo composite algorithm
 * @returns the native algorithm instance.
 */
static CompositeAlgorithm_t*
PyCompositeAlgorithm_GetNative(PyCompositeAlgorithm* pyalgorithm)
{
  RAVE_ASSERT((pyalgorithm != NULL), "pyalgorithm == NULL");
  return RAVE_OBJECT_COPY(pyalgorithm->algorithm);
}

/**
 * Creates a python composite algorithm from a native one.
 * @param[in] p - the native algorithm (MAY NOT BE NULL)
 * @returns the python algorithm.
 */
static PyCompositeAlgorithm*
PyCompositeAlgorithm_New(CompositeAlgorithm_t* p)
{
  PyCompositeAlgorithm* result = NULL;
  CompositeAlgorithm_t* cp = NULL;

  if (p == NULL) {
    RAVE_CRITICAL0("You can not create a composite algorithm without the native algorithm");
    raiseException_returnNULL(PyExc_MemoryError, "You can not create a composite algorithm without the native algorithm.");
  }
  cp = RAVE_OBJECT_COPY(p);
  result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
  if (result != NULL) {
    Py_INCREF(result);
  }

  if (result == NULL) {
    result = PyObject_NEW(PyCompositeAlgorithm, &PyCompositeAlgorithm_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->algorithm = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->algorithm, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyCompositeAlgorithm instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyCompositeAlgorithm.");
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
static void _pycompositealgorithm_dealloc(PyCompositeAlgorithm* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->algorithm, obj);
  RAVE_OBJECT_RELEASE(obj->algorithm);
  PyObject_Del(obj);
}

/**
 * Returns the name of this composite algorithm
 * @param[in] self - self
 * @param[in] args - N/A
 * @return the name of this algorithm
 */
static PyObject* _pycompositealgorithm_getName(PyCompositeAlgorithm* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyString_FromString(CompositeAlgorithm_getName(self->algorithm));
}

static PyObject* _pycompositealgorithm_initialize(PyCompositeAlgorithm* self, PyObject* args)
{
  return NULL;
}

/**
 * Processes the given lon/lat
 */
static PyObject* _pycompositealgorithm_process(PyCompositeAlgorithm* self, PyObject* args)
{
  Py_RETURN_NONE;
}

/**
 * All methods a area can have
 */
static struct PyMethodDef _pycompositealgorithm_methods[] =
{
  {"getName", (PyCFunction) _pycompositealgorithm_getName, 1},
  {"initialize", (PyCFunction) _pycompositealgorithm_initialize, 1},
  {"process", (PyCFunction) _pycompositealgorithm_process, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the area
 * @param[in] self - the area
 */
static PyObject* _pycompositealgorithm_getattro(PyCompositeAlgorithm* self, PyObject* name)
{
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the area
 */
static int _pycompositealgorithm_setattro(PyCompositeAlgorithm* self, PyObject* name, PyObject* val)
{
  return -1;
}

/*@} End of PooCompositeAlgorithm */

/*@{ Type definitions */
PyTypeObject PyCompositeAlgorithm_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "CompositeAlgorithmCore", /*tp_name*/
  sizeof(PyCompositeAlgorithm), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pycompositealgorithm_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pycompositealgorithm_getattro, /*tp_getattro*/
  (setattrofunc)_pycompositealgorithm_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  0,                            /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pycompositealgorithm_methods,/*tp_methods*/
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

MOD_INIT(_compositealgorithm)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyCompositeAlgorithm_API[PyCompositeAlgorithm_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyCompositeAlgorithm_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyCompositeAlgorithm_Type);

  MOD_INIT_DEF(module, "_compositealgorithm", NULL/*doc*/, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyCompositeAlgorithm_API[PyCompositeAlgorithm_Type_NUM] = (void*)&PyCompositeAlgorithm_Type;
  PyCompositeAlgorithm_API[PyCompositeAlgorithm_GetNative_NUM] = (void *)PyCompositeAlgorithm_GetNative;
  PyCompositeAlgorithm_API[PyCompositeAlgorithm_New_NUM] = (void*)PyCompositeAlgorithm_New;

  c_api_object = PyCapsule_New(PyCompositeAlgorithm_API, PyCompositeAlgorithm_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_compositealgorithm.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _compositealgorithm.error");
    return MOD_INIT_ERROR;
  }


  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
