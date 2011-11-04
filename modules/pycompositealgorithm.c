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
#include "Python.h"
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
static PyObject* _pycompositealgorithm_getattr(PyCompositeAlgorithm* self, char* name)
{
  PyObject* res = NULL;
  res = Py_FindMethod(_pycompositealgorithm_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the area
 */
static int _pycompositealgorithm_setattr(PyCompositeAlgorithm* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  //result = 0;
done:
  return result;
}

/*@} End of PooCompositeAlgorithm */

/*@{ Type definitions */
PyTypeObject PyCompositeAlgorithm_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "CompositeAlgorithmCore", /*tp_name*/
  sizeof(PyCompositeAlgorithm), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pycompositealgorithm_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pycompositealgorithm_getattr, /*tp_getattr*/
  (setattrfunc)_pycompositealgorithm_setattr, /*tp_setattr*/
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
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_compositealgorithm(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyCompositeAlgorithm_API[PyCompositeAlgorithm_API_pointers];
  PyObject *c_api_object = NULL;

  PyCompositeAlgorithm_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_compositealgorithm", functions);
  if (module == NULL) {
    return;
  }
  PyCompositeAlgorithm_API[PyCompositeAlgorithm_Type_NUM] = (void*)&PyCompositeAlgorithm_Type;
  PyCompositeAlgorithm_API[PyCompositeAlgorithm_GetNative_NUM] = (void *)PyCompositeAlgorithm_GetNative;
  PyCompositeAlgorithm_API[PyCompositeAlgorithm_New_NUM] = (void*)PyCompositeAlgorithm_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyCompositeAlgorithm_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_compositealgorithm.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _compositealgorithm.error");
  }
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
