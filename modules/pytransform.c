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
 * Python version of the Transform API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#include "Python.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYTRANSFORM_MODULE
#include "pytransform.h"

#include "pypolarscan.h"
#include "pypolarvolume.h"
#include "pycartesian.h"

#include "rave_debug.h"
#include "rave_alloc.h"

/**
 * Some helpful exception defines.
 */
#define raiseException_gotoTag(tag, type, msg) \
{PyErr_SetString(type, msg); goto tag;}

#define raiseException_returnNULL(type, msg) \
{PyErr_SetString(type, msg); return NULL;}

/**
 * Error object for reporting errors to the python interpreeter
 */
static PyObject *ErrorObject;

/*@{ Transform */
/**
 * Returns the native Transform_t instance.
 * @param[in] pytransform - the python transform instance
 * @returns the native transform instance.
 */
static Transform_t*
PyTransform_GetNative(PyTransform* pytransform)
{
  RAVE_ASSERT((pytransform != NULL), "pytransform == NULL");
  return RAVE_OBJECT_COPY(pytransform->transform);
}

/**
 * Creates a python transform from a native transform or will create an
 * initial native transform if p is NULL.
 * @param[in] p - the native transform (or NULL)
 * @returns the python transform product.
 */
static PyTransform*
PyTransform_New(Transform_t* p)
{
  PyTransform* result = NULL;
  Transform_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&Transform_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for transform.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for transform.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
  }

  result = PyObject_NEW(PyTransform, &PyTransform_Type);
  if (result == NULL) {
    RAVE_CRITICAL0("Failed to create PyTransform instance");
    raiseException_gotoTag(error, PyExc_MemoryError, "Failed to allocate memory for PyTransform.");
  }

  result->transform = RAVE_OBJECT_COPY(cp);
  RAVE_OBJECT_BIND(result->transform, result);
error:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the transformator
 * @param[in] obj the object to deallocate.
 */
static void _pytransform_dealloc(PyTransform* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  RAVE_OBJECT_UNBIND(obj->transform, obj);
  RAVE_OBJECT_RELEASE(obj->transform);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the transformator.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pytransform_new(PyObject* self, PyObject* args)
{
  PyTransform* result = PyTransform_New(NULL);
  return (PyObject*)result;
}

/**
 * Creates a ppi from a polar volume
 * @param[in] self the transformer
 * @param[in] args arguments for generating the ppi (polarvolume, cartesian)
 * @return Py_None on success, otherwise NULL
 */
static PyObject* _pytransform_ppi(PyTransform* self, PyObject* args)
{
  PyCartesian* cartesian = NULL;
  PyObject* pycartesian = NULL;
  PyPolarScan* scan = NULL;
  PyObject* pyscan = NULL;

  if(!PyArg_ParseTuple(args, "OO", &pyscan, &pycartesian)) {
    return NULL;
  }

  if (!PyPolarScan_Check(pyscan)) {
    raiseException_returnNULL(PyExc_TypeError, "First argument should be a polar scan")
  }

  if (!PyCartesian_Check(pycartesian)) {
    raiseException_returnNULL(PyExc_TypeError, "Second argument should be a cartesian product");
  }

  scan = (PyPolarScan*)pyscan;
  cartesian = (PyCartesian*)pycartesian;

  if (!Transform_ppi(self->transform, scan->scan, cartesian->cartesian)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to transform volume into a ppi");
  }

  Py_RETURN_NONE;
}

/**
 * Creates a cappi from a polar volume
 * @param[in] self the transformer
 * @param[in] args arguments for generating the cappi (polarvolume, cartesian, height in meters)
 * @return Py_None on success, otherwise NULL
 */
static PyObject* _pytransform_cappi(PyTransform* self, PyObject* args)
{
  PyCartesian* cartesian = NULL;
  PyObject* pycartesian = NULL;
  PyPolarVolume* pvol = NULL;
  PyObject* pypvol = NULL;
  double height = 0.0L;

  if(!PyArg_ParseTuple(args, "OOd", &pypvol, &pycartesian, &height)) {
    return NULL;
  }

  if (!PyPolarVolume_Check(pypvol)) {
    raiseException_returnNULL(PyExc_TypeError, "First argument should be a polar volume")
  }

  if (!PyCartesian_Check(pycartesian)) {
    raiseException_returnNULL(PyExc_TypeError, "Second argument should be a cartesian product");
  }

  pvol = (PyPolarVolume*)pypvol;
  cartesian = (PyCartesian*)pycartesian;

  if (!Transform_cappi(self->transform, pvol->pvol, cartesian->cartesian, height)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to transform volume into a cappi");
  }

  Py_RETURN_NONE;
}

/**
 * Creates a pseudo-cappi from a polar volume
 * @param[in] self the transformer
 * @param[in] args arguments for generating the pseudo-cappi (polarvolume, cartesian)
 * @return Py_None on success, otherwise NULL
 */
static PyObject* _pytransform_pcappi(PyTransform* self, PyObject* args)
{
  PyCartesian* cartesian = NULL;
  PyObject* pycartesian = NULL;
  PyPolarVolume* pvol = NULL;
  PyObject* pypvol = NULL;
  double height = 0.0L;

  if(!PyArg_ParseTuple(args, "OOd", &pypvol, &pycartesian,&height)) {
    return NULL;
  }

  if (!PyPolarVolume_Check(pypvol)) {
    raiseException_returnNULL(PyExc_TypeError, "First argument should be a polar volume")
  }

  if (!PyCartesian_Check(pycartesian)) {
    raiseException_returnNULL(PyExc_TypeError, "Second argument should be a cartesian product");
  }

  pvol = (PyPolarVolume*)pypvol;
  cartesian = (PyCartesian*)pycartesian;

  if (!Transform_pcappi(self->transform, pvol->pvol, cartesian->cartesian, height)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to transform volume into a cappi");
  }

  Py_RETURN_NONE;
}

/**
 * All methods a transformator can have
 */
static struct PyMethodDef _pytransform_methods[] =
{
  {"ppi", (PyCFunction) _pytransform_ppi, 1},
  {"cappi", (PyCFunction) _pytransform_cappi, 1},
  {"pcappi", (PyCFunction) _pytransform_pcappi, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the transformator
 * @param[in] self - the transform
 */
static PyObject* _pytransform_getattr(PyTransform* self, char* name)
{
  PyObject* res = NULL;

  if (strcmp("method", name) == 0) {
    return PyInt_FromLong(Transform_getMethod(self->transform));
  }

  res = Py_FindMethod(_pytransform_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the transformator
 */
static int _pytransform_setattr(PyTransform* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("method", name)==0) {
    if (PyInt_Check(val)) {
      if (!Transform_setMethod(self->transform, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "method must be in valid range");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"method must be a valid RaveTransformMethod");
    }
  }

  result = 0;
done:
  return result;
}
/*@} End of Transform */

/*@{ Type definitions */
PyTypeObject PyTransform_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "TransformCore", /*tp_name*/
  sizeof(PyTransform), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pytransform_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pytransform_getattr, /*tp_getattr*/
  (setattrfunc)_pytransform_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_pytransform_new, 1},
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_transform(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyTransform_API[PyTransform_API_pointers];
  PyObject *c_api_object = NULL;
  PyTransform_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_transform", functions);
  if (module == NULL) {
    return;
  }
  PyTransform_API[PyTransform_Type_NUM] = (void*)&PyTransform_Type;
  PyTransform_API[PyTransform_GetNative_NUM] = (void *)PyTransform_GetNative;
  PyTransform_API[PyTransform_New_NUM] = (void*)PyTransform_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyTransform_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_transform.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _transform.error");
  }

  import_pypolarvolume();
  import_pypolarscan();
  import_pycartesian();
  Rave_initializeDebugger();
}
/*@} End of Module setup */
