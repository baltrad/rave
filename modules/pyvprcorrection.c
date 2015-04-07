/* --------------------------------------------------------------------
Copyright (C) 2015 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the VPR Correction API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2015-03-23
 */
#include "Python.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYVPRCORRECTION_MODULE /**< include correct part of pyvprcorrection.h */
#include "pyvprcorrection.h"

#include "pypolarscan.h"
#include "pypolarvolume.h"
#include "pyrave_debug.h"
#include "rave_alloc.h"

/**
 * This modules name
 */
PYRAVE_DEBUG_MODULE("_vprcorrection");

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

/*@{ VprCorrection */
/**
 * Returns the native RaveVprCorrection_t instance.
 * @param[in] pyvprcorrection - the python vpr correction instance
 * @returns the native vpr correction instance.
 */
static RaveVprCorrection_t*
PyVprCorrection_GetNative(PyVprCorrection* pyvprcorrection)
{
  RAVE_ASSERT((pyvprcorrection != NULL), "pyvprcorrection == NULL");
  return RAVE_OBJECT_COPY(pyvprcorrection->vpr);
}

/**
 * Creates a python vpr correction from a native vpr correction or will create an
 * initial native vpr correction if p is NULL.
 * @param[in] p - the native vpr correction (or NULL)
 * @returns the python vpr correction product.
 */
static PyVprCorrection*
PyVprCorrection_New(RaveVprCorrection_t* p)
{
  PyVprCorrection* result = NULL;
  RaveVprCorrection_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RaveVprCorrection_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for vpr correction.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for vpr correction.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyVprCorrection, &PyVprCorrection_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->vpr = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->vpr, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyVprCorrection instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyVprCorrection.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the vpr correction
 * @param[in] obj the object to deallocate.
 */
static void _pyvprcorrection_dealloc(PyVprCorrection* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->vpr, obj);
  RAVE_OBJECT_RELEASE(obj->vpr);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the vpr correction instance.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyvprcorrection_new(PyObject* self, PyObject* args)
{
  PyVprCorrection* result = PyVprCorrection_New(NULL);
  return (PyObject*)result;
}

static PyObject* _pyvprcorrection_getReflectivityArray(PyVprCorrection* self, PyObject* val)
{
  PyObject* pyo = NULL;
  int nElem = 0;
  double* elems = NULL;
  PyObject* pylist = NULL;
  if (!PyArg_ParseTuple(val, "O", &pyo)) {
    return NULL;
  }
  if (!PyPolarVolume_Check(pyo)) {
    raiseException_returnNULL(PyExc_TypeError, "Must provide a polar volume to get a relevant reflectivity array");
  }
  elems = RaveVprCorrection_getReflectivityArray(self->vpr, ((PyPolarVolume*)pyo)->pvol, &nElem);
  if (elems != NULL && nElem > 0) {
    int i = 0;
    pylist = PyList_New(nElem);
    for (i = 0; i < nElem; i++) {
      PyList_SetItem(pylist, i, PyFloat_FromDouble(elems[i]));
    }
  } else {
    raiseException_gotoTag(done, PyExc_RuntimeError, "Could not generate reflectivity array");
  }
done:
  RAVE_FREE(elems);
  return pylist;
}

/**
 * All methods a transformator can have
 */
static struct PyMethodDef _pyvprcorrection_methods[] =
{
  {"minReflectivity", NULL},
  {"heightLimit", NULL},
  {"profileHeight", NULL},
  {"minDistance", NULL},
  {"maxDistance", NULL},
  {"getReflectivityArray", (PyCFunction) _pyvprcorrection_getReflectivityArray, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the vpr correction instance
 * @param[in] self - the vpr correction
 */
static PyObject* _pyvprcorrection_getattr(PyVprCorrection* self, char* name)
{
  PyObject* res = NULL;
  if (strcmp("minReflectivity", name) == 0) {
    return PyFloat_FromDouble(RaveVprCorrection_getMinReflectivity(self->vpr));
  }else if (strcmp("heightLimit", name) == 0) {
    return PyFloat_FromDouble(RaveVprCorrection_getHeightLimit(self->vpr));
  } else if (strcmp("profileHeight", name) == 0) {
    return PyFloat_FromDouble(RaveVprCorrection_getProfileHeight(self->vpr));
  } else if (strcmp("minDistance", name) == 0) {
    return PyFloat_FromDouble(RaveVprCorrection_getMinDistance(self->vpr));
  } else if (strcmp("maxDistance", name) == 0) {
    return PyFloat_FromDouble(RaveVprCorrection_getMaxDistance(self->vpr));
  }

  res = Py_FindMethod(_pyvprcorrection_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the vpr correction
 */
static int _pyvprcorrection_setattr(PyVprCorrection* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }

  if (strcmp("minReflectivity", name) == 0) {
    if (PyInt_Check(val)) {
      RaveVprCorrection_setMinReflectivity(self->vpr, (double)PyInt_AsLong(val));
    } else if (PyFloat_Check(val)) {
      RaveVprCorrection_setMinReflectivity(self->vpr, (double)PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"Min reflectivity must be a valid float");
    }
  } else if (strcmp("heightLimit", name) == 0) {
    if (PyInt_Check(val)) {
      RaveVprCorrection_setHeightLimit(self->vpr, (double)PyInt_AsLong(val));
    } else if (PyFloat_Check(val)) {
      RaveVprCorrection_setHeightLimit(self->vpr, (double)PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"Height must be a valid float");
    }
  } else if (strcmp("profileHeight", name) == 0) {
    if (PyInt_Check(val)) {
      RaveVprCorrection_setProfileHeight(self->vpr, (double)PyInt_AsLong(val));
    } else if (PyFloat_Check(val)) {
      RaveVprCorrection_setProfileHeight(self->vpr, (double)PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"Profile Height must be a valid float");
    }
  } else if (strcmp("minDistance", name) == 0) {
    if (PyInt_Check(val)) {
      RaveVprCorrection_setMinDistance(self->vpr, (double)PyInt_AsLong(val));
    } else if (PyFloat_Check(val)) {
      RaveVprCorrection_setMinDistance(self->vpr, (double)PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"Min Distance must be a valid float");
    }
  } else if (strcmp("maxDistance", name) == 0) {
    if (PyInt_Check(val)) {
      RaveVprCorrection_setMaxDistance(self->vpr, (double)PyInt_AsLong(val));
    } else if (PyFloat_Check(val)) {
      RaveVprCorrection_setMaxDistance(self->vpr, (double)PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"Max Distance must be a valid float");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, name);
  }

  result = 0;
done:
  return result;
}
/*@} End of Transform */

/*@{ Type definitions */
PyTypeObject PyVprCorrection_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "VprCorrectionCore", /*tp_name*/
  sizeof(PyVprCorrection), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyvprcorrection_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pyvprcorrection_getattr, /*tp_getattr*/
  (setattrfunc)_pyvprcorrection_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_pyvprcorrection_new, 1},
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_vprcorrection(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyVprCorrection_API[PyVprCorrection_API_pointers];
  PyObject *c_api_object = NULL;
  PyVprCorrection_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_vprcorrection", functions);
  if (module == NULL) {
    return;
  }
  PyVprCorrection_API[PyVprCorrection_Type_NUM] = (void*)&PyVprCorrection_Type;
  PyVprCorrection_API[PyVprCorrection_GetNative_NUM] = (void *)PyVprCorrection_GetNative;
  PyVprCorrection_API[PyVprCorrection_New_NUM] = (void*)PyVprCorrection_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyVprCorrection_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_vprcorrection.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _vprcorrection.error");
  }

  import_pypolarvolume();
  import_pypolarscan();
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
