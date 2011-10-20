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
 * Python version of the detection range API
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2011-02-18
 */
#include "Python.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYDETECTIONRANGE_MODULE        /**< to get correct part of pydetectionrange.h */
#include "pydetectionrange.h"

#include "pypolarvolume.h"
#include "pypolarscan.h"
#include "pyravefield.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_pydetectionrange");

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

/*@{ Detection range */
/**
 * Returns the native DetectionRange instance.
 * @param[in] pydetectionrange - the python detection range instance
 * @returns the native DetectionRange instance.
 */
static DetectionRange_t*
PyDetectionRange_GetNative(PyDetectionRange* pydetectionrange)
{
  RAVE_ASSERT((pydetectionrange != NULL), "pydetectionrange == NULL");
  return RAVE_OBJECT_COPY(pydetectionrange->dr);
}

/**
 * Creates a python detection range from a native detection range or will create an
 * initial native Detection range if p is NULL.
 * @param[in] p - the native detection range (or NULL)
 * @returns the python detection range generator.
 */
static PyDetectionRange*
PyDetectionRange_New(DetectionRange_t* p)
{
  PyDetectionRange* result = NULL;
  DetectionRange_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&DetectionRange_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for detection range.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for detection range.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyDetectionRange, &PyDetectionRange_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->dr = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->dr, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyDetectionRange instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for detection range.");
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
static void _pydetectionrange_dealloc(PyDetectionRange* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->dr, obj);
  RAVE_OBJECT_RELEASE(obj->dr);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the detection range.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pydetectionrange_new(PyObject* self, PyObject* args)
{
  PyDetectionRange* result = PyDetectionRange_New(NULL);
  return (PyObject*)result;
}

/**
 * Evaluates the echo tops
 * @param[in] self - self
 * @param[in] args - (a volume, scale (of the bins) and dbzn threshold)
 * @returns a python scan containing the tops
 */
static PyObject* _pydetectionrange_top(PyDetectionRange* self, PyObject* args)
{
  PyObject* object = NULL;
  PyObject* result = NULL;
  PyPolarVolume* volume = NULL;
  PolarScan_t* scan = NULL;
  double scale = 0.0, threshold = 0.0;
  if (!PyArg_ParseTuple(args, "Odd", &object, &scale, &threshold)) {
    return NULL;
  }
  if (PyPolarVolume_Check(object)) {
    volume = (PyPolarVolume*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "Top requires volume");
  }

  scan = DetectionRange_top(self->dr, volume->pvol, scale, threshold);
  if (scan == NULL) {
    raiseException_returnNULL(PyExc_Exception, "Failed to create top");
  }

  result = (PyObject*)PyPolarScan_New(scan);
  RAVE_OBJECT_RELEASE(scan);
  return result;
}

/**
 * Filters out unwanted values
 * @param[in] self - self
 * @param[in] args - (a hght scan retrieved from .top())
 * @returns a python scan containing the tops
 */
static PyObject* _pydetectionrange_filter(PyDetectionRange* self, PyObject* args)
{
  PyObject* object = NULL;
  PyPolarScan* pyscan = NULL;
  PolarScan_t* filteredscan = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "O", &object)) {
    return NULL;
  }
  if (PyPolarScan_Check(object)) {
    pyscan = (PyPolarScan*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "filter requires scan");
  }
  filteredscan = DetectionRange_filter(self->dr, pyscan->scan);
  if (filteredscan == NULL) {
    raiseException_returnNULL(PyExc_Exception, "Failed to filter scan");
  }
  result = (PyObject*)PyPolarScan_New(filteredscan);
  RAVE_OBJECT_RELEASE(filteredscan);
  return result;
}

/**
 * Performs the analyzing
 * @param[in] self - self
 * @param[in] args - (a hght scan, avgsector (int), sortage, samplepoint (double))
 * @returns a python rave field containing the tops
 */
static PyObject* _pydetectionrange_analyze(PyDetectionRange* self, PyObject* args)
{
  PyObject* object = NULL;
  PyPolarScan* pyscan = NULL;
  RaveField_t* analyzedfield = NULL;
  PyObject* result = NULL;
  int avgsector = 0;
  double sortage = 0.0L;
  double samplepoint = 0.0L;

  if (!PyArg_ParseTuple(args, "Oidd", &object, &avgsector, &sortage, &samplepoint)) {
    return NULL;
  }
  if (PyPolarScan_Check(object)) {
    pyscan = (PyPolarScan*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "filter requires scan");
  }
  analyzedfield = DetectionRange_analyze(self->dr, pyscan->scan, avgsector, sortage, samplepoint);
  if (analyzedfield == NULL) {
    raiseException_returnNULL(PyExc_Exception, "Failed to analyze field");
  }
  result = (PyObject*)PyRaveField_New(analyzedfield);
  RAVE_OBJECT_RELEASE(analyzedfield);
  return result;
}

/**
 * All methods a detection range generator
 */
static struct PyMethodDef _pydetectionrange_methods[] =
{
  {"lookupPath", NULL},
  {"analysis_minrange", NULL},
  {"analysis_maxrange", NULL},
  {"top", (PyCFunction) _pydetectionrange_top, 1},
  {"filter", (PyCFunction) _pydetectionrange_filter, 1},
  {"analyze", (PyCFunction) _pydetectionrange_analyze, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the detection range
 * @param[in] self - the detection range product
 */
static PyObject* _pydetectionrange_getattr(PyDetectionRange* self, char* name)
{
  PyObject* res = NULL;

  if (strcmp("lookupPath", name) == 0) {
    return PyString_FromString(DetectionRange_getLookupPath(self->dr));
  } else if (strcmp("analysis_minrange", name) == 0) {
    return PyFloat_FromDouble(DetectionRange_getAnalysisMinRange(self->dr));
  } else if (strcmp("analysis_maxrange", name) == 0) {
    return PyFloat_FromDouble(DetectionRange_getAnalysisMaxRange(self->dr));
  }

  res = Py_FindMethod(_pydetectionrange_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the detection range
 */
static int _pydetectionrange_setattr(PyDetectionRange* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("lookupPath", name) == 0) {
    if (PyString_Check(val)) {
      if (!DetectionRange_setLookupPath(self->dr, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "lookupPath could not be set");
      }
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"lookupPath must be of type string");
    }
  } else if (strcmp("analysis_minrange", name) == 0) {
    if (PyFloat_Check(val)) {
      DetectionRange_setAnalysisMinRange(self->dr, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      DetectionRange_setAnalysisMinRange(self->dr, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      DetectionRange_setAnalysisMinRange(self->dr, (double)PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "analysis_minrange must be a float or decimal value")
    }
  } else if (strcmp("analysis_maxrange", name) == 0) {
    if (PyFloat_Check(val)) {
      DetectionRange_setAnalysisMaxRange(self->dr, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      DetectionRange_setAnalysisMaxRange(self->dr, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      DetectionRange_setAnalysisMaxRange(self->dr, (double)PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "analysis_maxrange must be a float or decimal value")
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, name);
  }

  result = 0;
done:
  return result;
}

/*@} End of detection range generator */

/*@{ Type definitions */
PyTypeObject PyDetectionRange_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "DetectionRangeCore", /*tp_name*/
  sizeof(PyDetectionRange), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pydetectionrange_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pydetectionrange_getattr, /*tp_getattr*/
  (setattrfunc)_pydetectionrange_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_pydetectionrange_new, 1},
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_detectionrange(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyDetectionRange_API[PyDetectionRange_API_pointers];
  PyObject *c_api_object = NULL;
  PyDetectionRange_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_detectionrange", functions);
  if (module == NULL) {
    return;
  }
  PyDetectionRange_API[PyDetectionRange_Type_NUM] = (void*)&PyDetectionRange_Type;
  PyDetectionRange_API[PyDetectionRange_GetNative_NUM] = (void *)PyDetectionRange_GetNative;
  PyDetectionRange_API[PyDetectionRange_New_NUM] = (void*)PyDetectionRange_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyDetectionRange_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_pydetectionrange.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _pydetectionrange.error");
  }

  import_pypolarvolume();
  import_pypolarscan();
  import_pyravefield();
  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
