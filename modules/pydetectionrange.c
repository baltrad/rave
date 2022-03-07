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
#include "pyravecompat.h"
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
  char* paramname = "DBZH";
  if (!PyArg_ParseTuple(args, "Odd|s", &object, &scale, &threshold, &paramname)) {
    return NULL;
  }
  if (PyPolarVolume_Check(object)) {
    volume = (PyPolarVolume*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "Top requires volume");
  }

  scan = DetectionRange_top(self->dr, volume->pvol, scale, threshold, paramname);
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
  {"lookupPath", NULL, METH_VARARGS},
  {"analysis_minrange", NULL, METH_VARARGS},
  {"analysis_maxrange", NULL, METH_VARARGS},
  {"top", (PyCFunction) _pydetectionrange_top, 1,
    "top(pvol, scale, threshold[, quantity]) -> polar scan" // Odd|s", &object, &scale, &threshold, &paramname
    "Creates the echo top as a scan with the parameters quantity set to HGHT.\n\n"
    "pvol      - the polar volume\n"
    "scale     - the bin length\n"
    "threshold - the threshold for the values\n"
    "quantity  - Optional, the parameter quantity the calculation should be performed on. If not provided, default is DBZH."
  },
  {"filter", (PyCFunction) _pydetectionrange_filter, 1,
    "filter(scan) -> scan\n\n"
    "Filters out unwanted values. The provided scan should contain a HGHT parameter from the previous call to .top(). The returned value is a clone of provided scan with data filtered.\n\n"
    "scan - the scan with HGHT parameter as generated in the call to .top()"
  },
  {"analyze", (PyCFunction) _pydetectionrange_analyze, 1,
    "analyze(scan, avgsector, sortage, samplepoint) -> rave quality field\n\n"
    "Analyzes the detection ranges and returns a rave quality field with how/task set to se.smhi.detector.poo\n\n"
    "scan        - Scan that was retrieved from the call to filter()\n"
    "avgsector   - Width of the floating average azimuthal sector.\n"
    "sortage     - Defining the higher portion of sorted ray to be analysed, typically 0.05 - 0.2\n"
    "samplepoint - Define the position to pick a representative TOP value from highest\n"
    "              valid TOPs, typically near 0.5 (median) lower values (nearer to\n"
    "              highest TOP, 0.15) used in noisier radars like KOR."
  },
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the detection range
 * @param[in] self - the detection range product
 */
static PyObject* _pydetectionrange_getattro(PyDetectionRange* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("lookupPath", name) == 0) {
    return PyString_FromString(DetectionRange_getLookupPath(self->dr));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("analysis_minrange", name) == 0) {
    return PyFloat_FromDouble(DetectionRange_getAnalysisMinRange(self->dr));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("analysis_maxrange", name) == 0) {
    return PyFloat_FromDouble(DetectionRange_getAnalysisMaxRange(self->dr));
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the detection range
 */
static int _pydetectionrange_setattro(PyDetectionRange* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("lookupPath", name) == 0) {
    if (PyString_Check(val)) {
      if (!DetectionRange_setLookupPath(self->dr, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "lookupPath could not be set");
      }
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"lookupPath must be of type string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("analysis_minrange", name) == 0) {
    if (PyFloat_Check(val)) {
      DetectionRange_setAnalysisMinRange(self->dr, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      DetectionRange_setAnalysisMinRange(self->dr, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      DetectionRange_setAnalysisMinRange(self->dr, (double)PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "analysis_minrange must be a float or decimal value")
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("analysis_maxrange", name) == 0) {
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
    raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }

  result = 0;
done:
  return result;
}

/*@} End of detection range generator */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pydetectionrange_module_doc,
  "Provides an algorithm for calculating probability of overshooting. There are 3 member attributes that can be set:\n"
  " lookupPath        - The lookup path where the cache files are stored.\n"
  " analysis_minrange - Min radial range during the analysis stage in meters. Default is 10000.\n"
  " analysis_maxrange - Max radial range during the analysis stage in meters. Default is 240000.\n"
  "Usage:\n"
  " import _detectionrange\n"
  " generator = _detectionrange.new()\n"
  " pvol = _raveio.open(\"somepvol.h5\").object\n"
  " maxscan = pvol.getScanWithMaxDistance()\n"
  " top = generator.top(pvol, maxscan.rscale, -40.0)\n"
  " filtered = generator.filter(top)\n"
  " poofield = generator.analyze(filtered, 60.0, 0.1, 0.35)"
);
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyDetectionRange_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "DetectionRangeCore", /*tp_name*/
  sizeof(PyDetectionRange), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pydetectionrange_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pydetectionrange_getattro, /*tp_getattro*/
  (setattrofunc)_pydetectionrange_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pydetectionrange_module_doc, /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pydetectionrange_methods,              /*tp_methods*/
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
  {"new", (PyCFunction)_pydetectionrange_new, 1,
    "new() -> new instance of the DetectionRangeCore object\n\n"
    "Creates a new instance of the DetectionRangeCore object"
  },
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_detectionrange)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyDetectionRange_API[PyDetectionRange_API_pointers];
  PyObject *c_api_object = NULL;
  MOD_INIT_SETUP_TYPE(PyDetectionRange_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyDetectionRange_Type);

  MOD_INIT_DEF(module, "_detectionrange", _pydetectionrange_module_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyDetectionRange_API[PyDetectionRange_Type_NUM] = (void*)&PyDetectionRange_Type;
  PyDetectionRange_API[PyDetectionRange_GetNative_NUM] = (void *)PyDetectionRange_GetNative;
  PyDetectionRange_API[PyDetectionRange_New_NUM] = (void*)PyDetectionRange_New;

  c_api_object = PyCapsule_New(PyDetectionRange_API, PyDetectionRange_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_detectionrange.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _detectionrange.error");
    return MOD_INIT_ERROR;
  }

  import_pypolarvolume();
  import_pypolarscan();
  import_pyravefield();
  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
