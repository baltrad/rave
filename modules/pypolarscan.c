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
 * Python version of the PolarScan API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-08
 */
#include "Python.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYPOLARSCAN_MODULE   /**< to get correct part of pypolarscan,h */
#include "pypolarscan.h"

#include "pypolarscanparam.h"
#include "pyrave_debug.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_polarscan");

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

/// --------------------------------------------------------------------
/// Polar Scans
/// --------------------------------------------------------------------
/*@{ Polar Scans */
/**
 * Returns the native PolarScan_t instance.
 * @param[in] pypolarscan - the python polar scan instance
 * @returns the native polar scan instance.
 */
static PolarScan_t*
PyPolarScan_GetNative(PyPolarScan* pypolarscan)
{
  RAVE_ASSERT((pypolarscan != NULL), "pypolarscan == NULL");
  return RAVE_OBJECT_COPY(pypolarscan->scan);
}

/**
 * Creates a python polar scan from a native polar scan or will create an
 * initial native PolarScan if p is NULL.
 * @param[in] p - the native polar scan (or NULL)
 * @returns the python polar scan.
 */
static PyPolarScan* PyPolarScan_New(PolarScan_t* p)
{
  PyPolarScan* result = NULL;
  PolarScan_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&PolarScan_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for polar scan.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for polar scan.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyPolarScan, &PyPolarScan_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->scan = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->scan, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyPolarScan instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for polar scan.");
    }
  }
done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the polar scan
 * @param[in] obj the object to deallocate.
 */
static void _pypolarscan_dealloc(PyPolarScan* obj)
{
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->scan, obj);
  RAVE_OBJECT_RELEASE(obj->scan);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the polar scan.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pypolarscan_new(PyObject* self, PyObject* args)
{
  PyPolarScan* result = PyPolarScan_New(NULL);
  return (PyObject*)result;
}

static PyObject* _pypolarscan_addParameter(PyPolarScan* self, PyObject* args)
{
  PyObject* inptr = NULL;
  PyPolarScanParam* polarScanParam = NULL;

  if (!PyArg_ParseTuple(args, "O", &inptr)) {
    return NULL;
  }

  if (!PyPolarScanParam_Check(inptr)) {
    raiseException_returnNULL(PyExc_TypeError,"Added object must be of type PolarScanParamCore");
  }

  polarScanParam = (PyPolarScanParam*)inptr;

  if (!PolarScan_addParameter(self->scan, polarScanParam->scanparam)) {
    raiseException_returnNULL(PyExc_AttributeError, "Failed to add parameter to scan");
  }

  Py_RETURN_NONE;
}

static PyObject* _pypolarscan_removeParameter(PyPolarScan* self, PyObject* args)
{
  char* paramname = NULL;
  PolarScanParam_t* param = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "s", &paramname)) {
    return NULL;
  }

  if (!PolarScan_hasParameter(self->scan, paramname)) {
    Py_RETURN_NONE;
  }

  if((param = PolarScan_removeParameter(self->scan, paramname)) == NULL) {
    Py_RETURN_NONE;
  }

  if (param != NULL) {
    result = (PyObject*)PyPolarScanParam_New(param);
  }

  RAVE_OBJECT_RELEASE(param);

  return result;
}

static PyObject* _pypolarscan_getParameter(PyPolarScan* self, PyObject* args)
{
  char* paramname = NULL;
  PolarScanParam_t* param = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "s", &paramname)) {
    return NULL;
  }

  if (!PolarScan_hasParameter(self->scan, paramname)) {
    Py_RETURN_NONE;
  }

  if((param = PolarScan_getParameter(self->scan, paramname)) == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "Could not aquire parameter");
  }

  if (param != NULL) {
    result = (PyObject*)PyPolarScanParam_New(param);
  }

  RAVE_OBJECT_RELEASE(param);

  return result;
}

static PyObject* _pypolarscan_getParameterNames(PyPolarScan* self, PyObject* args)
{
  RaveList_t* paramnames = NULL;
  PyObject* result = NULL;
  int nparams = 0;
  int i = 0;
  paramnames = PolarScan_getParameterNames(self->scan);
  if (paramnames == NULL) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not get names");
  }
  nparams = RaveList_size(paramnames);
  result = PyList_New(0);
  for (i = 0; result != NULL && i < nparams; i++) {
    char* param = RaveList_get(paramnames, i);
    if (param != NULL) {
      PyObject* pyparamstr = PyString_FromString(param);
      if (pyparamstr == NULL) {
        goto fail;
      }
      if (PyList_Append(result, pyparamstr) != 0) {
        Py_DECREF(pyparamstr);
        goto fail;
      }
      Py_DECREF(pyparamstr);
    }
  }
  RaveList_freeAndDestroy(&paramnames);
  return result;
fail:
  RaveList_freeAndDestroy(&paramnames);
  Py_XDECREF(result);
  return NULL;
}

/**
 * Checks if the specified parameter name exists as a parameter
 * in the scan.
 * @param[in] self - self
 * @param[in] args - a tuple containing a python string
 * @returns Python true or false on success, otherwise NULL
 */
static PyObject* _pypolarscan_hasParameter(PyPolarScan* self, PyObject* args)
{
  char* paramname = NULL;

  if (!PyArg_ParseTuple(args, "s", &paramname)) {
    return NULL;
  }

  return PyBool_FromLong(PolarScan_hasParameter(self->scan, paramname));
}

/**
 * Calculates the azimuth index from an azimuth (in radians).
 * @param[in] self - this instance
 * @param[in] args - an azimuth value (in radians)
 * @returns the azimuth index or -1 if none could be determined.
 */
static PyObject* _pypolarscan_getAzimuthIndex(PyPolarScan* self, PyObject* args)
{
  double azimuth = 0.0L;
  int index = -1;

  if (!PyArg_ParseTuple(args, "d", &azimuth)) {
    return NULL;
  }

  index = PolarScan_getAzimuthIndex(self->scan, azimuth);
  if (index < 0) {
    raiseException_returnNULL(PyExc_ValueError, "Invalid azimuth");
  }

  return PyInt_FromLong(index);
}

/**
 * Calculates the range index from a specified range
 * @param[in] self - this instance
 * @param[in] args - the range (in meters)
 * @returns the range index or -1 if outside range
 */
static PyObject* _pypolarscan_getRangeIndex(PyPolarScan* self, PyObject* args)
{
  double range = 0.0L;
  int index = -1;

  if (!PyArg_ParseTuple(args, "d", &range)) {
    return NULL;
  }

  index = PolarScan_getRangeIndex(self->scan, range);

  return PyInt_FromLong(index);
}

/**
 * Returns the value at the specified ray and bin index.
 * @param[in] self - this instance
 * @param[in] args - bin index, ray index.
 * @returns a tuple of value type and value
 */
static PyObject* _pypolarscan_getValue(PyPolarScan* self, PyObject* args)
{
  double value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  int ray = 0, bin = 0;
  if (!PyArg_ParseTuple(args, "ii", &bin, &ray)) {
    return NULL;
  }

  type = PolarScan_getValue(self->scan, bin, ray, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Gets the parameter value for the provided bin and ray index.
 * @param[in] self - this instance
 * @param[in] args - quantity, bin index, ray index
 * @returns a tuple of value type and value
 */
static PyObject* _pypolarscan_getParameterValue(PyPolarScan* self, PyObject* args)
{
  double value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  int ray = 0, bin = 0;
  char* quantity = NULL;
  if (!PyArg_ParseTuple(args, "sii", &quantity, &bin, &ray)) {
    return NULL;
  }

  type = PolarScan_getParameterValue(self->scan, quantity, bin, ray, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Returns the converted value at the specified ray and bin index.
 * @param[in] self - this instance
 * @param[in] args - bin index, ray index.
 * @returns a tuple of value type and value
 */
static PyObject* _pypolarscan_getConvertedValue(PyPolarScan* self, PyObject* args)
{
  double value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  int ray = 0, bin = 0;
  if (!PyArg_ParseTuple(args, "ii", &bin, &ray)) {
    return NULL;
  }

  type = PolarScan_getConvertedValue(self->scan, bin, ray, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Returns the converted parameter value at the specified ray and bin index.
 * @param[in] self - this instance
 * @param[in] args - quantity, bin index, ray index.
 * @returns a tuple of value type and value
 */
static PyObject* _pypolarscan_getConvertedParameterValue(PyPolarScan* self, PyObject* args)
{
  double value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  int ray = 0, bin = 0;
  char* quantity = NULL;
  if (!PyArg_ParseTuple(args, "sii", &quantity, &bin, &ray)) {
    return NULL;
  }

  type = PolarScan_getConvertedParameterValue(self->scan, quantity, bin, ray, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Returns the value at the specified azimuth and range for this scan.
 * @param[in] self - this instance
 * @param[in] args - two doubles, azimuth (in radians) and range (in meters)
 * @returns a tuple of value type and value
 */
static PyObject* _pypolarscan_getValueAtAzimuthAndRange(PyPolarScan* self, PyObject* args)
{
  double value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  double a = 0, r = 0;
  if (!PyArg_ParseTuple(args, "dd", &a, &r)) {
    return NULL;
  }

  type = PolarScan_getValueAtAzimuthAndRange(self->scan, a, r, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Returns the value that is nearest to the specified longitude/latitude.
 * @param[in] self - this instance
 * @param[in] args - a tuple consisting of (longitude, latitude).
 * @returns a tuple of (value type, value) or NULL on failure
 */
static PyObject* _pypolarscan_getNearest(PyPolarScan* self, PyObject* args)
{
  double lon = 0.0L, lat = 0.0L, value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  if (!PyArg_ParseTuple(args, "(dd)", &lon, &lat)) {
    return NULL;
  }

  type = PolarScan_getNearest(self->scan, lon, lat, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Returns the nearest index for the provided longitude, latitude.
 * @param[in] self - this instance
 * @param[in] args - a tuple consisting of (longitude, latitude).
 * @returns a tuple of (bin index, ray index) or None if either bin index or ray index were oob
 */
static PyObject* _pypolarscan_getNearestIndex(PyPolarScan* self, PyObject* args)
{
  double lon = 0.0L, lat = 0.0L;
  int bin = 0, ray = 0;
  if (!PyArg_ParseTuple(args, "(dd)", &lon, &lat)) {
    return NULL;
  }

  if (PolarScan_getNearestIndex(self->scan, lon, lat, &bin, &ray) != 0) {
    return Py_BuildValue("(ii)",bin,ray);
  }

  Py_RETURN_NONE;
}

/**
 * All methods a polar scan can have
 */
static struct PyMethodDef _pypolarscan_methods[] =
{
  {"addParameter", (PyCFunction) _pypolarscan_addParameter, 1},
  {"removeParameter", (PyCFunction) _pypolarscan_removeParameter, 1},
  {"getParameter", (PyCFunction) _pypolarscan_getParameter, 1},
  {"getParameterNames", (PyCFunction) _pypolarscan_getParameterNames, 1},
  {"hasParameter", (PyCFunction) _pypolarscan_hasParameter, 1},
  {"getAzimuthIndex", (PyCFunction) _pypolarscan_getAzimuthIndex, 1},
  {"getRangeIndex", (PyCFunction) _pypolarscan_getRangeIndex, 1},
  {"getValue", (PyCFunction) _pypolarscan_getValue, 1},
  {"getParameterValue", (PyCFunction) _pypolarscan_getParameterValue, 1},
  {"getConvertedValue", (PyCFunction) _pypolarscan_getConvertedValue, 1},
  {"getConvertedParameterValue", (PyCFunction) _pypolarscan_getConvertedParameterValue, 1},
  {"getValueAtAzimuthAndRange", (PyCFunction) _pypolarscan_getValueAtAzimuthAndRange, 1},
  {"getNearest", (PyCFunction) _pypolarscan_getNearest, 1},
  {"getNearestIndex", (PyCFunction) _pypolarscan_getNearestIndex, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the polar scan
 */
static PyObject* _pypolarscan_getattr(PyPolarScan* self, char* name)
{
  PyObject* res;
  if (strcmp("elangle", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getElangle(self->scan));
  } else if (strcmp("nbins", name) == 0) {
    return PyInt_FromLong(PolarScan_getNbins(self->scan));
  } else if (strcmp("rscale", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getRscale(self->scan));
  } else if (strcmp("nrays", name) == 0) {
    return PyInt_FromLong(PolarScan_getNrays(self->scan));
  } else if (strcmp("rstart", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getRstart(self->scan));
  } else if (strcmp("a1gate", name) == 0) {
    return PyInt_FromLong(PolarScan_getA1gate(self->scan));
  } else if (strcmp("datatype", name) == 0) {
    return PyInt_FromLong(PolarScan_getDataType(self->scan));
  } else if (strcmp("beamwidth", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getBeamWidth(self->scan));
  } else if (strcmp("longitude", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getLongitude(self->scan));
  } else if (strcmp("latitude", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getLatitude(self->scan));
  } else if (strcmp("height", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getHeight(self->scan));
  } else if (strcmp("time", name) == 0) {
    const char* str = PolarScan_getTime(self->scan);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("date", name) == 0) {
    const char* str = PolarScan_getDate(self->scan);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("source", name) == 0) {
    const char* str = PolarScan_getSource(self->scan);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("defaultparameter", name) == 0) {
    const char* str = PolarScan_getDefaultParameter(self->scan);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  }
  res = Py_FindMethod(_pypolarscan_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pypolarscan_setattr(PyPolarScan* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("elangle", name)==0) {
    if (PyFloat_Check(val)) {
      PolarScan_setElangle(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"elangle must be of type float");
    }
  } else if (strcmp("rscale", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setRscale(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "rscale must be of type float");
    }
  } else if (strcmp("rstart", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setRstart(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "rstart must be of type float");
    }
  } else if (strcmp("a1gate", name) == 0) {
    if (PyInt_Check(val)) {
      PolarScan_setA1gate(self->scan, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"a1gate must be of type int");
    }
  } else if (strcmp("beamwidth", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setBeamWidth(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "beamwidth must be of type float");
    }
  } else if (strcmp("longitude", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setLongitude(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "longitude must be of type float");
    }
  } else if (strcmp("latitude", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setLatitude(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "latitude must be of type float");
    }
  } else if (strcmp("height", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setHeight(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "height must be of type float");
    }
  } else if (strcmp("time", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarScan_setTime(self->scan, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "time must be a string (HHmmss)");
      }
    } else if (val == Py_None) {
      PolarScan_setTime(self->scan, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "time must be a string (HHmmss)");
    }
  } else if (strcmp("date", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarScan_setDate(self->scan, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "date must be a string (YYYYMMSS)");
      }
    } else if (val == Py_None) {
      PolarScan_setDate(self->scan, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "date must be a string (YYYYMMSS)");
    }
  } else if (strcmp("source", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarScan_setSource(self->scan, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "source must be a string");
      }
    } else if (val == Py_None) {
      PolarScan_setSource(self->scan, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "source must be a string");
    }
  } else if (strcmp("defaultparameter", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarScan_setDefaultParameter(self->scan, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "source must be a string");
      }
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "source must be a string");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, name);
  }

  result = 0;
done:
  return result;
}
/*@} End of Polar Scans */

/// --------------------------------------------------------------------
/// Type definitions
/// --------------------------------------------------------------------
/*@{ Type definition */
PyTypeObject PyPolarScan_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "PolarScanCore", /*tp_name*/
  sizeof(PyPolarScan), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pypolarscan_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pypolarscan_getattr, /*tp_getattr*/
  (setattrfunc)_pypolarscan_setattr, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0,
  0, /*tp_as_mapping */
  0 /*tp_hash*/
};
/*@} End of Type definition */

/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pypolarscan_new, 1},
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_polarscan(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyPolarScan_API[PyPolarScan_API_pointers];
  PyObject *c_api_object = NULL;
  PyPolarScan_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_polarscan", functions);
  if (module == NULL) {
    return;
  }
  PyPolarScan_API[PyPolarScan_Type_NUM] = (void*)&PyPolarScan_Type;
  PyPolarScan_API[PyPolarScan_GetNative_NUM] = (void *)PyPolarScan_GetNative;
  PyPolarScan_API[PyPolarScan_New_NUM] = (void*)PyPolarScan_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyPolarScan_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_polarscan.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _polarscan.error");
  }

  import_pypolarscanparam();
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
