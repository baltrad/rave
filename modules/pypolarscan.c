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
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <float.h>

#define PYPOLARSCAN_MODULE   /**< to get correct part of pypolarscan,h */
#include "pypolarscan.h"

#include <arrayobject.h>
#include "pypolarscanparam.h"
#include "pyprojection.h"
#include "pyrave_debug.h"
#include "pyravefield.h"
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

static PyObject* _pypolarscan_clone(PyPolarScan* self, PyObject* args)
{
  PolarScan_t* cpy = NULL;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  cpy = RAVE_OBJECT_CLONE(self->scan);
  if (cpy == NULL) {
    raiseException_returnNULL(PyExc_RuntimeError, "Failed to clone scan");
  }
  result = (PyObject*)PyPolarScan_New(cpy);

  RAVE_OBJECT_RELEASE(cpy);

  return result;
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

static PyObject* _pypolarscan_removeAllParameters(PyPolarScan* self, PyObject* args)
{
  if (!PolarScan_removeAllParameters(self->scan)) {
    raiseException_returnNULL(PyExc_MemoryError, "Failed to remove all parameters");
  }
  Py_RETURN_NONE;
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

  index = PolarScan_getAzimuthIndex(self->scan, azimuth, PolarScanSelectionMethod_ROUND);
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

  index = PolarScan_getRangeIndex(self->scan, range, PolarScanSelectionMethod_FLOOR, 0);

  return PyInt_FromLong(index);
}

/**
 * Calculates the range from a specific range index
 * @param[in] self - this instance
 * @param[in] args - the range index as an integer
 * @return the range or a negative value if outside boundaries
 */
static PyObject* _pypolarscan_getRange(PyPolarScan* self, PyObject* args)
{
  double range = 0.0L;
  int index = -1;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }
  range = PolarScan_getRange(self->scan, index, 0);

  return PyFloat_FromDouble(range);
}

/**
 * Sets the value at the specified ray and bin index.
 * @param[in] self - this instance
 * @param[in] args - bin index, ray index.
 * @returns None on success otherwise NULL
 */
static PyObject* _pypolarscan_setValue(PyPolarScan* self, PyObject* args)
{
  double value = 0.0L;
  int ray = 0, bin = 0;
  if (!PyArg_ParseTuple(args, "(ii)d", &bin, &ray, &value)) {
    return NULL;
  }

  if (!PolarScan_setValue(self->scan, bin, ray, value)) {
    raiseException_returnNULL(PyExc_ValueError, "Could not set value");
  }

  Py_RETURN_NONE;
}

/**
 * Sets the parameter value for the provided bin and ray index.
 * @param[in] self - this instance
 * @param[in] args - quantity, bin index, ray index
 * @returns None on success otherwise NULL
 */
static PyObject* _pypolarscan_setParameterValue(PyPolarScan* self, PyObject* args)
{
  double value = 0.0L;
  int ray = 0, bin = 0;
  char* quantity = NULL;
  if (!PyArg_ParseTuple(args, "s(ii)d", &quantity, &bin, &ray, &value)) {
    return NULL;
  }

  if (!PolarScan_setParameterValue(self->scan, quantity, bin, ray, value)) {
    raiseException_returnNULL(PyExc_ValueError, "Could not set parameter value");
  }

  Py_RETURN_NONE;
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
 * Calculates the bin and ray index from a azimuth and range.
 * @param[in] self - self
 * @param[in] args - a python object containing (azimuth (in radians), range (in meters))
 * @returns a tuple (ray index, bin index) or None if outside boundaries
 */
static PyObject* _pypolarscan_getIndexFromAzimuthAndRange(PyPolarScan* self, PyObject* args)
{
  double a = 0.0L, r = 0.0L;
  int ray = -1, bin = -1;
  if (!PyArg_ParseTuple(args, "dd", &a, &r)) {
    return NULL;
  }

  if (PolarScan_getIndexFromAzimuthAndRange(self->scan, a, r, PolarScanSelectionMethod_ROUND, PolarScanSelectionMethod_FLOOR, 0, &ray, &bin)) {
    return Py_BuildValue("(ii)",ray,bin);
  }

  Py_RETURN_NONE;
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

  type = PolarScan_getValueAtAzimuthAndRange(self->scan, a, r, 0, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Returns the parameter value at the specified azimuth and range for this scan.
 * @param[in] self - this instance
 * @param[in] args - quantity, azimuth (in radians) and range (in meters)
 * @returns a tuple of value type and value
 */
static PyObject* _pypolarscan_getParameterValueAtAzimuthAndRange(PyPolarScan* self, PyObject* args)
{
  double value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  double a = 0, r = 0;
  char* quantity = NULL;
  if (!PyArg_ParseTuple(args, "sdd", &quantity, &a, &r)) {
    return NULL;
  }

  type = PolarScan_getParameterValueAtAzimuthAndRange(self->scan, quantity, a, r, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Returns the converted parameter value at the specified azimuth and range for this scan.
 * @param[in] self - this instance
 * @param[in] args - quantity, azimuth (in radians) and range (in meters)
 * @returns a tuple of value type and value
 */
static PyObject* _pypolarscan_getConvertedParameterValueAtAzimuthAndRange(PyPolarScan* self, PyObject* args)
{
  double value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  double a = 0, r = 0;
  char* quantity = NULL;
  if (!PyArg_ParseTuple(args, "sdd", &quantity, &a, &r)) {
    return NULL;
  }

  type = PolarScan_getConvertedParameterValueAtAzimuthAndRange(self->scan, quantity, a, r, &value);

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

  type = PolarScan_getNearest(self->scan, lon, lat, 0, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Returns the parameter value that is nearest to the specified longitude/latitude.
 * @param[in] self - this instance
 * @param[in] args - a tuple consisting of (longitude, latitude).
 * @returns a tuple of (value type, value) or NULL on failure
 */
static PyObject* _pypolarscan_getNearestParameterValue(PyPolarScan* self, PyObject* args)
{
  double lon = 0.0L, lat = 0.0L, value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  char* quantity = NULL;
  if (!PyArg_ParseTuple(args, "s(dd)", &quantity, &lon, &lat)) {
    return NULL;
  }

  type = PolarScan_getNearestParameterValue(self->scan, quantity, lon, lat, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Returns the converted parameter value that is nearest to the specified longitude/latitude.
 * @param[in] self - this instance
 * @param[in] args - a tuple consisting of (longitude, latitude).
 * @returns a tuple of (value type, value) or NULL on failure
 */
static PyObject* _pypolarscan_getNearestConvertedParameterValue(PyPolarScan* self, PyObject* args)
{
  double lon = 0.0L, lat = 0.0L, value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  char* quantity = NULL;

  if (!PyArg_ParseTuple(args, "s(dd)", &quantity, &lon, &lat)) {
    return NULL;
  }

  type = PolarScan_getNearestConvertedParameterValue(self->scan, quantity, lon, lat, &value, NULL);

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
 * Calculates the lon/lat from a specified ray/bin index.
 * @param[in] self - self
 * @param[in] args - tuple of two integers (bin, ray)
 * @returns a tuple of double (lon, lat) in radians
 */
static PyObject* _pypolarscan_getLonLatFromIndex(PyPolarScan* self, PyObject* args)
{
  double lon = 0.0L, lat = 0.0L;
  int bin = 0, ray = 0;
  if (!PyArg_ParseTuple(args, "ii", &bin, &ray)) {
    return NULL;
  }

  if (PolarScan_getLonLatFromIndex(self->scan, bin, ray, &lon, &lat) != 0) {
    return Py_BuildValue("(dd)",lon,lat);
  }

  Py_RETURN_NONE;
}

/**
 * Adds an attribute to the parameter. Name of the attribute should be in format
 * ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis etc.
 * Currently, the only supported values are double, long, string.
 * @param[in] self - this instance
 * @param[in] args - bin index, ray index.
 * @returns true or false depending if it works.
 */
static PyObject* _pypolarscan_addAttribute(PyPolarScan* self, PyObject* args)
{
  RaveAttribute_t* attr = NULL;
  char* name = NULL;
  PyObject* obj = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "sO", &name, &obj)) {
    return NULL;
  }

  attr = RAVE_OBJECT_NEW(&RaveAttribute_TYPE);
  if (attr == NULL) {
    return NULL;
  }

  if (!RaveAttribute_setName(attr, name)) {
    raiseException_gotoTag(done, PyExc_MemoryError, "Failed to set name");
  }

  if (PyLong_Check(obj) || PyInt_Check(obj)) {
    long value = PyLong_AsLong(obj);
    RaveAttribute_setLong(attr, value);
  } else if (PyFloat_Check(obj)) {
    double value = PyFloat_AsDouble(obj);
    RaveAttribute_setDouble(attr, value);
  } else if (PyString_Check(obj)) {
    char* value = PyString_AsString(obj);
    if (!RaveAttribute_setString(attr, value)) {
      raiseException_gotoTag(done, PyExc_AttributeError, "Failed to set string value");
    }
  } else if (PyArray_Check(obj)) {
    PyArrayObject* arraydata = (PyArrayObject*)obj;
    if (PyArray_NDIM(arraydata) != 1) {
      raiseException_gotoTag(done, PyExc_AttributeError, "Only allowed attribute arrays are 1-dimensional");
    }
    if (!RaveAttribute_setArrayFromData(attr, PyArray_DATA(arraydata), PyArray_DIM(arraydata, 0), translate_pyarraytype_to_ravetype(PyArray_TYPE(arraydata)))) {
      raiseException_gotoTag(done, PyExc_AttributeError, "Failed to set array data");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, "Unsupported data type");
  }

  if (!PolarScan_addAttribute(self->scan, attr)) {
    raiseException_gotoTag(done, PyExc_AttributeError, "Failed to add attribute");
  }

  result = PyBool_FromLong(1);
done:
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

/**
 * Returns an attribute with the specified name
 * @param[in] self - this instance
 * @param[in] args - name
 * @returns the attribute value for the name
 */
static PyObject* _pypolarscan_getAttribute(PyPolarScan* self, PyObject* args)
{
  RaveAttribute_t* attribute = NULL;
  char* name = NULL;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }

  attribute = PolarScan_getAttribute(self->scan, name);
  if (attribute != NULL) {
    RaveAttribute_Format format = RaveAttribute_getFormat(attribute);
    if (format == RaveAttribute_Format_Long) {
      long value = 0;
      RaveAttribute_getLong(attribute, &value);
      result = PyLong_FromLong(value);
    } else if (format == RaveAttribute_Format_Double) {
      double value = 0.0;
      RaveAttribute_getDouble(attribute, &value);
      result = PyFloat_FromDouble(value);
    } else if (format == RaveAttribute_Format_String) {
      char* value = NULL;
      RaveAttribute_getString(attribute, &value);
      result = PyString_FromString(value);
    } else if (format == RaveAttribute_Format_LongArray) {
      long* value = NULL;
      int len = 0;
      int i = 0;
      npy_intp dims[1];
      RaveAttribute_getLongArray(attribute, &value, &len);
      dims[0] = len;
      result = PyArray_SimpleNew(1, dims, PyArray_LONG);
      for (i = 0; i < len; i++) {
        *((long*) PyArray_GETPTR1(result, i)) = value[i];
      }
    } else if (format == RaveAttribute_Format_DoubleArray) {
      double* value = NULL;
      int len = 0;
      int i = 0;
      npy_intp dims[1];
      RaveAttribute_getDoubleArray(attribute, &value, &len);
      dims[0] = len;
      result = PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
      for (i = 0; i < len; i++) {
        *((double*) PyArray_GETPTR1(result, i)) = value[i];
      }
    } else {
      RAVE_CRITICAL1("Undefined format on requested attribute %s", name);
      raiseException_gotoTag(done, PyExc_AttributeError, "Undefined attribute");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, "No such attribute");
  }
done:
  RAVE_OBJECT_RELEASE(attribute);
  return result;
}

/**
 * Returns if there exists an attribute with the specified name
 * @param[in] self - this instance
 * @param[in] args - name
 * @returns True if attribute exists otherwise False
 */
static PyObject* _pypolarscan_hasAttribute(PyPolarScan* self, PyObject* args)
{
  char* name = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  return PyBool_FromLong((long)PolarScan_hasAttribute(self->scan, name));
}

/**
 * Returns a list of attribute names
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns a list of attribute names
 */
static PyObject* _pypolarscan_getAttributeNames(PyPolarScan* self, PyObject* args)
{
  RaveList_t* list = NULL;
  PyObject* result = NULL;
  int n = 0;
  int i = 0;

  list = PolarScan_getAttributeNames(self->scan);
  if (list == NULL) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not get attribute names");
  }
  n = RaveList_size(list);
  result = PyList_New(0);
  for (i = 0; result != NULL && i < n; i++) {
    char* name = RaveList_get(list, i);
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
  RaveList_freeAndDestroy(&list);
  return result;
fail:
  RaveList_freeAndDestroy(&list);
  Py_XDECREF(result);
  return NULL;
}

/**
 * Returns if the scan is valid or not (for storage purposes)
 * @param[in] self - this instance
 * @param[in] args - rave object type as integer
 * @returns true or false
 */
static PyObject* _pypolarscan_isValid(PyPolarScan* self, PyObject* args)
{
  Rave_ObjectType otype = Rave_ObjectType_UNDEFINED;

  if (!PyArg_ParseTuple(args, "i", &otype)) {
    return NULL;
  }

  return PyBool_FromLong(PolarScan_isValid(self->scan, otype));
}

/**
 * Adds a quality field to the scan
 * @param[in] self - this instance
 * @param[in] args - object, of type RaveFieldCore
 * @returns None
 */
static PyObject* _pypolarscan_addQualityField(PyPolarScan* self, PyObject* args)
{
  PyObject* inptr = NULL;
  PyRaveField* ravefield = NULL;

  if (!PyArg_ParseTuple(args, "O", &inptr)) {
    return NULL;
  }

  if (!PyRaveField_Check(inptr)) {
    raiseException_returnNULL(PyExc_TypeError,"Added object must be of type RaveFieldCore");
  }

  ravefield = (PyRaveField*)inptr;

  if (!PolarScan_addQualityField(self->scan, ravefield->field)) {
    raiseException_returnNULL(PyExc_AttributeError, "Failed to add quality field to scan");
  }

  Py_RETURN_NONE;
}

/**
 * Adds or replace quality field to the scan
 * @param[in] self - this instance
 * @param[in] args - object, of type RaveFieldCore
 * @returns None
 */
static PyObject* _pypolarscan_addOrReplaceQualityField(PyPolarScan* self, PyObject* args)
{
  PyObject* inptr = NULL;
  PyRaveField* ravefield = NULL;

  if (!PyArg_ParseTuple(args, "O", &inptr)) {
    return NULL;
  }

  if (!PyRaveField_Check(inptr)) {
    raiseException_returnNULL(PyExc_TypeError,"Added object must be of type RaveFieldCore");
  }

  ravefield = (PyRaveField*)inptr;

  if (!PolarScan_addOrReplaceQualityField(self->scan, ravefield->field)) {
    raiseException_returnNULL(PyExc_AttributeError, "Failed to add quality field to scan");
  }

  Py_RETURN_NONE;
}

/**
 * Returns the number of quality fields
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns The number of quality fields
 */
static PyObject* _pypolarscan_getNumberOfQualityFields(PyPolarScan* self, PyObject* args)
{
  return PyInt_FromLong(PolarScan_getNumberOfQualityFields(self->scan));
}

/**
 * Returns the number of quality fields
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns The number of quality fields
 */
static PyObject* _pypolarscan_getQualityField(PyPolarScan* self, PyObject* args)
{
  PyObject* result = NULL;
  RaveField_t* field = NULL;
  int index = 0;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  if (index < 0 || index >= PolarScan_getNumberOfQualityFields(self->scan)) {
    raiseException_returnNULL(PyExc_IndexError, "Index out of bounds");
  }

  if ((field = PolarScan_getQualityField(self->scan, index)) == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "Could not get quality field");
  }

  result = (PyObject*)PyRaveField_New(field);

  RAVE_OBJECT_RELEASE(field);

  return result;
}

/**
 * Removes the specified quality field from the scan
 * @param[in] self - this instance
 * @param[in] args - the index of the field to be removed
 * @returns None
 */
static PyObject* _pypolarscan_removeQualityField(PyPolarScan* self, PyObject* args)
{
  int index = 0;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  PolarScan_removeQualityField(self->scan, index);

  Py_RETURN_NONE;
}

/**
 * Returns a quality field based on the value of how/task that should be a
 * string.
 * @param[in] self - self
 * @param[in] args - the how/task value string
 * @return the field if found otherwise NULL
 */
static PyObject* _pypolarscan_getQualityFieldByHowTask(PyPolarScan* self, PyObject* args)
{
  PyObject* result = NULL;
  char* value = NULL;
  RaveField_t* field = NULL;

  if (!PyArg_ParseTuple(args, "s", &value)) {
    return NULL;
  }
  field = PolarScan_getQualityFieldByHowTask(self->scan, value);
  if (field == NULL) {
    raiseException_gotoTag(done, PyExc_NameError, "Could not locate quality field");
  }
  result = (PyObject*)PyRaveField_New(field);
done:
  RAVE_OBJECT_RELEASE(field);
  return result;
}

/**
 * Atempts to locate a quality field with how/task = value. First it will
 * check in the default parameter, then it will check the scan it self.
 * @param[in] self - self
 * @param[in] args - the how/task value string
 * @return the field if found, otherwise NULL
 */
static PyObject* _pypolarscan_findQualityFieldByHowTask(PyPolarScan* self, PyObject* args)
{
  PyObject* result = NULL;
  char* value = NULL;
  char* quantity = NULL;
  RaveField_t* field = NULL;

  if (!PyArg_ParseTuple(args, "s|s", &value, &quantity)) {
    return NULL;
  }
  field = PolarScan_findQualityFieldByHowTask(self->scan, value, quantity);
  if (field != NULL) {
    result = (PyObject*)PyRaveField_New(field);
  }

  RAVE_OBJECT_RELEASE(field);

  if (result != NULL) {
    return result;
  } else {
    Py_RETURN_NONE;
  }
}

static PyObject* _pypolarscan_getDistanceField(PyPolarScan* self, PyObject* args)
{
  PyObject* result = NULL;
  RaveField_t* field = NULL;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  field = PolarScan_getDistanceField(self->scan);
  if (field != NULL) {
    result = (PyObject*)PyRaveField_New(field);
  }

  RAVE_OBJECT_RELEASE(field);
  if (result == NULL) {
    raiseException_returnNULL(PyExc_RuntimeError, "Could not create distance field");
  }
  return result;
}

static PyObject* _pypolarscan_getHeightField(PyPolarScan* self, PyObject* args)
{
  PyObject* result = NULL;
  RaveField_t* field = NULL;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  field = PolarScan_getHeightField(self->scan);
  if (field != NULL) {
    result = (PyObject*)PyRaveField_New(field);
  }

  RAVE_OBJECT_RELEASE(field);
  if (result == NULL) {
    raiseException_returnNULL(PyExc_RuntimeError, "Could not create height field");
  }
  return result;
}

static PyObject* _pypolarscan_getMaxDistance(PyPolarScan* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyFloat_FromDouble(PolarScan_getMaxDistance(self->scan));
}

static PyObject* _pypolarscan_getDistance(PyPolarScan* self, PyObject* args)
{
  double lon = 0.0, lat = 0.0;
  if (!PyArg_ParseTuple(args, "(dd)", &lon, &lat)) {
    return NULL;
  }
  return PyFloat_FromDouble(PolarScan_getDistance(self->scan, lon, lat));
}

/**
 * All methods a polar scan can have
 */
static struct PyMethodDef _pypolarscan_methods[] =
{
  {"elangle", NULL},
  {"nbins", NULL},
  {"rscale", NULL},
  {"nrays", NULL},
  {"rstart", NULL},
  {"a1gate", NULL},
  {"datatype", NULL},
  {"beamwidth", NULL},
  {"longitude", NULL},
  {"latitude", NULL},
  {"height", NULL},
  {"time", NULL},
  {"date", NULL},
  {"starttime", NULL},
  {"startdate", NULL},
  {"endtime", NULL},
  {"enddate", NULL},
  {"source", NULL},
  {"defaultparameter", NULL},
  {"projection", NULL},
  {"addParameter", (PyCFunction) _pypolarscan_addParameter, 1,
    "addParameter(param)\n\n"
    "Adds a parameter with it's quantity set to this scan. The first time a parameter is added to a scan, it will get some basic properties (nrays, nbins, ...) that all "
    "subsequent parameters has to match.\n"
    "param - the polar scan param of type PolarScanParamCore."
  },
  {"removeParameter", (PyCFunction) _pypolarscan_removeParameter, 1,
    "removeParameter(quant)\n\n"
    "Removes the parameter with the specified quantity.\n\n"
    "quant - the quantity of the parameter to be removed."
  },
  {"removeAllParameters", (PyCFunction) _pypolarscan_removeAllParameters, 1,
    "removeAllParameters()\n\n"
    "Removes all parameters from this scan."
  },
  {"getParameter", (PyCFunction) _pypolarscan_getParameter, 1,
    "getParameter(quant) -> polar scan parameter\n\n"
    "Returns the parameter with specified quantity. If it doesn't exist, None will be returned.\n\n"
    "quant - the quantity of the parameter to be returned"
  },
  {"getParameterNames", (PyCFunction) _pypolarscan_getParameterNames, 1,
    "getParameterNames() -> list of strings\n\n"
    "Return all parameters that has been set in this scan."
  },
  {"hasParameter", (PyCFunction) _pypolarscan_hasParameter, 1,
    "hasParameter(quant) -> a boolean\n\n"
    "Returns True or False depending if this scan has a parameter with specified quantity or not.\n\n"
    "quant - the quantity to be queried for"
  },
  {"getAzimuthIndex", (PyCFunction) _pypolarscan_getAzimuthIndex, 1,
    "getAzimuthIndex(aziumth) -> index\n\n"
    "Calculates the azimuth index from an azimuth (in radians).\n\n"
    "azimuth - azimuth in radians"
  },
  {"getRangeIndex", (PyCFunction) _pypolarscan_getRangeIndex, 1,
    "getRangeIndex(range) -> index\n\n"
    "Calculates the range index from a specified range.\n\n"
    "range - the range in meter along the ray"
  },
  {"getRange", (PyCFunction) _pypolarscan_getRange, 1,
    "getRange(index) -> range in meters\n\n"
    "Calculates the range from a specific range index. Will return the range or a negative value if index is out of bounds.\n"
    "index - the index in the ray"
  },
  {"setValue", (PyCFunction) _pypolarscan_setValue, 1,
    "setValue(bin, ray, value)\n\n"
    "Sets the value at the specified ray and bin index.\n\n"
    "bin    - the bin index\n"
    "ray    - the ray index\n"
    "value  - the value to set"
  },
  {"setParameterValue", (PyCFunction) _pypolarscan_setParameterValue, 1,
    "setValue(quantity, bin, ray, value)\n\n"
    "Sets the value for the parameter with quantity at the specified ray and bin index.\n\n"
    "quantity - the parameter quantity"
    "bin      - the bin index\n"
    "ray      - the ray index\n"
    "value    - the value to set\n\n"
    "Throws ValueError if there is no such quantity"
  },
  {"getValue", (PyCFunction) _pypolarscan_getValue, 1,
    "getValue(bin,ray) -> tuple (type, value)\n\n"
    "Returns the value at the specified ray and bin index for the default parameter. The value type can be one of the value types defined as _rave.RaveValueType_XXXX\n\n"
    "bin - bin index\n"
    "ray - ray index\n"
  },
  {"getParameterValue", (PyCFunction) _pypolarscan_getParameterValue, 1,
    "getParameterValue(quantity, bin,ray) -> (type, value)\n\n"
    "Returns the value for the specified parameter (quantity) at the specified ray and bin index. The value type can be one of the value types defined as _rave.RaveValueType_XXXX\n\n"
    "quantity - the quantity for the parameter"
    "bin      - bin index\n"
    "ray      - ray index\n"
  },
  {"getConvertedValue", (PyCFunction) _pypolarscan_getConvertedValue, 1,
    "getConvertedValue(bin,ray) -> (type, value)\n\n"
    "Returns the converted value for the default parameter at the specified ray and bin index. The value is evaluated as offset + v*gain. The value type can be one of the value types defined as _rave.RaveValueType_XXXX\n\n"
    "bin      - bin index\n"
    "ray      - ray index\n"
  },
  {"getConvertedParameterValue", (PyCFunction) _pypolarscan_getConvertedParameterValue, 1,
    "getConvertedParameterValue(quantity,bin,ray) -> (type, value)\n\n"
    "Returns the converted value for the default parameter at the specified ray and bin index. The value is evaluated as offset + v*gain. The value type can be one of the value types defined as _rave.RaveValueType_XXXX\n\n"
    "quantity - the parameter quantity"
    "bin      - bin index\n"
    "ray      - ray index\n"
  },
  {"getIndexFromAzimuthAndRange", (PyCFunction) _pypolarscan_getIndexFromAzimuthAndRange, 1,
    "getIndexFromAzimuthAndRange(azimuth, range) -> (ray index, bin index)\n\n"
    "Calculates the bin and ray index from a azimuth and range.\n\n"
    "azimuth - azimuth in radians\n"
    "range   - range in meters"
  },
  {"getValueAtAzimuthAndRange", (PyCFunction) _pypolarscan_getValueAtAzimuthAndRange, 1,
    "getValueAtAzimuthAndRange(azimuth, range) -> (type, value)\n\n"
    "Returns the value for the default parameter at the specified azimuth and range for this scan.\n\n"
      "azimuth - azimuth in radians\n"
      "range   - range in meters"
  },
  {"getParameterValueAtAzimuthAndRange", (PyCFunction) _pypolarscan_getParameterValueAtAzimuthAndRange, 1,
    "getParameterValueAtAzimuthAndRange(quantity, azimuth, range) -> (type, value)\n\n"
    "Returns the value for the parameter as defined by quantity at the specified azimuth and range for this scan.\n\n"
      "azimuth - azimuth in radians\n"
      "range   - range in meters"
  },
  {"getConvertedParameterValueAtAzimuthAndRange", (PyCFunction) _pypolarscan_getConvertedParameterValueAtAzimuthAndRange, 1,
    "getConvertedParameterValueAtAzimuthAndRange(quantity, azimuth, range) -> (type, value)\n\n"
    "Returns the converted value (offset+gain*value) for the parameter as defined by quantity at the specified azimuth and range for this scan.\n\n"
    "quantity  - the parameter quantity\n"
    "azimuth   - azimuth in radians\n"
    "range     - range in meters"
  },
  {"getNearest", (PyCFunction) _pypolarscan_getNearest, 1,
    "getNearest((lon, lat)) -> (type, value)\n\n"
    "Returns the default parameters value that is nearest to the specified longitude/latitude.\n\n"
    "lon - longitude in radians\n"
    "lat - latitude in radians"
  },
  {"getNearestParameterValue", (PyCFunction) _pypolarscan_getNearestParameterValue, 1,
    "getNearestParameterValue(quantity, (lon, lat)) -> (type, value)\n\n"
    "Returns the specified parameters value that is nearest to the specified longitude/latitude.\n\n"
    "quantity - the parameter quantity"
    "lon      - longitude in radians\n"
    "lat      - latitude in radians"
  },
  {"getNearestConvertedParameterValue", (PyCFunction) _pypolarscan_getNearestConvertedParameterValue, 1,
    "getNearestConvertedParameterValue(quantity, (lon, lat)) -> (type, value)\n\n"
    "Returns the converted value (offset+gain*value) for the specified parameters value that is nearest to the specified longitude/latitude.\n\n"
    "quantity - the parameter quantity"
    "lon      - longitude in radians\n"
    "lat      - latitude in radians"
  },
  {"getNearestIndex", (PyCFunction) _pypolarscan_getNearestIndex, 1,
    "getNearestIndex((lon, lat)) -> (bin, ray)\n\n"
    "Returns the nearest index for the provided longitude, latitude.\n\n"
    "lon      - longitude in radians\n"
    "lat      - latitude in radians"
  },
  {"getLonLatFromIndex", (PyCFunction) _pypolarscan_getLonLatFromIndex, 1,
    "getLonLatFromIndex(bin,ray)) -> (lon, lat)\n\n"
    "Calculates the lon/lat from a specified ray/bin index.\n\n"
    "bin      - The bin index\n"
    "ray      - The ray index"
  },
  {"addAttribute", (PyCFunction) _pypolarscan_addAttribute, 1,
    "addAttribute(name, value) \n\n"
    "Adds an attribute to the scan. Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis etc. \n"
    "Currently, double, long, string and 1-dimensional arrays are supported.\n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr.\n"
    "value - Value to be associated with the name. Currently, double, long, string and 1-dimensional arrays are supported."
  },
  {"getAttribute", (PyCFunction) _pypolarscan_getAttribute, 1,
    "getAttribute(name) -> value \n\n"
    "Returns the value associated with the specified name \n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr."
  },
  {"hasAttribute", (PyCFunction) _pypolarscan_hasAttribute, 1,
    "hasAttribute(name) -> a boolean \n\n"
    "Returns if the specified name is defined within this polar scan\n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis.\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr.\n"
  },
  {"getAttributeNames", (PyCFunction) _pypolarscan_getAttributeNames, 1,
    "getAttributeNames() -> array of names \n\n"
    "Returns the attribute names associated with this scan"
  },
  {"isValid", (PyCFunction) _pypolarscan_isValid, 1,
    "isValid(otype) -> a boolean \n\n"
    "Validates this polar scan object to see if it is possible to write as specified type.\n\n"
    "otype  - The type we want to save as, can be one of ObjectType_PVOL or ObjectType_SCAN."
  },
  {"addQualityField", (PyCFunction) _pypolarscan_addQualityField, 1,
    "addQualityField(field) \n\n"
    "Adds a quality field to this polar scan. Note, there is no check for valid size or similar. Also, there is no check if same how/task is specified or the likes. \n\n"
    "field  - The RaveFieldCore field"
  },
  {"addOrReplaceQualityField", (PyCFunction) _pypolarscan_addOrReplaceQualityField, 1,
    "addOrReplaceQualityField(field) \n\n"
    "Adds or replaces the quality field in this polar scan. Note, there is no check for valid size or similar. Ensures that only one field with a specific how/task exists.\n\n"
    "field  - The RaveFieldCore field"
  },
  {"getNumberOfQualityFields", (PyCFunction) _pypolarscan_getNumberOfQualityFields, 1,
    "getNumberOfQualityFields() -> integer\n\n"
    "Returns the number of quality fields in this scan"
  },
  {"getQualityField", (PyCFunction) _pypolarscan_getQualityField, 1,
    "getQualityField(index) -> RaveFieldCore \n\n"
    "Returns the rave field at specified index\n\n"
    "index  - The rave field at specified position.\n\n"
    "Throws IndexError if the rave field not could be found"
  },
  {"removeQualityField", (PyCFunction) _pypolarscan_removeQualityField, 1,
    "removeQualityField(index) \n\n"
    "Removes the quality field at specified index\n\n"
    "index  - The rave field at specified position.\n\n"
  },
  {"getQualityFieldByHowTask", (PyCFunction) _pypolarscan_getQualityFieldByHowTask, 1,
    "getQualityFieldByHowTask(name) -> RaveFieldCore or None \n\n"
    "Returns the quality with the how/task attribute equal to name\n\n"
    "name  - The rave field with how/task name equal to name\n\n"
  },
  {"findQualityFieldByHowTask", (PyCFunction) _pypolarscan_findQualityFieldByHowTask, 1,
    "findQualityFieldByHowTask(name) -> RaveFieldCore or None \n\n"
    "Tries to locate any quality field with  how/task attribute equal to name. First, the current parameters quality fields are checked and then self.\n\n"
    "name  - The rave field with how/task name equal to name\n\n"
  },
  {"getDistanceField", (PyCFunction) _pypolarscan_getDistanceField, 1,
    "getDistanceField() -> RaveFieldCore\n\n"
    "Creates a distance field for this scan"
  },
  {"getHeightField", (PyCFunction) _pypolarscan_getHeightField, 1,
    "getHeightField() -> RaveFieldCore\n\n"
    "Creates a height field for this scan"
  },
  {"getMaxDistance", (PyCFunction) _pypolarscan_getMaxDistance, 1,
    "getMaxDistance() -> max distance at ground level\n\n"
    "Returns the maximum distance (at ground level) that this scan will cover."
  },
  {"getDistance", (PyCFunction) _pypolarscan_getDistance, 1,
    "getDistance((lon,lat)) --> distance from origin of this scan to the specified lon/lat pair\n\n"
    "Returns the distance in meters along the surface from the radar to the specified lon/lat coordinate pair.\n\n"
    "lon - Longitude in radians\n"
    "lat - Latitude in radians"
  },
  {"clone", (PyCFunction) _pypolarscan_clone, 1,
    "clone() -> PolarScanCore\n\n"
    "Creates a clone of self"
  },
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the polar scan
 */
static PyObject* _pypolarscan_getattro(PyPolarScan* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("elangle", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getElangle(self->scan));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("nbins", name) == 0) {
    return PyInt_FromLong(PolarScan_getNbins(self->scan));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("rscale", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getRscale(self->scan));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("nrays", name) == 0) {
    return PyInt_FromLong(PolarScan_getNrays(self->scan));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("rstart", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getRstart(self->scan));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("a1gate", name) == 0) {
    return PyInt_FromLong(PolarScan_getA1gate(self->scan));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("datatype", name) == 0) {
    return PyInt_FromLong(PolarScan_getDataType(self->scan));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("beamwidth", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getBeamwidth(self->scan));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("longitude", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getLongitude(self->scan));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("latitude", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getLatitude(self->scan));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("height", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getHeight(self->scan));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("time", name) == 0) {
    const char* str = PolarScan_getTime(self->scan);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("date", name) == 0) {
    const char* str = PolarScan_getDate(self->scan);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("starttime", name) == 0) {
    const char* str = PolarScan_getStartTime(self->scan);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("startdate", name) == 0) {
    const char* str = PolarScan_getStartDate(self->scan);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("endtime", name) == 0) {
    const char* str = PolarScan_getEndTime(self->scan);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("enddate", name) == 0) {
    const char* str = PolarScan_getEndDate(self->scan);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("source", name) == 0) {
    const char* str = PolarScan_getSource(self->scan);
    if (str != NULL) {
      return PyRaveAPI_StringOrUnicode_FromASCII(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("defaultparameter", name) == 0) {
    const char* str = PolarScan_getDefaultParameter(self->scan);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("projection", name) == 0) {
    Projection_t* projection = PolarScan_getProjection(self->scan);
    if (projection != NULL) {
      PyProjection* result = PyProjection_New(projection);
      RAVE_OBJECT_RELEASE(projection);
      return (PyObject*)result;
    } else {
      Py_RETURN_NONE;
    }
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pypolarscan_setattro(PyPolarScan* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("elangle", name)==0) {
    if (PyFloat_Check(val)) {
      PolarScan_setElangle(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"elangle must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("rscale", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setRscale(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "rscale must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("rstart", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setRstart(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "rstart must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("a1gate", name) == 0) {
    if (PyInt_Check(val)) {
      PolarScan_setA1gate(self->scan, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"a1gate must be of type int");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("beamwidth", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setBeamwidth(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "beamwidth must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("longitude", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setLongitude(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "longitude must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("latitude", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setLatitude(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "latitude must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("height", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setHeight(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "height must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("time", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarScan_setTime(self->scan, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "time must be a string (HHmmss)");
      }
    } else if (val == Py_None) {
      PolarScan_setTime(self->scan, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "time must be a string (HHmmss)");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("date", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarScan_setDate(self->scan, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "date must be a string (YYYYMMSS)");
      }
    } else if (val == Py_None) {
      PolarScan_setDate(self->scan, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "date must be a string (YYYYMMSS)");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("starttime", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarScan_setStartTime(self->scan, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "starttime must be a string (HHmmss)");
      }
    } else if (val == Py_None) {
      PolarScan_setStartTime(self->scan, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "starttime must be a string (HHmmss)");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("startdate", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarScan_setStartDate(self->scan, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "startdate must be a string (YYYYMMSS)");
      }
    } else if (val == Py_None) {
      PolarScan_setStartDate(self->scan, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "startdate must be a string (YYYYMMSS)");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("endtime", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarScan_setEndTime(self->scan, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "endtime must be a string (HHmmss)");
      }
    } else if (val == Py_None) {
      PolarScan_setEndTime(self->scan, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "endtime must be a string (HHmmss)");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("enddate", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarScan_setEndDate(self->scan, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "enddate must be a string (YYYYMMSS)");
      }
    } else if (val == Py_None) {
      PolarScan_setEndDate(self->scan, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "enddate must be a string (YYYYMMSS)");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("source", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarScan_setSource(self->scan, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "source must be a string");
      }
    } else if (val == Py_None) {
      PolarScan_setSource(self->scan, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "source must be a string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("defaultparameter", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarScan_setDefaultParameter(self->scan, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "source must be a string");
      }
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "source must be a string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("projection", name) == 0) {
    if (PyProjection_Check(val)) {
      PolarScan_setProjection(self->scan, ((PyProjection*)val)->projection);
    } else if (val == Py_None) {
      PolarScan_setProjection(self->scan, NULL);
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }

  result = 0;
done:
  return result;
}

static PyObject* _pypolarscan_isPolarScan(PyObject* self, PyObject* args)
{
  PyObject* inobj = NULL;
  if (!PyArg_ParseTuple(args,"O", &inobj)) {
    return NULL;
  }
  if (PyPolarScan_Check(inobj)) {
    return PyBool_FromLong(1);
  }
  return PyBool_FromLong(0);
}

/*@} End of Polar Scans */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pypolarscan_type_doc,
    "The polar scan represents one polar scan. There are several member attributes "
    "associated with a polar scan as well a number of child objects like parameters, quality fields and more.\n"
    "Since a lot of RAVE has been developed with ODIM H5 in mind, it is also possible to add arbitrary attributes in "
    "various groups, e.g. c.addAttribute(\"how/this\", 1.2) and so on.\n\n"
    "A list of avilable member attributes are described below. For information about member functions, check each functions doc.\n"
    "\n"
    "elangle          - Elevation angle in radians.\n"
    "nbins            - Number of bins in the data set.\n"
    "nrays            - Number of rays in the data set.\n"
    "rstart           - Range where the ray start for the scan.\n"
    "a1gate           - the a1gate\n"
    "datatype         - The data type for the selected default parameter. If no default parameter set, the UNDEFINED is returned.\n"
    "beamwidth        - The beamwidth. Default is 1.0 * M_PI/360.0.\n"
    "longitude        - The longitude for where this polar scan originates from (lon0) in radians\n"
    "latitude         - The latitude for where this polar scan originates from (lat0) in radians\n"""
    "height           - The height above sea level where this polar scan originates from (alt0) in meters\n"
    "time             - Time this polar scan should represent as a string with format HHmmSS\n"
    "date             - Date this polar scan should represent as a string in the format YYYYMMDD\n"
    "starttime        - Time the collection of this polar scan started as a string with format HHmmSS\n"
    "startdate        - Date the collection of this polar scan started as a string in the format YYYYMMDD\n"
    "endtime          - Time the collection of this polar scan ended as a string with format HHmmSS\n"
    "enddate          - Date the collection of this polar scan ended as a string in the format YYYYMMDD\n"
    "source           - The source for this product. Defined as what/source in ODIM H5. I.e. a comma separated list of various identifiers. For example. NOD:seang,WMO:1234,....\n"
    "defaultparameter - Since a polar scan is a container of a number of different parameters, this setting allows the user to define a default parameter that will allow for operations directly in the scan instead of getting the parameter.\n"
    "\n"
    "The most common usage of this class is probably to load a ODIM H5 scan and perform operations on this object. However, to create a new instance:\n"
    "import _polarscan\n"
    "scan = _polarscan.new()\n"
    );
/*@} End of Documentation about the type */
/// --------------------------------------------------------------------
/// Type definitions
/// --------------------------------------------------------------------
/*@{ Type definition */
PyTypeObject PyPolarScan_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "PolarScanCore", /*tp_name*/
  sizeof(PyPolarScan), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pypolarscan_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pypolarscan_getattro, /*tp_getattro*/
  (setattrofunc)_pypolarscan_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pypolarscan_type_doc,        /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pypolarscan_methods,         /*tp_methods*/
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
/*@} End of Type definition */

/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pypolarscan_new, 1,
    "new() -> new instance of the PolarScanCore object\n\n"
    "Creates a new instance of the PolarScanCore object"
  },
  {"isPolarScan", (PyCFunction)_pypolarscan_isPolarScan, 1,
    "isPolarScan(object) -> boolean\n\n"
    "Checks if provided object is of PolarScanCore type or not.\n\n"
    "object - the object to check"
  },
  {NULL,NULL} /*Sentinel*/
};


MOD_INIT(_polarscan)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyPolarScan_API[PyPolarScan_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyPolarScan_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyPolarScan_Type);

  MOD_INIT_DEF(module, "_polarscan", _pypolarscan_type_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyPolarScan_API[PyPolarScan_Type_NUM] = (void*)&PyPolarScan_Type;
  PyPolarScan_API[PyPolarScan_GetNative_NUM] = (void *)PyPolarScan_GetNative;
  PyPolarScan_API[PyPolarScan_New_NUM] = (void*)PyPolarScan_New;

  c_api_object = PyCapsule_New(PyPolarScan_API, PyPolarScan_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_polarscan.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _polarscan.error");
    return MOD_INIT_ERROR;
  }

  import_array(); /*To make sure I get access to Numeric*/
  import_pypolarscanparam();
  import_pyprojection();
  import_pyravefield();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}

/*@} End of Module setup */
