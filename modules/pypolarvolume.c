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
 * Python version of the PolarVolume API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-08
 */
#include "Python.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYPOLARVOLUME_MODULE   /**< to get correct part in pypolarvolume.h */
#include "pypolarvolume.h"

#include <arrayobject.h>
#include "pypolarscan.h"
#include "pyrave_debug.h"
#include "pyravefield.h"
#include "rave_alloc.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_polarvolume");

/**
 * Sets a python exception and goto tag
 */
#define raiseException_gotoTag(tag, type, msg) \
{PyErr_SetString(type, msg); goto tag;}

/**
 * Sets python exception and returns NULL
 */
#define raiseException_returnNULL(type, msg) \
{PyErr_SetString(type, msg); return NULL;}

/**
 * Error object for reporting errors to the python interpreeter
 */
static PyObject *ErrorObject;

/// --------------------------------------------------------------------
/// Polar Volumes
/// --------------------------------------------------------------------
/*@{ Polar Volumes */
/**
 * Returns the native PolarVolume_t instance.
 * @param[in] pypolarvolume - the python polar volume instance
 * @returns the native polar volume instance.
 */
static PolarVolume_t*
PyPolarVolume_GetNative(PyPolarVolume* pypolarvolume)
{
  RAVE_ASSERT((pypolarvolume != NULL), "pypolarvolume == NULL");
  return RAVE_OBJECT_COPY(pypolarvolume->pvol);
}

/**
 * Creates a python polar volume from a native polar volume or will create an
 * initial native PolarVolume if p is NULL.
 * @param[in] p - the native polar volume (or NULL)
 * @returns the python polar volume.
 */
static PyPolarVolume*
PyPolarVolume_New(PolarVolume_t* p)
{
  PyPolarVolume* result = NULL;
  PolarVolume_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&PolarVolume_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for polar volume.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for polar volume.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyPolarVolume, &PyPolarVolume_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->pvol = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->pvol, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyPolarVolume instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for polar volume.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the polar volume
 * @param[in] obj the object to deallocate.
 */
static void _pypolarvolume_dealloc(PyPolarVolume* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->pvol, obj);
  RAVE_OBJECT_RELEASE(obj->pvol);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the polar volume.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pypolarvolume_new(PyObject* self, PyObject* args)
{
  PyPolarVolume* result = PyPolarVolume_New(NULL);
  return (PyObject*)result;
}

/**
 * Returns the distance from the radar to the specified lon/lat coordinate.
 * @param[in] self - self
 * @param[in] args - a tuple with (lon,lat) in radians.
 * @returns the distance in meters.
 */
static PyObject* _pypolarvolume_getDistance(PyPolarVolume* self, PyObject* args)
{
  double lon = 0.0L, lat = 0.0L;
  double distance = 0.0L;
  if (!PyArg_ParseTuple(args, "(dd)", &lon,&lat)) {
    return NULL;
  }
  distance = PolarVolume_getDistance(self->pvol, lon, lat);

  return PyFloat_FromDouble(distance);
}

/**
 * Adds one scan to a volume.
 * @param[in] self - the polar volume
 * @param[in] args - the scan, must be of type PolarScanCore
 * @return NULL on failure
 */
static PyObject* _pypolarvolume_addScan(PyPolarVolume* self, PyObject* args)
{
  PyObject* inptr = NULL;
  PyPolarScan* polarScan = NULL;

  if (!PyArg_ParseTuple(args, "O", &inptr)) {
    return NULL;
  }

  if (!PyPolarScan_Check(inptr)) {
    raiseException_returnNULL(PyExc_TypeError,"Added object must be of type PolarScanCore");
  }

  polarScan = (PyPolarScan*)inptr;

  if (!PolarVolume_addScan(self->pvol, polarScan->scan)) {
    raiseException_returnNULL(PyExc_MemoryError, "Failed to add scan to volume");
  }

  Py_RETURN_NONE;
}

/**
 * Returns the scan at the provided index.
 * @param[in] self - the polar volume
 * @param[in] args - the index must be >= 0 and < getNumberOfScans
 * @return NULL on failure
 */
static PyObject* _pypolarvolume_getScan(PyPolarVolume* self, PyObject* args)
{
  int index = -1;
  PolarScan_t* scan = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  if (index < 0 || index >= PolarVolume_getNumberOfScans(self->pvol)) {
    raiseException_returnNULL(PyExc_IndexError, "Index out of range");
  }

  if((scan = PolarVolume_getScan(self->pvol, index)) == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "Could not aquire scan");
  }

  if (scan != NULL) {
    result = (PyObject*)PyPolarScan_New(scan);
  }

  RAVE_OBJECT_RELEASE(scan);

  return result;
}

/**
 * Returns the number of scans for this volume
 * @param[in] self - the polar volume
 * @param[in] args - not used
 * @return NULL on failure or a PyInteger
 */
static PyObject* _pypolarvolume_getNumberOfScans(PyPolarVolume* self, PyObject* args)
{
  return PyInt_FromLong(PolarVolume_getNumberOfScans(self->pvol));
}

static PyObject* _pypolarvolume_removeScan(PyPolarVolume* self, PyObject* args)
{
  int index = -1;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }
  if (!PolarVolume_removeScan(self->pvol, index)) {
    raiseException_returnNULL(PyExc_IndexError, "Failed to remove scan");
  }
  Py_RETURN_NONE;
}

/**
 * Locates the scan that covers the longest distance
 * @param[in] self - self
 * @param[in] args - N/A
 * @return the scan with longest distance, otherwise NULL
 */
static PyObject* _pypolarvolume_getScanWithMaxDistance(PyPolarVolume* self, PyObject* args)
{
  PyObject* result = NULL;
  PolarScan_t* scan = NULL;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  scan = PolarVolume_getScanWithMaxDistance(self->pvol);
  if (scan == NULL) {
    raiseException_returnNULL(PyExc_RuntimeError, "Could not find any scan");
  }
  result = (PyObject*)PyPolarScan_New(scan);
  RAVE_OBJECT_RELEASE(scan);
  return result;
}

/**
 * Returns 1 if the scans in the volume are sorted in ascending order, otherwise
 * 0 will be returned.
 * @param[in] self - the polar volume
 * @param[in] args - not used
 * @returns NULL on failure or a PyInteger where 1 indicates that the scans are sorted after elevation angle
 * in ascending order.
 */
static PyObject* _pypolarvolume_isAscendingScans(PyPolarVolume* self, PyObject* args)
{
  return PyBool_FromLong(PolarVolume_isAscendingScans(self->pvol));
}

/**
 * Returns 1 if is is possible to perform transform operations on this volume or 0 if it isn't.
 * @param[in] self - the polar volume
 * @param[in] args - not used
 * @returns NULL on failure or a PyInteger where 1 indicates that it is possible to perform a transformation.
 */
static PyObject* _pypolarvolume_isTransformable(PyPolarVolume* self, PyObject* args)
{
  return PyBool_FromLong(PolarVolume_isTransformable(self->pvol));
}

/**
 * Sorts the scans by comparing elevations in either ascending or descending order
 * @param[in] self - the polar volume
 * @param[in] args - 1 if soring should be done in ascending order, otherwise descending
 * @return NULL on failure or a Py_None
 */
static PyObject* _pypolarvolume_sortByElevations(PyPolarVolume* self, PyObject* args)
{
  int order = 0;

  if (!PyArg_ParseTuple(args, "i", &order)) {
    return NULL;
  }

  PolarVolume_sortByElevations(self->pvol, order);

  Py_RETURN_NONE;
}

/**
 * Gets the scan that got an elevation angle that is closest to the specified angle.
 * @param[in] self - the polar volume
 * @param[in] args - the elevation angle (in radians) and an integer where 1 means only include elevations that are within the min-max elevations
 * @return a scan or NULL on failure
 */
static PyObject* _pypolarvolume_getScanClosestToElevation(PyPolarVolume* self, PyObject* args)
{
  double elevation = 0.0L;
  int inside = 0;
  PolarScan_t* scan = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "di", &elevation, &inside)) {
    return NULL;
  }

  scan = PolarVolume_getScanClosestToElevation(self->pvol, elevation, inside);
  if (scan != NULL) {
    result = (PyObject*)PyPolarScan_New(scan);
  }

  RAVE_OBJECT_RELEASE(scan);
  if (result != NULL) {
    return result;
  } else {
    Py_RETURN_NONE;
  }
}

/**
 * Gets the nearest value at the specified lon/lat/height.
 * @param[in] self - the polar volume
 * @param[in] args - the lon/lat as a tuple in radians), the height and an indicator if elevation must be within min-max elevation or not
 * @return a tuple of (valuetype, value)
 */
static PyObject* _pypolarvolume_getNearest(PyPolarVolume* self, PyObject* args)
{
  double lon = 0.0L, lat = 0.0L, height = 0.0L;
  double v = 0.0L;
  int insidee = 0;
  RaveValueType vtype = RaveValueType_NODATA;

  if (!PyArg_ParseTuple(args, "(dd)di", &lon,&lat,&height,&insidee)) {
    return NULL;
  }

  vtype = PolarVolume_getNearest(self->pvol, lon, lat, height, insidee, &v);

  return Py_BuildValue("(id)", vtype, v);
}

/**
 * Gets the nearest parameter value at the specified lon/lat/height.
 * @param[in] self - the polar volume
 * @param[in] args - quantity, the lon/lat as a tuple in radians), the height and an indicator if elevation must be within min-max elevation or not
 * @return a tuple of (valuetype, value)
 */
static PyObject* _pypolarvolume_getNearestParameterValue(PyPolarVolume* self, PyObject* args)
{
  double lon = 0.0L, lat = 0.0L, height = 0.0L;
  double v = 0.0L;
  char* quantity = NULL;
  int insidee = 0;
  RaveValueType vtype = RaveValueType_NODATA;

  if (!PyArg_ParseTuple(args, "s(dd)di", &quantity, &lon,&lat,&height,&insidee)) {
    return NULL;
  }

  vtype = PolarVolume_getNearestParameterValue(self->pvol, quantity, lon, lat, height, insidee, &v);

  return Py_BuildValue("(id)", vtype, v);
}

/**
 * Gets the nearest converted parameter value at the specified lon/lat/height.
 * @param[in] self - the polar volume
 * @param[in] args - quantity, the lon/lat as a tuple in radians), the height and an indicator if elevation must be within min-max elevation or not
 * @return a tuple of (valuetype, value)
 */
static PyObject* _pypolarvolume_getNearestConvertedParameterValue(PyPolarVolume* self, PyObject* args)
{
  double lon = 0.0L, lat = 0.0L, height = 0.0L;
  double v = 0.0L;
  char* quantity = NULL;
  int insidee = 0;
  RaveValueType vtype = RaveValueType_NODATA;

  if (!PyArg_ParseTuple(args, "s(dd)di", &quantity, &lon,&lat,&height,&insidee)) {
    return NULL;
  }

  vtype = PolarVolume_getNearestConvertedParameterValue(self->pvol, quantity, lon, lat, height, insidee, &v, NULL);

  return Py_BuildValue("(id)", vtype, v);
}

/**
 * Gets the vertical max value for the specified lon/lat coordinate.
 * @param[in] self - the polar volume
 * @param[in] args - quantity, the lon/lat as a tuple in radians.
 */
static PyObject* _pypolarvolume_getConvertedVerticalMaxValue(PyPolarVolume* self, PyObject* args)
{
  double lon = 0.0L, lat = 0.0L;
  double v = 0.0L;
  char* quantity = NULL;
  RaveValueType vtype = RaveValueType_NODATA;

  if (!PyArg_ParseTuple(args, "s(dd)", &quantity, &lon, &lat)) {
    return NULL;
  }

  vtype = PolarVolume_getConvertedVerticalMaxValue(self->pvol, quantity, lon, lat, &v, NULL);

  return Py_BuildValue("(id)", vtype, v);
}

/**
 * Adds an attribute to the parameter. Name of the attribute should be in format
 * ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis etc.
 * Currently, the only supported values are double, long, string.
 * @param[in] self - this instance
 * @param[in] args - bin index, ray index.
 * @returns true or false depending if it works.
 */
static PyObject* _pypolarvolume_addAttribute(PyPolarVolume* self, PyObject* args)
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

  if (!PolarVolume_addAttribute(self->pvol, attr)) {
    raiseException_gotoTag(done, PyExc_AttributeError, "Failed to add attribute");
  }

  result = PyBool_FromLong(1);
done:
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

static PyObject* _pypolarvolume_getAttribute(PyPolarVolume* self, PyObject* args)
{
  RaveAttribute_t* attribute = NULL;
  char* name = NULL;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }

  attribute = PolarVolume_getAttribute(self->pvol, name);
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

static PyObject* _pypolarvolume_getAttributeNames(PyPolarVolume* self, PyObject* args)
{
  RaveList_t* list = NULL;
  PyObject* result = NULL;
  int n = 0;
  int i = 0;

  list = PolarVolume_getAttributeNames(self->pvol);
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

static PyObject* _pypolarvolume_hasAttribute(PyPolarVolume* self, PyObject* args)
{
  RaveAttribute_t* attribute = NULL;
  char* name = NULL;
  long result = 0;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  attribute = PolarVolume_getAttribute(self->pvol, name);
  if (attribute != NULL) {
    result = 1;
  }
  RAVE_OBJECT_RELEASE(attribute);
  return PyBool_FromLong(result);
}

static PyObject* _pypolarvolume_isValid(PyPolarVolume* self, PyObject* args)
{
  return PyBool_FromLong(PolarVolume_isValid(self->pvol));
}

static PyObject* _pypolarvolume_getDistanceField(PyPolarVolume* self, PyObject* args)
{
  PyObject* result = NULL;
  RaveField_t* field = NULL;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  field = PolarVolume_getDistanceField(self->pvol);
  if (field != NULL) {
    result = (PyObject*)PyRaveField_New(field);
  }

  RAVE_OBJECT_RELEASE(field);
  if (result == NULL) {
    raiseException_returnNULL(PyExc_RuntimeError, "Could not create distance field");
  }
  return result;
}

static PyObject* _pypolarvolume_getHeightField(PyPolarVolume* self, PyObject* args)
{
  PyObject* result = NULL;
  RaveField_t* field = NULL;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  field = PolarVolume_getHeightField(self->pvol);
  if (field != NULL) {
    result = (PyObject*)PyRaveField_New(field);
  }

  RAVE_OBJECT_RELEASE(field);
  if (result == NULL) {
    raiseException_returnNULL(PyExc_RuntimeError, "Could not create height field");
  }
  return result;
}

/**
 * All methods a polar volume can have
 */
static struct PyMethodDef _pypolarvolume_methods[] =
{
  {"longitude", NULL},
  {"latitude", NULL},
  {"height", NULL},
  {"time", NULL},
  {"date", NULL},
  {"source", NULL},
  {"paramname", NULL},
  {"beamwidth", NULL},
  {"getDistance", (PyCFunction) _pypolarvolume_getDistance, 1},
  {"addScan", (PyCFunction) _pypolarvolume_addScan, 1},
  {"getScan", (PyCFunction) _pypolarvolume_getScan, 1},
  {"getNumberOfScans", (PyCFunction) _pypolarvolume_getNumberOfScans, 1},
  {"removeScan", (PyCFunction) _pypolarvolume_removeScan, 1},
  {"getScanWithMaxDistance", (PyCFunction) _pypolarvolume_getScanWithMaxDistance, 1},
  {"isAscendingScans", (PyCFunction) _pypolarvolume_isAscendingScans, 1},
  {"isTransformable", (PyCFunction) _pypolarvolume_isTransformable, 1},
  {"sortByElevations", (PyCFunction) _pypolarvolume_sortByElevations, 1},
  {"getScanClosestToElevation", (PyCFunction) _pypolarvolume_getScanClosestToElevation, 1},
  {"getNearest", (PyCFunction) _pypolarvolume_getNearest, 1},
  {"getNearestParameterValue", (PyCFunction) _pypolarvolume_getNearestParameterValue, 1},
  {"getNearestConvertedParameterValue", (PyCFunction) _pypolarvolume_getNearestConvertedParameterValue, 1},
  {"getConvertedVerticalMaxValue", (PyCFunction)_pypolarvolume_getConvertedVerticalMaxValue, 1},
  {"addAttribute", (PyCFunction) _pypolarvolume_addAttribute, 1},
  {"getAttribute", (PyCFunction) _pypolarvolume_getAttribute, 1},
  {"getAttributeNames", (PyCFunction) _pypolarvolume_getAttributeNames, 1},
  {"hasAttribute", (PyCFunction) _pypolarvolume_hasAttribute, 1},
  {"isValid", (PyCFunction) _pypolarvolume_isValid, 1},
  {"getDistanceField", (PyCFunction) _pypolarvolume_getDistanceField, 1},
  {"getHeightField", (PyCFunction) _pypolarvolume_getHeightField, 1},
  {NULL, NULL} /* sentinel */
};

/**
 * Returns the specified attribute in the polar volume
 */
static PyObject* _pypolarvolume_getattr(PyPolarVolume* self, char* name)
{
  PyObject* res = NULL;
  if (strcmp("longitude", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getLongitude(self->pvol));
  } else if (strcmp("latitude", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getLatitude(self->pvol));
  } else if (strcmp("height", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getHeight(self->pvol));
  } else if (strcmp("beamwidth", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getBeamwidth(self->pvol));
  } else if (strcmp("time", name) == 0) {
    const char* str = PolarVolume_getTime(self->pvol);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("date", name) == 0) {
    const char* str = PolarVolume_getDate(self->pvol);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("source", name) == 0) {
    const char* str = PolarVolume_getSource(self->pvol);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("paramname", name) == 0) {
    const char* str = PolarVolume_getDefaultParameter(self->pvol);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  }

  res = Py_FindMethod(_pypolarvolume_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pypolarvolume_setattr(PyPolarVolume* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("longitude", name)==0) {
    if (PyFloat_Check(val)) {
      PolarVolume_setLongitude(self->pvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"longitude must be of type float");
    }
  } else if (strcmp("latitude", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarVolume_setLatitude(self->pvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "latitude must be of type float");
    }
  } else if (strcmp("height", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarVolume_setHeight(self->pvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "height must be of type float");
    }
  } else if (strcmp("beamwidth", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarVolume_setBeamwidth(self->pvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "beamwidth must be of type float");
    }
  } else if (strcmp("time", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarVolume_setTime(self->pvol, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set time");
      }
    } else if (val == Py_None) {
        PolarVolume_setTime(self->pvol, NULL);
    } else {
        raiseException_gotoTag(done, PyExc_ValueError, "time should be specified as a string (HHmmss)");
    }
  } else if (strcmp("date", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarVolume_setDate(self->pvol, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set date");
      }
    } else if (val == Py_None) {
      PolarVolume_setDate(self->pvol, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "date should be specified as a string (YYYYMMDD)");
    }
  } else if (strcmp("source", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarVolume_setSource(self->pvol, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set source");
      }
    } else if (val == Py_None) {
      PolarVolume_setSource(self->pvol, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "source should be specified as a string");
    }
  } else if (strcmp("paramname", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarVolume_setDefaultParameter(self->pvol, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set default parameter");
      }
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "paramname should be specified as a string");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, name);
  }

  result = 0;
done:
  return result;
}

static PyObject* _pypolarvolume_isPolarVolume(PyObject* self, PyObject* args)
{
  PyObject* inobj = NULL;
  if (!PyArg_ParseTuple(args,"O", &inobj)) {
    return NULL;
  }
  if (PyPolarVolume_Check(inobj)) {
    return PyBool_FromLong(1);
  }
  return PyBool_FromLong(0);
}
/*@} End of Polar Volumes */

/// --------------------------------------------------------------------
/// Type definitions
/// --------------------------------------------------------------------
/*@{ Type definitions */
PyTypeObject PyPolarVolume_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "PolarVolumeCore", /*tp_name*/
  sizeof(PyPolarVolume), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pypolarvolume_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pypolarvolume_getattr, /*tp_getattr*/
  (setattrfunc)_pypolarvolume_setattr, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0,
  0, /*tp_as_mapping */
  0 /*tp_hash*/
};
/*@} End of Type definitions */

/// --------------------------------------------------------------------
/// Module setup
/// --------------------------------------------------------------------
/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pypolarvolume_new, 1},
  {"isPolarVolume", (PyCFunction)_pypolarvolume_isPolarVolume, 1},
  {NULL,NULL} /*Sentinel*/
};

/**
 * Initializes polar volume.
 */
void init_polarvolume(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyPolarVolume_API[PyPolarVolume_API_pointers];
  PyObject *c_api_object = NULL;
  PyPolarVolume_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_polarvolume", functions);
  if (module == NULL) {
    return;
  }
  PyPolarVolume_API[PyPolarVolume_Type_NUM] = (void*)&PyPolarVolume_Type;
  PyPolarVolume_API[PyPolarVolume_GetNative_NUM] = (void *)PyPolarVolume_GetNative;
  PyPolarVolume_API[PyPolarVolume_New_NUM] = (void*)PyPolarVolume_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyPolarVolume_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_polarvolume.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _polarvolume.error");
  }

  import_array(); /*To make sure I get access to Numeric*/
  import_pypolarscan();
  import_pyravefield();

  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
