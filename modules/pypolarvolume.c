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
#include "pyravecompat.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYPOLARVOLUME_MODULE   /**< to get correct part in pypolarvolume.h */
#include "pypolarvolume.h"

#include <arrayobject.h>
#include "pypolarscan.h"
#include "pyrave_debug.h"
#include "pyravefield.h"
#include "pyravedata2d.h"
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

static PyObject* _pypolarvolume_clone(PyPolarVolume* obj, PyObject* args)
{
  PolarVolume_t* cpy = NULL;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  cpy = RAVE_OBJECT_CLONE(obj->pvol);
  if (cpy == NULL) {
    raiseException_returnNULL(PyExc_RuntimeError, "Failed to clone volume");
  }
  result = (PyObject*)PyPolarVolume_New(cpy);

  RAVE_OBJECT_RELEASE(cpy);

  return result;
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

static PyObject* _pypolarvolume_getMaxDistance(PyPolarVolume* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyFloat_FromDouble(PolarVolume_getMaxDistance(self->pvol));
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
  {"beamwH", NULL},
  {"beamwV", NULL},
  {"use_azimuthal_nav_information", NULL},
  {"getDistance", (PyCFunction) _pypolarvolume_getDistance, 1,
    "getDistance((lon,lat)) --> distance from origin of this volume to the specified lon/lat pair\n\n"
    "Returns the distance in meters along the surface from the radar to the specified lon/lat coordinate pair.\n\n"
    "lon - Longitude in radians\n"
    "lat - Latitude in radians"
  },
  {"getMaxDistance", (PyCFunction) _pypolarvolume_getMaxDistance, 1,
    "getMaxDistance() -> max distance at ground level\n\n"
    "Returns the maximum distance (at ground level) that this volume will cover."
  },
  {"addScan", (PyCFunction) _pypolarvolume_addScan, 1,
    "addScan(scan)\n\n"
    "Adds a scan to the volume\n\n"
    "scan - a PolarScanCore"
  },
  {"getScan", (PyCFunction) _pypolarvolume_getScan, 1,
    "getScan(index) -> scan\n\n"
    "Returns the polar scan at specified index\n\n"
    "index - the index"
  },
  {"getNumberOfScans", (PyCFunction) _pypolarvolume_getNumberOfScans, 1,
    "getNumberOfScans() -> integer\n\n"
    "Returns the number of scans that this volume contains."
  },
  {"removeScan", (PyCFunction) _pypolarvolume_removeScan, 1,
    "removeScan(index)\n\n"
    "Removes the scan at specified index\n\n"
    "index - the index of the scan to remove"
  },
  {"getScanWithMaxDistance", (PyCFunction) _pypolarvolume_getScanWithMaxDistance, 1,
    "getScanWithMaxDistance() -> a scan\n\n"
    "Locates the scan that covers the longest distance."
  },
  {"isAscendingScans", (PyCFunction) _pypolarvolume_isAscendingScans, 1,
    "isAscendingScans() -> boolean\n\n"
    "Returns True if the scans in the volume are sorted in ascending order, otherwise False will be returned."
  },
  {"isTransformable", (PyCFunction) _pypolarvolume_isTransformable, 1,
    "isTransformable() -> boolean\n\n"
    "Returns True if is is possible to perform transform operations on this volume or False if it isn't."
  },
  {"sortByElevations", (PyCFunction) _pypolarvolume_sortByElevations, 1,
    "sortByElevations(order)\n\n"
    "Sorts the scans in the volume according to elevation angle.\n\n"
    "order - 1 if soring should be done in ascending order, otherwise descending ordering will be performed."
  },
  {"getScanClosestToElevation", (PyCFunction) _pypolarvolume_getScanClosestToElevation, 1,
    "getScanClosestToElevation(elevation, insidee) -> scan\n\n"
    "Returns the scan with elevation closest to the specified elevation. This function requires that the scans are ordered in ascending order, otherwise the behaviour will be undefined.\n\n"
    "elevation - the elevation angle in radians\n"
    "insidee   - if 1, then elevations must be within the min - max elevations. If 0, then either min or max will always be returned if not within min-max."
  },
  {"getNearest", (PyCFunction) _pypolarvolume_getNearest, 1,
    "getNearest((lon,lat), height, insidee) -> (type, value)\n\n"
    "Gets the nearest value at the specified lon/lat/height.\n\n"
    "lon    - longitude in radians\n"
    "lat    - latitude in radians\n"
    "height - height above sea level\n"
    "inside - if elevation must be within min-max elevation or not"
  },
  {"getNearestParameterValue", (PyCFunction) _pypolarvolume_getNearestParameterValue, 1,
    "getNearestParameterValue(quantity, (lon,lat), height, insidee) -> (type,value)\n\n"
    "Gets the nearest value at the specified lon/lat/height for the specified quantity.\n\n"
    "quantity - the parameter of interest\n"
    "lon      - longitude in radians\n"
    "lat      - latitude in radians\n"
    "height   - height above sea level\n"
    "inside   - if elevation must be within min-max elevation or not"
  },
  {"getNearestConvertedParameterValue", (PyCFunction) _pypolarvolume_getNearestConvertedParameterValue, 1,
    "getNearestConvertedParameterValue(quantity, (lon,lat), height, insidee) -> (type,value)\n\n"
    "Gets the nearest converted value (offset+v*gain) at the specified lon/lat/height for the specified quantity.\n\n"
    "quantity - the parameter of interest\n"
    "lon      - longitude in radians\n"
    "lat      - latitude in radians\n"
    "height   - height above sea level\n"
    "inside   - if elevation must be within min-max elevation or not"
  },
  {"getConvertedVerticalMaxValue", (PyCFunction)_pypolarvolume_getConvertedVerticalMaxValue, 1,
    "getConvertedVerticalMaxValue(quantity, (lon,lat)) -> (type,value)\n\n"
    "Gets the vertical converted maximum value (offset+v*gain) at the specified lon/lat for the specified quantity.\n\n"
    "quantity - the parameter of interest\n"
    "lon      - longitude in radians\n"
    "lat      - latitude in radians\n"
  },
  {"addAttribute", (PyCFunction) _pypolarvolume_addAttribute, 1,
    "addAttribute(name, value) \n\n"
    "Adds an attribute to the volume. Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis etc. \n"
    "Currently, double, long, string and 1-dimensional arrays are supported.\n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr.\n"
    "value - Value to be associated with the name. Currently, double, long, string and 1-dimensional arrays are supported."
  },
  {"getAttribute", (PyCFunction) _pypolarvolume_getAttribute, 1,
    "getAttribute(name) -> value \n\n"
    "Returns the value associated with the specified name \n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr."
  },
  {"getAttributeNames", (PyCFunction) _pypolarvolume_getAttributeNames, 1,
    "getAttributeNames() -> array of names \n\n"
    "Returns the attribute names associated with this volume"
  },
  {"hasAttribute", (PyCFunction) _pypolarvolume_hasAttribute, 1,
    "hasAttribute(name) -> a boolean \n\n"
    "Returns if the specified name is defined within this polar volume\n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis.\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr."
  },
  {"isValid", (PyCFunction) _pypolarvolume_isValid, 1,
    "isValid() -> a boolean \n\n"
    "Validates this polar volume to see if it is possible to write as a volume.\n\n"
  },
  {"getDistanceField", (PyCFunction) _pypolarvolume_getDistanceField, 1,
    "getDistanceField() -> RaveFieldCore\n\n"
    "Creates a distance field for this volume"
  },
  {"getHeightField", (PyCFunction) _pypolarvolume_getHeightField, 1,
    "getHeightField() -> RaveFieldCore\n\n"
    "Creates a height field for this volume"
  },
  {"clone", (PyCFunction) _pypolarvolume_clone, 1,
    "clone() -> PolarVolumeCore\n\n"
    "Creates a clone of self"
  },
  {NULL, NULL} /* sentinel */
};

/**
 * Returns the specified attribute in the polar volume
 */
static PyObject* _pypolarvolume_getattro(PyPolarVolume* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("longitude", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getLongitude(self->pvol));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("latitude", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getLatitude(self->pvol));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("height", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getHeight(self->pvol));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("beamwidth", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getBeamwidth(self->pvol));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("beamwH", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getBeamwH(self->pvol));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("beamwV", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getBeamwV(self->pvol));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("time", name) == 0) {
    const char* str = PolarVolume_getTime(self->pvol);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("date", name) == 0) {
    const char* str = PolarVolume_getDate(self->pvol);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("source", name) == 0) {
    const char* str = PolarVolume_getSource(self->pvol);
    if (str != NULL) {
      return PyRaveAPI_StringOrUnicode_FromASCII(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("paramname", name) == 0) {
    const char* str = PolarVolume_getDefaultParameter(self->pvol);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("use_azimuthal_nav_information", name) == 0) {
    return PyBool_FromLong(PolarVolume_useAzimuthalNavInformation(self->pvol));
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pypolarvolume_setattro(PyPolarVolume* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("longitude", name)==0) {
    if (PyFloat_Check(val)) {
      PolarVolume_setLongitude(self->pvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"longitude must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("latitude", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarVolume_setLatitude(self->pvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "latitude must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("height", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarVolume_setHeight(self->pvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "height must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("beamwidth", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarVolume_setBeamwidth(self->pvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "beamwidth must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("beamwH", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarVolume_setBeamwH(self->pvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "beamwH must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("beamwV", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarVolume_setBeamwV(self->pvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "beamwV must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("time", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarVolume_setTime(self->pvol, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set time");
      }
    } else if (val == Py_None) {
        PolarVolume_setTime(self->pvol, NULL);
    } else {
        raiseException_gotoTag(done, PyExc_ValueError, "time should be specified as a string (HHmmss)");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("date", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarVolume_setDate(self->pvol, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set date");
      }
    } else if (val == Py_None) {
      PolarVolume_setDate(self->pvol, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "date should be specified as a string (YYYYMMDD)");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("source", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarVolume_setSource(self->pvol, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set source");
      }
    } else if (val == Py_None) {
      PolarVolume_setSource(self->pvol, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "source should be specified as a string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("paramname", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarVolume_setDefaultParameter(self->pvol, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set default parameter");
      }
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "paramname should be specified as a string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("use_azimuthal_nav_information", name) == 0) {
    if (PyBool_Check(val)) {
      if (PyObject_IsTrue(val)) {
        PolarVolume_setUseAzimuthalNavInformation(self->pvol, 1);
      } else {
        PolarVolume_setUseAzimuthalNavInformation(self->pvol, 0);
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "use_azimuthal_nav_information must be of type bool");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));
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

/*@{ Documentation about the type */
PyDoc_STRVAR(_pypolarvolume_type_doc,
    "The polar volume is a container for polar scans. There are a few member attributes and like polar scans and other classes "
    "it is possible to add arbitrary attributes in the 3 groups how, what and where. For example what/sthis.\n\n"
    "\n"
    "longitude        - The longitude for where this polar scan originates from (lon0) in radians\n"
    "latitude         - The latitude for where this polar scan originates from (lat0) in radians\n"""
    "height           - The height above sea level where this polar scan originates from (alt0) in meters\n"
    "time             - Time this polar scan should represent as a string with format HHmmSS\n"
    "date             - Date this polar scan should represent as a string in the format YYYYMMDD\n"
    "source           - The source for this product. Defined as what/source in ODIM H5. I.e. a comma separated list of various identifiers. For example. NOD:seang,WMO:1234,....\n"
    "paramname        - The default parameter. Default parameter is used when operating on this volume unless the parameter (quantity) explicitly has been specified in the function.\n"
    "                 - All scans that currently are held by this volume and eventual new ones will all get the same default parameter.\n"
    "beamwidth        - DEPRECATED, Use beamwH! Beamwidth for the volume. All scans will get the specified beamwidth. If you only want to make the beamwidth affect an individual\n"
    "                   scan, modify the scan directly.\n"
    "beamwH           - Horizontal beamwidth for the volume. All scans will get the specified beamwidth. If you only want to make the beamwidth affect an individual\n"
    "                   scan, modify the scan directly.\n"
    "beamwV           - Vertical beamwidth for the volume. All scans will get the specified beamwidth. If you only want to make the beamwidth affect an individual\n"
    "                   scan, modify the scan directly.\n"
    "use_azimuthal_nav_information - If setting. Then all currently added scans will get this value. It will not affect scans added after. If reading this value\n"
    "                   it will return True if at least one of the currently set scans returns True. Otherwise this attribute will be False.\n"
    "\n"
    "Usage:\n"
    "import _polarvolume\n"
    "vol = _polarvolume.new()\n"
    );
/*@} End of Documentation about the type */

/// --------------------------------------------------------------------
/// Type definitions
/// --------------------------------------------------------------------
/*@{ Type definitions */
PyTypeObject PyPolarVolume_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "PolarVolumeCore", /*tp_name*/
  sizeof(PyPolarVolume), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pypolarvolume_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pypolarvolume_getattro, /*tp_getattro*/
  (setattrofunc)_pypolarvolume_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pypolarvolume_type_doc,      /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pypolarvolume_methods,         /*tp_methods*/
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

/// --------------------------------------------------------------------
/// Module setup
/// --------------------------------------------------------------------
/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pypolarvolume_new, 1,
    "new() -> new instance of the PolarVolumeCore object\n\n"
    "Creates a new instance of the PolarVolumeCore object"
  },
  {"isPolarVolume", (PyCFunction)_pypolarvolume_isPolarVolume, 1,
    "isPolarVolume(object) -> boolean\n\n"
    "Checks if provided object is of PolarVolumeCore type or not.\n\n"
    "object - the object to check"
  },
  {NULL,NULL} /*Sentinel*/
};

/**
 * Initializes polar volume.
 */
MOD_INIT(_polarvolume)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyPolarVolume_API[PyPolarVolume_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyPolarVolume_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyPolarVolume_Type);
  MOD_INIT_DEF(module, "_polarvolume", _pypolarvolume_type_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyPolarVolume_API[PyPolarVolume_Type_NUM] = (void*)&PyPolarVolume_Type;
  PyPolarVolume_API[PyPolarVolume_GetNative_NUM] = (void *)PyPolarVolume_GetNative;
  PyPolarVolume_API[PyPolarVolume_New_NUM] = (void*)PyPolarVolume_New;

  c_api_object = PyCapsule_New(PyPolarVolume_API, PyPolarVolume_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_polarvolume.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _polarvolume.error");
    return MOD_INIT_ERROR;
  }

  import_array(); /*To make sure I get access to Numeric*/
  import_pypolarscan();
  import_pyravefield();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
