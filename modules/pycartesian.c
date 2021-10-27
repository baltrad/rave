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
 * Python version of the Cartesian API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#include "cartesian_odim_io.h"

#define PYCARTESIAN_MODULE        /**< to get correct part of pycartesian.h */
#include "pycartesian.h"

#include "pyprojection.h"
#include "pyarea.h"
#include "pyravefield.h"
#include "pycartesianparam.h"
#include <arrayobject.h>
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_cartesian");

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

/*@{ Cartesian products */
/**
 * Returns the native Cartesian_t instance.
 * @param[in] pycartesian - the python cartesian instance
 * @returns the native cartesian instance.
 */
static Cartesian_t*
PyCartesian_GetNative(PyCartesian* pycartesian)
{
  RAVE_ASSERT((pycartesian != NULL), "pycartesian == NULL");
  return RAVE_OBJECT_COPY(pycartesian->cartesian);
}

/**
 * Creates a python cartesian from a native cartesian or will create an
 * initial native Cartesian if p is NULL.
 * @param[in] p - the native cartesian (or NULL)
 * @returns the python cartesian product.
 */
static PyCartesian*
PyCartesian_New(Cartesian_t* p)
{
  PyCartesian* result = NULL;
  Cartesian_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&Cartesian_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for cartesian.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for cartesian.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyCartesian, &PyCartesian_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->cartesian = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->cartesian, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyCartesian instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for cartesian.");
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
static void _pycartesian_dealloc(PyCartesian* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->cartesian, obj);
  RAVE_OBJECT_RELEASE(obj->cartesian);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the cartesian.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pycartesian_new(PyObject* self, PyObject* args)
{
  PyCartesian* result = PyCartesian_New(NULL);
  return (PyObject*)result;
}

/**
 * Initializes a cartesian product with the settings as described by the
 * area definition.
 */
static PyObject* _pycartesian_init(PyCartesian* self, PyObject* args)
{
  PyObject* inarea = NULL;

  if (!PyArg_ParseTuple(args, "O", &inarea)) {
    return NULL;
  }

  if (!PyArea_Check(inarea)) {
    raiseException_returnNULL(PyExc_TypeError, "First argument must be a PyAreaCore instance");
  }

  Cartesian_init(self->cartesian, ((PyArea*)inarea)->area);

  Py_RETURN_NONE;
}

/**
 * Returns the x location defined by area extent and x scale and the provided x position.
 * @param[in] self this instance.
 * @param[in] args - x position
 * @return the x location on success, otherwise NULL
 */
static PyObject* _pycartesian_getLocationX(PyCartesian* self, PyObject* args)
{
  long x = 0;
  double xloc = 0.0;
  if (!PyArg_ParseTuple(args, "l", &x)) {
    return NULL;
  }

  xloc = Cartesian_getLocationX(self->cartesian, x);

  return PyFloat_FromDouble(xloc);
}

/**
 * Returns the y location defined by area extent and y scale and the provided y position.
 * @param[in] self this instance.
 * @param[in] args - y position
 * @return the y location on success, otherwise NULL
 */
static PyObject* _pycartesian_getLocationY(PyCartesian* self, PyObject* args)
{
  long y = 0;
  double yloc = 0.0;
  if (!PyArg_ParseTuple(args, "l", &y)) {
    return NULL;
  }

  yloc = Cartesian_getLocationY(self->cartesian, y);

  return PyFloat_FromDouble(yloc);
}

/**
 * Returns the x index from the surface coordinate x location
 * @param[in] self this instance.
 * @param[in] args - x location
 * @return the x index on success, otherwise NULL
 */
static PyObject* _pycartesian_getIndexX(PyCartesian* self, PyObject* args)
{
  long x = 0;
  double xloc = 0.0;
  if (!PyArg_ParseTuple(args, "d", &xloc)) {
    return NULL;
  }

  x = Cartesian_getIndexX(self->cartesian, xloc);

  return PyLong_FromLong(x);
}

/**
 * Returns the y index from the surface coordinate y location
 * @param[in] self this instance.
 * @param[in] args - y location
 * @return the y index on success, otherwise NULL
 */
static PyObject* _pycartesian_getIndexY(PyCartesian* self, PyObject* args)
{
  long y = 0;
  double yloc = 0.0;
  if (!PyArg_ParseTuple(args, "d", &yloc)) {
    return NULL;
  }

  y = Cartesian_getIndexY(self->cartesian, yloc);

  return PyLong_FromLong(y);
}

static PyObject* _pycartesian_getExtremeLonLatBoundaries(PyCartesian* self, PyObject* args)
{
  double ulLon = 0.0, ulLat = 0.0, lrLon = 0.0, lrLat = 0.0;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  if (!Cartesian_getExtremeLonLatBoundaries(self->cartesian, &ulLon, &ulLat, &lrLon, &lrLat)) {
    raiseException_returnNULL(PyExc_ValueError, "Could not get extreme boundaries for cartesian product");
  }
  return Py_BuildValue("(dd)(dd)",ulLon,ulLat,lrLon,lrLat);
}

/**
 * sets the value at the specified position
 * @param[in] self this instance.
 * @param[in] args - tuple (x, y) and v
 * @return 0 on failure, otherwise 1
 */
static PyObject* _pycartesian_setValue(PyCartesian* self, PyObject* args)
{
  long x = 0, y = 0;
  double v = 0.0L;
  int result = 0;
  if (!PyArg_ParseTuple(args, "(ll)d", &x, &y, &v)) {
    return NULL;
  }

  result = Cartesian_setValue(self->cartesian, x, y, v);

  return PyInt_FromLong(result);
}

/**
 * sets the converted value at the specified position. Would be same
 * as setValue((x, y), (v - offset)/gain).
 * @param[in] self this instance.
 * @param[in] args - tuple (x, y) and v
 * @return 0 on failure, otherwise 1
 */
static PyObject* _pycartesian_setConvertedValue(PyCartesian* self, PyObject* args)
{
  long x = 0, y = 0;
  double v = 0.0L;
  int result = 0;
  if (!PyArg_ParseTuple(args, "(ll)d", &x, &y, &v)) {
    return NULL;
  }

  result = Cartesian_setConvertedValue(self->cartesian, x, y, v);

  return PyInt_FromLong(result);
}

/**
 * returns the value at the specified position
 * @param[in] self this instance.
 * @param[in] args - tuple (x, y) and v
 * @return 0 on failure, otherwise 1
 */
static PyObject* _pycartesian_getValue(PyCartesian* self, PyObject* args)
{
  long x = 0, y = 0;
  double v = 0.0L;
  RaveValueType result = RaveValueType_NODATA;
  if (!PyArg_ParseTuple(args, "(ll)", &x, &y)) {
    return NULL;
  }

  result = Cartesian_getValue(self->cartesian, x, y, &v);

  return Py_BuildValue("(id)", result, v);
}

/**
 * returns the converted value at the specified position
 * @param[in] self this instance.
 * @param[in] args - tuple (x, y) and v
 * @return 0 on failure, otherwise 1
 */
static PyObject* _pycartesian_getConvertedValue(PyCartesian* self, PyObject* args)
{
  long x = 0, y = 0;
  double v = 0.0L;
  RaveValueType result = RaveValueType_NODATA;
  if (!PyArg_ParseTuple(args, "(ll)", &x, &y)) {
    return NULL;
  }

  result = Cartesian_getConvertedValue(self->cartesian, x, y, &v);

  return Py_BuildValue("(id)", result, v);
}

/**
 * returns the value at the specified position as defined by the area definition
 * @param[in] self this instance.
 * @param[in] args - tuple (lx, ly) and v
 * @return 0 on failure, otherwise 1
 */
static PyObject* _pycartesian_getValueAtLocation(PyCartesian* self, PyObject* args)
{
  double lx = 0, ly = 0;
  double v = 0.0L;
  RaveValueType result = RaveValueType_NODATA;
  if (!PyArg_ParseTuple(args, "(dd)", &lx, &ly)) {
    return NULL;
  }

  result = Cartesian_getValueAtLocation(self->cartesian, lx, ly, &v);

  return Py_BuildValue("(id)", result, v);
}

/**
 * returns the converted value at the specified position as defined by the area definition
 * @param[in] self this instance.
 * @param[in] args - tuple (lx, ly) and v
 * @return 0 on failure, otherwise 1
 */
static PyObject* _pycartesian_getConvertedValueAtLocation(PyCartesian* self, PyObject* args)
{
  double lx = 0, ly = 0;
  double v = 0.0L;
  RaveValueType result = RaveValueType_NODATA;
  if (!PyArg_ParseTuple(args, "(dd)", &lx, &ly)) {
    return NULL;
  }

  result = Cartesian_getConvertedValueAtLocation(self->cartesian, lx, ly, &v);

  return Py_BuildValue("(id)", result, v);
}

/**
 * returns the converted value at the specified lon/lat position as defined by the area definition
 * @param[in] self this instance.
 * @param[in] args - tuple (lon, lat) and v
 * @return 0 on failure, otherwise 1
 */
static PyObject* _pycartesian_getConvertedValueAtLonLat(PyCartesian* self, PyObject* args)
{
  double lon = 0, lat = 0;
  double v = 0.0L;
  RaveValueType result = RaveValueType_NODATA;
  if (!PyArg_ParseTuple(args, "(dd)", &lon, &lat)) {
    return NULL;
  }

  result = Cartesian_getConvertedValueAtLonLat(self->cartesian, lon, lat, &v);

  return Py_BuildValue("(id)", result, v);
}

/**
 * returns the quality value at the specified location as defined by the area definition
 * @param[in] self this instance.
 * @param[in] args - tuple (lx, ly) and fieldname
 * @return the quality value if found otherwise None
 */
static PyObject* _pycartesian_getQualityValueAtLocation(PyCartesian* self, PyObject* args)
{
  double lx = 0, ly = 0;
  double v = 0.0L;
  char* fieldname = NULL;
  RaveValueType result = RaveValueType_NODATA;
  if (!PyArg_ParseTuple(args, "(dd)s", &lx, &ly,&fieldname)) {
    return NULL;
  }

  result = Cartesian_getQualityValueAtLocation(self->cartesian, lx, ly, fieldname, &v);

  if (result == 0) {
    Py_RETURN_NONE;
  } else {
    return PyFloat_FromDouble(v);
  }
}

/**
 * returns the scaled quality value at the specified location as defined by the area definition
 * @param[in] self this instance.
 * @param[in] args - tuple (lx, ly) and fieldname
 * @return the quality value if found otherwise None
 */
static PyObject* _pycartesian_getConvertedQualityValueAtLocation(PyCartesian* self, PyObject* args)
{
  double lx = 0, ly = 0;
  double v = 0.0L;
  char* fieldname = NULL;
  RaveValueType result = RaveValueType_NODATA;
  if (!PyArg_ParseTuple(args, "(dd)s", &lx, &ly,&fieldname)) {
    return NULL;
  }

  result = Cartesian_getConvertedQualityValueAtLocation(self->cartesian, lx, ly, fieldname, &v);

  if (result == 0) {
    Py_RETURN_NONE;
  } else {
    return PyFloat_FromDouble(v);
  }
}

/**
 * returns the quality value at the specified lon/lat position as defined by the area definition
 * @param[in] self this instance.
 * @param[in] args - tuple (lon, lat) and fieldname
 * @return the quality value if found otherwise None
 */
static PyObject* _pycartesian_getQualityValueAtLonLat(PyCartesian* self, PyObject* args)
{
  double lon = 0, lat = 0;
  double v = 0.0L;
  char* fieldname = NULL;
  RaveValueType result = RaveValueType_NODATA;
  if (!PyArg_ParseTuple(args, "(dd)s", &lon, &lat,&fieldname)) {
    return NULL;
  }

  result = Cartesian_getQualityValueAtLonLat(self->cartesian, lon, lat, fieldname, &v);

  if (result == 0) {
    Py_RETURN_NONE;
  } else {
    return PyFloat_FromDouble(v);
  }
}

/**
 * returns the scaled quality value at the specified lon/lat position as defined by the area definition
 * @param[in] self this instance.
 * @param[in] args - tuple (lon, lat) and fieldname
 * @return the quality value if found otherwise None
 */
static PyObject* _pycartesian_getConvertedQualityValueAtLonLat(PyCartesian* self, PyObject* args)
{
  double lon = 0, lat = 0;
  double v = 0.0L;
  char* fieldname = NULL;
  RaveValueType result = RaveValueType_NODATA;
  if (!PyArg_ParseTuple(args, "(dd)s", &lon, &lat,&fieldname)) {
    return NULL;
  }

  result = Cartesian_getConvertedQualityValueAtLonLat(self->cartesian, lon, lat, fieldname, &v);

  if (result == 0) {
    Py_RETURN_NONE;
  } else {
    return PyFloat_FromDouble(v);
  }
}

static PyObject* _pycartesian_isTransformable(PyCartesian* self, PyObject* args)
{
  return PyBool_FromLong(Cartesian_isTransformable(self->cartesian));
}

static PyObject* _pycartesian_getMean(PyCartesian* self, PyObject* args)
{
  long x = 0, y = 0;
  int N = 0;
  double v = 0.0L;
  RaveValueType result = RaveValueType_NODATA;
  if (!PyArg_ParseTuple(args, "(ll)i", &x, &y, &N)) {
    return NULL;
  }
  result = Cartesian_getMean(self->cartesian, x, y, N, &v);

  return Py_BuildValue("(id)", result, v);
}

/**
 * Adds an attribute to the volume. Name of the attribute should be in format
 * ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis etc.
 * Currently, the only supported values are double, long, string.
 * @param[in] self - this instance
 * @param[in] args - bin index, ray index.
 * @returns true or false depending if it works.
 */
static PyObject* _pycartesian_addAttribute(PyCartesian* self, PyObject* args)
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
    char* value = (char*)PyString_AsString(obj);
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

  if (!Cartesian_addAttribute(self->cartesian, attr)) {
    raiseException_gotoTag(done, PyExc_AttributeError, "Failed to add attribute");
  }

  result = PyBool_FromLong(1);
done:
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

static PyObject* _pycartesian_getAttribute(PyCartesian* self, PyObject* args)
{
  RaveAttribute_t* attribute = NULL;
  char* name = NULL;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }

  attribute = Cartesian_getAttribute(self->cartesian, name);
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

static PyObject* _pycartesian_getAttributeNames(PyCartesian* self, PyObject* args)
{
  RaveList_t* list = NULL;
  PyObject* result = NULL;
  int n = 0;
  int i = 0;

  list = Cartesian_getAttributeNames(self->cartesian);
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

static PyObject* _pycartesian_hasAttribute(PyCartesian* self, PyObject* args)
{
  char* name = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  return PyBool_FromLong(Cartesian_hasAttribute(self->cartesian, name));
}

static PyObject* _pycartesian_isValid(PyCartesian* self, PyObject* args)
{
  Rave_ObjectType otype = Rave_ObjectType_UNDEFINED;

  if (!PyArg_ParseTuple(args, "i", &otype)) {
    return NULL;
  }
  if (otype == Rave_ObjectType_IMAGE) {
    return PyBool_FromLong(CartesianOdimIO_isValidImage(self->cartesian));
  } else if (otype == Rave_ObjectType_COMP) {
    return PyBool_FromLong(CartesianOdimIO_isValidVolumeImage(self->cartesian));
  } else if (otype == Rave_ObjectType_CVOL) {
    return PyBool_FromLong(CartesianOdimIO_isValidVolumeImage(self->cartesian));
  }
  return PyBool_FromLong(CartesianOdimIO_isValidImage(self->cartesian));
}

/**
 * Adds a quality field to the cartesian product
 * @param[in] self - this instance
 * @param[in] args - object, of type RaveFieldCore
 * @returns None
 */
static PyObject* _pycartesian_addQualityField(PyCartesian* self, PyObject* args)
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

  if (!Cartesian_addQualityField(self->cartesian, ravefield->field)) {
    raiseException_returnNULL(PyExc_AttributeError, "Failed to add quality field to cartesian");
  }

  Py_RETURN_NONE;
}

/**
 * Returns the number of quality fields
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns The number of quality fields
 */
static PyObject* _pycartesian_getNumberOfQualityFields(PyCartesian* self, PyObject* args)
{
  return PyInt_FromLong(Cartesian_getNumberOfQualityFields(self->cartesian));
}

/**
 * Returns the number of quality fields
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns The number of quality fields
 */
static PyObject* _pycartesian_getQualityField(PyCartesian* self, PyObject* args)
{
  PyObject* result = NULL;
  RaveField_t* field = NULL;
  int index = 0;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  if (index < 0 || index >= Cartesian_getNumberOfQualityFields(self->cartesian)) {
    raiseException_returnNULL(PyExc_IndexError, "Index out of bounds");
  }

  if ((field = Cartesian_getQualityField(self->cartesian, index)) == NULL) {
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
static PyObject* _pycartesian_removeQualityField(PyCartesian* self, PyObject* args)
{
  int index = 0;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  Cartesian_removeQualityField(self->cartesian, index);

  Py_RETURN_NONE;
}

/**
 * Gets the quality field with the specified how/task value
 * @param[in] self - this instance
 * @param[in] args - the how/task string value
 * @returns the quality field or None if there is no such quality field
 */
static PyObject* _pycartesian_getQualityFieldByHowTask(PyCartesian* self, PyObject* args)
{
  PyObject* result = NULL;
  RaveField_t* field = NULL;

  char* howtask = NULL;
  if (!PyArg_ParseTuple(args, "s", &howtask)) {
    return NULL;
  }

  field = Cartesian_getQualityFieldByHowTask(self->cartesian, howtask);

  if (field != NULL) {
    result = (PyObject*)PyRaveField_New(field);
  }

  RAVE_OBJECT_RELEASE(field);

  if (result != NULL) {
    return result;
  }

  Py_RETURN_NONE;
}

/**
 * Finds the quality field with the specified how/task value by first checking in the current
 * parameter and if there is no such quality field, self is checked.
 * @param[in] self - this instance
 * @param[in] args - the how/task string value
 * @returns the quality field or None if there is no such quality field
 */
static PyObject* _pycartesian_findQualityFieldByHowTask(PyCartesian* self, PyObject* args)
{
  PyObject* result = NULL;
  RaveField_t* field = NULL;

  char* howtask = NULL;
  if (!PyArg_ParseTuple(args, "s", &howtask)) {
    return NULL;
  }

  field = Cartesian_findQualityFieldByHowTask(self->cartesian, howtask);

  if (field != NULL) {
    result = (PyObject*)PyRaveField_New(field);
  }

  RAVE_OBJECT_RELEASE(field);

  if (result != NULL) {
    return result;
  }

  Py_RETURN_NONE;
}

/**
 * Adds a parameter to the cartesian product.
 * @param[in] self - self
 * @param[in] args - an CartesianParamCore object
 * @return NONE on success otherwise NULL
 */
static PyObject* _pycartesian_addParameter(PyCartesian* self, PyObject* args)
{
  PyObject* inptr = NULL;
  PyCartesianParam* param = NULL;

  if (!PyArg_ParseTuple(args, "O", &inptr)) {
    return NULL;
  }

  if (!PyCartesianParam_Check(inptr)) {
    raiseException_returnNULL(PyExc_TypeError,"Added object must be of type CartesianParamCore");
  }

  param = (PyCartesianParam*)inptr;

  if (!Cartesian_addParameter(self->cartesian, param->param)) {
    raiseException_returnNULL(PyExc_AttributeError, "Failed to add parameter to cartesian");
  }

  Py_RETURN_NONE;
}

/**
 * Creates a parameter that is added to the cartesian product. This
 * call requires that the cartesian product has been initialized.
 * @param[in] self - self
 * @param[in] args - a string defining quantity and a type defining array data type
 * @return None on success otherwise NULL
 */
static PyObject* _pycartesian_createParameter(PyCartesian* self, PyObject* args)
{
  char* quantity = NULL;
  RaveDataType type = RaveDataType_UNDEFINED;
  CartesianParam_t* param = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "si", &quantity, &type)) {
    return NULL;
  }

  param = Cartesian_createParameter(self->cartesian, quantity, type, 0);
  if (param != NULL) {
    result = (PyObject*)PyCartesianParam_New(param);
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, "Could not create parameter, has cartesian been initialized?");
  }

done:
  RAVE_OBJECT_RELEASE(param);
  return result;
}

/**
 * Returns the parameter with specified quantity.
 * @param[in] self - self
 * @param[in] args - the quantity as a string
 * @return the associated CartesianParamCore instance if found otherwise None
 */
static PyObject* _pycartesian_getParameter(PyCartesian* self, PyObject* args)
{
  char* paramname = NULL;
  CartesianParam_t* param = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "s", &paramname)) {
    return NULL;
  }

  if (!Cartesian_hasParameter(self->cartesian, paramname)) {
    Py_RETURN_NONE;
  }

  if((param = Cartesian_getParameter(self->cartesian, paramname)) == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "Could not aquire parameter");
  }

  if (param != NULL) {
    result = (PyObject*)PyCartesianParam_New(param);
  }

  RAVE_OBJECT_RELEASE(param);

  return result;
}

/**
 * Returns True or False depending on if the cartesian product has a
 * parameter with specified quantity or not.
 * @param[in] self - self
 * @param[in] args - a string specifying the quantity
 * @return True if product has the specified parameter
 */
static PyObject* _pycartesian_hasParameter(PyCartesian* self, PyObject* args)
{
  char* paramname = NULL;

  if (!PyArg_ParseTuple(args, "s", &paramname)) {
    return NULL;
  }

  return PyBool_FromLong(Cartesian_hasParameter(self->cartesian, paramname));
}

/**
 * Removes the parameter with the specified name
 * @param[in] self - self
 * @param[in] args - a string defining the parameter name (quantity)
 * @return None on success or NULL on failure
 */
static PyObject* _pycartesian_removeParameter(PyCartesian* self, PyObject* args)
{
  char* paramname = NULL;

  if (!PyArg_ParseTuple(args, "s", &paramname)) {
    return NULL;
  }

  Cartesian_removeParameter(self->cartesian, paramname);

  Py_RETURN_NONE;
}

/**
 * Returns the number of parameters that exists in this product
 * @param[in] self - self
 * @param[in] args - NA
 * @return the number of parameters
 */
static PyObject* _pycartesian_getParameterCount(PyCartesian* self, PyObject* args)
{
  return PyInt_FromLong(Cartesian_getParameterCount(self->cartesian));
}

/**
 * Returns the parameter names that exists in this product
 * @param[in] self - self
 * @param[in] args - NA
 * @return a list of parameter names
 */
static PyObject* _pycartesian_getParameterNames(PyCartesian* self, PyObject* args)
{
  RaveList_t* paramnames = NULL;
  PyObject* result = NULL;
  int nparams = 0;
  int i = 0;

  paramnames = Cartesian_getParameterNames(self->cartesian);
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
 * Clones self
 * @param[in] self - self
 * @param[in] args - NA
 * @return a clone of self
 */
static PyObject* _pycartesian_clone(PyCartesian* self, PyObject* args)
{
  PyObject* pyresult = NULL;
  Cartesian_t* result = RAVE_OBJECT_CLONE(self->cartesian);
  if (result != NULL) {
    pyresult = (PyObject*)PyCartesian_New(result);
  }
  RAVE_OBJECT_RELEASE(result);
  return pyresult;
}

/**
 * All methods a cartesian product can have
 */
static struct PyMethodDef _pycartesian_methods[] =
{
  {"time", NULL},
  {"date", NULL},
  {"objectType", NULL},
  {"product", NULL},
  {"source", NULL},
  {"prodname", NULL},
  {"xsize", NULL},
  {"ysize", NULL},
  {"xscale", NULL},
  {"yscale", NULL},
  {"areaextent", NULL},
  {"projection", NULL},
  {"starttime", NULL},
  {"startdate", NULL},
  {"endtime", NULL},
  {"enddate", NULL},
  {"defaultParameter", NULL},
  {"init", (PyCFunction) _pycartesian_init, 1,
      "init(inarea)\n\n"
      "Initializes this cartesian product with the area \n\n"
      "inarea - The area definition to be used for this cartesian product"
  },
  {"getLocationX", (PyCFunction) _pycartesian_getLocationX, 1,
    "getLocationX(x) -> cartesian x coordinate\n\n"
    "Returns the location within the area as identified by a x-position. Evaluated as: upperLeft.x + xscale * x \n\n"
    "x - The x index in the area. I.e. x >= 0 and x < xsize."
  },
  {"getLocationY", (PyCFunction) _pycartesian_getLocationY, 1,
    "getLocationY(y) -> cartesian y coordinate\n\n"
    "Returns the location within the area as identified by a y-position. Evaluated as: upperLeft.y - yscale * y \n\n"
    "y - The y index in the area. I.e. y >= 0 and y < ysize."
  },
  {"getIndexX", (PyCFunction) _pycartesian_getIndexX, 1,
    "getIndexX(x) -> x index\n\n"
    "Returns the index within the area as identified by a x-coordinate. Evaluated as: (x - lowerLeft.x)/xscale \n\n"
    "x - The x coordinate in the area."
  },
  {"getIndexY", (PyCFunction) _pycartesian_getIndexY, 1,
    "getIndexY(y) -> y index\n\n"
    "Returns the index within the area as identified by a y-coordinate. Evaluated as: (upperRight.y - y)/yscale \n\n"
    "y - The y coordinate in the area."
  },
  {"getExtremeLonLatBoundaries", (PyCFunction) _pycartesian_getExtremeLonLatBoundaries, 1,
    "getExtremeLonLatBoundaries() -> (ullon, ullat),(lrlon,lrlat)\n\n"
    "Determines the extreme lon lat boundaries for this area. I.e. the outer boundaries of this cartesian image "
    "will be steped over until the absolute min/max lon/lat positions are found for this image.\n"
    "Note, that the bounding box returned will be in a different setup than area extent"
  },
  {"setValue", (PyCFunction) _pycartesian_setValue, 1,
    "setValue((x,y),value) -> 1 on success otherwise 0\n\n"
    "Sets the value at the specified position. \n\n"
    "(x,y) - tuple with x & y position\n"
    "value - the value that should be set at specified position."
  },
  {"setConvertedValue", (PyCFunction) _pycartesian_setConvertedValue, 1,
    "setConvertedValue((x,y),value) -> 1 on success otherwise 0\n\n"
    "Sets the value at the specified position with gain & offset applied. Would be same as setValue((x, y), (v - offset)/gain). \n\n"
    "(x,y) - tuple with x & y position\n"
    "value - the value with offset/gain applied that should be set at specified position."
  },
  {"getValue", (PyCFunction) _pycartesian_getValue, 1,
    "getValue((x,y)) -> the value at the specified x and y position.\n\n"
    "Returns the value at the specified x and y position. \n\n"
    "(x,y) - tuple with x & y position\n"
  },
  {"getConvertedValue", (PyCFunction) _pycartesian_getConvertedValue, 1,
    "getConvertedValue((x,y)) -> the value at the specified x and y position.\n\n"
    "Returns the converted value at the specified x and y position. \n\n"
    "(x,y) - tuple with x & y position\n"
  },
  {"getValueAtLocation", (PyCFunction) _pycartesian_getValueAtLocation, 1,
    "getValueAtLocation((x,y)) -> the value at the specified x and y coordinate.\n\n"
    "Returns the value from the location as defined by the area definition. Same as calling c.getValue((c.getIndexX(),c.getIndexY())\n\n"
    "(x,y) - tuple with x & y coordinate\n"
  },
  {"getConvertedValueAtLocation", (PyCFunction) _pycartesian_getConvertedValueAtLocation, 1,
    "getConvertedValueAtLocation((x,y)) -> the converted value at the specified x and y position.\n\n"
    "Returns the value from the location as defined by the area definition. Same as calling c.getConvertedValue(c.getIndexX(), c.getIndexY() \n\n"
    "(x,y) - tuple with x & y coordinate\n"
  },
  {"getConvertedValueAtLonLat", (PyCFunction) _pycartesian_getConvertedValueAtLonLat, 1,
    "getConvertedValueAtLonLat((lon,lat)) -> the converted value at the specified lon/lat (in radians) position.\n\n"
    "Returns the value from the lon/lat coordinate. \n\n"
    "(lon,lat) - tuple with lon/lat coordinate in radians\n"
  },
  {"getQualityValueAtLocation", (PyCFunction) _pycartesian_getQualityValueAtLocation, 1,
    "getQualityValueAtLocation((x,y), fieldname) -> the quality value at the specified x/y coordinate.\n\n"
    "Returns the quality value from the specified quality field and location \n\n"
    "(x, y) - tuple with x/y coordinate \n"
    "fieldname  - how/task name of the quality field"
  },
  {"getConvertedQualityValueAtLocation", (PyCFunction) _pycartesian_getConvertedQualityValueAtLocation, 1,
    "getConvertedQualityValueAtLocation((x,y), fieldname) -> the converted quality value at the specified x/y coordinate.\n\n"
    "Returns the quality value from the specified quality field and location. Since offset & gain is not mandatory in the quality field. If they are missing, gain will be 1.0 and offset 0.0. \n\n"
    "(x, y) - tuple with x/y coordinate \n"
    "fieldname  - how/task name of the quality field"
  },
  {"getQualityValueAtLonLat", (PyCFunction) _pycartesian_getQualityValueAtLonLat, 1,
    "getQualityValueAtLonLat((lon,lat), fieldname) -> the quality value at the specified lon/lat coordinate.\n\n"
    "Returns the quality value from the specified quality field and location. \n\n"
    "(lon, lat) - tuple with lon/lat coordinate \n"
    "fieldname  - how/task name of the quality field"
  },
  {"getConvertedQualityValueAtLonLat", (PyCFunction) _pycartesian_getConvertedQualityValueAtLonLat, 1,
    "getConvertedQualityValueAtLonLat((lon,lat), fieldname) -> the converted quality value at the specified lon/lat coordinate.\n\n"
    "Returns the quality value from the specified quality field and location.  Since offset & gain is not mandatory in the quality field. If they are missing, gain will be 1.0 and offset 0.0. \n\n"
    "(lon, lat) - tuple with lon/lat coordinate \n"
    "fieldname  - how/task name of the quality field"
  },
  {"isTransformable", (PyCFunction) _pycartesian_isTransformable, 1,
    "isTransformable() -> a boolean.\n\n"
    "Returns if all preconditions are met in order to perform a transformation. \n\n"
  },
  {"getMean", (PyCFunction) _pycartesian_getMean, 1,
    "getMean((x,y), N) -> (datatype, the mean value) \n\n"
    "Returns the mean value over a NxN square around the specified x and y position. \n\n"
    "(x,y) - tuple with x/y position \n"
    "N     - Number of pixels in horizontal and vertical (NxN) direction around x,y"
  },
  {"addAttribute", (PyCFunction) _pycartesian_addAttribute, 1,
    "addAttribute(name, value) \n\n"
    "Adds an attribute to the volume. Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis etc. \n"
    "Currently, double, long, string and 1-dimensional arrays are supported.\n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr.\n"
    "value - Value to be associated with the name. Currently, double, long, string and 1-dimensional arrays are supported."
  },
  {"getAttribute", (PyCFunction) _pycartesian_getAttribute, 1,
    "getAttribute(name) -> value \n\n"
    "Returns the value associated with the specified name \n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr.\n"
  },
  {"getAttributeNames", (PyCFunction) _pycartesian_getAttributeNames, 1,
    "getAttributeNames() -> array of names \n\n"
    "Returns the attribute names associated with this cartesian object"
  },
  {"hasAttribute", (PyCFunction) _pycartesian_hasAttribute, 1,
    "hasAttribute(name) -> a boolean \n\n"
    "Returns if the specified name is defined within this cartesian object\n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis.\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr.\n"
  },
  {"isValid", (PyCFunction) _pycartesian_isValid, 1,
    "isValid(otype) -> a boolean \n\n"
    "Validates this cartesian object to see if it is possible to write as specified type.\n\n"
    "otype  - The type we want to save as, can be one of ObjectType_IMAGE, ObjectType_COMP or ObjectType_CVOL. Any other and it is assumed that it should be written as an IMAGE"
  },
  {"addQualityField", (PyCFunction) _pycartesian_addQualityField, 1,
    "addQualityField(field) \n\n"
    "Adds a quality field to this cartesian product. Note, there is no check for valid size or similar. Also, there is no check if same how/task is specified or the likes. \n\n"
    "field  - The RaveFieldCore field"
  },
  {"getNumberOfQualityFields", (PyCFunction) _pycartesian_getNumberOfQualityFields, 1,
    "getNumberOfQualityFields() -> integer\n\n"
    "Returns the number of quality fields in this cartesian product"
  },
  {"getQualityField", (PyCFunction) _pycartesian_getQualityField, 1,
    "getQualityField(index) -> RaveFieldCore \n\n"
    "Returns the rave field at specified index\n\n"
    "index  - The rave field at specified position.\n\n"
    "Throws IndexError if the rave field not could be found"
  },
  {"removeQualityField", (PyCFunction) _pycartesian_removeQualityField, 1,
    "removeQualityField(index) \n\n"
    "Removes the quality field at specified index\n\n"
    "index  - The rave field at specified position.\n\n"
  },
  {"getQualityFieldByHowTask", (PyCFunction) _pycartesian_getQualityFieldByHowTask, 1,
    "getQualityFieldByHowTask(name) -> RaveFieldCore or None \n\n"
    "Returns the quality with the how/task attribute equal to name\n\n"
    "name  - The rave field with how/task name equal to name\n\n"
  },
  {"findQualityFieldByHowTask", (PyCFunction) _pycartesian_findQualityFieldByHowTask, 1,
    "findQualityFieldByHowTask(name) -> RaveFieldCore or None \n\n"
    "Tries to locate any quality field with  how/task attribute equal to name. First, the current parameters quality fields are checked and then self.\n\n"
    "name  - The rave field with how/task name equal to name\n\n"
  },
  {"addParameter", (PyCFunction)_pycartesian_addParameter, 1,
    "addParameter(parameter) \n\n"
    "Adds a parameter to this cartesian product. Note, the quantity is essential in the cartesian parameter since that will be identifying each parameter.\n"
    "If a parameter with same quantity already exists in the cartesian product. That cartesian parameter will be replaced.\n\n"
    "parameter  - The CartesianParamCore instance"
  },
  {"createParameter", (PyCFunction)_pycartesian_createParameter, 1,
    "createParameter(quantity, type) -> parameter\n\n"
    "Creates a parameter with specified quantity and value type with same geometry as self.\n"
    "If a parameter with same quantity already exists in the cartesian product. The created cartesian parameter will be added to the internals of self.\n\n"
    "quantity  - A string representing the quantity like TH, DBZH, ...\n"
    "type      - The data type of the created field, e.g. _rave.RaveDataType_UCHAR, ...."
  },
  {"getParameter", (PyCFunction)_pycartesian_getParameter, 1,
    "getParameter(quantity) -> CartesianParamCore\n\n"
    "Returns the parameter with specified quantity if it exists\n\n"
    "quantity  - The quantity of the requested parameter\n\n"
    "Throws IndexError if no parameter exists with specified quantity"
  },
  {"hasParameter", (PyCFunction)_pycartesian_hasParameter, 1,
    "hasParameter(quantity) -> boolean\n\n"
    "Returns True or False depending if a parameter with specified quantity exists in this cartesian product. \n\n"
    "quantity  - The quantity of the requested parameter\n\n"
  },
  {"removeParameter", (PyCFunction)_pycartesian_removeParameter, 1,
    "hasParameter(quantity)\n\n"
    "Removes the parameter with the specified quantity if it exists. \n\n"
    "quantity  - The quantity of the parameter that should be removed\n\n"
  },
  {"getParameterCount", (PyCFunction)_pycartesian_getParameterCount, 1,
    "getParameterCount() -> number of parameters\n\n"
    "Returns the number of parameters that has been added to this cartesian product. \n\n"
  },
  {"getParameterNames", (PyCFunction)_pycartesian_getParameterNames, 1,
    "getParameterNames() -> list of quantities for parameters existing in this product\n\n"
    "Returns a list of quantities for the parameters that exists in this product"
  },
  {"clone", (PyCFunction)_pycartesian_clone, 1,
    "clone() -> a clone of self (CartesianCore)\n\n"
    "Creates a duplicate of self."
  },
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the cartesian
 * @param[in] self - the cartesian product
 */
static PyObject* _pycartesian_getattro(PyCartesian* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("time", name) == 0) {
    if (Cartesian_getTime(self->cartesian) != NULL) {
      return PyString_FromString(Cartesian_getTime(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("date", name) == 0) {
    if (Cartesian_getDate(self->cartesian) != NULL) {
      return PyString_FromString(Cartesian_getDate(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("objectType", name) == 0) {
    return PyInt_FromLong(Cartesian_getObjectType(self->cartesian));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("product", name) == 0) {
    return PyInt_FromLong(Cartesian_getProduct(self->cartesian));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("source", name) == 0) {
    if (Cartesian_getSource(self->cartesian) != NULL) {
      return PyRaveAPI_StringOrUnicode_FromASCII(Cartesian_getSource(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "prodname") == 0) {
    if (Cartesian_getProdname(self->cartesian) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyRaveAPI_StringOrUnicode_FromASCII(Cartesian_getProdname(self->cartesian));
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("xsize", name) == 0) {
    return PyInt_FromLong(Cartesian_getXSize(self->cartesian));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("ysize", name) == 0) {
    return PyInt_FromLong(Cartesian_getYSize(self->cartesian));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("xscale", name) == 0) {
    return PyFloat_FromDouble(Cartesian_getXScale(self->cartesian));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("yscale", name) == 0) {
    return PyFloat_FromDouble(Cartesian_getYScale(self->cartesian));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("areaextent", name) == 0) {
    double llX = 0.0, llY = 0.0, urX = 0.0, urY = 0.0;
    Cartesian_getAreaExtent(self->cartesian, &llX, &llY, &urX, &urY);
    return Py_BuildValue("(dddd)", llX, llY, urX, urY);
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("projection", name) == 0) {
    Projection_t* projection = Cartesian_getProjection(self->cartesian);
    if (projection != NULL) {
      PyProjection* result = PyProjection_New(projection);
      RAVE_OBJECT_RELEASE(projection);
      return (PyObject*)result;
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("starttime", name) == 0) {
    if (Cartesian_getStartTime(self->cartesian) != NULL) {
      return PyString_FromString(Cartesian_getStartTime(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("startdate", name) == 0) {
    if (Cartesian_getStartDate(self->cartesian) != NULL) {
      return PyString_FromString(Cartesian_getStartDate(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("endtime", name) == 0) {
    if (Cartesian_getEndTime(self->cartesian) != NULL) {
      return PyString_FromString(Cartesian_getEndTime(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("enddate", name) == 0) {
    if (Cartesian_getEndDate(self->cartesian) != NULL) {
      return PyString_FromString(Cartesian_getEndDate(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("defaultParameter", name) == 0) {
    if (Cartesian_getDefaultParameter(self->cartesian) != NULL) {
      return PyString_FromString(Cartesian_getDefaultParameter(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  }

  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pycartesian_setattro(PyCartesian* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("time", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setTime(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "time must be in the format HHmmss");
      }
    } else if (val == Py_None) {
      Cartesian_setTime(self->cartesian, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"time must be of type string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("date", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setDate(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "date must be in the format YYYYMMSS");
      }
    } else if (val == Py_None) {
      Cartesian_setDate(self->cartesian, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"date must be of type string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("objectType", name) == 0) {
    if (PyInt_Check(val)) {
      if (!Cartesian_setObjectType(self->cartesian, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "objectType not supported");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "objectType must be a valid object type")
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("product", name) == 0) {
    if (PyInt_Check(val)) {
      if (!Cartesian_setProduct(self->cartesian, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "product not supported");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "product must be a valid product type")
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("source", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setSource(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "Failed to set source");
      }
    } else if (val == Py_None) {
      Cartesian_setSource(self->cartesian, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"source must be of type string");
    }
  } else  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "prodname") == 0) {
      if (PyString_Check(val)) {
        if (!Cartesian_setProdname(self->cartesian, PyString_AsString(val))) {
          raiseException_gotoTag(done, PyExc_MemoryError, "Could not set prodname");
        }
      } else if (val == Py_None) {
        Cartesian_setProdname(self->cartesian, NULL);
      } else {
        raiseException_gotoTag(done, PyExc_TypeError,"prodname must be of type string");
      }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("xscale", name)==0) {
    if (PyFloat_Check(val)) {
      Cartesian_setXScale(self->cartesian, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"xscale must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("yscale", name)==0) {
    if (PyFloat_Check(val)) {
      Cartesian_setYScale(self->cartesian, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"yscale must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("areaextent", name) == 0) {
    double llX = 0.0, llY = 0.0, urX = 0.0, urY = 0.0;
    if (!PyArg_ParseTuple(val, "dddd", &llX, &llY, &urX, &urY)) {
      goto done;
    }
    Cartesian_setAreaExtent(self->cartesian, llX, llY, urX, urY);
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("projection", name) == 0) {
    if (PyProjection_Check(val)) {
      Cartesian_setProjection(self->cartesian, ((PyProjection*)val)->projection);
    } else if (val == Py_None) {
      Cartesian_setProjection(self->cartesian, NULL);
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("starttime", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setStartTime(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "starttime must be in the format HHmmss");
      }
    } else if (val == Py_None) {
      Cartesian_setStartTime(self->cartesian, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"time must be of type string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("startdate", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setStartDate(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "startdate must be in the format YYYYMMSS");
      }
    } else if (val == Py_None) {
      Cartesian_setStartDate(self->cartesian, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"date must be of type string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("endtime", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setEndTime(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "endtime must be in the format HHmmss");
      }
    } else if (val == Py_None) {
      Cartesian_setEndTime(self->cartesian, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"endtime must be of type string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("enddate", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setEndDate(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "enddate must be in the format YYYYMMSS");
      }
    } else if (val == Py_None) {
      Cartesian_setEndDate(self->cartesian, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"enddate must be of type string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("defaultParameter", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setDefaultParameter(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set defaultParameter");
      }
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"defaultParameter must be of type string");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }

  result = 0;
done:
  return result;
}

static PyObject* _pycartesian_isCartesian(PyObject* self, PyObject* args)
{
  PyObject* inobj = NULL;
  if (!PyArg_ParseTuple(args,"O", &inobj)) {
    return NULL;
  }
  if (PyCartesian_Check(inobj)) {
    return PyBool_FromLong(1);
  }
  return PyBool_FromLong(0);
}

/*@} End of Cartesian products */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pycartesian_type_doc,
    "The cartesian product represents a cartesian object in one or another way. There are several member attributes "
    "associated with a cartesian product as well a number of child objects like parameters, quality fields and more.\n"
    "Since a lot of RAVE has been developed with ODIM H5 in mind, it is also possible to add arbitrary attributes in "
    "various groups, e.g. c.addAttribute(\"how/this\", 1.2) and so on.\n\n"
    "A list of avilable member attributes are described below. For information about member functions, check each functions doc.\n"
    "\n"
    "time             - Time this cartesian product should represent as a string with format HHmmSS\n"
    "date             - Date this cartesian product should represent as a string in the format YYYYMMDD\n"
    "objectType       - The object type as defined in ODIM H5 this cartesian product should be defined as. Can be _rave.Rave_ObjectType_IMAGE or _raveRave_ObjectType_COMP\n"
    "product          - The product this cartesian product should represent as defined in ODIM H5. Can be for example _rave.Rave_ProductType_PPI or _rave.Rave_ProductType_PCAPPI\n"
    "source           - The source for this product. Defined as what/source in ODIM H5. I.e. a comma separated list of various identifiers. For example. NOD:seang,WMO:1234,....\n"
    "prodname         - The product name\n"
    "xsize            - The xsize of the area represented. ReadOnly, initialization occurs with for example the init-function.\n"
    "ysize            - The ysize of the area represented. ReadOnly, initialization occurs with for example the init-function.\n"
    "xscale           - The scale in meters in x-direction.\n"
    "yscale           - The scale in meters in y-direction.\n"
    "areaextent       - A tuple of four representing the outer boundaries of this cartesian product. Defined as (lower left X, lower left Y, upper right X, upper right Y).\n"
    "projection       - The projection object of type ProjectionCore that defines what projection that this cartesian product is defined with.\n"
    "starttime        - Start time for this product as a string with format HHmmSS.\n"
    "startdate        - Start date for this product as a string with format YYYYMMDD.\n"
    "endtime          - End time for this product as a string with format HHmmSS.\n"
    "enddate          - End date for this product as a string with format YYYYMMDD.\n"
    "defaultParameter - Since a cartesian product doesn't contain data by itself and instead contains a number of parameters like TH, DBZH, .... This setting allows the user to work directly with a parameter through the cartesian API.\n"
    "\n"
    "Usage:\n"
    " import _arearegistry, _projectionregistry\n"
    " reg = _arearegistry.load(\"area_registry.xml\", \n"
    "                         _projectionregistry.load(\"projection_registry.xml\"))\n"
    " c.init(reg.getByName(\"swegmaps_2000\"))\n"
    " th   = c.createParameter(\"TH\", _rave.RaveDataType_UCHAR)\n"
    " dbzh = c.createParameter(\"DBZH\", _rave.RaveDataType_UCHAR)\n"
    " c.defaultParameter = \"TH\"\n"
    " c.setValue((2,2), 1.0) # Sets 1.0 in parameter TH\n"
    " if th.getValue((2,2))[1] == c.getValue((2,2))[1]:\n"
    "   print(\"TH is same as cartesian\") # Will be written \n"
    " if th.getValue((2,2))[1] == c.getValue((2,2))[1]:\n"
    "   print(\"DBZH is same as cartesian\")  # Will not be written \n"
    );
/*@} End of Documentation about the type */


/*@{ Type definitions */
PyTypeObject PyCartesian_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "CartesianCore", /*tp_name*/
  sizeof(PyCartesian), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pycartesian_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pycartesian_getattro, /*tp_getattro*/
  (setattrofunc)_pycartesian_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pycartesian_type_doc,        /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pycartesian_methods,         /*tp_methods*/
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
  {"new", (PyCFunction)_pycartesian_new, 1,
    "new() -> new instance of the CartesianCore object\n\n"
    "Creates a new instance of the CartesianCore object"
  },
  {"isCartesian", (PyCFunction)_pycartesian_isCartesian, 1,
    "isCartesian(object) -> boolean\n\n"
    "Checks if provided object is of CartesianCore type or not.\n\n"
    "object - the object to check"
  },
  {NULL,NULL} /*Sentinel*/
};

/*@{ Documentation about the module */
PyDoc_STRVAR(_pycartesian_module_doc,
    "Represents a cartesian product\n"
    "\n"
    "Usage:\n"
    " import _cartesian\n"
    " c = _cartesian.new()\n"
    " c.init(area) # If area exists, then use it to initialize the cartesian product\n"
    );
/*@} End of Documentation about the module */

MOD_INIT(_cartesian)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyCartesian_API[PyCartesian_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyCartesian_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyCartesian_Type);

  MOD_INIT_DEF(module, "_cartesian", _pycartesian_module_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyCartesian_API[PyCartesian_Type_NUM] = (void*)&PyCartesian_Type;
  PyCartesian_API[PyCartesian_GetNative_NUM] = (void *)PyCartesian_GetNative;
  PyCartesian_API[PyCartesian_New_NUM] = (void*)PyCartesian_New;

  c_api_object = PyCapsule_New(PyCartesian_API, PyCartesian_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_cartesian.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _cartesian.error");
    return MOD_INIT_ERROR;
  }

  import_array(); /*To make sure I get access to Numeric*/
  import_pyprojection();
  import_pyarea();
  import_pyravefield();
  import_pycartesianparam();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
