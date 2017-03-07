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
 * Python version of the Cartesian Parameter API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2012-02-07
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYCARTESIANPARAM_MODULE        /**< to get correct part of pycartesianparam.h */
#include "pycartesianparam.h"

#include "pyravefield.h"

#include "arrayobject.h"
#include <arrayobject.h>
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_cartesianparam");

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
 * Returns the native CartesianParam_t instance.
 * @param[in] pyparam - the python cartesian parameter instance
 * @returns the native cartesian parameter instance.
 */
static CartesianParam_t*
PyCartesianParam_GetNative(PyCartesianParam* pyparam)
{
  RAVE_ASSERT((pyparam != NULL), "pyparam == NULL");
  return RAVE_OBJECT_COPY(pyparam->param);
}

/**
 * Creates a python cartesian parameter from a native cartesian parameter or will create an
 * initial native Cartesian parameter if p is NULL.
 * @param[in] p - the native cartesian parameter (or NULL)
 * @returns the python cartesian parameter.
 */
static PyCartesianParam*
PyCartesianParam_New(CartesianParam_t* p)
{
  PyCartesianParam* result = NULL;
  CartesianParam_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&CartesianParam_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for cartesian parameter.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for cartesian parameter.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyCartesianParam, &PyCartesianParam_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->param = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->param, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyCartesianParam instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for cartesian parameter.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the cartesian parameter
 * @param[in] obj the object to deallocate.
 */
static void _pycartesianparam_dealloc(PyCartesianParam* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->param, obj);
  RAVE_OBJECT_RELEASE(obj->param);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the cartesian parameter.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pycartesianparam_new(PyObject* self, PyObject* args)
{
  PyCartesianParam* result = PyCartesianParam_New(NULL);
  return (PyObject*)result;
}

/**
 * Sets the data array that should be used for this parameter.
 * @param[in] self this instance.
 * @param[in] args - the array
 * @return Py_None on success, otherwise NULL
 */
static PyObject* _pycartesianparam_setData(PyCartesianParam* self, PyObject* args)
{
  PyObject* inarray = NULL;
  PyArrayObject* arraydata = NULL;
  RaveDataType datatype = RaveDataType_UNDEFINED;
  long xsize = 0;
  long ysize = 0;
  unsigned char* data = NULL;

  if (!PyArg_ParseTuple(args, "O", &inarray)) {
    return NULL;
  }

  if (!PyArray_Check(inarray)) {
    raiseException_returnNULL(PyExc_TypeError, "Data must be of arrayobject type")
  }

  arraydata = (PyArrayObject*)inarray;

  if (PyArray_NDIM(arraydata) != 2) {
    raiseException_returnNULL(PyExc_ValueError, "A cartesian parameter must be of rank 2");
  }

  datatype = translate_pyarraytype_to_ravetype(PyArray_TYPE(arraydata));

  if (PyArray_ITEMSIZE(arraydata) != get_ravetype_size(datatype)) {
    raiseException_returnNULL(PyExc_TypeError, "numpy and rave does not have same data sizes");
  }

  xsize  = PyArray_DIM(arraydata, 1);
  ysize  = PyArray_DIM(arraydata, 0);
  data   = PyArray_DATA(arraydata);

  if (!CartesianParam_setData(self->param, xsize, ysize, data, datatype)) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not allocate memory");
  }

  Py_RETURN_NONE;
}

static PyObject* _pycartesianparam_getData(PyCartesianParam* self, PyObject* args)
{
  long xsize = 0, ysize = 0;
  RaveDataType type = RaveDataType_UNDEFINED;
  PyObject* result = NULL;
  npy_intp dims[2] = {0,0};
  int arrtype = 0;
  void* data = NULL;

  xsize = CartesianParam_getXSize(self->param);
  ysize = CartesianParam_getYSize(self->param);
  type = CartesianParam_getDataType(self->param);
  data = CartesianParam_getData(self->param);

  dims[1] = (npy_intp)xsize;
  dims[0] = (npy_intp)ysize;
  arrtype = translate_ravetype_to_pyarraytype(type);

  if (data == NULL) {
    raiseException_returnNULL(PyExc_IOError, "cartesian parameter does not have any data");
  }

  if (arrtype == PyArray_NOTYPE) {
    raiseException_returnNULL(PyExc_IOError, "Could not translate data type");
  }
  result = PyArray_SimpleNew(2, dims, arrtype);
  if (result == NULL) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not create resulting array");
  }
  if (result != NULL) {
    int nbytes = xsize*ysize*PyArray_ITEMSIZE(result);
    memcpy(((PyArrayObject*)result)->data, (unsigned char*)CartesianParam_getData(self->param), nbytes);
  }
  return result;
}

/**
 * sets the value at the specified position
 * @param[in] self this instance.
 * @param[in] args - tuple (x, y) and v
 * @return 0 on failure, otherwise 1
 */
static PyObject* _pycartesianparam_setValue(PyCartesianParam* self, PyObject* args)
{
  long x = 0, y = 0;
  double v = 0.0L;
  int result = 0;
  if (!PyArg_ParseTuple(args, "(ll)d", &x, &y, &v)) {
    return NULL;
  }

  result = CartesianParam_setValue(self->param, x, y, v);

  return PyInt_FromLong(result);
}

/**
 * sets the converted value at the specified position. Would be same
 * as setValue((x, y), (v - offset)/gain).
 * @param[in] self this instance.
 * @param[in] args - tuple (x, y) and v
 * @return 0 on failure, otherwise 1
 */
static PyObject* _pycartesianparam_setConvertedValue(PyCartesianParam* self, PyObject* args)
{
  long x = 0, y = 0;
  double v = 0.0L;
  int result = 0;
  if (!PyArg_ParseTuple(args, "(ll)d", &x, &y, &v)) {
    return NULL;
  }

  result = CartesianParam_setConvertedValue(self->param, x, y, v);

  return PyInt_FromLong(result);
}

/**
 * returns the value at the specified position
 * @param[in] self this instance.
 * @param[in] args - tuple (x, y) and v
 * @return 0 on failure, otherwise 1
 */
static PyObject* _pycartesianparam_getValue(PyCartesianParam* self, PyObject* args)
{
  long x = 0, y = 0;
  double v = 0.0L;
  RaveValueType result = RaveValueType_NODATA;
  if (!PyArg_ParseTuple(args, "(ll)", &x, &y)) {
    return NULL;
  }

  result = CartesianParam_getValue(self->param, x, y, &v);

  return Py_BuildValue("(id)", result, v);
}

/**
 * returns the converted value at the specified position
 * @param[in] self this instance.
 * @param[in] args - tuple (x, y) and v
 * @return 0 on failure, otherwise 1
 */
static PyObject* _pycartesianparam_getConvertedValue(PyCartesianParam* self, PyObject* args)
{
  long x = 0, y = 0;
  double v = 0.0L;
  RaveValueType result = RaveValueType_NODATA;
  if (!PyArg_ParseTuple(args, "(ll)", &x, &y)) {
    return NULL;
  }

  result = CartesianParam_getConvertedValue(self->param, x, y, &v);

  return Py_BuildValue("(id)", result, v);
}

static PyObject* _pycartesianparam_isTransformable(PyCartesianParam* self, PyObject* args)
{
  return PyBool_FromLong(CartesianParam_isTransformable(self->param));
}

static PyObject* _pycartesianparam_getMean(PyCartesianParam* self, PyObject* args)
{
  long x = 0, y = 0;
  int N = 0;
  double v = 0.0L;
  RaveValueType result = RaveValueType_NODATA;
  if (!PyArg_ParseTuple(args, "(ll)i", &x, &y, &N)) {
    return NULL;
  }
  result = CartesianParam_getMean(self->param, x, y, N, &v);

  return Py_BuildValue("(id)", result, v);
}

/**
 * Adds an attribute to the parameter. Name of the attribute should be in format
 * ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis etc.
 * Currently, the only supported values are double, long, string.
 * @param[in] self - this instance
 * @param[in] args - bin index, ray index.
 * @returns true or false depending if it works.
 */
static PyObject* _pycartesianparam_addAttribute(PyCartesianParam* self, PyObject* args)
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

  if (!CartesianParam_addAttribute(self->param, attr)) {
    raiseException_gotoTag(done, PyExc_AttributeError, "Failed to add attribute");
  }

  result = PyBool_FromLong(1);
done:
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

static PyObject* _pycartesianparam_getAttribute(PyCartesianParam* self, PyObject* args)
{
  RaveAttribute_t* attribute = NULL;
  char* name = NULL;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }

  attribute = CartesianParam_getAttribute(self->param, name);
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

static PyObject* _pycartesianparam_getAttributeNames(PyCartesianParam* self, PyObject* args)
{
  RaveList_t* list = NULL;
  PyObject* result = NULL;
  int n = 0;
  int i = 0;

  list = CartesianParam_getAttributeNames(self->param);
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
 * Adds a quality field to the cartesian parameter
 * @param[in] self - this instance
 * @param[in] args - object, of type RaveFieldCore
 * @returns None
 */
static PyObject* _pycartesianparam_addQualityField(PyCartesianParam* self, PyObject* args)
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

  if (!CartesianParam_addQualityField(self->param, ravefield->field)) {
    raiseException_returnNULL(PyExc_AttributeError, "Failed to add quality field to cartesian parameter");
  }

  Py_RETURN_NONE;
}

/**
 * Returns the number of quality fields
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns The number of quality fields
 */
static PyObject* _pycartesianparam_getNumberOfQualityFields(PyCartesianParam* self, PyObject* args)
{
  return PyInt_FromLong(CartesianParam_getNumberOfQualityFields(self->param));
}

/**
 * Returns the number of quality fields
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns The number of quality fields
 */
static PyObject* _pycartesianparam_getQualityField(PyCartesianParam* self, PyObject* args)
{
  PyObject* result = NULL;
  RaveField_t* field = NULL;
  int index = 0;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  if (index < 0 || index >= CartesianParam_getNumberOfQualityFields(self->param)) {
    raiseException_returnNULL(PyExc_IndexError, "Index out of bounds");
  }

  if ((field = CartesianParam_getQualityField(self->param, index)) == NULL) {
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
static PyObject* _pycartesianparam_removeQualityField(PyCartesianParam* self, PyObject* args)
{
  int index = 0;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  CartesianParam_removeQualityField(self->param, index);

  Py_RETURN_NONE;
}

/**
 * Returns a quality field based on the value of how/task that should be a
 * string.
 * @param[in] self - self
 * @param[in] args - the how/task value string
 * @return the field if found otherwise NULL
 */
static PyObject* _pycartesianparam_getQualityFieldByHowTask(PyCartesianParam* self, PyObject* args)
{
  PyObject* result = NULL;
  char* value = NULL;
  RaveField_t* field = NULL;

  if (!PyArg_ParseTuple(args, "s", &value)) {
    return NULL;
  }
  field = CartesianParam_getQualityFieldByHowTask(self->param, value);
  if (field == NULL) {
    raiseException_gotoTag(done, PyExc_NameError, "Could not locate quality field");
  }
  result = (PyObject*)PyRaveField_New(field);
done:
  RAVE_OBJECT_RELEASE(field);
  return result;
}

/**
 * All methods a cartesian product can have
 */
static struct PyMethodDef _pycartesianparam_methods[] =
{
  {"xsize", NULL},
  {"ysize", NULL},
  {"quantity", NULL},
  {"gain", NULL},
  {"offset", NULL},
  {"nodata", NULL},
  {"undetect", NULL},
  {"datatype", NULL},
  {"setData", (PyCFunction) _pycartesianparam_setData, 1},
  {"getData", (PyCFunction) _pycartesianparam_getData, 1},
  {"setValue", (PyCFunction) _pycartesianparam_setValue, 1},
  {"setConvertedValue", (PyCFunction) _pycartesianparam_setConvertedValue, 1},
  {"getValue", (PyCFunction) _pycartesianparam_getValue, 1},
  {"getConvertedValue", (PyCFunction) _pycartesianparam_getConvertedValue, 1},
  {"isTransformable", (PyCFunction) _pycartesianparam_isTransformable, 1},
  {"getMean", (PyCFunction) _pycartesianparam_getMean, 1},
  {"addAttribute", (PyCFunction) _pycartesianparam_addAttribute, 1},
  {"getAttribute", (PyCFunction) _pycartesianparam_getAttribute, 1},
  {"getAttributeNames", (PyCFunction) _pycartesianparam_getAttributeNames, 1},
  {"addQualityField", (PyCFunction) _pycartesianparam_addQualityField, 1},
  {"getNumberOfQualityFields", (PyCFunction) _pycartesianparam_getNumberOfQualityFields, 1},
  {"getQualityField", (PyCFunction) _pycartesianparam_getQualityField, 1},
  {"removeQualityField", (PyCFunction) _pycartesianparam_removeQualityField, 1},
  {"getQualityFieldByHowTask", (PyCFunction) _pycartesianparam_getQualityFieldByHowTask, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the cartesian
 * @param[in] self - the cartesian product
 */
static PyObject* _pycartesianparam_getattro(PyCartesianParam* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "xsize") == 0) {
    return PyInt_FromLong(CartesianParam_getXSize(self->param));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "ysize") == 0) {
    return PyInt_FromLong(CartesianParam_getYSize(self->param));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "quantity") == 0) {
    if (CartesianParam_getQuantity(self->param) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(CartesianParam_getQuantity(self->param));
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "gain") == 0) {
    return PyFloat_FromDouble(CartesianParam_getGain(self->param));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "offset") == 0) {
    return PyFloat_FromDouble(CartesianParam_getOffset(self->param));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "nodata") == 0) {
    return PyFloat_FromDouble(CartesianParam_getNodata(self->param));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "undetect") == 0) {
    return PyFloat_FromDouble(CartesianParam_getUndetect(self->param));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "datatype") == 0) {
    return PyInt_FromLong(CartesianParam_getDataType(self->param));
  }

  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pycartesianparam_setattro(PyCartesianParam* self, PyObject *name, PyObject *val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "quantity") == 0) {
    if (PyString_Check(val)) {
      if (!CartesianParam_setQuantity(self->param, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_MemoryError, "Could not set quantity");
      }
    } else if (val == Py_None) {
      CartesianParam_setQuantity(self->param, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"quantity must be of type string");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "gain") == 0) {
    if (PyFloat_Check(val)) {
      CartesianParam_setGain(self->param, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "gain must be of type float");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "offset") == 0) {
    if (PyFloat_Check(val)) {
      CartesianParam_setOffset(self->param, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "offset must be of type float");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "nodata") == 0) {
    if (PyFloat_Check(val)) {
      CartesianParam_setNodata(self->param, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "nodata must be of type float");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "undetect") == 0) {
    if (PyFloat_Check(val)) {
      CartesianParam_setUndetect(self->param, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "undetect must be of type float");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }

  result = 0;
done:
  return result;
}

/*@} End of Cartesian products */

/*@{ Type definitions */
PyTypeObject PyCartesianParam_Type =
{
   PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "CartesianParamCore", /*tp_name*/
  sizeof(PyCartesianParam), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pycartesianparam_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pycartesianparam_getattro, /*tp_getattro*/
  (setattrofunc)_pycartesianparam_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  0,                            /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pycartesianparam_methods,    /*tp_methods*/
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
  0,                            /*tp_is_gc*/};
/*@} End of Type definitions */

/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pycartesianparam_new, 1},
  {NULL,NULL} /*Sentinel*/
};
#ifdef KALLE
PyMODINIT_FUNC
init_cartesianparam(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyCartesianParam_API[PyCartesianParam_API_pointers];
  PyObject *c_api_object = NULL;
  PyCartesianParam_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_cartesianparam", functions);
  if (module == NULL) {
    return;
  }
  PyCartesianParam_API[PyCartesianParam_Type_NUM] = (void*)&PyCartesianParam_Type;
  PyCartesianParam_API[PyCartesianParam_GetNative_NUM] = (void *)PyCartesianParam_GetNative;
  PyCartesianParam_API[PyCartesianParam_New_NUM] = (void*)PyCartesianParam_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyCartesianParam_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_cartesianparam.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _cartesianparam.error");
  }

  import_array(); /*To make sure I get access to Numeric*/
  import_pyravefield();
  PYRAVE_DEBUG_INITIALIZE;
}
#endif

MOD_INIT(_cartesianparam)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyCartesianParam_API[PyCartesianParam_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyCartesianParam_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyCartesianParam_Type);

  MOD_INIT_DEF(module, "_cartesianparam", NULL/*doc*/, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyCartesianParam_API[PyCartesianParam_Type_NUM] = (void*)&PyCartesianParam_Type;
  PyCartesianParam_API[PyCartesianParam_GetNative_NUM] = (void *)PyCartesianParam_GetNative;
  PyCartesianParam_API[PyCartesianParam_New_NUM] = (void*)PyCartesianParam_New;

  c_api_object = PyCapsule_New(PyCartesianParam_API, PyCartesianParam_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_cartesianparam.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _cartesianparam.error");
    return MOD_INIT_ERROR;
  }

  import_array(); /*To make sure I get access to Numeric*/
  import_pyravefield();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}

/*@} End of Module setup */
