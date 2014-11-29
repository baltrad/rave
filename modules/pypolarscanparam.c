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

#define PYPOLARSCANPARAM_MODULE   /**< to get correct part of pypolarscanparam,h */
#include "pypolarscanparam.h"

#include <arrayobject.h>
#include "pyrave_debug.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"
#include "pyravefield.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_polarscanparam");

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
/// Polar Scan Param
/// --------------------------------------------------------------------
/*@{ Polar Scan Param */
/**
 * Returns the native PolarScanParam_t instance.
 * @param[in] pypolarscanparam - the python polar scan param instance
 * @returns the native polar scan param instance.
 */
static PolarScanParam_t*
PyPolarScanParam_GetNative(PyPolarScanParam* pypolarscanparam)
{
  RAVE_ASSERT((pypolarscanparam != NULL), "pypolarscanparam == NULL");
  return RAVE_OBJECT_COPY(pypolarscanparam->scanparam);
}

/**
 * Creates a python polar scan param from a native polar scan param or will create an
 * initial native PolarScanParam if p is NULL.
 * @param[in] p - the native polar scan param (or NULL)
 * @returns the python polar scan param .
 */
static PyPolarScanParam* PyPolarScanParam_New(PolarScanParam_t* p)
{
  PyPolarScanParam* result = NULL;
  PolarScanParam_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&PolarScanParam_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for polar scan param.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for polar scan param.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyPolarScanParam, &PyPolarScanParam_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->scanparam = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->scanparam, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyPolarScanParam instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for polar scan param.");
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
static void _pypolarscanparam_dealloc(PyPolarScanParam* obj)
{
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->scanparam, obj);
  RAVE_OBJECT_RELEASE(obj->scanparam);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the polar scan.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pypolarscanparam_new(PyObject* self, PyObject* args)
{
  PyPolarScanParam* result = PyPolarScanParam_New(NULL);
  return (PyObject*)result;
}

/**
 * Translates a py rave field into a py polar scan parameter. It will only atempt to find
 * to use what/gain, what/offset, what/nodata and what/undetect. If these doesn't
 * exist, default values will be used.
 * @param[in] field - the field to convert into a polar scan parameter
 * @returns a polar scan parameter on success otherwise NULL
 */
static PyObject* _pypolarscanparam_fromField(PyObject* self, PyObject* args)
{
  PyObject* pyobj = NULL;
  PolarScanParam_t* param = NULL;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "O", &pyobj)) {
    return NULL;
  }
  if (!PyRaveField_Check(pyobj)) {
    raiseException_returnNULL(PyExc_AttributeError, "Indata should be a rave field");
  }
  param = PolarScanParam_fromField(((PyRaveField*)pyobj)->field);
  if (param == NULL) {
    raiseException_returnNULL(PyExc_RuntimeError, "Could not create parameter from field");
  }
  result = (PyObject*)PyPolarScanParam_New(param);
  RAVE_OBJECT_RELEASE(param);
  return result;
}

static PyObject* _pypolarscanparam_setData(PyPolarScanParam* self, PyObject* args)
{
  PyObject* inarray = NULL;
  PyArrayObject* arraydata = NULL;
  RaveDataType datatype = RaveDataType_UNDEFINED;
  long nbins = 0;
  long nrays = 0;
  unsigned char* data = NULL;

  if (!PyArg_ParseTuple(args, "O", &inarray)) {
    return NULL;
  }

  if (!PyArray_Check(inarray)) {
    raiseException_returnNULL(PyExc_TypeError, "Data must be of arrayobject type")
  }

  arraydata = (PyArrayObject*)inarray;

  if (PyArray_NDIM(arraydata) != 2) {
    raiseException_returnNULL(PyExc_ValueError, "A scan param must be of rank 2");
  }

  datatype = translate_pyarraytype_to_ravetype(PyArray_TYPE(arraydata));

  if (PyArray_ITEMSIZE(arraydata) != get_ravetype_size(datatype)) {
    raiseException_returnNULL(PyExc_TypeError, "numpy and rave does not have same data sizes");
  }

  nbins  = PyArray_DIM(arraydata, 1);
  nrays  = PyArray_DIM(arraydata, 0);
  data   = PyArray_DATA(arraydata);

  if (!PolarScanParam_setData(self->scanparam, nbins, nrays, data, datatype)) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not allocate memory");
  }

  Py_RETURN_NONE;
}

static PyObject* _pypolarscanparam_getData(PyPolarScanParam* self, PyObject* args)
{
  long nbins = 0, nrays = 0;
  RaveDataType type = RaveDataType_UNDEFINED;
  PyObject* result = NULL;
  npy_intp dims[2] = {0,0};
  int arrtype = 0;
  void* data = NULL;

  nbins = PolarScanParam_getNbins(self->scanparam);
  nrays = PolarScanParam_getNrays(self->scanparam);
  type = PolarScanParam_getDataType(self->scanparam);
  data = PolarScanParam_getData(self->scanparam);

  dims[0] = (npy_intp)nrays;
  dims[1] = (npy_intp)nbins;
  arrtype = translate_ravetype_to_pyarraytype(type);

  if (data == NULL) {
    raiseException_returnNULL(PyExc_IOError, "polar scan does not have any data");
  }

  if (arrtype == PyArray_NOTYPE) {
    raiseException_returnNULL(PyExc_IOError, "Could not translate data type");
  }
  result = PyArray_SimpleNew(2, dims, arrtype);
  if (result == NULL) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not create resulting array");
  }
  if (result != NULL) {
    int nbytes = nbins*nrays*((PyArrayObject*)result)->descr->elsize;
    memcpy(((PyArrayObject*)result)->data, PolarScanParam_getData(self->scanparam), nbytes);
  }

  return result;
}

/**
 * Returns the value at the specified ray and bin index.
 * @param[in] self - this instance
 * @param[in] args - bin index, ray index.
 * @returns a tuple of value type and value
 */
static PyObject* _pypolarscanparam_getValue(PyPolarScanParam* self, PyObject* args)
{
  double value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  int ray = 0, bin = 0;
  if (!PyArg_ParseTuple(args, "ii", &bin, &ray)) {
    return NULL;
  }

  type = PolarScanParam_getValue(self->scanparam, bin, ray, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Returns the converted value at the specified ray and bin index.
 * @param[in] self - this instance
 * @param[in] args - bin index, ray index.
 * @returns a tuple of value type and value
 */
static PyObject* _pypolarscanparam_getConvertedValue(PyPolarScanParam* self, PyObject* args)
{
  double value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  int ray = 0, bin = 0;
  if (!PyArg_ParseTuple(args, "ii", &bin, &ray)) {
    return NULL;
  }

  type = PolarScanParam_getConvertedValue(self->scanparam, bin, ray, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * sets the value at the specified position
 * @param[in] self this instance.
 * @param[in] args - tuple (bin, ray) and v
 * @return 0 on failure, otherwise 1
 */
static PyObject* _pypolarscanparam_setValue(PyPolarScanParam* self, PyObject* args)
{
  long bin = 0, ray = 0;
  double v = 0.0L;
  if (!PyArg_ParseTuple(args, "(ll)d", &bin, &ray, &v)) {
    return NULL;
  }

  if(!PolarScanParam_setValue(self->scanparam, bin, ray, v)) {
    raiseException_returnNULL(PyExc_ValueError, "Could not set value");
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
static PyObject* _pypolarscanparam_addAttribute(PyPolarScanParam* self, PyObject* args)
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

  if (!PolarScanParam_addAttribute(self->scanparam, attr)) {
    raiseException_gotoTag(done, PyExc_AttributeError, "Failed to add attribute");
  }

  result = PyBool_FromLong(1);
done:
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

static PyObject* _pypolarscanparam_getAttribute(PyPolarScanParam* self, PyObject* args)
{
  RaveAttribute_t* attribute = NULL;
  char* name = NULL;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }

  attribute = PolarScanParam_getAttribute(self->scanparam, name);
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
static PyObject* _pypolarscanparam_hasAttribute(PyPolarScanParam* self, PyObject* args)
{
  char* name = NULL;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  return PyBool_FromLong((long)PolarScanParam_hasAttribute(self->scanparam, name));
}

static PyObject* _pypolarscanparam_getAttributeNames(PyPolarScanParam* self, PyObject* args)
{
  RaveList_t* list = NULL;
  PyObject* result = NULL;
  int n = 0;
  int i = 0;

  list = PolarScanParam_getAttributeNames(self->scanparam);
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
 * Adds a quality field to the scan param
 * @param[in] self - this instance
 * @param[in] args - object, of type RaveFieldCore
 * @returns None
 */
static PyObject* _pypolarscanparam_addQualityField(PyPolarScanParam* self, PyObject* args)
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

  if (!PolarScanParam_addQualityField(self->scanparam, ravefield->field)) {
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
static PyObject* _pypolarscanparam_getNumberOfQualityFields(PyPolarScanParam* self, PyObject* args)
{
  return PyInt_FromLong(PolarScanParam_getNumberOfQualityFields(self->scanparam));
}

/**
 * Returns the number of quality fields
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns The number of quality fields
 */
static PyObject* _pypolarscanparam_getQualityField(PyPolarScanParam* self, PyObject* args)
{
  PyObject* result = NULL;
  RaveField_t* field = NULL;
  int index = 0;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  if (index < 0 || index >= PolarScanParam_getNumberOfQualityFields(self->scanparam)) {
    raiseException_returnNULL(PyExc_IndexError, "Index out of bounds");
  }

  if ((field = PolarScanParam_getQualityField(self->scanparam, index)) == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "Could not get quality field");
  }

  result = (PyObject*)PyRaveField_New(field);

  RAVE_OBJECT_RELEASE(field);

  return result;
}

/**
 * Removes the specified quality field from the scan param
 * @param[in] self - this instance
 * @param[in] args - the index of the field to be removed
 * @returns None
 */
static PyObject* _pypolarscanparam_removeQualityField(PyPolarScanParam* self, PyObject* args)
{
  int index = 0;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  PolarScanParam_removeQualityField(self->scanparam, index);

  Py_RETURN_NONE;
}

/**
 * Converts the data field portion and the corresponding attributes into
 * a rave field. I.e. no quality fields will be affected.
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns the rave field on success otherwise NULL
 */
static PyObject* _pypolarscanparam_toField(PyPolarScanParam* self, PyObject* args)
{
  PyObject* result = NULL;
  RaveField_t* field = NULL;

  if (!PyArg_ParseTuple(args, "")) {
      return NULL;
    }
  field = PolarScanParam_toField(self->scanparam);
  if (field == NULL) {
    raiseException_returnNULL(PyExc_RuntimeError, "Failed to convert parameter into a field");
  }
  result = (PyObject*)PyRaveField_New(field);
  RAVE_OBJECT_RELEASE(field);
  return result;
}

/**
 * Converter for 64-bit float (from BUFR) to 8-bit uint,
 * primarily for reverting reflectivity data back to what they once were.
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns None
 */
static PyObject* _pypolarscanparam_convertDataDoubleToUchar(PyPolarScanParam* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
      return NULL;
    }
  if (!PolarScanParam_convertDataDoubleToUchar(self->scanparam)) {
    raiseException_returnNULL(PyExc_TypeError, "Failed to convert dataset, not double data?");
  }
  Py_RETURN_NONE;
}

/**
 * All methods a polar scan can have
 */
static struct PyMethodDef _pypolarscanparam_methods[] =
{
  {"nbins", NULL},
  {"nrays", NULL},
  {"quantity", NULL},
  {"gain", NULL},
  {"offset", NULL},
  {"nodata", NULL},
  {"undetect", NULL},
  {"datatype", NULL},
  {"setData", (PyCFunction) _pypolarscanparam_setData, 1},
  {"getData", (PyCFunction) _pypolarscanparam_getData, 1},
  {"getValue", (PyCFunction) _pypolarscanparam_getValue, 1},
  {"getConvertedValue", (PyCFunction) _pypolarscanparam_getConvertedValue, 1},
  {"setValue", (PyCFunction) _pypolarscanparam_setValue, 1},
  {"addAttribute", (PyCFunction) _pypolarscanparam_addAttribute, 1},
  {"getAttribute", (PyCFunction) _pypolarscanparam_getAttribute, 1},
  {"hasAttribute", (PyCFunction) _pypolarscanparam_hasAttribute, 1},
  {"getAttributeNames", (PyCFunction) _pypolarscanparam_getAttributeNames, 1},
  {"addQualityField", (PyCFunction) _pypolarscanparam_addQualityField, 1},
  {"getNumberOfQualityFields", (PyCFunction) _pypolarscanparam_getNumberOfQualityFields, 1},
  {"getQualityField", (PyCFunction) _pypolarscanparam_getQualityField, 1},
  {"removeQualityField", (PyCFunction) _pypolarscanparam_removeQualityField, 1},
  {"toField", (PyCFunction)_pypolarscanparam_toField, 1},
  {"convertDataDoubleToUchar", (PyCFunction)_pypolarscanparam_convertDataDoubleToUchar, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the polar scan
 */
static PyObject* _pypolarscanparam_getattr(PyPolarScanParam* self, char* name)
{
  PyObject* res;
  if (strcmp("nbins", name) == 0) {
    return PyInt_FromLong(PolarScanParam_getNbins(self->scanparam));
  } else if (strcmp("nrays", name) == 0) {
    return PyInt_FromLong(PolarScanParam_getNrays(self->scanparam));
  } else if (strcmp("quantity", name) == 0) {
    const char* str = PolarScanParam_getQuantity(self->scanparam);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("gain", name) == 0) {
    return PyFloat_FromDouble(PolarScanParam_getGain(self->scanparam));
  } else if (strcmp("offset", name) == 0) {
    return PyFloat_FromDouble(PolarScanParam_getOffset(self->scanparam));
  } else if (strcmp("nodata", name) == 0) {
    return PyFloat_FromDouble(PolarScanParam_getNodata(self->scanparam));
  } else if (strcmp("undetect", name) == 0) {
    return PyFloat_FromDouble(PolarScanParam_getUndetect(self->scanparam));
  } else if (strcmp("datatype", name) == 0) {
    return PyInt_FromLong(PolarScanParam_getDataType(self->scanparam));
  }

  res = Py_FindMethod(_pypolarscanparam_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pypolarscanparam_setattr(PyPolarScanParam* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("quantity", name) == 0) {
    if (PyString_Check(val)) {
      if (!PolarScanParam_setQuantity(self->scanparam, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "quantity must be a string");
      }
    } else if (val == Py_None) {
      PolarScanParam_setQuantity(self->scanparam, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "quantity must be a string");
    }
  } else if (strcmp("gain", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScanParam_setGain(self->scanparam, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "gain must be of type float");
    }
  } else if (strcmp("offset", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScanParam_setOffset(self->scanparam, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "offset must be of type float");
    }
  } else if (strcmp("nodata", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScanParam_setNodata(self->scanparam, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "nodata must be of type float");
    }
  } else if (strcmp("undetect", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScanParam_setUndetect(self->scanparam, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "undetect must be of type float");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, name);
  }

  result = 0;
done:
  return result;
}
/*@} End of Polar Scan Param */

/// --------------------------------------------------------------------
/// Type definitions
/// --------------------------------------------------------------------
/*@{ Type definition */
PyTypeObject PyPolarScanParam_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "PolarScanParamCore", /*tp_name*/
  sizeof(PyPolarScanParam), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pypolarscanparam_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pypolarscanparam_getattr, /*tp_getattr*/
  (setattrfunc)_pypolarscanparam_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_pypolarscanparam_new, 1},
  {"fromField", (PyCFunction)_pypolarscanparam_fromField, 1},
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_polarscanparam(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyPolarScanParam_API[PyPolarScanParam_API_pointers];
  PyObject *c_api_object = NULL;
  PyPolarScanParam_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_polarscanparam", functions);
  if (module == NULL) {
    return;
  }
  PyPolarScanParam_API[PyPolarScanParam_Type_NUM] = (void*)&PyPolarScanParam_Type;
  PyPolarScanParam_API[PyPolarScanParam_GetNative_NUM] = (void *)PyPolarScanParam_GetNative;
  PyPolarScanParam_API[PyPolarScanParam_New_NUM] = (void*)PyPolarScanParam_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyPolarScanParam_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_polarscanparam.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _polarscanparam.error");
  }

  import_array(); /*To make sure I get access to Numeric*/
  import_pyravefield();
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
