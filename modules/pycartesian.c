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
#include "Python.h"
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

  param = Cartesian_createParameter(self->cartesian, quantity, type);
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
 * All methods a cartesian product can have
 */
static struct PyMethodDef _pycartesian_methods[] =
{
  {"time", NULL},
  {"date", NULL},
  {"objectType", NULL},
  {"product", NULL},
  {"source", NULL},
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
  {"init", (PyCFunction) _pycartesian_init, 1},
  {"getLocationX", (PyCFunction) _pycartesian_getLocationX, 1},
  {"getLocationY", (PyCFunction) _pycartesian_getLocationY, 1},
  {"getIndexX", (PyCFunction) _pycartesian_getIndexX, 1},
  {"getIndexY", (PyCFunction) _pycartesian_getIndexY, 1},
  {"setValue", (PyCFunction) _pycartesian_setValue, 1},
  {"setConvertedValue", (PyCFunction) _pycartesian_setConvertedValue, 1},
  {"getValue", (PyCFunction) _pycartesian_getValue, 1},
  {"getConvertedValue", (PyCFunction) _pycartesian_getConvertedValue, 1},
  {"isTransformable", (PyCFunction) _pycartesian_isTransformable, 1},
  {"getMean", (PyCFunction) _pycartesian_getMean, 1},
  {"addAttribute", (PyCFunction) _pycartesian_addAttribute, 1},
  {"getAttribute", (PyCFunction) _pycartesian_getAttribute, 1},
  {"getAttributeNames", (PyCFunction) _pycartesian_getAttributeNames, 1},
  {"isValid", (PyCFunction) _pycartesian_isValid, 1},
  {"addQualityField", (PyCFunction) _pycartesian_addQualityField, 1},
  {"getNumberOfQualityFields", (PyCFunction) _pycartesian_getNumberOfQualityFields, 1},
  {"getQualityField", (PyCFunction) _pycartesian_getQualityField, 1},
  {"removeQualityField", (PyCFunction) _pycartesian_removeQualityField, 1},
  {"addParameter", (PyCFunction)_pycartesian_addParameter, 1},
  {"createParameter", (PyCFunction)_pycartesian_createParameter, 1},
  {"getParameter", (PyCFunction)_pycartesian_getParameter, 1},
  {"hasParameter", (PyCFunction)_pycartesian_hasParameter, 1},
  {"removeParameter", (PyCFunction)_pycartesian_removeParameter, 1},
  {"getParameterCount", (PyCFunction)_pycartesian_getParameterCount, 1},
  {"getParameterNames", (PyCFunction)_pycartesian_getParameterNames, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the cartesian
 * @param[in] self - the cartesian product
 */
static PyObject* _pycartesian_getattr(PyCartesian* self, char* name)
{
  PyObject* res = NULL;

  if (strcmp("time", name) == 0) {
    if (Cartesian_getTime(self->cartesian) != NULL) {
      return PyString_FromString(Cartesian_getTime(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("date", name) == 0) {
    if (Cartesian_getDate(self->cartesian) != NULL) {
      return PyString_FromString(Cartesian_getDate(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("objectType", name) == 0) {
    return PyInt_FromLong(Cartesian_getObjectType(self->cartesian));
  } else if (strcmp("product", name) == 0) {
    return PyInt_FromLong(Cartesian_getProduct(self->cartesian));
  } else if (strcmp("source", name) == 0) {
    if (Cartesian_getSource(self->cartesian) != NULL) {
      return PyString_FromString(Cartesian_getSource(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("xsize", name) == 0) {
    return PyInt_FromLong(Cartesian_getXSize(self->cartesian));
  } else if (strcmp("ysize", name) == 0) {
    return PyInt_FromLong(Cartesian_getYSize(self->cartesian));
  } else if (strcmp("xscale", name) == 0) {
    return PyFloat_FromDouble(Cartesian_getXScale(self->cartesian));
  } else if (strcmp("yscale", name) == 0) {
    return PyFloat_FromDouble(Cartesian_getYScale(self->cartesian));
  } else if (strcmp("areaextent", name) == 0) {
    double llX = 0.0, llY = 0.0, urX = 0.0, urY = 0.0;
    Cartesian_getAreaExtent(self->cartesian, &llX, &llY, &urX, &urY);
    return Py_BuildValue("(dddd)", llX, llY, urX, urY);
  } else if (strcmp("projection", name) == 0) {
    Projection_t* projection = Cartesian_getProjection(self->cartesian);
    if (projection != NULL) {
      PyProjection* result = PyProjection_New(projection);
      RAVE_OBJECT_RELEASE(projection);
      return (PyObject*)result;
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("starttime", name) == 0) {
    if (Cartesian_getStartTime(self->cartesian) != NULL) {
      return PyString_FromString(Cartesian_getStartTime(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("startdate", name) == 0) {
    if (Cartesian_getStartDate(self->cartesian) != NULL) {
      return PyString_FromString(Cartesian_getStartDate(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("endtime", name) == 0) {
    if (Cartesian_getEndTime(self->cartesian) != NULL) {
      return PyString_FromString(Cartesian_getEndTime(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("enddate", name) == 0) {
    if (Cartesian_getEndDate(self->cartesian) != NULL) {
      return PyString_FromString(Cartesian_getEndDate(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("defaultParameter", name) == 0) {
    if (Cartesian_getDefaultParameter(self->cartesian) != NULL) {
      return PyString_FromString(Cartesian_getDefaultParameter(self->cartesian));
    } else {
      Py_RETURN_NONE;
    }
  }

  res = Py_FindMethod(_pycartesian_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pycartesian_setattr(PyCartesian* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("time", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setTime(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "time must be in the format HHmmss");
      }
    } else if (val == Py_None) {
      Cartesian_setTime(self->cartesian, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"time must be of type string");
    }
  } else if (strcmp("date", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setDate(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "date must be in the format YYYYMMSS");
      }
    } else if (val == Py_None) {
      Cartesian_setDate(self->cartesian, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"date must be of type string");
    }
  } else if (strcmp("objectType", name) == 0) {
    if (PyInt_Check(val)) {
      if (!Cartesian_setObjectType(self->cartesian, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "objectType not supported");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "objectType must be a valid object type")
    }
  } else if (strcmp("product", name) == 0) {
    if (PyInt_Check(val)) {
      if (!Cartesian_setProduct(self->cartesian, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "product not supported");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "product must be a valid product type")
    }
  } else if (strcmp("source", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setSource(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "Failed to set source");
      }
    } else if (val == Py_None) {
      Cartesian_setSource(self->cartesian, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"source must be of type string");
    }
  } else if (strcmp("xscale", name)==0) {
    if (PyFloat_Check(val)) {
      Cartesian_setXScale(self->cartesian, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"xscale must be of type float");
    }
  } else if (strcmp("yscale", name)==0) {
    if (PyFloat_Check(val)) {
      Cartesian_setYScale(self->cartesian, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"yscale must be of type float");
    }
  } else if (strcmp("areaextent", name) == 0) {
    double llX = 0.0, llY = 0.0, urX = 0.0, urY = 0.0;
    if (!PyArg_ParseTuple(val, "dddd", &llX, &llY, &urX, &urY)) {
      goto done;
    }
    Cartesian_setAreaExtent(self->cartesian, llX, llY, urX, urY);
  } else if (strcmp("projection", name) == 0) {
    if (PyProjection_Check(val)) {
      Cartesian_setProjection(self->cartesian, ((PyProjection*)val)->projection);
    } else if (val == Py_None) {
      Cartesian_setProjection(self->cartesian, NULL);
    }
  } else if (strcmp("starttime", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setStartTime(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "starttime must be in the format HHmmss");
      }
    } else if (val == Py_None) {
      Cartesian_setStartTime(self->cartesian, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"time must be of type string");
    }
  } else if (strcmp("startdate", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setStartDate(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "startdate must be in the format YYYYMMSS");
      }
    } else if (val == Py_None) {
      Cartesian_setStartDate(self->cartesian, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"date must be of type string");
    }
  } else if (strcmp("endtime", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setEndTime(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "endtime must be in the format HHmmss");
      }
    } else if (val == Py_None) {
      Cartesian_setEndTime(self->cartesian, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"endtime must be of type string");
    }
  } else if (strcmp("enddate", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setEndDate(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "enddate must be in the format YYYYMMSS");
      }
    } else if (val == Py_None) {
      Cartesian_setEndDate(self->cartesian, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"enddate must be of type string");
    }
  } else if (strcmp("defaultParameter", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setDefaultParameter(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set defaultParameter");
      }
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"defaultParameter must be of type string");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, name);
  }

  result = 0;
done:
  return result;
}

/*@} End of Cartesian products */

/*@{ Type definitions */
PyTypeObject PyCartesian_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "CartesianCore", /*tp_name*/
  sizeof(PyCartesian), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pycartesian_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pycartesian_getattr, /*tp_getattr*/
  (setattrfunc)_pycartesian_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_pycartesian_new, 1},
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_cartesian(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyCartesian_API[PyCartesian_API_pointers];
  PyObject *c_api_object = NULL;
  PyCartesian_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_cartesian", functions);
  if (module == NULL) {
    return;
  }
  PyCartesian_API[PyCartesian_Type_NUM] = (void*)&PyCartesian_Type;
  PyCartesian_API[PyCartesian_GetNative_NUM] = (void *)PyCartesian_GetNative;
  PyCartesian_API[PyCartesian_New_NUM] = (void*)PyCartesian_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyCartesian_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_cartesian.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _cartesian.error");
  }

  import_array(); /*To make sure I get access to Numeric*/
  import_pyprojection();
  import_pyarea();
  import_pyravefield();
  import_pycartesianparam();
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
