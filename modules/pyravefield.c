/* --------------------------------------------------------------------
Copyright (C) 2009-2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the RaveField API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-07-05
 */
#include "Python.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYRAVEFIELD_MODULE        /**< to get correct part of pycartesian.h */
#include "pyravefield.h"

#include <arrayobject.h>
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_ravefield");

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

/*@{ Rave field */
/**
 * Returns the native RaveField_t instance.
 * @param[in] pyfield - self
 * @returns the native cartesian instance.
 */
static RaveField_t* PyRaveField_GetNative(PyRaveField* pyfield)
{
  RAVE_ASSERT((pyfield != NULL), "pyfield == NULL");
  return RAVE_OBJECT_COPY(pyfield->field);
}

/**
 * Creates a python rave field from a native rave field or will create an
 * initial native RaveField if p is NULL.
 * @param[in] p - the native rave field (or NULL)
 * @returns the python rave field.
 */
static PyRaveField* PyRaveField_New(RaveField_t* p)
{
  PyRaveField* result = NULL;
  RaveField_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RaveField_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for rave field.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for rave field.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyRaveField, &PyRaveField_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->field = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->field, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyRaveField instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for rave field.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the rave field
 * @param[in] obj the object to deallocate.
 */
static void _pyravefield_dealloc(PyRaveField* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->field, obj);
  RAVE_OBJECT_RELEASE(obj->field);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the rave field.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyravefield_new(PyObject* self, PyObject* args)
{
  PyRaveField* result = PyRaveField_New(NULL);
  return (PyObject*)result;
}

/**
 * Sets the data
 * @param[in] self this instance.
 * @param[in] args arguments for creation
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyravefield_setData(PyRaveField* self, PyObject* args)
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
    raiseException_returnNULL(PyExc_ValueError, "A cartesian product must be of rank 2");
  }

  datatype = translate_pyarraytype_to_ravetype(PyArray_TYPE(arraydata));

  if (PyArray_ITEMSIZE(arraydata) != get_ravetype_size(datatype)) {
    raiseException_returnNULL(PyExc_TypeError, "numpy and rave does not have same data sizes");
  }

  xsize  = PyArray_DIM(arraydata, 1);
  ysize  = PyArray_DIM(arraydata, 0);
  data   = PyArray_DATA(arraydata);

  if (!RaveField_setData(self->field, xsize, ysize, data, datatype)) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not allocate memory");
  }

  Py_RETURN_NONE;
}

static PyObject* _pyravefield_getData(PyRaveField* self, PyObject* args)
{
  long xsize = 0, ysize = 0;
  RaveDataType type = RaveDataType_UNDEFINED;
  PyObject* result = NULL;
  npy_intp dims[2] = {0,0};
  int arrtype = 0;
  void* data = NULL;

  xsize = RaveField_getXsize(self->field);
  ysize = RaveField_getYsize(self->field);
  type = RaveField_getDataType(self->field);
  data = RaveField_getData(self->field);

  dims[1] = (npy_intp)xsize;
  dims[0] = (npy_intp)ysize;
  arrtype = translate_ravetype_to_pyarraytype(type);

  if (data == NULL) {
    raiseException_returnNULL(PyExc_IOError, "rave field does not have any data");
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
    memcpy(((PyArrayObject*)result)->data, (unsigned char*)RaveField_getData(self->field), nbytes);
  }
  return result;
}

static PyObject* _pyravefield_setValue(PyRaveField* self, PyObject* args)
{
  long x = 0, y = 0;
  double value = 0.0;
  if (!PyArg_ParseTuple(args, "lld", &x, &y, &value)) {
    return NULL;
  }

  if (!RaveField_setValue(self->field, x, y, value)) {
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject* _pyravefield_getValue(PyRaveField* self, PyObject* args)
{
  double value = 0.0L;
  long x = 0, y = 0;
  RaveValueType type = RaveValueType_NODATA;

  if (!PyArg_ParseTuple(args, "ll", &x, &y)) {
    return NULL;
  }

  type = RaveField_getValue(self->field, x, y, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Adds an attribute to the rave field. Name of the attribute should be in format
 * ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis etc.
 * Currently, the only supported values are double, long, string.
 * @param[in] self - this instance
 * @param[in] args - bin index, ray index.
 * @returns true or false depending if it works.
 */
static PyObject* _pyravefield_addAttribute(PyRaveField* self, PyObject* args)
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
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, "Unsupported data type");
  }

  if (!RaveField_addAttribute(self->field, attr)) {
    raiseException_gotoTag(done, PyExc_AttributeError, "Failed to add attribute");
  }

  result = PyBool_FromLong(1);
done:
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

static PyObject* _pyravefield_getAttribute(PyRaveField* self, PyObject* args)
{
  RaveAttribute_t* attribute = NULL;
  char* name = NULL;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }

  attribute = RaveField_getAttribute(self->field, name);
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

static PyObject* _pyravefield_getAttributeNames(PyRaveField* self, PyObject* args)
{
  RaveList_t* list = NULL;
  PyObject* result = NULL;
  int n = 0;
  int i = 0;

  list = RaveField_getAttributeNames(self->field);
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
 * All methods a cartesian product can have
 */
static struct PyMethodDef _pyravefield_methods[] =
{
  {"xsize", NULL},
  {"ysize", NULL},
  {"datatype", NULL},
  {"setData", (PyCFunction) _pyravefield_setData, 1},
  {"getData", (PyCFunction) _pyravefield_getData, 1},
  {"setValue", (PyCFunction) _pyravefield_setValue, 1},
  {"getValue", (PyCFunction) _pyravefield_getValue, 1},
  {"addAttribute", (PyCFunction) _pyravefield_addAttribute, 1},
  {"getAttribute", (PyCFunction) _pyravefield_getAttribute, 1},
  {"getAttributeNames", (PyCFunction) _pyravefield_getAttributeNames, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the rave field
 * @param[in] self - the rave field
 */
static PyObject* _pyravefield_getattr(PyRaveField* self, char* name)
{
  PyObject* res = NULL;
  if (strcmp("xsize", name) == 0) {
    return PyInt_FromLong(RaveField_getXsize(self->field));
  } else if (strcmp("ysize", name) == 0) {
    return PyInt_FromLong(RaveField_getYsize(self->field));
  } else if (strcmp("datatype", name) == 0) {
    return PyInt_FromLong(RaveField_getDataType(self->field));
  }

  res = Py_FindMethod(_pyravefield_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pyravefield_setattr(PyRaveField* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }

  raiseException_gotoTag(done, PyExc_AttributeError, name);

  result = 0;
done:
  return result;
}

/*@} End of rave field */

/*@{ Type definitions */
PyTypeObject PyRaveField_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "RaveFieldCore", /*tp_name*/
  sizeof(PyRaveField), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyravefield_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pyravefield_getattr, /*tp_getattr*/
  (setattrfunc)_pyravefield_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_pyravefield_new, 1},
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_ravefield(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyRaveField_API[PyRaveField_API_pointers];
  PyObject *c_api_object = NULL;
  PyRaveField_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_ravefield", functions);
  if (module == NULL) {
    return;
  }
  PyRaveField_API[PyRaveField_Type_NUM] = (void*)&PyRaveField_Type;
  PyRaveField_API[PyRaveField_GetNative_NUM] = (void *)PyRaveField_GetNative;
  PyRaveField_API[PyRaveField_New_NUM] = (void*)PyRaveField_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyRaveField_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_ravefield.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _ravefield.error");
  }

  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */

