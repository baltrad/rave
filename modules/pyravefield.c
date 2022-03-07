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
#include "pyravecompat.h"
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
  int result = 0;

  if (!PyArg_ParseTuple(args, "ll", &x, &y)) {
    return NULL;
  }

  result = RaveField_getValue(self->field, x, y, &value);

  return Py_BuildValue("(id)", result, value);
}

static PyObject* _pyravefield_getConvertedValue(PyRaveField* self, PyObject* args)
{
  double value = 0.0L;
  long x = 0, y = 0;
  int result = 0;

  if (!PyArg_ParseTuple(args, "ll", &x, &y)) {
    return NULL;
  }

  result = RaveField_getConvertedValue(self->field, x, y, &value);

  return Py_BuildValue("(id)", result, value);
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
    const char* value = PyString_AsString(obj);
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

static PyObject* _pyravefield_hasAttribute(PyRaveField* self, PyObject* args)
{
  char* name = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  return PyBool_FromLong(RaveField_hasAttribute(self->field, name));
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

static PyObject* _pyravefield_removeAttributes(PyRaveField* self, PyObject* args)
{
  RaveField_removeAttributes(self->field);
  Py_RETURN_NONE;
}


/**
 * Concatenates two fields x-wise.
 * @param[in] self - self
 * @param[in] args - the other rave field object
 * @return a rave field object on success otherwise NULL
 */
static PyObject* _pyravefield_concatx(PyRaveField* self, PyObject* args)
{
  PyObject* result = NULL;
  PyObject* pyin = NULL;
  RaveField_t *field = NULL;
  if (!PyArg_ParseTuple(args, "O", &pyin)) {
    return NULL;
  }
  if (!PyRaveField_Check(pyin)) {
    raiseException_returnNULL(PyExc_ValueError, "Argument must be another rave field");
  }
  field = RaveField_concatX(self->field, ((PyRaveField*)pyin)->field);
  if (field == NULL) {
    raiseException_gotoTag(done, PyExc_ValueError, "Failed to concatenate fields");
  }

  result = (PyObject*)PyRaveField_New(field);
done:
  RAVE_OBJECT_RELEASE(field);
  return result;
}

static PyObject* _pyravefield_circshiftData(PyRaveField* self, PyObject* args)
{
  long x = 0, y = 0;
  int result = 0;

  if (!PyArg_ParseTuple(args, "ll", &x, &y)) {
    return NULL;
  }

  result = RaveField_circshiftData(self->field, x, y);
  if (!result) {
    raiseException_returnNULL(PyExc_ValueError, "Failed to run circular shift on field");
  }

  Py_RETURN_NONE;
}


MOD_DIR_FORWARD_DECLARE(PyRaveField);

/**
 * All methods a cartesian product can have
 */
static struct PyMethodDef _pyravefield_methods[] =
{
  {"xsize", NULL, METH_VARARGS},
  {"ysize", NULL, METH_VARARGS},
  {"datatype", NULL, METH_VARARGS},
  {"setData", (PyCFunction) _pyravefield_setData, 1,
    "setData(array)\n\n"
    "Initializes the parameter with a datafield as defined by a 2-dimensional numpy array and datatype.\n\n"
    "array - The 2 dimensional numpy array."
  },
  {"getData", (PyCFunction) _pyravefield_getData, 1,
    "getData() -> a numpy array\n\n"
    "Returns a 2 dimensional data array with the data set."
  },
  {"setValue", (PyCFunction) _pyravefield_setValue, 1,
    "setValue(x,y,value) -> 1 on success otherwise 0\n\n"
    "Sets the value at the specified position. \n\n"
    "x     - x position\n"
    "y     - y position\n"
    "value - the value that should be set at specified position."
  },
  {"getValue", (PyCFunction) _pyravefield_getValue, 1,
    "getValue(x,y) -> the value at the specified x and y position.\n\n"
    "Returns the value at the specified x and y position. \n\n"
    "x - x position\n"
    "y - y position\n"
  },
  {"getConvertedValue", (PyCFunction) _pyravefield_getConvertedValue, 1,
    "getConvertedValue(x,y) -> the converted value at the specified x and y position.\n\n"
    "Returns the converted value (what/offset + what/gain*v) at the specified x and y position. Since what/offset and what/gain are optional, they are assumed to have 0.0 and 1.0 respectively if they are missing.\n\n"
    "x - x position\n"
    "y - y position\n"
  },
  {"addAttribute", (PyCFunction) _pyravefield_addAttribute, 1,
    "addAttribute(name, value) \n\n"
    "Adds an attribute to the field. Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis etc. \n"
    "Currently, double, long, string and 1-dimensional arrays are supported.\n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis\n"
    "value - Value to be associated with the name. Currently, double, long, string and 1-dimensional arrays are supported."
  },
  {"getAttribute", (PyCFunction) _pyravefield_getAttribute, 1,
    "getAttribute(name) -> value \n\n"
    "Returns the value associated with the specified name \n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis\n"
  },
  {"hasAttribute", (PyCFunction) _pyravefield_hasAttribute, 1,
    "hasAttribute(name) -> a boolean \n\n"
    "Returns if the specified name is defined within this rave field\n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis"
  },
  {"getAttributeNames", (PyCFunction) _pyravefield_getAttributeNames, 1,
    "getAttributeNames() -> array of names \n\n"
    "Returns the attribute names associated with this field"
  },
  {"removeAttributes", (PyCFunction) _pyravefield_removeAttributes, 1,
    "removeAttributes()\n\n"
    "Removes all attributes associated with self."
  },
  {"concatx", (PyCFunction) _pyravefield_concatx, 1,
    "concatx(other) -> rave field core\n\n"
    "Concatenates self with other x-wise. This requires that the fields have same ysize and same datatype. Will \n\n"
    "other - the other field that self should be concatenated with. Requires that other has same ysize and datatype as self."
  },
  {"circshiftData", (PyCFunction) _pyravefield_circshiftData, 1,
    "circshiftData(x,y)\n\n"
    "Performs a circular shift of self in both x & y dimension to modify the internal data field.\n\n"
    "x - the number of steps to be shifted in x-direction. Can be both positive and negative\n"
    "y - the number of steps to be shifted in y-direction. Can be both positive and negative"
  },
  {"__dir__", (PyCFunction) MOD_DIR_REFERENCE(PyRaveField), METH_NOARGS},
  {NULL, NULL } /* sentinel */
};

MOD_DIR_FUNCTION(PyRaveField, _pyravefield_methods)

/**
 * Returns the specified attribute in the rave field
 * @param[in] self - the rave field
 */
static PyObject* _pyravefield_getattro(PyRaveField* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "xsize") == 0) {
    return PyInt_FromLong(RaveField_getXsize(self->field));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "ysize") == 0) {
    return PyInt_FromLong(RaveField_getYsize(self->field));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "datatype") == 0) {
    return PyInt_FromLong(RaveField_getDataType(self->field));
  }

  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the attribute value
 */
static int _pyravefield_setattro(PyObject *self, PyObject *name, PyObject *value)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));

  result = 0;
done:
  return result;
}

/*@} End of rave field */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyravefield_type_doc,
    "A data container that is used as for example quality fields or other similar constructs.\n\n"
    "The only 3 member attributes that are accessible are:\n"
    "xsize     - xsize of data field (read only)\n"
    "ysize     - ysize of data field (read only)\n"
    "datatype  - data type (read only)\n"
    "\n"
    "These attributes will be set when initializing the field with setData.\n"
    "\n"
    "Since a lot of RAVE has been developed with ODIM H5 in mind, it is also possible to add arbitrary attributes in "
    "various groups, e.g. c.addAttribute(\"how/this\", 1.2) and so on.\n\n"
    "\n"
    "Usage:\n"
    " import _ravefield, numpy\n"
    " dfield = _ravefield.new()\n"
    " dfield.setData(numpy.array([[1,2],[3,4]],numpy.uint8))"
    );
/*@} End of Documentation about the module */


/*@{ Type definitions */
PyTypeObject PyRaveField_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0)
  "RaveFieldCore",                  /*tp_name*/
  sizeof(PyRaveField),              /*tp_size*/
  0,                                /*tp_itemsize*/
  /* methods */
  (destructor)_pyravefield_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)0,                   /*tp_getattr*/
  (setattrfunc)0,                   /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0, /*tp_as_sequence */
  0, /*tp_as_mapping */
  (hashfunc)0, /*tp_hash*/
  (ternaryfunc)0, /*tp_call*/
  (reprfunc)0, /*tp_str*/
  (getattrofunc)_pyravefield_getattro, /*tp_getattro*/
  (setattrofunc)_pyravefield_setattro, /*tp_setattro*/
  0, /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyravefield_type_doc, /*tp_doc*/
  (traverseproc)0, /*tp_traverse*/
  (inquiry)0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  _pyravefield_methods, /*tp_methods*/
  0,                    /*tp_members*/
  0,                      /*tp_getset*/
  0,                      /*tp_base*/
  0,                      /*tp_dict*/
  0,                      /*tp_descr_get*/
  0,                      /*tp_descr_set*/
  0,                      /*tp_dictoffset*/
  0,                      /*tp_init*/
  0,                      /*tp_alloc*/
  0,                      /*tp_new*/
  0,                      /*tp_free*/
  0,                      /*tp_is_gc*/
};
/*@} End of Type definitions */

/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pyravefield_new, 1,
    "new() -> new instance of the RaveFieldCore object\n\n"
    "Creates a new instance of the RaveFieldCore object"
  },
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_ravefield)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyRaveField_API[PyRaveField_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyRaveField_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyRaveField_Type);

  MOD_INIT_DEF(module, "_ravefield", _pyravefield_type_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyRaveField_API[PyRaveField_Type_NUM] = (void*)&PyRaveField_Type;
  PyRaveField_API[PyRaveField_GetNative_NUM] = (void *)PyRaveField_GetNative;
  PyRaveField_API[PyRaveField_New_NUM] = (void*)PyRaveField_New;

  c_api_object = PyCapsule_New(PyRaveField_API, PyRaveField_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_ravefield.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _area.error");
    return MOD_INIT_ERROR;
  }

  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */

