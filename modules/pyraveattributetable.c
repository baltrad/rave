/* --------------------------------------------------------------------
Copyright (C) 2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the rave attribute table
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2022-03-30
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <float.h>

#define PY_RAVE_ATTRIBUTE_TABLE_MODULE    /**< to get correct part in pyraveattributetable.h */
#include <arrayobject.h>
#include "pyraveattributetable.h"
#include "rave_alloc.h"
#include "pyrave_debug.h"
#include "raveutil.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_attributetable");

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

/*@{ RaveAttributeTable */
/**
 * Returns the native RaveAttributeTable_t instance.
 * @param[in] pytable - the python rave attribute table
 * @returns the native rave attribute table
 */
static RaveAttributeTable_t*
PyRaveAttributeTable_GetNative(PyRaveAttributeTable* pytable)
{
  RAVE_ASSERT((pytable != NULL), "pytable == NULL");
  return RAVE_OBJECT_COPY(pytable->table);
}

/**
 * Creates a python attribute table from a native table or will create an
 * initial native attribute table  if p is NULL.
 * @param[in] p - the native attribute table (or NULL)
 * @returns the python attribute table.
 */
static PyRaveAttributeTable*
PyRaveAttributeTable_New(RaveAttributeTable_t* p)
{
  PyRaveAttributeTable* result = NULL;
  RaveAttributeTable_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RaveAttributeTable_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for rave attribute table.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for rave attribute table.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyRaveAttributeTable, &PyRaveAttributeTable_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->table = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->table, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyRaveAttributeTable instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyRaveAttributeTable.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the attribute table
 * @param[in] obj the object to deallocate.
 */
static void _pyattributetable_dealloc(PyRaveAttributeTable* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->table, obj);
  RAVE_OBJECT_RELEASE(obj->table);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the attribute table.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyattributetable_new(PyObject* self, PyObject* args)
{
  PyRaveAttributeTable* result = PyRaveAttributeTable_New(NULL);
  return (PyObject*)result;
}

static PyObject* _pyattributeinternal_createPyAttributeTuple(RaveAttribute_t* attr)
{
  PyObject* result = NULL;

  RaveAttribute_Format format = RaveAttribute_getFormat(attr);
  if (format == RaveAttribute_Format_Long) {
    long value = 0;
    RaveAttribute_getLong(attr, &value);
    result = Py_BuildValue("(si)", RaveAttribute_getName(attr), value);
  } else if (format == RaveAttribute_Format_Double) {
    double value = 0.0;
    RaveAttribute_getDouble(attr, &value);
    result = Py_BuildValue("(sd)", RaveAttribute_getName(attr), value);
  } else if (format == RaveAttribute_Format_String) {
    char* value = NULL;
    RaveAttribute_getString(attr, &value);
    result = Py_BuildValue("(ss)", RaveAttribute_getName(attr), value);
  } else if (format == RaveAttribute_Format_LongArray) {
    long* value = NULL;
    int len = 0;
    int i = 0;
    PyObject* arr = NULL;
    npy_intp dims[1];
    RaveAttribute_getLongArray(attr, &value, &len);
    dims[0] = len;
    arr = PyArray_SimpleNew(1, dims, PyArray_LONG);
    for (i = 0; i < len; i++) {
      *((long*) PyArray_GETPTR1(arr, i)) = value[i];
    }
    result = Py_BuildValue("(sO)", RaveAttribute_getName(attr), (PyObject*)arr);
    Py_XDECREF(arr);
  } else if (format == RaveAttribute_Format_DoubleArray) {
    double* value = NULL;
    int len = 0;
    int i = 0;
    PyObject* arr = NULL;
    npy_intp dims[1];
    RaveAttribute_getDoubleArray(attr, &value, &len);
    dims[0] = len;
    arr = PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
    for (i = 0; i < len; i++) {
      *((double*) PyArray_GETPTR1(arr, i)) = value[i];
    }
    result = Py_BuildValue("(sO)", RaveAttribute_getName(attr), (PyObject*)arr);
    Py_XDECREF(arr);
  }

  return result;
}

/**
 * Adds an attribute to the parameter. Name of the attribute should be in format
 * ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis etc.
 * Currently, the only supported values are double, long, string.
 * @param[in] self - this instance
 * @param[in] args - bin index, ray index.
 * @returns true or false depending if it works.
 */
static PyObject* _pyattributetable_addAttribute(PyRaveAttributeTable* self, PyObject* args)
{
  RaveAttribute_t* attr = NULL;
  char* name = NULL;
  PyObject* obj = NULL;
  PyObject* result = NULL;
  RaveIO_ODIM_Version version = RaveIO_ODIM_Version_UNDEFINED;
  RaveAttribute_t* translated = NULL;

  if (!PyArg_ParseTuple(args, "sO|i", &name, &obj, &version)) {
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

  if (version != RaveIO_ODIM_Version_UNDEFINED) {
    if (!RaveAttributeTable_addAttributeVersion(self->table, attr, version, &translated)) {
      raiseException_gotoTag(done, PyExc_AttributeError, "Failed to add attribute");
    }
  } else {
    if (!RaveAttributeTable_addAttribute(self->table, attr, &translated)) {
      raiseException_gotoTag(done, PyExc_AttributeError, "Failed to add attribute");
    }
  }

  if (translated == NULL) {
    translated = RAVE_OBJECT_COPY(attr);
  }

  if (translated != NULL) {
    RaveAttribute_Format format = RaveAttribute_getFormat(translated);
    if (format == RaveAttribute_Format_Long) {
      long value = 0;
      RaveAttribute_getLong(translated, &value);
      result = Py_BuildValue("(si)", RaveAttribute_getName(translated), value);
    } else if (format == RaveAttribute_Format_Double) {
      double value = 0.0;
      RaveAttribute_getDouble(translated, &value);
      result = Py_BuildValue("(sd)", RaveAttribute_getName(translated), value);
    } else if (format == RaveAttribute_Format_String) {
      char* value = NULL;
      RaveAttribute_getString(translated, &value);
      result = Py_BuildValue("(ss)", RaveAttribute_getName(translated), value);
    } else if (format == RaveAttribute_Format_LongArray) {
      long* value = NULL;
      int len = 0;
      int i = 0;
      PyObject* arr = NULL;
      npy_intp dims[1];
      RaveAttribute_getLongArray(translated, &value, &len);
      dims[0] = len;
      arr = PyArray_SimpleNew(1, dims, PyArray_LONG);
      for (i = 0; i < len; i++) {
        *((long*) PyArray_GETPTR1(arr, i)) = value[i];
      }
      result = Py_BuildValue("(sO)", RaveAttribute_getName(translated), (PyObject*)arr);
      Py_XDECREF(arr);
    } else if (format == RaveAttribute_Format_DoubleArray) {
      double* value = NULL;
      int len = 0;
      int i = 0;
      PyObject* arr = NULL;
      npy_intp dims[1];
      RaveAttribute_getDoubleArray(translated, &value, &len);
      dims[0] = len;
      arr = PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
      for (i = 0; i < len; i++) {
        *((double*) PyArray_GETPTR1(arr, i)) = value[i];
      }
      result = Py_BuildValue("(sO)", RaveAttribute_getName(translated), (PyObject*)arr);
      Py_XDECREF(arr);
    } else {
      RAVE_CRITICAL1("Undefined format on requested attribute %s", name);
      raiseException_gotoTag(done, PyExc_AttributeError, "Undefined attribute");
    }
  }

done:
  RAVE_OBJECT_RELEASE(attr);
  RAVE_OBJECT_RELEASE(translated);

  return result;
}

/**
 * Returns an attribute with the specified name
 * @param[in] self - this instance
 * @param[in] args - name
 * @returns the attribute value for the name
 */
static PyObject* _pyattributetable_getAttribute(PyRaveAttributeTable* self, PyObject* args)
{
  RaveAttribute_t* attribute = NULL;
  char* name = NULL;
  PyObject* result = NULL;
  RaveIO_ODIM_Version version = RaveIO_ODIM_Version_UNDEFINED;

  if (!PyArg_ParseTuple(args, "s|i", &name, &version)) {
    return NULL;
  }
  if (version != RaveIO_ODIM_Version_UNDEFINED) {
    attribute = RaveAttributeTable_getAttributeVersion(self->table, name, version);
  } else {
    attribute = RaveAttributeTable_getAttribute(self->table, name);
  }

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
 * Returns the size of this table
 * @param[in] self - this instance
 * @param[in] args - name
 * @returns the size
 */
static PyObject* _pyattributetable_size(PyRaveAttributeTable* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyInt_FromLong(RaveAttributeTable_size(self->table));
}

/**
 * Returns if there exists an attribute with the specified name
 * @param[in] self - this instance
 * @param[in] args - name
 * @returns True if attribute exists otherwise False
 */
static PyObject* _pyattributetable_removeAttribute(PyRaveAttributeTable* self, PyObject* args)
{
  char* name = NULL;
  RaveAttribute_t* removed = NULL;

  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  removed = RaveAttributeTable_removeAttribute(self->table, name);
  RAVE_OBJECT_RELEASE(removed);
  Py_RETURN_NONE;
}

/**
 * Returns if there exists an attribute with the specified name
 * @param[in] self - this instance
 * @param[in] args - name
 * @returns True if attribute exists otherwise False
 */
static PyObject* _pyattributetable_clear(PyRaveAttributeTable* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  RaveAttributeTable_clear(self->table);

  Py_RETURN_NONE;
}

/**
 * Returns if there exists an attribute with the specified name
 * @param[in] self - this instance
 * @param[in] args - name
 * @returns True if attribute exists otherwise False
 */
static PyObject* _pyattributetable_hasAttribute(PyRaveAttributeTable* self, PyObject* args)
{
  char* name = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  return PyBool_FromLong((long)RaveAttributeTable_hasAttribute(self->table, name));
}

static PyObject* _pyattributetable_shiftAttribute(PyRaveAttributeTable* self, PyObject* args)
{
  char* name = NULL;
  int i = 0;
  if (!PyArg_ParseTuple(args, "si", &name, &i)) {
    return NULL;
  }
  if (!RaveAttributeTable_shiftAttribute(self->table, name, i)) {
    raiseException_returnNULL(PyExc_RuntimeError, "Failed to shift attribute");
  }
  Py_RETURN_NONE;
}
/**
 * Returns a list of attribute names
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns a list of attribute names
 */
static PyObject* _pyattributetable_getAttributeNames(PyRaveAttributeTable* self, PyObject* args)
{
  RaveList_t* list = NULL;
  PyObject* result = NULL;
  int n = 0;
  int i = 0;
  RaveIO_ODIM_Version version = RaveIO_ODIM_Version_UNDEFINED;

  if (!PyArg_ParseTuple(args, "|i", &version)) {
    return NULL;
  }
  if (version != RaveIO_ODIM_Version_UNDEFINED) {
    list = RaveAttributeTable_getAttributeNamesVersion(self->table, version);
  } else {
    list = RaveAttributeTable_getAttributeNames(self->table);
  }

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

static PyObject* _pyattributetable_getValues(PyRaveAttributeTable* self, PyObject* args)
{
  RaveObjectList_t* list = NULL;
  int n = 0;
  int i = 0;
  RaveIO_ODIM_Version version = RaveIO_ODIM_Version_UNDEFINED;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "|i", &version)) {
    return NULL;
  }

  if (version != RaveIO_ODIM_Version_UNDEFINED) {
    list = RaveAttributeTable_getValuesVersion(self->table, version);
  } else {
    list = RaveAttributeTable_getValues(self->table);
  }

  if (list == NULL) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not get attribute values");
  }
  n = RaveObjectList_size(list);
  result = PyList_New(0);
  for (i = 0; result != NULL && i < n; i++) {
    RaveAttribute_t* attr = (RaveAttribute_t*)RaveObjectList_get(list, i);
    if (attr != NULL) {
      PyObject* pyv = _pyattributeinternal_createPyAttributeTuple(attr);
      if (pyv != NULL) {
        if (PyList_Append(result, pyv) != 0) {
          Py_DECREF(pyv);
          goto fail;
        }
      }
      Py_DECREF(pyv);
    }
    RAVE_OBJECT_RELEASE(attr);
  }

  RAVE_OBJECT_RELEASE(list);
  return result;
fail:
  RAVE_OBJECT_RELEASE(list);
  Py_XDECREF(result);
  return NULL;
}


/**
 * All methods a table can have
 */
static struct PyMethodDef _pyattributetable_methods[] =
{
  {"version", NULL, METH_VARARGS},
  {"addAttribute", (PyCFunction) _pyattributetable_addAttribute, 1,
    "addAttribute(name, value) \n\n"
    "Adds an attribute to the table. Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis etc. \n"
    "Currently, double, long, string and 1-dimensional arrays are supported.\n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr.\n"
    "value - Value to be associated with the name. Currently, double, long, string and 1-dimensional arrays are supported."
  },
  {"getAttribute", (PyCFunction) _pyattributetable_getAttribute, 1,
    "getAttribute(name) -> value \n\n"
    "Returns the value associated with the specified name \n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr."
  },
  {"size", (PyCFunction) _pyattributetable_size, 1,
    "size() -> value \n\n"
    "Returns the number of attributes in this table \n\n"
  },
  {"removeAttribute", (PyCFunction) _pyattributetable_removeAttribute, 1,
    "removeAttribute(name)\n\n"
    "Removes the attribute with the specified name \n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr."
  },
  {"clear", (PyCFunction) _pyattributetable_clear, 1,
    "clear() -> N/A \n\n"
    "Removes all attributes from table \n\n"
  },
  {"hasAttribute", (PyCFunction) _pyattributetable_hasAttribute, 1,
    "hasAttribute(name) -> a boolean \n\n"
    "Returns if the specified name is defined within this polar scan\n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis.\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr.\n"
  },
  {"shiftAttribute", (PyCFunction) _pyattributetable_shiftAttribute, 1,
    "shiftAttribute(name, nx) \n\n"
    "Performs a circular shift of an array attribute. if nx < 0, then shift is performed counter clockwise, if nx > 0, shift is performed clock wise, if 0, no shift is performed.\n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis.\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr.\n"
    "nx    - Number of positions to shift\n"
  },
  {"getAttributeNames", (PyCFunction) _pyattributetable_getAttributeNames, 1,
    "getAttributeNames(|version) -> array of names \n\n"
    "version - optional, specified version of names to be returned.\n"
    "Returns the attribute names associated with this table and eventual version"
  },
  {"getValues", (PyCFunction) _pyattributetable_getValues, 1,
    "getValues(|version) -> array of tuples (name,value) \n\n"
    "version - optional, specified version of values to be returned.\n"
    "Returns the attribute values associated with this table and eventual version"
  },
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the table
 * @param[in] self - the registry
 */
static PyObject* _pyattributetable_getattro(PyRaveAttributeTable* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("version", name) == 0) {
    return PyInt_FromLong(RaveAttributeTable_getVersion(self->table));
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the specified attribute in the registry
 */
static int _pyattributetable_setattro(PyRaveAttributeTable* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("version", name)==0) {
    if (PyInt_Check(val)) {
      if (!RaveAttributeTable_setVersion(self->table, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_TypeError,"Could not set requested version");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"version must be of type int");
    }
  }

  result = 0;
done:
  return result;
}
/*@} End of RaveAttributeTable */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyattributetable_type_doc,
    "The attribute table gives the user the possibility to manage attributes and handle different versions of the ODIM specification.\n"
    "\n"
    "After the attribute has been created, you are able to add, remove and other miscellaneous operations related to attributes.\n"
    );
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyRaveAttributeTable_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "AttributeTableCore", /*tp_name*/
  sizeof(PyRaveAttributeTable), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyattributetable_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pyattributetable_getattro, /*tp_getattro*/
  (setattrofunc)_pyattributetable_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyattributetable_type_doc,     /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyattributetable_methods,      /*tp_methods*/
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
  {"new", (PyCFunction)_pyattributetable_new, 1,
      "new() -> new instance of the AttributeTableCore object\n\n"
      "Creates a new instance of the AttributeTableCore object"},
  {NULL,NULL} /*Sentinel*/
};

/*@{ Documentation about the module */
PyDoc_STRVAR(_pyattributetable_module_doc,
    "This class provides functionality for managing an attribute table.\n"
    "\n"
    );
/*@} End of Documentation about the module */


MOD_INIT(_attributetable)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyRaveAttributeTable_API[PyRaveAttributeTable_API_pointers];
  PyObject *c_api_object = NULL;
  MOD_INIT_SETUP_TYPE(PyRaveAttributeTable_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyRaveAttributeTable_Type);

  MOD_INIT_DEF(module, "_attributetable", _pyattributetable_module_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyRaveAttributeTable_API[PyRaveAttributeTable_Type_NUM] = (void*)&PyRaveAttributeTable_Type;
  PyRaveAttributeTable_API[PyRaveAttributeTable_GetNative_NUM] = (void *)PyRaveAttributeTable_GetNative;
  PyRaveAttributeTable_API[PyRaveAttributeTable_New_NUM] = (void*)PyRaveAttributeTable_New;

  c_api_object = PyCapsule_New(PyRaveAttributeTable_API, PyRaveAttributeTable_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_attributetable.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _attributetable.error");
    return MOD_INIT_ERROR;
  }

  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
