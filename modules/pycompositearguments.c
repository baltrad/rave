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
 * Python version of the Composite arguments API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024"-12-13
 */
#include "rave_types.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "compositearguments.h"
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYCOMPOSITEARGUMENTS_MODULE        /**< to get correct part of pycompositearguments.h */
#include "pycompositearguments.h"
#include "pypolarvolume.h"
#include "pypolarscan.h"
#include "pyarea.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_compositearguments");

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

/*@{ Composite generator */
/**
 * Returns the native CartesianGenerator_t instance.
 * @param[in] pyargs - the python composite arguments instance
 * @returns the native cartesian instance.
 */
static CompositeArguments_t*
PyCompositeArguments_GetNative(PyCompositeArguments* pyargs)
{
  RAVE_ASSERT((pyargs != NULL), "pyargs == NULL");
  return RAVE_OBJECT_COPY(pyargs->args);
}

/**
 * Creates a python composite arguments from a native composite arguments or will create an
 * initial native CompositeArguments if p is NULL.
 * @param[in] p - the native composite arguments (or NULL)
 * @returns the python composite product arguments.
 */
static PyCompositeArguments*
PyCompositeArguments_New(CompositeArguments_t* p)
{
  PyCompositeArguments* result = NULL;
  CompositeArguments_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&CompositeArguments_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for composite arguments.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for composite generator.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyCompositeArguments, &PyCompositeArguments_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->args = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->args, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyCompositeArguments instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for composite generator.");
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
static void _pycompositearguments_dealloc(PyCompositeArguments* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->args, obj);
  RAVE_OBJECT_RELEASE(obj->args);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the composite.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pycompositearguments_new(PyObject* self, PyObject* args)
{
  PyCompositeArguments* result = PyCompositeArguments_New(NULL);
  return (PyObject*)result;
}

// static PyObject* _pyompositearguments_addPlugin(PyCompositeArguments* self, PyObject* args)
// {
//   char* id = NULL;
//   PyObject* pyplugin = NULL;
//   if (!PyArg_ParseTuple(args, "sO", &id, &pyplugin)) {
//     return NULL;
//   }

//   if (!PyCompositeArgumentsPlugin_Check(pyplugin)) {
//     raiseException_returnNULL(PyExc_TypeError, "object must be a CompositeArgumentsPlugin");
//   }

//   if (!CompositeArguments_addPlugin(self->generator, id, ((PyCompositeArgumentsPlugin*)pyplugin)->plugin)) {
//     raiseException_returnNULL(PyExc_AttributeError, "Could not add plugin to generator");
//   }

//   Py_RETURN_NONE;
// }

// static PyObject* _pyCompositeArguments_generate(PyCompositeArguments* self, PyObject* args)
// {
//   if (!PyArg_ParseTuple(args, "")) {
//     return NULL;
//   }

//   CompositeArguments_generate(self->generator, NULL, NULL);

//   Py_RETURN_NONE;
// }

/**
 * Adds an attribute to the composite arguments. Name of the attribute.
 * Currently, the only supported values are double, long, string.
 * @param[in] self - this instance
 * @param[in] args - bin index, ray index.
 * @returns true or false depending if it works.
 */
static PyObject* _pycompositearguments_addArgument(PyCompositeArguments* self, PyObject* args)
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

  if (!CompositeArguments_addArgument(self->args, attr)) {
    raiseException_gotoTag(done, PyExc_AttributeError, "Failed to add argument");
  }

  result = PyBool_FromLong(1);
done:
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

static PyObject* _pycompositearguments_getArgument(PyCompositeArguments* self, PyObject* args)
{
  RaveAttribute_t* attribute = NULL;
  char* name = NULL;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }

  attribute = CompositeArguments_getArgument(self->args, name);
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
      result = PyArray_SimpleNew(1, dims, NPY_LONG);
      for (i = 0; i < len; i++) {
        *((long*) PyArray_GETPTR1((PyArrayObject*)result, i)) = value[i];
      }
    } else if (format == RaveAttribute_Format_DoubleArray) {
      double* value = NULL;
      int len = 0;
      int i = 0;
      npy_intp dims[1];
      RaveAttribute_getDoubleArray(attribute, &value, &len);
      dims[0] = len;
      result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
      for (i = 0; i < len; i++) {
        *((double*) PyArray_GETPTR1((PyArrayObject*)result, i)) = value[i];
      }
    } else {
      RAVE_CRITICAL1("Undefined format on requested argument %s", name);
      raiseException_gotoTag(done, PyExc_AttributeError, "Undefined argument");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, "No such argument");
  }
done:
  RAVE_OBJECT_RELEASE(attribute);
  return result;
}

/**
 * Adds a parameter to the composite arguments
 * @param[in] self - self
 * @param[in] args - <quantity as a string>, <gain as double>, <offset as double>, <minvalue as double>
 * @return None on success otherwise NULL
 */
static PyObject* _pycompositearguments_addParameter(PyCompositeArguments* self, PyObject* args)
{
  char* quantity = NULL;
  double gain = 1.0, offset = 0.0;
  RaveDataType datatype = RaveDataType_UCHAR;
  double nodata = 255.0, undetect = 0.0;
  if (!PyArg_ParseTuple(args, "sdd|ddd", &quantity, &gain, &offset, &datatype, &nodata, &undetect)) {
    return NULL;
  }
  if (!CompositeArguments_addParameter(self->args, quantity, gain, offset, datatype, nodata, undetect)) {
    raiseException_returnNULL(PyExc_AttributeError, "Could not add parameter");
  }

  Py_RETURN_NONE;
}

/**
 * Returns if the composite arguments will composite specified parameter
 * @param[in] self - self
 * @param[in] args - <quantity as a string>
 * @return True or False
 */
static PyObject* _pycompositearguments_hasParameter(PyCompositeArguments* self, PyObject* args)
{
  char* quantity = NULL;
  if (!PyArg_ParseTuple(args, "s", &quantity)) {
    return NULL;
  }
  return PyBool_FromLong(CompositeArguments_hasParameter(self->args, quantity));
}

/**
 * Returns the parameter value for specified quantity.
 * @param[in] self - self
 * @param[in] args - <index as int>
 * @return A tuple containing (<gain as double>,<offset as double>,<datatype as int>,<nodata as double>,<undetect as double>)
 */
static PyObject* _pycompositearguments_getParameter(PyCompositeArguments* self, PyObject* args)
{
  char* quantity = NULL;
  double gain = 1.0, offset = 0.0, nodata = 255.0, undetect = 0.0;
  RaveDataType datatype = RaveDataType_UCHAR;

  if (!PyArg_ParseTuple(args, "s", &quantity)) {
    return NULL;
  }

  if (quantity == NULL) {
    raiseException_returnNULL(PyExc_KeyError, "Must specify a quantity when getting parameter settings");
  }

  if (!CompositeArguments_getParameter(self->args, quantity, &gain, &offset, &datatype, &nodata, &undetect)) {
    raiseException_returnNULL(PyExc_KeyError, "No parameter with specified quantity set");
  }

  return Py_BuildValue("(ddidd)", gain, offset, datatype, nodata, undetect);
}


/**
 * Returns the number of parameters this generator will process
 * @param[in] self - self
 * @param[in] args - N/A
 * @return The number of parameters
 */
static PyObject* _pycompositearguments_getParameterCount(PyCompositeArguments* self, PyObject* args)
{
  return PyLong_FromLong(CompositeArguments_getParameterCount(self->args));
}

/**
 * Returns the parameter at specified index.
 * @param[in] self - self
 * @param[in] args - <index as int>
 * @return A tuple containing (<quantity as string>,<gain as double>,<offset as double>)
 */
static PyObject* _pycompositearguments_getParameterAtIndex(PyCompositeArguments* self, PyObject* args)
{
  int i = 0;
  const char* quantity;
  double gain = 1.0, offset = 0.0, nodata = 255.0, undetect = 0.0;
  RaveDataType datatype = RaveDataType_UCHAR;

  if (!PyArg_ParseTuple(args, "i", &i)) {
    return NULL;
  }
  quantity = CompositeArguments_getParameterAtIndex(self->args, i, &gain, &offset, &datatype, &nodata, &undetect);
  if (quantity == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "No parameter at specified index");
  }

  return Py_BuildValue("(sddidd)", quantity, gain, offset, datatype, nodata, undetect);
}

/**
 * Adds an object to the object list
 * @param[in] self - self
 * @param[in] args - object
 * @return
 */
static PyObject* _pycompositearguments_addObject(PyCompositeArguments* self, PyObject* args)
{
  PyObject* obj = NULL;
  RaveCoreObject* rcobject = NULL;

  if(!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }

  if (PyPolarVolume_Check(obj)) {
    rcobject = (RaveCoreObject*)((PyPolarVolume*)obj)->pvol;
  } else if (PyPolarScan_Check(obj)) {
    rcobject = (RaveCoreObject*)((PyPolarScan*)obj)->scan;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "only supported objects are volumes and scans");
  }

  if (!CompositeArguments_addObject(self->args, rcobject)) {
    raiseException_returnNULL(PyExc_MemoryError, "failed to add object to composite arguments");
  }

  Py_RETURN_NONE;
}

/**
 * Returns the number of objects in the arguments
 * @param[in] self - self
 * @param[in] args - object
 * @return number of objects
 */
static PyObject* _pycompositearguments_getNumberOfObjects(PyCompositeArguments* self, PyObject* args)
{
  if(!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyLong_FromLong(CompositeArguments_getNumberOfObjects(self->args));
}

/**
 * Returns the object as specified index
 * @param[in] self - self
 * @param[in] args - index
 * @return the object at specified index or IndexError
 */
static PyObject* _pycompositearguments_getObject(PyCompositeArguments* self, PyObject* args)
{
  int index = 0;
  PyObject* res = NULL;

  if(!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }
  if (index >= 0 && index < CompositeArguments_getNumberOfObjects(self->args)) {
    RaveCoreObject* object = CompositeArguments_getObject(self->args, index);
    if (object != NULL) {
      if (RAVE_OBJECT_CHECK_TYPE(object, &PolarVolume_TYPE)) {
        res = (PyObject*)PyPolarVolume_New((PolarVolume_t*)object);
      } else if (RAVE_OBJECT_CHECK_TYPE(object, &PolarScan_TYPE)) {
        res = (PyObject*)PyPolarScan_New((PolarScan_t*)object);
      } else {
        PyErr_SetString(PyExc_NotImplementedError, "support lacking for object type");
      }
      RAVE_OBJECT_RELEASE(object);
    }    
  }
  return res;
}

static PyObject* _pycompositearguments_addQualityFlag(PyCompositeArguments* self, PyObject* args)
{
  char* qualityflag = NULL;
  if (!PyArg_ParseTuple(args, "s", &qualityflag)) {
    return NULL;
  }
  if (!CompositeArguments_addQualityFlag(self->args, (const char*)qualityflag)) {
    raiseException_returnNULL(PyExc_RuntimeError, "Could not add quality flag to arguments");
  }
  Py_RETURN_NONE;
}

static PyObject* _pycompositearguments_setQualityFlags(PyCompositeArguments* self, PyObject* args)
{
  PyObject* pyqualityflags = NULL;
  char** flags = NULL;
  Py_ssize_t nnames = 0;
  int i = 0;

  if (!PyArg_ParseTuple(args, "O", &pyqualityflags)) {
    return NULL;
  }
  if (!PyList_Check(pyqualityflags)) {
    raiseException_returnNULL(PyExc_AttributeError, "Must provide a list of strings")
  }
  if (PyObject_Length(pyqualityflags) > 0) {
    nnames = PyObject_Length(pyqualityflags);
    flags = RAVE_MALLOC(sizeof(char*) * nnames);
    if (flags == NULL) {
      raiseException_returnNULL(PyExc_MemoryError, "Could not allocate array for strings");
    }
    memset(flags, 0, sizeof(char*)*nnames);
    for (i = 0; i < nnames; i++) {
      PyObject* pystr = PyList_GetItem(pyqualityflags, i);
      char* dupstr = NULL;
      if (pystr == NULL || !PyString_Check(pystr)) {
        raiseException_gotoTag(done, PyExc_AttributeError, "The list should only contain strings");
      }
      dupstr = RAVE_STRDUP(PyString_AsString(pystr));
      if (dupstr != NULL) {
        flags[i] = dupstr;
      } else {
        raiseException_gotoTag(done, PyExc_MemoryError, "Could not allocate memory for string");
      }        
      dupstr = NULL;
    }
    if (!CompositeArguments_setQualityFlags(self->args, (const char**)flags, nnames)) {
      raiseException_gotoTag(done, PyExc_RuntimeError, "Could not set quality flags");
    }
  }
  Py_RETURN_NONE;
done:
  if (flags != NULL) {
    for (i = 0; i < nnames; i++) {
      if (flags[i] != NULL) {
        RAVE_FREE(flags[i]);
      }
    }
    RAVE_FREE(flags);
  }
  return NULL;
}

static PyObject* _pycompositearguments_getQualityFlagAt(PyCompositeArguments* self, PyObject* args)
{
  int index = 0;
  const char* qualityflagname;

  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }
  qualityflagname = CompositeArguments_getQualityFlagAt(self->args, index);
  if (qualityflagname == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "Could not read quality flag at specified position");
  }
  return Py_BuildValue("s", qualityflagname);
}

static PyObject* _pycompositearguments_removeQualityFlag(PyCompositeArguments* self, PyObject* args)
{
  char* qualityflag = NULL;
  if (!PyArg_ParseTuple(args, "s", &qualityflag)) {
    return NULL;
  }
  if (!CompositeArguments_removeQualityFlag(self->args, qualityflag)) {
    raiseException_returnNULL(PyExc_RuntimeError, "Could not remove quality flag from arguments");
  }
  Py_RETURN_NONE;
}

static PyObject* _pycompositearguments_removeQualityFlagAt(PyCompositeArguments* self, PyObject* args)
{
  int index = 0;

  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }
  if (!CompositeArguments_removeQualityFlagAt(self->args, index)) {
    raiseException_returnNULL(PyExc_RuntimeError, "Could not remove quality flag from arguments");
  }
  Py_RETURN_NONE;
}

static PyObject* _pycompositearguments_getNumberOfQualityFlags(PyCompositeArguments* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyLong_FromLong(CompositeArguments_getNumberOfQualityFlags(self->args));
}

/**
 * All methods a cartesian product can have
 */
static struct PyMethodDef _pycompositearguments_methods[] =
{
  {"area", NULL, METH_VARARGS},
  {"product", NULL, METH_VARARGS},
  {"time", NULL, METH_VARARGS},
  {"date", NULL, METH_VARARGS},
  {"height", NULL, METH_VARARGS},
  {"elangle", NULL, METH_VARARGS},
  {"range", NULL, METH_VARARGS},
  {"strategy", NULL, METH_VARARGS},
  {"addArgument", (PyCFunction) _pycompositearguments_addArgument, 1,
    "addArgument(name, value) \n\n"
    "Adds an argument.\n"
    "Currently, double, long, string and 1-dimensional arrays are supported.\n\n"
    "name  - Name of the argument.\n"
    "value - Value to be associated with the name. Currently, double, long, string and 1-dimensional arrays are supported."
  },
  {"getArgument", (PyCFunction) _pycompositearguments_getArgument, 1,
    "getArgument(name) -> value \n\n"
    "Returns the value associated with the specified name \n\n"
    "name  - Name of the argument.\n"
  },
  {"addParameter", (PyCFunction)_pycompositearguments_addParameter, 1,
    "addParameter(quantity, gain, offset)\n\n" // "sdd", &
    "Adds one parameter (quantity) that should be processed in the run.\n\n"
    "quantity   - the parameter quantity\n"
    "gain       - the gain to be used for the parameter\n"
    "offset     - the offset to be used for the parameter\n"
  },
  {"hasParameter", (PyCFunction)_pycompositearguments_hasParameter, 1,
    "hasParameter(quantity) -> boolean\n\n"
    "Returns if this composite generator is going to process specified parameter\n\n"
    "quantity   - the parameter quantity"
  },
  {"getParameter", (PyCFunction)_pycompositearguments_getParameter, 1,
    "getParameter(quantity) -> (gain, offset)\n\n"
    "Returns information about the parameter with specified quantity. Returned value will be a tuple of quantity, gain and offset."
  },
  {"getParameterCount", (PyCFunction)_pycompositearguments_getParameterCount, 1,
    "getParameterCount() -> integer\n\n"
    "Returns the number of parameters that are going to be processed."
  },
  {"getParameterAtIndex", (PyCFunction)_pycompositearguments_getParameterAtIndex, 1,
    "getParameterAtIndex(index) -> (quantity, gain, offset)\n\n"
    "Returns information about the parameter at index. Returned value will be a tuple of quantity, gain and offset."
  },
  {"addObject", (PyCFunction)_pycompositearguments_addObject, 1,
    "addObject(object)\n\n"
    "Adds an object to the list of objects."
  },
  {"getNumberOfObjects", (PyCFunction)_pycompositearguments_getNumberOfObjects, 1,
    "getNumberOfObjects() -> integer\n\n"
    "Returns the number of objects that have been added to the arguments."
  },
  {"getObject", (PyCFunction)_pycompositearguments_getObject, 1,
    "getObject(index) -> object\n\n"
    "Returns the object at provided position."
  },
  {"addQualityFlag", (PyCFunction)_pycompositearguments_addQualityFlag, 1,
    "addQualityFlag(qualityname)\n\n"
    "Adds a quality flag to the arguments.\n"
    "qualityname - the name of the quality field"
  },
  {"setQualityFlags", (PyCFunction)_pycompositearguments_setQualityFlags, 1,
    "setQualityFlags(qualitynames)\n\n"
    "Sets the quality flags in the arguments.\n"
    "qualitynames - a list containing strings"
  },
  {"getQualityFlagAt", (PyCFunction)_pycompositearguments_getQualityFlagAt, 1,
    "getQualityFlagAt(index)\n\n"
    "Returns the quality flag at specified position.\n"
    "index - the index of the quality field"
  },
  {"removeQualityFlag", (PyCFunction)_pycompositearguments_removeQualityFlag, 1,
    "removeQualityFlag(qualityname)\n\n"
    "Removes the quality flag with specified name.\n"
    "qualityname - the name of the quality field to be removed"
  },
  {"removeQualityFlagAt", (PyCFunction)_pycompositearguments_removeQualityFlagAt, 1,
    "removeQualityFlagAt(index)\n\n"
    "Removes the quality flag at specified index.\n"
    "index - the quality flag to remove"
  },
  {"getNumberOfQualityFlags", (PyCFunction)_pycompositearguments_getNumberOfQualityFlags, 1,
    "getNumberOfQualityFlags() -> int\n\n"
    "Returns the number of quality flags.\n"
  },
  

  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the cartesian
 * @param[in] self - the cartesian product
 */

static PyObject* _pycompositearguments_getattro(PyCompositeArguments* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "area") == 0) {
    Area_t* area = CompositeArguments_getArea(self->args);
    if (area != NULL) {
      PyArea* result = PyArea_New(area);
      RAVE_OBJECT_RELEASE(area);
      return (PyObject*)result;
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "product") == 0) {
    if (CompositeArguments_getProduct(self->args) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(CompositeArguments_getProduct(self->args));
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("time", name) == 0) {
    if (CompositeArguments_getTime(self->args) != NULL) {
      return PyString_FromString(CompositeArguments_getTime(self->args));
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("date", name) == 0) {
    if (CompositeArguments_getDate(self->args) != NULL) {
      return PyString_FromString(CompositeArguments_getDate(self->args));
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("height", name) == 0) {
    return PyFloat_FromDouble(CompositeArguments_getHeight(self->args));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("elangle", name) == 0) {
    return PyFloat_FromDouble(CompositeArguments_getElevationAngle(self->args));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("range", name) == 0) {
    return PyFloat_FromDouble(CompositeArguments_getRange(self->args));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("strategy", name) == 0) {
    if (CompositeArguments_getStrategy(self->args) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(CompositeArguments_getStrategy(self->args));
    }
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pycompositearguments_setattro(PyCompositeArguments* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "area") == 0) {
    if (PyArea_Check(val)) {
      CompositeArguments_setArea(self->args, ((PyArea*)val)->area);
    } else if (val == Py_None) {
      CompositeArguments_setArea(self->args, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "area must be of AreaCore type or None");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "product") == 0) {
    if (PyString_Check(val)) {
      if (!CompositeArguments_setProduct(self->args, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "Failed to set product string");
      }
    } else if (val == Py_None) {
      CompositeArguments_setProduct(self->args, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"product must be a string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("time", name) == 0) {
    if (PyString_Check(val)) {
      if (!CompositeArguments_setTime(self->args, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "time must be in the format HHmmss");
      }
    } else if (val == Py_None) {
      CompositeArguments_setTime(self->args, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"time must be of type string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("date", name) == 0) {
    if (PyString_Check(val)) {
      if (!CompositeArguments_setDate(self->args, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "date must be in the format YYYYMMSS");
      }
    } else if (val == Py_None) {
      CompositeArguments_setDate(self->args, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"date must be of type string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("height", name) == 0) {
    if (PyFloat_Check(val)) {
      CompositeArguments_setHeight(self->args, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      CompositeArguments_setHeight(self->args, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      CompositeArguments_setHeight(self->args, (double)PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"height must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("elangle", name) == 0) {
    if (PyFloat_Check(val)) {
      CompositeArguments_setElevationAngle(self->args, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      CompositeArguments_setElevationAngle(self->args, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      CompositeArguments_setElevationAngle(self->args, (double)PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "elangle must be a float or decimal value")
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("range", name) == 0) {
    if (PyFloat_Check(val)) {
      CompositeArguments_setRange(self->args, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      CompositeArguments_setRange(self->args, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      CompositeArguments_setRange(self->args, (double)PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "range must be a float or decimal value")
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "strategy") == 0) {
    if (PyString_Check(val)) {
      if (!CompositeArguments_setStrategy(self->args, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "Failed to set strategy string");
      }
    } else if (val == Py_None) {
      CompositeArguments_setStrategy(self->args, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"strategy must be a string or None");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }

//   } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("product", name) == 0) {
//     if (PyInt_Check(val)) {
//       Composite_setProduct(self->composite, PyInt_AsLong(val));
//     } else {
//       raiseException_gotoTag(done, PyExc_TypeError, "product must be a valid product type")
//     }
//   } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("selection_method", name) == 0) {
//     if (!PyInt_Check(val) || !Composite_setSelectionMethod(self->composite, PyInt_AsLong(val))) {
//       raiseException_gotoTag(done, PyExc_ValueError, "not a valid selection method");
//     }
//   } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("interpolation_method", name) == 0) {
//     if (!PyInt_Check(val) || !Composite_setInterpolationMethod(self->composite, PyInt_AsLong(val))) {
//       raiseException_gotoTag(done, PyExc_ValueError, "not a valid interpolation method");
//     }
//   } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("interpolate_undetect", name) == 0) {
//     if (PyBool_Check(val)) {
//       if (PyObject_IsTrue(val)) {
//         Composite_setInterpolateUndetect(self->composite, 1);
//       } else {
//         Composite_setInterpolateUndetect(self->composite, 0);
//       }
//     } else {
//       raiseException_gotoTag(done, PyExc_ValueError, "interpolate_undetect must be a bool");
//     }
//   } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("time", name) == 0) {
//     if (PyString_Check(val)) {
//       if (!Composite_setTime(self->composite, PyString_AsString(val))) {
//         raiseException_gotoTag(done, PyExc_ValueError, "time must be in the format HHmmss");
//       }
//     } else if (val == Py_None) {
//       Composite_setTime(self->composite, NULL);
//     } else {
//       raiseException_gotoTag(done, PyExc_ValueError,"time must be of type string");
//     }
//   } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("date", name) == 0) {
//     if (PyString_Check(val)) {
//       if (!Composite_setDate(self->composite, PyString_AsString(val))) {
//         raiseException_gotoTag(done, PyExc_ValueError, "date must be in the format YYYYMMSS");
//       }
//     } else if (val == Py_None) {
//       Composite_setDate(self->composite, NULL);
//     } else {
//       raiseException_gotoTag(done, PyExc_ValueError,"date must be of type string");
//     }
//   } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("quality_indicator_field_name", name) == 0) {
//     if (PyString_Check(val)) {
//       if (!Composite_setQualityIndicatorFieldName(self->composite, PyString_AsString(val))) {
//         raiseException_gotoTag(done, PyExc_MemoryError, "Failed to set quality indicator field name");
//       }
//     } else if (val == Py_None) {
//       Composite_setQualityIndicatorFieldName(self->composite, NULL);
//     } else {
//       raiseException_gotoTag(done, PyExc_ValueError, "quality_indicator_field_name must be a string");
//     }
//   } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("algorithm", name) == 0) {
//     if (val == Py_None) {
//       Composite_setAlgorithm(self->composite, NULL);
//     } else if (PyCompositeAlgorithm_Check(val)) {
//       Composite_setAlgorithm(self->composite, ((PyCompositeAlgorithm*)val)->algorithm);
//     } else {
//       raiseException_gotoTag(done, PyExc_TypeError, "algorithm must either be None or a CompositeAlgorithm");
//     }
//   } else {
//     raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));
//   }

  result = 0;
done:
  return result;
}

static PyObject* _pycompositearguments_isCompositeArguments(PyObject* self, PyObject* args)
{
  PyObject* inobj = NULL;
  if (!PyArg_ParseTuple(args,"O", &inobj)) {
    return NULL;
  }
  if (PyCompositeArguments_Check(inobj)) {
    return PyBool_FromLong(1);
  }
  return PyBool_FromLong(0);
}
/*@} End of Composite product generator */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pycompositearguments_type_doc,
    "The composite type provides the possibility to create cartesian composites from a number of polar objects.\n"
    "To generate the composite, one or many polar scans or polar volumes has to be added to the generator. Then generate should be called with the expected area and an optional list of how/task quality field names.\n"
    "There are a few attributes that can be set besides the functions.\n"
    " height                       - The height in meters that should be used when generating a composite like CAPPI, PCAPPI or PMAX.\n"
    " elangle                      - The elevation angle in radians that should be used when generating a composite like PPI."
    " range                        - The range that should be used when generating the Pseudo MAX. This range is the limit in meters\n"
    "                                for when the vertical max should be used. When outside this range, the PCAPPI value is used instead.\n"
    " product                      - The product type that should be generated when generating the composite.\n"
    "                                Height/Elevation angle and range are used in combination with the products.\n"
    "                                PPI requires elevation angle\n"
    "                                CAPPI, PCAPPI and PMAX requires height above sea level\n"
    "                                PMAX also requires range in meters\n"
    " selection_method             - The selection method to use when there are more than one radar covering same point. I.e. if for example taking distance to radar or height above sea level. Currently the following methods are available\n"
    "       _pycomposite.SelectionMethod_NEAREST - Value from the nearest radar is selected.\n"
    "       _pycomposite.SelectionMethod_HEIGHT  - Value from radar which scan is closest to the sea level at current point.\n"
    " interpolation_method         - Interpolation method is used to choose how to interpolate the surrounding values. The default behaviour is NEAREST.\n"
    "       _pycomposite.InterpolationMethod_NEAREST                  - Nearest value is used\n"
    "       _pycomposite.InterpolationMethod_LINEAR_HEIGHT            - Value calculated by performing a linear interpolation between the closest positions above and below\n"
    "       _pycomposite.InterpolationMethod_LINEAR_RANGE             - Value calculated by performing a linear interpolation between the closest positions before\n"
    "                                                                   and beyond in the range dimension of the ray\n"
    "       _pycomposite.InterpolationMethod_LINEAR_AZIMUTH           - Value calculated by performing a linear interpolation between the closest positions on each\n"
    "                                                                   side of the position, i.e., interpolation between consecutive rays\n"
    "       _pycomposite.InterpolationMethod_LINEAR_RANGE_AND_AZIMUTH - Value calculated by performing a linear interpolation in azimuth and range directions.\n"
    "       _pycomposite.InterpolationMethod_LINEAR_3D                - Value calculated by performing a linear interpolation in height, azimuth and range directions.\n"
    "       _pycomposite.InterpolationMethod_QUADRATIC_HEIGHT         - Value calculated by performing a quadratic interpolation between the closest positions before and beyond in\n"
    "                                                                   the range dimension of the ray. Quadratic interpolation means that inverse distance weights raised to the\n"
    "                                                                   power of 2 are used in value interpolation.\n"
    "       _pycomposite.InterpolationMethod_QUADRATIC_3D             - Value calculated by performing a quadratic interpolation in height, azimuth and range\n"
    "                                                                   directions. Quadratic interpolation means that inverse distance weights raised to the\n"
    "                                                                   power of 2 are used in value interpolation.\n"
    ""
    " interpolate_undetect         - If undetect should be used in interpolation or not.\n"
    "                                If undetect not should be included in the interpolation, the behavior will be the following:\n"
    "                                * If all values are UNDETECT, then result will be UNDETECT.\n"
    "                                * If only one value is DATA, then use that value.\n"
    "                                * If more than one value is DATA, then interpolation.\n"
    "                                * If all values are NODATA, then NODATA.\n"
    "                                * If all values are either NODATA or UNDETECT, then UNDETECT.\n"
    ""
    " date                         - The nominal date as a string in format YYYYMMDD\n"
    " time                         - The nominal time as a string in format HHmmss\n"
    " quality_indicator_field_name - If this field name is set, then the composite will be generated by first using the quality indicator field for determining\n"
    "                                radar usage. If the field name is None, then the selection method will be used instead.\n"
    "\n"
    "Usage:\n"
    " import _pycomposite\n"
    " generator = _pycomposite.new()\n"
    " generator.selection_method = _pycomposite.SelectionMethod_HEIGHT\n"
    " generator.product = \"PCAPPI\"\n"
    " generator.height = 500.0\n"
    " generator.date = \"20200201\"\n"
    " generator.date = \"100000\"\n"
    " generator.addParameter(\"DBZH\", 2.0, 3.0, -30.0)\n"
    " generator.add(_rave.open(\"se1_pvol_20200201100000.h5\").object)\n"
    " generator.add(_rave.open(\"se2_pvol_20200201100000.h5\").object)\n"
    " generator.add(_rave.open(\"se3_pvol_20200201100000.h5\").object)\n"
    " result = generator.generate(myarea, [\"se.smhi.composite.distance.radar\",\"pl.imgw.radvolqc.spike\"])\n"
    );
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyCompositeArguments_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "CompositeArgumentsCore", /*tp_name*/
  sizeof(PyCompositeArguments), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pycompositearguments_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pycompositearguments_getattro, /*tp_getattro*/
  (setattrofunc)_pycompositearguments_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pycompositearguments_type_doc,        /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pycompositearguments_methods,              /*tp_methods*/
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
  {"new", (PyCFunction)_pycompositearguments_new, 1,
    "new() -> new instance of the CompositeArgumentsCore object\n\n"
    "Creates a new instance of the CompositeArgumentsCore object"
  },
  {"isCompositeArguments", (PyCFunction)_pycompositearguments_isCompositeArguments, 1,
      "isArea(obj) -> True if object is an area, otherwise False\n\n"
      "Checks if the provided object is a python area object or not.\n\n"
      "obj - the object to check."},  
  {NULL,NULL} /*Sentinel*/
};

/**
 * Adds constants to the dictionary (probably the modules dictionary).
 * @param[in] dictionary - the dictionary the long should be added to
 * @param[in] name - the name of the constant
 * @param[in] value - the value
 */
// static void add_long_constant(PyObject* dictionary, const char* name, long value)
// {
//   PyObject* tmp = NULL;
//   tmp = PyInt_FromLong(value);
//   if (tmp != NULL) {
//     PyDict_SetItemString(dictionary, name, tmp);
//   }
//   Py_XDECREF(tmp);
// }

MOD_INIT(_compositearguments)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyCompositeArguments_API[PyCompositeArguments_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyCompositeArguments_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyCompositeArguments_Type);

  MOD_INIT_DEF(module, "_compositearguments", _pycompositearguments_type_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyCompositeArguments_API[PyCompositeArguments_Type_NUM] = (void*)&PyCompositeArguments_Type;
  PyCompositeArguments_API[PyCompositeArguments_GetNative_NUM] = (void *)&PyCompositeArguments_GetNative;
  PyCompositeArguments_API[PyCompositeArguments_New_NUM] = (void*)&PyCompositeArguments_New;

  c_api_object = PyCapsule_New(PyCompositeArguments_API, PyCompositeArguments_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_compositearguments.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _compositearguments.error");
    return MOD_INIT_ERROR;
  }

  import_pyarea();
  import_pypolarvolume();
  import_pypolarscan();
  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
