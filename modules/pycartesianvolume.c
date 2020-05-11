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
 * Python version of the CartesianVolume API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-06-23
 */
#include "pyravecompat.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cartesian_odim_io.h"

#define PYCARTESIANVOLUME_MODULE   /**< to get correct part in pycartesianvolume.h */
#include "pycartesianvolume.h"

#include <arrayobject.h>
#include "pyprojection.h"
#include "pycartesian.h"
#include "pyrave_debug.h"
#include "rave_alloc.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_cartesianvolume");

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
/// Cartesian Volumes
/// --------------------------------------------------------------------
/*@{ Cartesian Volumes */
/**
 * Returns the native CartesianVolume_t instance.
 * @param[in] pycartesianvolume - the python polar volume instance
 * @returns the native cartesian volume instance.
 */
static CartesianVolume_t*
PyCartesianVolume_GetNative(PyCartesianVolume* pycartesianvolume)
{
  RAVE_ASSERT((pycartesianvolume != NULL), "pycartesianvolume == NULL");
  return RAVE_OBJECT_COPY(pycartesianvolume->cvol);
}

/**
 * Creates a python cartesian volume from a native cartesian volume or will create an
 * initial native CartesianVolume if p is NULL.
 * @param[in] p - the native cartesian volume (or NULL)
 * @returns the python cartesian volume.
 */
static PyCartesianVolume*
PyCartesianVolume_New(CartesianVolume_t* p)
{
  PyCartesianVolume* result = NULL;
  CartesianVolume_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&CartesianVolume_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for cartesoian volume.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for cartesian volume.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyCartesianVolume, &PyCartesianVolume_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->cvol = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->cvol, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyCartesianVolume instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for cartesian volume.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the cartesian volume
 * @param[in] obj the object to deallocate.
 */
static void _pycartesianvolume_dealloc(PyCartesianVolume* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->cvol, obj);
  RAVE_OBJECT_RELEASE(obj->cvol);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the polar volume.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pycartesianvolume_new(PyObject* self, PyObject* args)
{
  PyCartesianVolume* result = PyCartesianVolume_New(NULL);
  return (PyObject*)result;
}

/**
 * Adds one cartesian image to a volume.
 * @param[in] self - the cartesian volume
 * @param[in] args - the image, must be of type CartesianCore
 * @return NULL on failure
 */
static PyObject* _pycartesianvolume_addImage(PyCartesianVolume* self, PyObject* args)
{
  PyObject* inptr = NULL;
  PyCartesian* cartesian = NULL;

  if (!PyArg_ParseTuple(args, "O", &inptr)) {
    return NULL;
  }

  if (!PyCartesian_Check(inptr)) {
    raiseException_returnNULL(PyExc_TypeError,"Added object must be of type CartesianCore");
  }

  cartesian = (PyCartesian*)inptr;

  if (!CartesianVolume_addImage(self->cvol, cartesian->cartesian)) {
    raiseException_returnNULL(PyExc_MemoryError, "Failed to add image to volume");
  }

  Py_RETURN_NONE;
}

/**
 * Returns the image at the provided index.
 * @param[in] self - the cartesian volume
 * @param[in] args - the index must be >= 0 and < getNumberOfScans
 * @return NULL on failure
 */
static PyObject* _pycartesianvolume_getImage(PyCartesianVolume* self, PyObject* args)
{
  int index = -1;
  Cartesian_t* cartesian = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  if (index < 0 || index >= CartesianVolume_getNumberOfImages(self->cvol)) {
    raiseException_returnNULL(PyExc_IndexError, "Index out of range");
  }

  if((cartesian = CartesianVolume_getImage(self->cvol, index)) == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "Could not aquire image");
  }

  if (cartesian != NULL) {
    result = (PyObject*)PyCartesian_New(cartesian);
  }

  RAVE_OBJECT_RELEASE(cartesian);

  return result;
}

/**
 * Returns the number of images for this volume
 * @param[in] self - the cartesian volume
 * @param[in] args - not used
 * @return NULL on failure or a PyInteger
 */
static PyObject* _pycartesianvolume_getNumberOfImages(PyCartesianVolume* self, PyObject* args)
{
  return PyInt_FromLong(CartesianVolume_getNumberOfImages(self->cvol));
}

/**
 * Adds an attribute to the volume. Name of the attribute should be in format
 * ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis etc.
 * Currently, the only supported values are double, long, string.
 * @param[in] self - this instance
 * @param[in] args - bin index, ray index.
 * @returns true or false depending if it works.
 */
static PyObject* _pycartesianvolume_addAttribute(PyCartesianVolume* self, PyObject* args)
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

  if (!CartesianVolume_addAttribute(self->cvol, attr)) {
    raiseException_gotoTag(done, PyExc_AttributeError, "Failed to add attribute");
  }

  result = PyBool_FromLong(1);
done:
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

static PyObject* _pycartesianvolume_getAttribute(PyCartesianVolume* self, PyObject* args)
{
  RaveAttribute_t* attribute = NULL;
  char* name = NULL;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }

  attribute = CartesianVolume_getAttribute(self->cvol, name);
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

static PyObject* _pycartesianvolume_hasAttribute(PyCartesianVolume* self, PyObject* args)
{
  RaveAttribute_t* attribute = NULL;
  char* name = NULL;
  long result = 0;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  attribute = CartesianVolume_getAttribute(self->cvol, name);
  if (attribute != NULL) {
    result = 1;
  }
  RAVE_OBJECT_RELEASE(attribute);
  return PyBool_FromLong(result);
}

static PyObject* _pycartesianvolume_getAttributeNames(PyCartesianVolume* self, PyObject* args)
{
  RaveList_t* list = NULL;
  PyObject* result = NULL;
  int n = 0;
  int i = 0;

  list = CartesianVolume_getAttributeNames(self->cvol);
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

static PyObject* _pycartesianvolume_isValid(PyCartesianVolume* self, PyObject* args)
{
  return PyBool_FromLong(CartesianOdimIO_isValidVolume(self->cvol));
}

/**
 * All methods a polar volume can have
 */
static struct PyMethodDef _pycartesianvolume_methods[] =
{
  {"time", NULL},
  {"date", NULL},
  {"source", NULL},
  {"objectType", NULL},
  {"xscale", NULL},
  {"yscale", NULL},
  {"zscale", NULL},
  {"zstart", NULL},
  {"xsize", NULL},
  {"ysize", NULL},
  {"zsize", NULL},
  {"projection", NULL},
  {"areaextent", NULL},
  {"addImage", (PyCFunction) _pycartesianvolume_addImage, 1,
    "addImage(cartesian)\n\n"
    "Adds a cartesian object to the volume. When adding the first object, xsize/ysize will be set. Then the following objects that are added has to have same xsize & ysize"
  },
  {"getImage", (PyCFunction) _pycartesianvolume_getImage, 1,
    "getImage(index) -> CartesianCore object\n\n"
    "Returns the cartesian object at index.\n\n"
    "index - the index that has to be >= 0 and < getNumberOfImages()."
  },
  {"getNumberOfImages", (PyCFunction) _pycartesianvolume_getNumberOfImages, 1,
    "getNumberOfImages() -> the number of images\n\n"
    "Returns the number of images in this volume."
  },
  {"addAttribute", (PyCFunction) _pycartesianvolume_addAttribute, 1,
    "addAttribute(name, value) \n\n"
    "Adds an attribute to the volume. Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis etc. \n"
    "Currently, double, long, string and 1-dimensional arrays are supported.\n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr.\n"
    "value - Value to be associated with the name. Currently, double, long, string and 1-dimensional arrays are supported."
  },
  {"getAttribute", (PyCFunction) _pycartesianvolume_getAttribute, 1,
    "getAttribute(name) -> value \n\n"
    "Returns the value associated with the specified name \n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr."
  },
  {"hasAttribute", (PyCFunction) _pycartesianvolume_hasAttribute, 1,
    "hasAttribute(name) -> a boolean \n\n"
    "Returns if the specified name is defined within this cartesian volume\n\n"
    "name  - Name of the attribute should be in format ^(how|what|where)/[A-Za-z0-9_.]$. E.g how/something, what/sthis.\n"
    "        In the case of how-groups, it is also possible to specify subgroups, like how/subgroup/attr or how/subgroup/subgroup/attr.\n"
  },
  {"getAttributeNames", (PyCFunction) _pycartesianvolume_getAttributeNames, 1,
    "getAttributeNames() -> array of names \n\n"
    "Returns the attribute names associated with this cartesian volume"
  },
  {"isValid", (PyCFunction) _pycartesianvolume_isValid, 1,
    "isValid() -> boolean\n\n"
    "Validates the volume to see if it contains necessary information like sizes & scales are set. That start and end date/times are set and so on."
  },
  {NULL, NULL} /* sentinel */
};

/**
 * Returns the specified attribute in the polar volume
 */
static PyObject* _pycartesianvolume_getattro(PyCartesianVolume* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("time", name) == 0) {
    const char* str = CartesianVolume_getTime(self->cvol);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("date", name) == 0) {
    const char* str = CartesianVolume_getDate(self->cvol);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("source", name) == 0) {
    const char* str = CartesianVolume_getSource(self->cvol);
    if (str != NULL) {
      return PyRaveAPI_StringOrUnicode_FromASCII(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("projection", name) == 0) {
    Projection_t* projection = CartesianVolume_getProjection(self->cvol);
    if (projection != NULL) {
      PyProjection* result = PyProjection_New(projection);
      RAVE_OBJECT_RELEASE(projection);
      return (PyObject*)result;
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("objectType", name) == 0) {
    return PyInt_FromLong(CartesianVolume_getObjectType(self->cvol));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("xscale", name) == 0) {
    return PyFloat_FromDouble(CartesianVolume_getXScale(self->cvol));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("yscale", name) == 0) {
    return PyFloat_FromDouble(CartesianVolume_getYScale(self->cvol));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("zscale", name) == 0) {
    return PyFloat_FromDouble(CartesianVolume_getZScale(self->cvol));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("zstart", name) == 0) {
    return PyFloat_FromDouble(CartesianVolume_getZStart(self->cvol));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("xsize", name) == 0) {
    return PyInt_FromLong(CartesianVolume_getXSize(self->cvol));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("ysize", name) == 0) {
    return PyInt_FromLong(CartesianVolume_getYSize(self->cvol));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("zsize", name) == 0) {
    return PyInt_FromLong(CartesianVolume_getZSize(self->cvol));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("areaextent", name) == 0) {
    double llX = 0.0, llY = 0.0, urX = 0.0, urY = 0.0;
    CartesianVolume_getAreaExtent(self->cvol, &llX, &llY, &urX, &urY);
    return Py_BuildValue("(dddd)", llX, llY, urX, urY);
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pycartesianvolume_setattro(PyCartesianVolume* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("time", name) == 0) {
    if (PyString_Check(val)) {
      if (!CartesianVolume_setTime(self->cvol, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set time");
      }
    } else if (val == Py_None) {
        CartesianVolume_setTime(self->cvol, NULL);
    } else {
        raiseException_gotoTag(done, PyExc_ValueError, "time should be specified as a string (HHmmss)");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("date", name) == 0) {
    if (PyString_Check(val)) {
      if (!CartesianVolume_setDate(self->cvol, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set date");
      }
    } else if (val == Py_None) {
      CartesianVolume_setDate(self->cvol, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "date should be specified as a string (YYYYMMDD)");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("source", name) == 0) {
    if (PyString_Check(val)) {
      if (!CartesianVolume_setSource(self->cvol, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set source");
      }
    } else if (val == Py_None) {
      CartesianVolume_setSource(self->cvol, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "source should be specified as a string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("xscale", name)==0) {
    if (PyFloat_Check(val)) {
      CartesianVolume_setXScale(self->cvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"xscale must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("yscale", name)==0) {
    if (PyFloat_Check(val)) {
      CartesianVolume_setYScale(self->cvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"yscale must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("zscale", name)==0) {
    if (PyFloat_Check(val)) {
      CartesianVolume_setZScale(self->cvol, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      CartesianVolume_setZScale(self->cvol, (double)PyLong_AsLong(val));
    } else if (PyInt_Check(val)) {
      CartesianVolume_setZScale(self->cvol, (double)PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"zscale must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("zstart", name)==0) {
    if (PyFloat_Check(val)) {
      CartesianVolume_setZStart(self->cvol, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      CartesianVolume_setZStart(self->cvol, (double)PyLong_AsLong(val));
    } else if (PyInt_Check(val)) {
      CartesianVolume_setZStart(self->cvol, (double)PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"zstart must be of type float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("projection", name) == 0) {
    if (PyProjection_Check(val)) {
      CartesianVolume_setProjection(self->cvol, ((PyProjection*)val)->projection);
    } else if (val == Py_None) {
      CartesianVolume_setProjection(self->cvol, NULL);
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("objectType", name) == 0) {
    if (PyInt_Check(val)) {
      if (!CartesianVolume_setObjectType(self->cvol, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "objectType not supported");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "objectType must be a valid object type")
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("areaextent", name) == 0) {
    double llX = 0.0, llY = 0.0, urX = 0.0, urY = 0.0;
    if (!PyArg_ParseTuple(val, "dddd", &llX, &llY, &urX, &urY)) {
      goto done;
    }
    CartesianVolume_setAreaExtent(self->cvol, llX, llY, urX, urY);
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }

  result = 0;
done:
  return result;
}

static PyObject* _pycartesianvolume_isCartesianVolume(PyObject* self, PyObject* args)
{
  PyObject* inobj = NULL;
  if (!PyArg_ParseTuple(args,"O", &inobj)) {
    return NULL;
  }
  if (PyCartesianVolume_Check(inobj)) {
    return PyBool_FromLong(1);
  }
  return PyBool_FromLong(0);
}

/*@} End of Cartesian Volumes */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pycartesianvolume_type_doc,
    "The cartesian volume is a container for cartesian products.  The member attributes should reflect some basic features of the included "
    "cartesian images as well as some basic information like object type.\n"
    "Since the parameter probably should contain a lot of attributes as defined in the ODIM H5 specification, these can be "
    "added within the attribute mapping (how/, what/, where/) groups. E.g. addAttribute(\"how/sthis\", 1.2).\n"
    "A list of avilable member attributes are described below. For information about member functions, check each functions doc.\n"
    "\n"
    "time             - Time this cartesian product should represent as a string with format HHmmSS\n"
    "date             - Date this cartesian product should represent as a string in the format YYYYMMDD\n"
    "source           - The source for this product. Defined as what/source in ODIM H5. I.e. a comma separated list of various identifiers. For example. NOD:seang,WMO:1234,....\n"
    "objectType       - The object type as defined in ODIM H5 this cartesian product should be defined as. Can be  _rave.Rave_ObjectType_CVOL or _rave.Rave_ObjectType_COMP\n"
    "xscale           - The scale in meters in x-direction.\n"
    "yscale           - The scale in meters in y-direction.\n"
    "areaextent       - A tuple of four representing the outer boundaries of this cartesian product. Defined as (lower left X, lower left Y, upper right X, upper right Y).\n"
    "projection       - The projection object of type ProjectionCore that defines what projection that this cartesian product is defined with.\n"
    "xsize            - The xsize of the area represented. ReadOnly, initialization occurs when adding first image.\n"
    "ysize            - The ysize of the area represented. ReadOnly, initialization occurs when adding first image.\n"
    "\n"
    "Usage:\n"
    " import _cartesianvolume\n"
    " vol = _cartesianvolume.new()\n"
    " vol.addImage(cartesian1)\n"
    " vol.addImage(cartesian2)\n"
    " ..."
    );
/*@} End of Documentation about the type */


/// --------------------------------------------------------------------
/// Type definitions
/// --------------------------------------------------------------------
/*@{ Type definitions */
PyTypeObject PyCartesianVolume_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "CartesianVolumeCore", /*tp_name*/
  sizeof(PyCartesianVolume), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pycartesianvolume_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pycartesianvolume_getattro, /*tp_getattro*/
  (setattrofunc)_pycartesianvolume_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pycartesianvolume_type_doc,  /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pycartesianvolume_methods,   /*tp_methods*/
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

/// --------------------------------------------------------------------
/// Module setup
/// --------------------------------------------------------------------
/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pycartesianvolume_new, 1,
    "new() -> new instance of the CartesianVolumeCore object\n\n"
    "Creates a new instance of the CartesianVolumeCore object"
  },
  {"isCartesianVolume", (PyCFunction)_pycartesianvolume_isCartesianVolume, 1,
    "isCartesianVolume(object) -> boolean\n\n"
    "Tests if the provided object is a cartesian volume or not.\n\n"
    "object - the object to be tested."
  },
  {NULL,NULL} /*Sentinel*/
};

/**
 * Initializes polar volume.
 */
MOD_INIT(_cartesianvolume)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyCartesianVolume_API[PyCartesianVolume_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyCartesianVolume_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyCartesianVolume_Type);

  MOD_INIT_DEF(module, "_cartesianvolume", _pycartesianvolume_type_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyCartesianVolume_API[PyCartesianVolume_Type_NUM] = (void*)&PyCartesianVolume_Type;
  PyCartesianVolume_API[PyCartesianVolume_GetNative_NUM] = (void *)PyCartesianVolume_GetNative;
  PyCartesianVolume_API[PyCartesianVolume_New_NUM] = (void*)PyCartesianVolume_New;

  c_api_object = PyCapsule_New(PyCartesianVolume_API, PyCartesianVolume_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_cartesianvolume.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _cartesianvolume.error");
    return MOD_INIT_ERROR;
  }

  import_array(); /*To make sure I get access to Numeric*/
  import_pyprojection();
  import_pycartesian();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
