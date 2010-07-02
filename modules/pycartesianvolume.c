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
#include "Python.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYCARTESIANVOLUME_MODULE   /**< to get correct part in pycartesianvolume.h */
#include "pycartesianvolume.h"

#include "pyprojection.h"
#include "pycartesian.h"
#include "pyrave_debug.h"
#include "rave_alloc.h"

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
  {"projection", NULL},
  {"areaextent", NULL},
  {"addImage", (PyCFunction) _pycartesianvolume_addImage, 1},
  {"getImage", (PyCFunction) _pycartesianvolume_getImage, 1},
  {"getNumberOfImages", (PyCFunction) _pycartesianvolume_getNumberOfImages, 1},
  {"addAttribute", (PyCFunction) _pycartesianvolume_addAttribute, 1},
  {"getAttribute", (PyCFunction) _pycartesianvolume_getAttribute, 1},
  {"getAttributeNames", (PyCFunction) _pycartesianvolume_getAttributeNames, 1},
  {NULL, NULL} /* sentinel */
};

/**
 * Returns the specified attribute in the polar volume
 */
static PyObject* _pycartesianvolume_getattr(PyCartesianVolume* self, char* name)
{
  PyObject* res = NULL;
  if (strcmp("time", name) == 0) {
    const char* str = CartesianVolume_getTime(self->cvol);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("date", name) == 0) {
    const char* str = CartesianVolume_getDate(self->cvol);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("source", name) == 0) {
    const char* str = CartesianVolume_getSource(self->cvol);
    if (str != NULL) {
      return PyString_FromString(str);
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("projection", name) == 0) {
    Projection_t* projection = CartesianVolume_getProjection(self->cvol);
    if (projection != NULL) {
      PyProjection* result = PyProjection_New(projection);
      RAVE_OBJECT_RELEASE(projection);
      return (PyObject*)result;
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("objectType", name) == 0) {
    return PyInt_FromLong(CartesianVolume_getObjectType(self->cvol));
  } else if (strcmp("xscale", name) == 0) {
    return PyFloat_FromDouble(CartesianVolume_getXScale(self->cvol));
  } else if (strcmp("yscale", name) == 0) {
    return PyFloat_FromDouble(CartesianVolume_getYScale(self->cvol));
  } else if (strcmp("areaextent", name) == 0) {
    double llX = 0.0, llY = 0.0, urX = 0.0, urY = 0.0;
    CartesianVolume_getAreaExtent(self->cvol, &llX, &llY, &urX, &urY);
    return Py_BuildValue("(dddd)", llX, llY, urX, urY);
  }

  res = Py_FindMethod(_pycartesianvolume_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pycartesianvolume_setattr(PyCartesianVolume* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("time", name) == 0) {
    if (PyString_Check(val)) {
      if (!CartesianVolume_setTime(self->cvol, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set time");
      }
    } else if (val == Py_None) {
        CartesianVolume_setTime(self->cvol, NULL);
    } else {
        raiseException_gotoTag(done, PyExc_ValueError, "time should be specified as a string (HHmmss)");
    }
  } else if (strcmp("date", name) == 0) {
    if (PyString_Check(val)) {
      if (!CartesianVolume_setDate(self->cvol, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set date");
      }
    } else if (val == Py_None) {
      CartesianVolume_setDate(self->cvol, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "date should be specified as a string (YYYYMMDD)");
    }
  } else if (strcmp("source", name) == 0) {
    if (PyString_Check(val)) {
      if (!CartesianVolume_setSource(self->cvol, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "could not set source");
      }
    } else if (val == Py_None) {
      CartesianVolume_setSource(self->cvol, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "source should be specified as a string");
    }
  } else if (strcmp("xscale", name)==0) {
    if (PyFloat_Check(val)) {
      CartesianVolume_setXScale(self->cvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"xscale must be of type float");
    }
  } else if (strcmp("yscale", name)==0) {
    if (PyFloat_Check(val)) {
      CartesianVolume_setYScale(self->cvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"yscale must be of type float");
    }
  } else if (strcmp("projection", name) == 0) {
    if (PyProjection_Check(val)) {
      CartesianVolume_setProjection(self->cvol, ((PyProjection*)val)->projection);
    } else if (val == Py_None) {
      CartesianVolume_setProjection(self->cvol, NULL);
    }
  } else if (strcmp("objectType", name) == 0) {
    if (PyInt_Check(val)) {
      if (!CartesianVolume_setObjectType(self->cvol, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "objectType not supported");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "objectType must be a valid object type")
    }
  } else if (strcmp("areaextent", name) == 0) {
    double llX = 0.0, llY = 0.0, urX = 0.0, urY = 0.0;
    if (!PyArg_ParseTuple(val, "dddd", &llX, &llY, &urX, &urY)) {
      goto done;
    }
    CartesianVolume_setAreaExtent(self->cvol, llX, llY, urX, urY);
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, name);
  }

  result = 0;
done:
  return result;
}

/*@} End of Cartesian Volumes */

/// --------------------------------------------------------------------
/// Type definitions
/// --------------------------------------------------------------------
/*@{ Type definitions */
PyTypeObject PyCartesianVolume_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "CartesianVolumeCore", /*tp_name*/
  sizeof(PyCartesianVolume), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pycartesianvolume_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pycartesianvolume_getattr, /*tp_getattr*/
  (setattrfunc)_pycartesianvolume_setattr, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0,
  0, /*tp_as_mapping */
  0 /*tp_hash*/
};
/*@} End of Type definitions */

/// --------------------------------------------------------------------
/// Module setup
/// --------------------------------------------------------------------
/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pycartesianvolume_new, 1},
  {NULL,NULL} /*Sentinel*/
};

/**
 * Initializes polar volume.
 */
void init_cartesianvolume(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyCartesianVolume_API[PyCartesianVolume_API_pointers];
  PyObject *c_api_object = NULL;
  PyCartesianVolume_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_cartesianvolume", functions);
  if (module == NULL) {
    return;
  }
  PyCartesianVolume_API[PyCartesianVolume_Type_NUM] = (void*)&PyCartesianVolume_Type;
  PyCartesianVolume_API[PyCartesianVolume_GetNative_NUM] = (void *)PyCartesianVolume_GetNative;
  PyCartesianVolume_API[PyCartesianVolume_New_NUM] = (void*)PyCartesianVolume_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyCartesianVolume_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_cartesianvolume.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _cartesianvolume.error");
  }

  import_pyprojection();
  import_pycartesian();
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
