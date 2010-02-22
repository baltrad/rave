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

#define PYCARTESIAN_MODULE        /**< to get correct part of pycartesian.h */
#include "pycartesian.h"

#include "pyprojection.h"
#include "pyarea.h"
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
  RaveDataType type = RaveDataType_UNDEFINED;

  if (!PyArg_ParseTuple(args, "Oi", &inarea, &type)) {
    return NULL;
  }

  if (!PyArea_Check(inarea)) {
    raiseException_returnNULL(PyExc_TypeError, "First argument must be a PyAreaCore instance");
  }

  if (!Cartesian_init(self->cartesian, ((PyArea*)inarea)->area, type)) {
    raiseException_returnNULL(PyExc_ValueError, "Failed to initialize cartesian product");
  }

  Py_RETURN_NONE;
}

/**
 * Sets the data array that should be used for this product.
 * @param[in] self this instance.
 * @param[in] args - the array
 * @return Py_None on success, otherwise NULL
 */
static PyObject* _pycartesian_setData(PyCartesian* self, PyObject* args)
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

  if (!Cartesian_setData(self->cartesian, xsize, ysize, data, datatype)) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not allocate memory");
  }

  Py_RETURN_NONE;
}

static PyObject* _pycartesian_getData(PyCartesian* self, PyObject* args)
{
  long xsize = 0, ysize = 0;
  RaveDataType type = RaveDataType_UNDEFINED;
  PyObject* result = NULL;
  npy_intp dims[2] = {0,0};
  int arrtype = 0;
  void* data = NULL;

  xsize = Cartesian_getXSize(self->cartesian);
  ysize = Cartesian_getYSize(self->cartesian);
  type = Cartesian_getDataType(self->cartesian);
  data = Cartesian_getData(self->cartesian);

  dims[1] = (npy_intp)xsize;
  dims[0] = (npy_intp)ysize;
  arrtype = translate_ravetype_to_pyarraytype(type);

  if (data == NULL) {
    raiseException_returnNULL(PyExc_IOError, "cartesian product does not have any data");
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
    memcpy(((PyArrayObject*)result)->data, (unsigned char*)Cartesian_getData(self->cartesian), nbytes);
  }
  return result;
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
 * sets the value at the specified position
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
 * All methods a cartesian product can have
 */
static struct PyMethodDef _pycartesian_methods[] =
{
  {"init", (PyCFunction) _pycartesian_init, 1},
  {"setData", (PyCFunction) _pycartesian_setData, 1},
  {"getData", (PyCFunction) _pycartesian_getData, 1},
  {"getLocationX", (PyCFunction) _pycartesian_getLocationX, 1},
  {"getLocationY", (PyCFunction) _pycartesian_getLocationY, 1},
  {"setValue", (PyCFunction) _pycartesian_setValue, 1},
  {"getValue", (PyCFunction) _pycartesian_getValue, 1},
  {"isTransformable", (PyCFunction) _pycartesian_isTransformable, 1},
  {"getMean", (PyCFunction) _pycartesian_getMean, 1},
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
  } else if (strcmp("quantity", name) == 0) {
    if (Cartesian_getQuantity(self->cartesian) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(Cartesian_getQuantity(self->cartesian));
    }
  } else if (strcmp("gain", name) == 0) {
    return PyFloat_FromDouble(Cartesian_getGain(self->cartesian));
  } else if (strcmp("offset", name) == 0) {
    return PyFloat_FromDouble(Cartesian_getOffset(self->cartesian));
  } else if (strcmp("nodata", name) == 0) {
    return PyFloat_FromDouble(Cartesian_getNodata(self->cartesian));
  } else if (strcmp("undetect", name) == 0) {
    return PyFloat_FromDouble(Cartesian_getUndetect(self->cartesian));
  } else if (strcmp("datatype", name) == 0) {
    return PyInt_FromLong(Cartesian_getDataType(self->cartesian));
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
  } else if (strcmp("quantity", name) == 0) {
    if (PyString_Check(val)) {
      if (!Cartesian_setQuantity(self->cartesian, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_MemoryError, "Could not set quantity");
      }
    } else if (val == Py_None) {
      Cartesian_setQuantity(self->cartesian, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"quantity must be of type string");
    }
  } else if (strcmp("gain", name) == 0) {
    if (PyFloat_Check(val)) {
      Cartesian_setGain(self->cartesian, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "gain must be of type float");
    }
  } else if (strcmp("offset", name) == 0) {
    if (PyFloat_Check(val)) {
      Cartesian_setOffset(self->cartesian, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "offset must be of type float");
    }
  } else if (strcmp("nodata", name) == 0) {
    if (PyFloat_Check(val)) {
      Cartesian_setNodata(self->cartesian, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "nodata must be of type float");
    }
  } else if (strcmp("undetect", name) == 0) {
    if (PyFloat_Check(val)) {
      Cartesian_setUndetect(self->cartesian, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "undetect must be of type float");
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
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
