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

#define PYPOLARSCAN_MODULE
#include "pypolarscan.h"

#include <arrayobject.h>
#include "pyrave_debug.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"

PYRAVE_DEBUG_MODULE("_polarscan");

/**
 * Some helpful exception defines.
 */
#define raiseException_gotoTag(tag, type, msg) \
{PyErr_SetString(type, msg); goto tag;}

#define raiseException_returnNULL(type, msg) \
{PyErr_SetString(type, msg); return NULL;}

/**
 * Error object for reporting errors to the python interpreeter
 */
static PyObject *ErrorObject;

/// --------------------------------------------------------------------
/// Polar Scans
/// --------------------------------------------------------------------
/*@{ Polar Scans */
/**
 * Returns the native PolarScan_t instance.
 * @param[in] pypolarscan - the python polar scan instance
 * @returns the native polar scan instance.
 */
static PolarScan_t*
PyPolarScan_GetNative(PyPolarScan* pypolarscan)
{
  RAVE_ASSERT((pypolarscan != NULL), "pypolarscan == NULL");
  return RAVE_OBJECT_COPY(pypolarscan->scan);
}

/**
 * Creates a python polar scan from a native polar scan or will create an
 * initial native PolarScan if p is NULL.
 * @param[in] p - the native polar scan (or NULL)
 * @returns the python polar scan.
 */
static PyPolarScan*
PyPolarScan_New(PolarScan_t* p)
{
  PyPolarScan* result = NULL;
  PolarScan_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&PolarScan_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for polar scan.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for polar scan.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyPolarScan, &PyPolarScan_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->scan = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->scan, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyPolarScan instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for polar scan.");
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
static void _pypolarscan_dealloc(PyPolarScan* obj)
{
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->scan, obj);
  RAVE_OBJECT_RELEASE(obj->scan);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the polar scan.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pypolarscan_new(PyObject* self, PyObject* args)
{
  PyPolarScan* result = PyPolarScan_New(NULL);
  return (PyObject*)result;
}

static PyObject* _pypolarscan_setData(PyPolarScan* self, PyObject* args)
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
    raiseException_returnNULL(PyExc_ValueError, "A scan must be of rank 2");
  }

  datatype = translate_pyarraytype_to_ravetype(PyArray_TYPE(arraydata));

  if (PyArray_ITEMSIZE(arraydata) != get_ravetype_size(datatype)) {
    raiseException_returnNULL(PyExc_TypeError, "numpy and rave does not have same data sizes");
  }

  nbins  = PyArray_DIM(arraydata, 1);
  nrays  = PyArray_DIM(arraydata, 0);
  data   = PyArray_DATA(arraydata);

  if (!PolarScan_setData(self->scan, nbins, nrays, data, datatype)) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not allocate memory");
  }

  Py_RETURN_NONE;
}

static PyObject* _pypolarscan_getData(PyPolarScan* self, PyObject* args)
{
  long nbins = 0, nrays = 0;
  RaveDataType type = RaveDataType_UNDEFINED;
  PyObject* result = NULL;
  npy_intp dims[2] = {0,0};
  int arrtype = 0;
  void* data = NULL;

  nbins = PolarScan_getNbins(self->scan);
  nrays = PolarScan_getNrays(self->scan);
  type = PolarScan_getDataType(self->scan);
  data = PolarScan_getData(self->scan);

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
    memcpy(((PyArrayObject*)result)->data, PolarScan_getData(self->scan), nbytes);
  }

  return result;
}

/**
 * Calculates the azimuth index from an azimuth (in radians).
 * @param[in] self - this instance
 * @param[in] args - an azimuth value (in radians)
 * @returns the azimuth index or -1 if none could be determined.
 */
static PyObject* _pypolarscan_getAzimuthIndex(PyPolarScan* self, PyObject* args)
{
  double azimuth = 0.0L;
  int index = -1;

  if (!PyArg_ParseTuple(args, "d", &azimuth)) {
    return NULL;
  }

  index = PolarScan_getAzimuthIndex(self->scan, azimuth);
  if (index < 0) {
    raiseException_returnNULL(PyExc_ValueError, "Invalid azimuth");
  }

  return PyInt_FromLong(index);
}

/**
 * Calculates the range index from a specified range
 * @param[in] self - this instance
 * @param[in] args - the range (in meters)
 * @returns the range index or -1 if outside range
 */
static PyObject* _pypolarscan_getRangeIndex(PyPolarScan* self, PyObject* args)
{
  double range = 0.0L;
  int index = -1;

  if (!PyArg_ParseTuple(args, "d", &range)) {
    return NULL;
  }

  index = PolarScan_getRangeIndex(self->scan, range);

  return PyInt_FromLong(index);
}

/**
 * Returns the value at the specified ray and bin index.
 * @param[in] self - this instance
 * @param[in] args - ray index, bin index.
 * @returns a tuple of value type and value
 */
static PyObject* _pypolarscan_getValueAtIndex(PyPolarScan* self, PyObject* args)
{
  double value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  int ray = 0, bin = 0;
  if (!PyArg_ParseTuple(args, "ii", &ray, &bin)) {
    return NULL;
  }

  type = PolarScan_getValueAtIndex(self->scan, ray, bin, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Returns the converted value at the specified ray and bin index.
 * @param[in] self - this instance
 * @param[in] args - ray index, bin index.
 * @returns a tuple of value type and value
 */
static PyObject* _pypolarscan_getConvertedValueAtIndex(PyPolarScan* self, PyObject* args)
{
  double value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  int ray = 0, bin = 0;
  if (!PyArg_ParseTuple(args, "ii", &ray, &bin)) {
    return NULL;
  }

  type = PolarScan_getConvertedValueAtIndex(self->scan, ray, bin, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Returns the value at the specified azimuth and range for this scan.
 * @param[in] self - this instance
 * @param[in] args - two doubles, azimuth (in radians) and range (in meters)
 * @returns a tuple of value type and value
 */
static PyObject* _pypolarscan_getValueAtAzimuthAndRange(PyPolarScan* self, PyObject* args)
{
  double value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  double a = 0, r = 0;
  if (!PyArg_ParseTuple(args, "dd", &a, &r)) {
    return NULL;
  }

  type = PolarScan_getValueAtAzimuthAndRange(self->scan, a, r, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Returns the value that is nearest to the specified longitude/latitude.
 * @param[in] self - this instance
 * @param[in] args - a tuple consisting of (longitude, latitude).
 * @returns a tuple of (value type, value) or NULL on failure
 */
static PyObject* _pypolarscan_getNearest(PyPolarScan* self, PyObject* args)
{
  double lon = 0.0L, lat = 0.0L, value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  if (!PyArg_ParseTuple(args, "(dd)", &lon, &lat)) {
    return NULL;
  }

  type = PolarScan_getNearest(self->scan, lon, lat, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * All methods a polar scan can have
 */
static struct PyMethodDef _pypolarscan_methods[] =
{
  {"setData", (PyCFunction) _pypolarscan_setData, 1},
  {"getData", (PyCFunction) _pypolarscan_getData, 1},
  {"getAzimuthIndex", (PyCFunction) _pypolarscan_getAzimuthIndex, 1},
  {"getRangeIndex", (PyCFunction) _pypolarscan_getRangeIndex, 1},
  {"getValueAtIndex", (PyCFunction) _pypolarscan_getValueAtIndex, 1},
  {"getConvertedValueAtIndex", (PyCFunction) _pypolarscan_getConvertedValueAtIndex, 1},
  {"getValueAtAzimuthAndRange", (PyCFunction) _pypolarscan_getValueAtAzimuthAndRange, 1},
  {"getNearest", (PyCFunction) _pypolarscan_getNearest, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the polar scan
 */
static PyObject* _pypolarscan_getattr(PyPolarScan* self, char* name)
{
  PyObject* res;
  if (strcmp("elangle", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getElangle(self->scan));
  } else if (strcmp("nbins", name) == 0) {
    return PyInt_FromLong(PolarScan_getNbins(self->scan));
  } else if (strcmp("rscale", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getRscale(self->scan));
  } else if (strcmp("nrays", name) == 0) {
    return PyInt_FromLong(PolarScan_getNrays(self->scan));
  } else if (strcmp("rstart", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getRstart(self->scan));
  } else if (strcmp("a1gate", name) == 0) {
    return PyInt_FromLong(PolarScan_getA1gate(self->scan));
  } else if (strcmp("quantity", name) == 0) {
    return PyString_FromString(PolarScan_getQuantity(self->scan));
  } else if (strcmp("gain", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getGain(self->scan));
  } else if (strcmp("offset", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getOffset(self->scan));
  } else if (strcmp("nodata", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getNodata(self->scan));
  } else if (strcmp("undetect", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getUndetect(self->scan));
  } else if (strcmp("datatype", name) == 0) {
    return PyInt_FromLong(PolarScan_getDataType(self->scan));
  } else if (strcmp("beamwidth", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getBeamWidth(self->scan));
  } else if (strcmp("longitude", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getLongitude(self->scan));
  } else if (strcmp("latitude", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getLatitude(self->scan));
  } else if (strcmp("height", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getHeight(self->scan));
  }
  res = Py_FindMethod(_pypolarscan_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pypolarscan_setattr(PyPolarScan* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("elangle", name)==0) {
    if (PyFloat_Check(val)) {
      PolarScan_setElangle(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"elangle must be of type float");
    }
  } else if (strcmp("nbins", name) == 0) {
    if (PyInt_Check(val)) {
      PolarScan_setNbins(self->scan, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"nbins must be of type int");
    }
  } else if (strcmp("rscale", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setRscale(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "rscale must be of type float");
    }
  } else if (strcmp("nrays", name) == 0) {
    if (PyInt_Check(val)) {
      PolarScan_setNrays(self->scan, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"nrays must be of type int");
    }
  } else if (strcmp("rstart", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setRstart(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "rstart must be of type float");
    }
  } else if (strcmp("a1gate", name) == 0) {
    if (PyInt_Check(val)) {
      PolarScan_setA1gate(self->scan, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"a1gate must be of type int");
    }
  } else if (strcmp("quantity", name) == 0) {
    if (PyString_Check(val)) {
      PolarScan_setQuantity(self->scan, PyString_AsString(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"quantity must be of type string");
    }
  } else if (strcmp("gain", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setGain(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "gain must be of type float");
    }
  } else if (strcmp("offset", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setOffset(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "offset must be of type float");
    }
  } else if (strcmp("nodata", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setNodata(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "nodata must be of type float");
    }
  } else if (strcmp("undetect", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setUndetect(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "undetect must be of type float");
    }
  } else if (strcmp("datatype", name) == 0) {
    if (PyInt_Check(val)) {
      if (!PolarScan_setDataType(self->scan, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "datatype must be in valid range");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "datatype must be of type RaveDataType");
    }
  } else if (strcmp("beamwidth", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setBeamWidth(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "beamwidth must be of type float");
    }
  } else if (strcmp("longitude", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setLongitude(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "longitude must be of type float");
    }
  } else if (strcmp("latitude", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setLatitude(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "latitude must be of type float");
    }
  } else if (strcmp("height", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarScan_setHeight(self->scan, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "height must be of type float");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, name);
  }

  result = 0;
done:
  return result;
}
/*@} End of Polar Scans */

/// --------------------------------------------------------------------
/// Type definitions
/// --------------------------------------------------------------------
/*@{ Type definition */
PyTypeObject PyPolarScan_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "PolarScanCore", /*tp_name*/
  sizeof(PyPolarScan), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pypolarscan_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pypolarscan_getattr, /*tp_getattr*/
  (setattrfunc)_pypolarscan_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_pypolarscan_new, 1},
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_polarscan(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyPolarScan_API[PyPolarScan_API_pointers];
  PyObject *c_api_object = NULL;
  PyPolarScan_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_polarscan", functions);
  if (module == NULL) {
    return;
  }
  PyPolarScan_API[PyPolarScan_Type_NUM] = (void*)&PyPolarScan_Type;
  PyPolarScan_API[PyPolarScan_GetNative_NUM] = (void *)PyPolarScan_GetNative;
  PyPolarScan_API[PyPolarScan_New_NUM] = (void*)PyPolarScan_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyPolarScan_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_polarscan.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _polarscan.error");
  }

  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
