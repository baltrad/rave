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
 * Defines a polar volume and the operations that can be performed
 * on this object.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-14
 */
#include <Python.h>
#include <arrayobject.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "raveutil.h"
#include "rave.h"
#include "polarscan.h"
#include "polarvolume.h"
#include "cartesian.h"
#include "rave_debug.h"
#include "rave_alloc.h"

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

typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  PolarScan_t* scan; /**< the scan type */
} PolarScan;

/**
 * The polar volume object
 */
typedef struct {
   PyObject_HEAD /*Always have to be on top*/
   PolarVolume_t* pvol;
} PolarVolume;

/**
 * The cartesian product object
 */
typedef struct {
   PyObject_HEAD /*Always have to be on top*/
   Cartesian_t* cartesian;
} Cartesian;

/**
 * PolarScan represents one scan in a pvol
 */
staticforward PyTypeObject PolarScan_Type;

/**
 * PolarVolume represents one pvol
 */
staticforward PyTypeObject PolarVolume_Type;

/**
 * Cartesian represents one cartesian product
 */
staticforward PyTypeObject Cartesian_Type;

/**
 * Checks if the object is a PolarScan type
 */
#define PolarScan_Check(op) ((op)->ob_type == &PolarScan_Type)

/**
 * Checks if the object is a PolarVolume type
 */
#define PolarVolume_Check(op) ((op)->ob_type == &PolarVolume_Type)

/**
 * Checks if the object is a PolarVolume type
 */
#define Cartesian_Check(op) ((op)->ob_type == &Cartesian_Type)

/// --------------------------------------------------------------------
/// Polar Scans
/// --------------------------------------------------------------------
/*@{ Polar Scans */

/**
 * Deallocates the polar scan
 * @param[in] obj the object to deallocate.
 */
static void _polarscan_dealloc(PolarScan* obj)
{
  if (obj == NULL) {
    return;
  }
  PolarScan_release(obj->scan); // Will trigger onscan_release (I hope)
}

/**
 * Creates a new instance of the polar scan.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _polarscan_new(PyObject* self, PyObject* args)
{
  PolarScan* result = NULL;
  result = PyObject_NEW(PolarScan, &PolarScan_Type);
  if (result == NULL) {
    return NULL;
  }
  result->scan = PolarScan_new();
  if (result->scan == NULL) {
    RAVE_CRITICAL0("Could not allocate scan");
    PyObject_Del(result);
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate scan");
  }
  // Keep track of pyobjects..
  PolarScan_setVoidPtr(result->scan, (void*)result);
  return (PyObject*)result;
}

static PyObject* _polarscan_setData(PolarScan* self, PyObject* args)
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

/**
 * All methods a polar scan can have
 */
static struct PyMethodDef _polarscan_methods[] =
{
  { "setData", (PyCFunction) _polarscan_setData, 1},
  { NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the polar scan
 */
static PyObject* _polarscan_getattr(PolarScan* self, char* name)
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
  }

  res = Py_FindMethod(_polarscan_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _polarscan_setattr(PolarScan* self, char* name, PyObject* val)
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
  }

  result = 0;
done:
  return result;
}
/*@} End of Polar Scans */

/// --------------------------------------------------------------------
/// Polar Volumes
/// --------------------------------------------------------------------
/*@{ Polar Volumes */

/**
 * Deallocates the polar volume
 * @param[in] obj the object to deallocate.
 */
static void _polarvolume_dealloc(PolarVolume* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PolarVolume_release(obj->pvol);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the polar volume.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _polarvolume_new(PyObject* self, PyObject* args)
{
  PolarVolume* result = NULL;
  result = PyObject_NEW(PolarVolume, &PolarVolume_Type);
  if (result == NULL) {
    return NULL;
  }
  result->pvol = PolarVolume_new();
  if (result->pvol == NULL) {
    RAVE_CRITICAL0("Could not allocate volume");
    PyObject_Del(result);
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate volume");
  }
  return (PyObject*)result;
}

/**
 * Adds one scan to a volume.
 * @param[in] self - the polar volume
 * @param[in] args - the scan, must be of type PolarScanCore
 * @return NULL on failure
 */
static PyObject* _polarvolume_addScan(PolarVolume* self, PyObject* args)
{
  PyObject* inptr = NULL;
  PolarScan* polarScan = NULL;

  if (!PyArg_ParseTuple(args, "O", &inptr)) {
    return NULL;
  }

  if (!PolarScan_Check(inptr)) {
    raiseException_returnNULL(PyExc_TypeError,"Added object must be of type PolarScanCore");
  }

  polarScan = (PolarScan*)inptr;

  if (!PolarVolume_addScan(self->pvol, polarScan->scan)) {
    raiseException_returnNULL(PyExc_MemoryError, "Failed to add scan to volume");
  }

  Py_RETURN_NONE;
}

/**
 * Returns the scan at the provided index.
 * @param[in] self - the polar volume
 * @param[in] args - the index must be >= 0 and < getNumberOfScans
 * @return NULL on failure
 */
static PyObject* _polarvolume_getScan(PolarVolume* self, PyObject* args)
{
  int index = -1;
  PolarScan_t* scan = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  if (index < 0 || index >= PolarVolume_getNumberOfScans(self->pvol)) {
    raiseException_returnNULL(PyExc_IndexError, "Index out of range");
  }

  if(!PolarVolume_getScan(self->pvol, index, &scan)) {
    raiseException_returnNULL(PyExc_IndexError, "Could not aquire scan");
  }

  if (scan != NULL) {
    result = (PyObject*)PolarScan_getVoidPtr(scan);
    Py_INCREF(result);
  }

  PolarScan_release(scan);

  return result;
}

/**
 * Returns the number of scans for this volume
 * @param[in] self - the polar volume
 * @param[in] args - not used
 * @return NULL on failure or a PyInteger
 */
static PyObject* _polarvolume_getNumberOfScans(PolarVolume* self, PyObject* args)
{
  return PyInt_FromLong(PolarVolume_getNumberOfScans(self->pvol));
}

/**
 * Sorts the scans by comparing elevations in either ascending or descending order
 * @param[in] self - the polar volume
 * @param[in] args - 1 if soring should be done in ascending order, otherwise descending
 * @return NULL on failure or a Py_None
 */
static PyObject* _polarvolume_sortByElevations(PolarVolume* self, PyObject* args)
{
  int order = 0;

  if (!PyArg_ParseTuple(args, "i", &order)) {
    return NULL;
  }

  PolarVolume_sortByElevations(self->pvol, order);

  Py_RETURN_NONE;
}

/**
 * Creates a cappi from a polar volume
 * @param[in] self the polar volume
 * @param[in] args arguments for generating the cappi
 * @return the object on success, otherwise NULL
 */
static PyObject* _polarvolume_cappi(PolarVolume* self, PyObject* args)
{
  Cartesian* cartesian = NULL;
  PyObject* pycartesian = NULL;

  if(!PyArg_ParseTuple(args, "O", &pycartesian)) {
    return NULL;
  }

  if (!Cartesian_Check(pycartesian)) {
    raiseException_returnNULL(PyExc_TypeError, "Must provide cartesian product");
  }

  cartesian = (Cartesian*)pycartesian;

  if (!PolarVolume_cappi(self->pvol, cartesian->cartesian)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to transform volume into a cappi");
  }

  Py_RETURN_NONE;
}

/**
 * All methods a polar volume can have
 */
static struct PyMethodDef _polarvolume_methods[] =
{
  { "addScan", (PyCFunction) _polarvolume_addScan, 1},
  { "getScan", (PyCFunction) _polarvolume_getScan, 1},
  { "getNumberOfScans", (PyCFunction) _polarvolume_getNumberOfScans, 1},
  { "sortByElevations", (PyCFunction) _polarvolume_sortByElevations, 1},
  { "cappi", (PyCFunction) _polarvolume_cappi, 1 },
  { NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the polar volume
 */
static PyObject* _polarvolume_getattr(PolarVolume* self, char* name)
{
  PyObject* res = NULL;
  if (strcmp("longitude", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getLongitude(self->pvol));
  } else if (strcmp("latitude", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getLatitude(self->pvol));
  } else if (strcmp("height", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getHeight(self->pvol));
  }

  res = Py_FindMethod(_polarvolume_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _polarvolume_setattr(PolarVolume* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("longitude", name)==0) {
    if (PyFloat_Check(val)) {
      PolarVolume_setLongitude(self->pvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"longitude must be of type float");
    }
  } else if (strcmp("latitude", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarVolume_setLatitude(self->pvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "latitude must be of type float");
    }
  } else if (strcmp("height", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarVolume_setHeight(self->pvol, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "height must be of type float");
    }
  }

  result = 0;
done:
  return result;
}

/*@} End of Polar Volumes */

/// --------------------------------------------------------------------
/// Cartesian products
/// --------------------------------------------------------------------
/*@{ Cartesian products */
/**
 * Deallocates the cartesian product
 * @param[in] obj the object to deallocate.
 */
static void _cartesian_dealloc(Cartesian* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  Cartesian_release(obj->cartesian);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the cartesian product.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _cartesian_new(PyObject* self, PyObject* args)
{
  Cartesian* result = NULL;
  result = PyObject_NEW(Cartesian, &Cartesian_Type);
  if (result == NULL) {
    return NULL;
  }
  result->cartesian = Cartesian_new();
  if (result->cartesian == NULL) {
    RAVE_CRITICAL0("Could not allocate cartesian product");
    PyObject_Del(result);
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate cartesian product");
  }
  return (PyObject*)result;
}

/**
 * Sets the data array that should be used for this product.
 * @param[in] self this instance.
 * @param[in] args - the array
 * @return Py_None on success, otherwise NULL
 */
static PyObject* _cartesian_setData(Cartesian* self, PyObject* args)
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

/**
 * Returns the x location defined by area extent and x scale and the provided x position.
 * @param[in] self this instance.
 * @param[in] args - x position
 * @return the x location on success, otherwise NULL
 */
static PyObject* _cartesian_getLocationX(Cartesian* self, PyObject* args)
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
static PyObject* _cartesian_getLocationY(Cartesian* self, PyObject* args)
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
 * All methods a cartesian product can have
 */
static struct PyMethodDef _cartesian_methods[] =
{
  { "setData", (PyCFunction) _cartesian_setData, 1},
  { "getLocationX", (PyCFunction) _cartesian_getLocationX, 1},
  { "getLocationY", (PyCFunction) _cartesian_getLocationY, 1},
  { NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the cartesian
 * @param[in] self - the cartesian product
 */
static PyObject* _cartesian_getattr(Cartesian* self, char* name)
{
  PyObject* res = NULL;

  if (strcmp("xsize", name) == 0) {
    return PyInt_FromLong(Cartesian_getXSize(self->cartesian));
  } else if (strcmp("ysize", name) == 0) {
    return PyInt_FromLong(Cartesian_getYSize(self->cartesian));
  } else if (strcmp("xscale", name) == 0) {
    return PyFloat_FromDouble(Cartesian_getXScale(self->cartesian));
  } else if (strcmp("yscale", name) == 0) {
    return PyFloat_FromDouble(Cartesian_getYScale(self->cartesian));
  } else if (strcmp("quantity", name) == 0) {
    return PyString_FromString(Cartesian_getQuantity(self->cartesian));
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
  }

  res = Py_FindMethod(_cartesian_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _cartesian_setattr(Cartesian* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("xsize", name)==0) {
    if (PyInt_Check(val)) {
      Cartesian_setXSize(self->cartesian, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"xsize must be of type int");
    }
  } else if (strcmp("ysize", name)==0) {
    if (PyInt_Check(val)) {
      Cartesian_setYSize(self->cartesian, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"ysize must be of type int");
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
      Cartesian_setQuantity(self->cartesian, PyString_AsString(val));
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
  } else if (strcmp("datatype", name) == 0) {
    if (PyInt_Check(val)) {
      if (!Cartesian_setDataType(self->cartesian, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "datatype must be in valid range");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "datatype must be of type RaveDataType");
    }
  } else if (strcmp("areaextent", name) == 0) {
    double llX = 0.0, llY = 0.0, urX = 0.0, urY = 0.0;
    if (!PyArg_ParseTuple(val, "dddd", &llX, &llY, &urX, &urY)) {
      goto done;
    }
    Cartesian_setAreaExtent(self->cartesian, llX, llY, urX, urY);
  }

  result = 0;
done:
  return result;
}

/*@} End of Cartesian products */

/// --------------------------------------------------------------------
/// Type definitions
/// --------------------------------------------------------------------
/*@{ Type definitions */
statichere PyTypeObject PolarScan_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "PolarScanCore", /*tp_name*/
  sizeof(PolarScan), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_polarscan_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_polarscan_getattr, /*tp_getattr*/
  (setattrfunc)_polarscan_setattr, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0,
  0, /*tp_as_mapping */
  0 /*tp_hash*/
};

statichere PyTypeObject PolarVolume_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "PolarVolumeCore", /*tp_name*/
  sizeof(PolarVolume), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_polarvolume_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_polarvolume_getattr, /*tp_getattr*/
  (setattrfunc)_polarvolume_setattr, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0,
  0, /*tp_as_mapping */
  0 /*tp_hash*/
};

statichere PyTypeObject Cartesian_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "CartesianCore", /*tp_name*/
  sizeof(Cartesian), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_cartesian_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_cartesian_getattr, /*tp_getattr*/
  (setattrfunc)_cartesian_setattr, /*tp_setattr*/
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
  {"volume", (PyCFunction)_polarvolume_new, 1},
  {"scan", (PyCFunction)_polarscan_new, 1},
  {"cartesian", (PyCFunction)_cartesian_new, 1},
  {NULL,NULL} /*Sentinel*/
};

/**
 * Adds constants to the dictionary (probably the modules dictionary).
 * @param[in] dictionary - the dictionary the long should be added to
 * @param[in] name - the name of the constant
 * @param[in] value - the value
 */
static void add_long_constant(PyObject* dictionary, const char* name, long value)
{
  PyObject* tmp = NULL;
  tmp = PyInt_FromLong(value);
  if (tmp != NULL) {
    PyDict_SetItemString(dictionary, name, tmp);
  }
  Py_XDECREF(tmp);
}

/**
 * Initializes polar volume.
 */
void init_rave(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  PolarVolume_Type.ob_type = &PyType_Type;
  PolarScan_Type.ob_type = &PyType_Type;
  Cartesian_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_rave", functions);
  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_rave.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _rave.error");
  }

  Rave_initializeDebugger();

  if (atexit(rave_alloc_print_statistics) != 0) {
    fprintf(stderr, "Could not set atexit function");
  }

  // Initialize some constants
  add_long_constant(dictionary, "RaveDataType_UNDEFINED", RaveDataType_UNDEFINED);
  add_long_constant(dictionary, "RaveDataType_CHAR", RaveDataType_CHAR);
  add_long_constant(dictionary, "RaveDataType_UCHAR", RaveDataType_UCHAR);
  add_long_constant(dictionary, "RaveDataType_SHORT", RaveDataType_SHORT);
  add_long_constant(dictionary, "RaveDataType_INT", RaveDataType_INT);
  add_long_constant(dictionary, "RaveDataType_LONG", RaveDataType_LONG);
  add_long_constant(dictionary, "RaveDataType_FLOAT", RaveDataType_FLOAT);
  add_long_constant(dictionary, "RaveDataType_DOUBLE", RaveDataType_DOUBLE);

  import_array(); /*To make sure I get access to Numeric*/
}
/*@} End of Module setup */
