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
 * Python wrapper for the rave product generation framework.
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
#include "transform.h"
#include "projection.h"
#include "rave_io.h"
#include "raveobject_list.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "hlhdf.h"
#include "hlhdf_debug.h"

/**
 * Some helpful exception defines.
 */
#define raiseException_gotoTag(tag, type, msg) \
{PyErr_SetString(type, msg); goto tag;}

#define raiseException_returnNULL(type, msg) \
{PyErr_SetString(type, msg); return NULL;}

static int PolarScan_alloc = 0;
static int PolarScan_dealloc = 0;
static int PolarVolume_alloc = 0;
static int PolarVolume_dealloc = 0;
static int Cartesian_alloc = 0;
static int Cartesian_dealloc = 0;
static int Transform_alloc = 0;
static int Transform_dealloc = 0;
static int Projection_alloc = 0;
static int Projection_dealloc = 0;
static int RaveIO_alloc = 0;
static int RaveIO_dealloc = 0;
static int RaveList_alloc = 0;
static int RaveList_dealloc = 0;


/**
 * Error object for reporting errors to the python interpreeter
 */
static PyObject *ErrorObject;

/**
 * A projection
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  Projection_t* projection;
} Projection;

/**
 * The polar scan
 */
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
   //Projection* projection;
} Cartesian;

/**
 * The transformator
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  Transform_t* transform;
} Transform;

/**
 * The RaveIO
 */
typedef struct {
  PyObject_HEAD /* Always has to be on top */
  RaveIO_t* raveio;
} RaveIO;

/**
 * The RaveList
 */
typedef struct {
  PyObject_HEAD /* Always has to be on top */
  RaveObjectList_t* list;
} RaveList;

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
 * Transform represents one transformator
 */
staticforward PyTypeObject Transform_Type;

/**
 * Projection represents one projection
 */
staticforward PyTypeObject Projection_Type;

/**
 * RaveIO represents the Rave IO operations
 */
staticforward PyTypeObject RaveIO_Type;

/**
 * RaveList represents the Rave Object List operations
 */
staticforward PyTypeObject RaveList_Type;

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

/**
 * Checks if the object is a PolarVolume type
 */
#define Transform_Check(op) ((op)->ob_type == &Transform_Type)

/**
 * Checks if the object is a Projection type
 */
#define Projection_Check(op) ((op)->ob_type == &Projection_Type)

/**
 * Checks if the object is a RaveIO type
 */
#define RaveIO_Check(op) ((op)->ob_type == &RaveIO_Type)

/**
 * Checks if the object is a RaveIO type
 */
#define RaveList_Check(op) ((op)->ob_type == &RaveList_Type)

/*@{ Forward declarations */
static Projection* _projection_createPyObject(Projection_t* projection);
/*@} End of Forward declarations */

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
  RAVE_OBJECT_UNBIND(obj->scan, obj);
  RAVE_OBJECT_RELEASE(obj->scan);
  PyObject_Del(obj);
  PolarScan_dealloc++;
}

/**
 * Creates the polar scan python object from a polar scan.
 * @param[in] scan - the polar scan
 * @returns a Python Polar scan on success.
 */
static PolarScan* _polarscan_createPyObject(PolarScan_t* scan)
{
  PolarScan* result = NULL;
  if (scan == NULL) {
    RAVE_CRITICAL0("Trying to create a python polar scan without the scan");
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for polar scan");
  }
  result = PyObject_NEW(PolarScan, &PolarScan_Type);
  if (result == NULL) {
    RAVE_CRITICAL0("Failed to allocate memory for polar scan");
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for polar scan");
  }
  result->scan = RAVE_OBJECT_COPY(scan);
  RAVE_OBJECT_BIND(result->scan, result);
  PolarScan_alloc++;
  return result;
}

/**
 * Creates a new instance of the polar scan.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _polarscan_new(PyObject* self, PyObject* args)
{
  PolarScan_t* scan = NULL;
  PolarScan* result = NULL;
  scan = RAVE_OBJECT_NEW(&PolarScan_TYPE);
  if (scan == NULL) {
    RAVE_CRITICAL0("Failed to allocate polar scan");
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for polar scan");
  }

  result = _polarscan_createPyObject(scan);
  if (result == NULL) {
    RAVE_CRITICAL0("Failed to allocate py polar scan");
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for py polar scan");
  }

  RAVE_OBJECT_RELEASE(scan);
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

static PyObject* _polarscan_getData(PolarScan* self, PyObject* args)
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
static PyObject* _polarscan_getAzimuthIndex(PolarScan* self, PyObject* args)
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
static PyObject* _polarscan_getRangeIndex(PolarScan* self, PyObject* args)
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
 * Returns the value at the specified azimuth and range index.
 * @param[in] self - this instance
 * @param[in] args - azimuth index, range index.
 * @returns a tuple of value type and value
 */
static PyObject* _polarscan_getValueAtIndex(PolarScan* self, PyObject* args)
{
  double value = 0.0L;
  RaveValueType type = RaveValueType_NODATA;
  int ai = 0, ri = 0;
  if (!PyArg_ParseTuple(args, "ii", &ai, &ri)) {
    return NULL;
  }

  type = PolarScan_getValueAtIndex(self->scan, ai, ri, &value);

  return Py_BuildValue("(id)", type, value);
}

/**
 * Returns the value at the specified azimuth and range for this scan.
 * @param[in] self - this instance
 * @param[in] args - two doubles, azimuth (in radians) and range (in meters)
 * @returns a tuple of value type and value
 */
static PyObject* _polarscan_getValueAtAzimuthAndRange(PolarScan* self, PyObject* args)
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
static PyObject* _polarscan_getNearest(PolarScan* self, PyObject* args)
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
static struct PyMethodDef _polarscan_methods[] =
{
  {"setData", (PyCFunction) _polarscan_setData, 1},
  {"getData", (PyCFunction) _polarscan_getData, 1},
  {"getAzimuthIndex", (PyCFunction) _polarscan_getAzimuthIndex, 1},
  {"getRangeIndex", (PyCFunction) _polarscan_getRangeIndex, 1},
  {"getValueAtIndex", (PyCFunction) _polarscan_getValueAtIndex, 1},
  {"getValueAtAzimuthAndRange", (PyCFunction) _polarscan_getValueAtAzimuthAndRange, 1},
  {"getNearest", (PyCFunction) _polarscan_getNearest, 1},
  {NULL, NULL } /* sentinel */
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
  } else if (strcmp("longitude", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getLongitude(self->scan));
  } else if (strcmp("latitude", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getLatitude(self->scan));
  } else if (strcmp("height", name) == 0) {
    return PyFloat_FromDouble(PolarScan_getHeight(self->scan));
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
  RAVE_OBJECT_UNBIND(obj->pvol, obj);
  RAVE_OBJECT_RELEASE(obj->pvol);
  PyObject_Del(obj);
  PolarVolume_dealloc++;
}

/**
 * Creates the polar volume python object from a polar volume.
 * @param[in] pvol - the polar volume
 * @returns a Python Polar Volume on success.
 */
static PolarVolume* _polarvolume_createPyObject(PolarVolume_t* pvol)
{
  PolarVolume* result = NULL;
  if (pvol == NULL) {
    RAVE_CRITICAL0("Trying to create a python polar volume without the volume");
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for polar volume");
  }
  result = PyObject_NEW(PolarVolume, &PolarVolume_Type);
  if (result == NULL) {
    RAVE_CRITICAL0("Failed to allocate memory for polar volume");
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for polar volume");
  }
  result->pvol = RAVE_OBJECT_COPY(pvol);
  RAVE_OBJECT_BIND(result->pvol, result);
  PolarVolume_alloc++;
  return result;
}

/**
 * Creates a new instance of the polar volume.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _polarvolume_new(PyObject* self, PyObject* args)
{
  PolarVolume_t* pvol = NULL;
  PolarVolume* result = NULL;

  pvol = RAVE_OBJECT_NEW(&PolarVolume_TYPE);
  if (pvol == NULL) {
    RAVE_CRITICAL0("Failed to allocate polar volume");
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for polar volume");
  }

  result = _polarvolume_createPyObject(pvol);
  if (result == NULL) {
    RAVE_CRITICAL0("Failed to allocate py polar volume");
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for py polar volume");
  }

  RAVE_OBJECT_RELEASE(pvol);
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

  if((scan = PolarVolume_getScan(self->pvol, index)) == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "Could not aquire scan");
  }

  if (scan != NULL) {
    result = RAVE_OBJECT_GETBINDING(scan);
    if (result == NULL) {
      result = (PyObject*)_polarscan_createPyObject(scan);
    } else {
      Py_INCREF(result);
    }
  }

  RAVE_OBJECT_RELEASE(scan);

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
 * Returns 1 if the scans in the volume are sorted in ascending order, otherwise
 * 0 will be returned.
 * @param[in] self - the polar volume
 * @param[in] args - not used
 * @returns NULL on failure or a PyInteger where 1 indicates that the scans are sorted after elevation angle
 * in ascending order.
 */
static PyObject* _polarvolume_isAscendingScans(PolarVolume* self, PyObject* args)
{
  return PyBool_FromLong(PolarVolume_isAscendingScans(self->pvol));
}

/**
 * Returns 1 if is is possible to perform transform operations on this volume or 0 if it isn't.
 * @param[in] self - the polar volume
 * @param[in] args - not used
 * @returns NULL on failure or a PyInteger where 1 indicates that it is possible to perform a transformation.
 */
static PyObject* _polarvolume_isTransformable(PolarVolume* self, PyObject* args)
{
  return PyBool_FromLong(PolarVolume_isTransformable(self->pvol));
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
 * Gets the scan that got an elevation angle that is closest to the specified angle.
 * @param[in] self - the polar volume
 * @param[in] args - the elevation angle (in radians) and an integer where 1 means only include elevations that are within the min-max elevations
 * @return a scan or NULL on failure
 */
static PyObject* _polarvolume_getScanClosestToElevation(PolarVolume* self, PyObject* args)
{
  double elevation = 0.0L;
  int inside = 0;
  PolarScan_t* scan = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "di", &elevation, &inside)) {
    return NULL;
  }

  scan = PolarVolume_getScanClosestToElevation(self->pvol, elevation, inside);
  if (scan != NULL) {
    result = RAVE_OBJECT_GETBINDING(scan);
    if (result == NULL) {
      result = (PyObject*)_polarscan_createPyObject(scan);
    } else {
      Py_INCREF(result);
    }
  }

  RAVE_OBJECT_RELEASE(scan);
  if (result != NULL) {
    return result;
  } else {
    Py_RETURN_NONE;
  }
}

/**
 * Gets the nearest value at the specified lon/lat/height.
 * @param[in] self - the polar volume
 * @param[in] args - the lon/lat as a tuple in radians), the height and an indicator if elevation must be within min-max elevation or not
 * @return a tuple of (valuetype, value)
 */
static PyObject* _polarvolume_getNearest(PolarVolume* self, PyObject* args)
{
  double lon = 0.0L, lat = 0.0L, height = 0.0L;
  double v = 0.0L;
  int insidee = 0;
  RaveValueType vtype = RaveValueType_NODATA;

  if (!PyArg_ParseTuple(args, "(dd)di", &lon,&lat,&height,&insidee)) {
    return NULL;
  }

  vtype = PolarVolume_getNearest(self->pvol, lon, lat, height, insidee, &v);

  return Py_BuildValue("(id)", vtype, v);
}

/**
 * All methods a polar volume can have
 */
static struct PyMethodDef _polarvolume_methods[] =
{
  {"addScan", (PyCFunction) _polarvolume_addScan, 1},
  {"getScan", (PyCFunction) _polarvolume_getScan, 1},
  {"getNumberOfScans", (PyCFunction) _polarvolume_getNumberOfScans, 1},
  {"isAscendingScans", (PyCFunction) _polarvolume_isAscendingScans, 1},
  {"isTransformable", (PyCFunction) _polarvolume_isTransformable, 1},
  {"sortByElevations", (PyCFunction) _polarvolume_sortByElevations, 1},
  {"getScanClosestToElevation", (PyCFunction) _polarvolume_getScanClosestToElevation, 1},
  {"getNearest", (PyCFunction) _polarvolume_getNearest, 1},
  {NULL, NULL} /* sentinel */
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
  //Py_XDECREF(obj->projection);
  RAVE_OBJECT_UNBIND(obj->cartesian, obj);
  RAVE_OBJECT_RELEASE(obj->cartesian);
  PyObject_Del(obj);
  Cartesian_dealloc++;
}

/**
 * Creates the polar scan python object from a polar scan.
 * @param[in] scan - the polar scan
 * @returns a Python Polar scan on success.
 */
static Cartesian* _cartesian_createPyObject(Cartesian_t* cartesian)
{
  Cartesian* result = NULL;
  if (cartesian == NULL) {
    RAVE_CRITICAL0("Trying to create a python cartesian without the cartesian");
    raiseException_returnNULL(PyExc_MemoryError, "Trying to create a python cartesian without the cartesian");
  }
  result = PyObject_NEW(Cartesian, &Cartesian_Type);
  if (result == NULL) {
    RAVE_CRITICAL0("Failed to allocate memory for cartesian");
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for cartesian");
  }
  result->cartesian = RAVE_OBJECT_COPY(cartesian);
  RAVE_OBJECT_BIND(result->cartesian, result);
  Cartesian_alloc++;
  return result;
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
  Cartesian_t* cartesian = NULL;

  cartesian = RAVE_OBJECT_NEW(&Cartesian_TYPE);
  if (cartesian == NULL) {
    RAVE_CRITICAL0("Failed to allocate cartesian");
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for cartesian");
  }

  result = _cartesian_createPyObject(cartesian);
  if (result == NULL) {
    RAVE_CRITICAL0("Failed to allocate py cartesian");
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for py cartesian");
  }

  RAVE_OBJECT_RELEASE(cartesian);
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

static PyObject* _cartesian_getData(Cartesian* self, PyObject* args)
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

  dims[0] = (npy_intp)ysize;
  dims[1] = (npy_intp)xsize;
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
 * sets the value at the specified position
 * @param[in] self this instance.
 * @param[in] args - tuple (x, y) and v
 * @return 0 on failure, otherwise 1
 */
static PyObject* _cartesian_setValue(Cartesian* self, PyObject* args)
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
static PyObject* _cartesian_getValue(Cartesian* self, PyObject* args)
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

static PyObject* _cartesian_isTransformable(Cartesian* self, PyObject* args)
{
  return PyBool_FromLong(Cartesian_isTransformable(self->cartesian));
}

/**
 * All methods a cartesian product can have
 */
static struct PyMethodDef _cartesian_methods[] =
{
  {"setData", (PyCFunction) _cartesian_setData, 1},
  {"getData", (PyCFunction) _cartesian_getData, 1},
  {"getLocationX", (PyCFunction) _cartesian_getLocationX, 1},
  {"getLocationY", (PyCFunction) _cartesian_getLocationY, 1},
  {"setValue", (PyCFunction) _cartesian_setValue, 1},
  {"getValue", (PyCFunction) _cartesian_getValue, 1},
  {"isTransformable", (PyCFunction) _cartesian_isTransformable, 1},
  {NULL, NULL } /* sentinel */
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
  } else if (strcmp("projection", name) == 0) {
    Projection_t* projection = Cartesian_getProjection(self->cartesian);
    if (projection != NULL) {
      Projection* result = RAVE_OBJECT_GETBINDING(projection);
      if (result == NULL) {
        result = _projection_createPyObject(projection);
      } else {
        Py_INCREF(result);
      }
      RAVE_OBJECT_RELEASE(projection);
      return (PyObject*)result;
    } else {
      Py_RETURN_NONE;
    }
#ifdef KALLE
    if (self->projection != NULL) {
      Py_INCREF(self->projection);
      return (PyObject*)self->projection;
    } else {
      Py_RETURN_NONE;
    }
#endif
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
  } else if (strcmp("projection", name) == 0) {
    if (Projection_Check(val)) {
      Cartesian_setProjection(self->cartesian, ((Projection*)val)->projection);
    } else if (val == Py_None) {
      Cartesian_setProjection(self->cartesian, NULL);
    }
#ifdef KALLE
    if (Projection_Check(val)) {
      Py_XDECREF(self->projection);
      self->projection = ((Projection*)val);
      Py_INCREF(self->projection);
      Cartesian_setProjection(self->cartesian, ((Projection*)val)->projection);
    } else if (val == Py_None) {
      Py_XDECREF(self->projection);
      Cartesian_setProjection(self->cartesian, NULL);
      self->projection = NULL;
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "projection must be of type ProjectionCore");
    }
#endif
  }

  result = 0;
done:
  return result;
}

/*@} End of Cartesian products */

/// --------------------------------------------------------------------
/// Transform
/// --------------------------------------------------------------------
/*@{ Transform */

/**
 * Deallocates the transformator
 * @param[in] obj the object to deallocate.
 */
static void _transform_dealloc(Transform* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  RAVE_OBJECT_RELEASE(obj->transform);
  PyObject_Del(obj);
  Transform_dealloc++;
}

/**
 * Creates a new instance of the transformator.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _transform_new(PyObject* self, PyObject* args)
{
  Transform* result = NULL;
  result = PyObject_NEW(Transform, &Transform_Type);
  if (result == NULL) {
    return NULL;
  }
  result->transform = RAVE_OBJECT_NEW(&Transform_TYPE);
  if (result->transform == NULL) {
    RAVE_CRITICAL0("Could not allocate transform");
    PyObject_Del(result);
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate transform");
  }
  Transform_alloc++;
  return (PyObject*)result;
}

/**
 * Creates a ppi from a polar volume
 * @param[in] self the transformer
 * @param[in] args arguments for generating the ppi (polarvolume, cartesian)
 * @return Py_None on success, otherwise NULL
 */
static PyObject* _transform_ppi(Transform* self, PyObject* args)
{
  Cartesian* cartesian = NULL;
  PyObject* pycartesian = NULL;
  PolarScan* scan = NULL;
  PyObject* pyscan = NULL;

  if(!PyArg_ParseTuple(args, "OO", &pyscan, &pycartesian)) {
    return NULL;
  }

  if (!PolarScan_Check(pyscan)) {
    raiseException_returnNULL(PyExc_TypeError, "First argument should be a polar scan")
  }

  if (!Cartesian_Check(pycartesian)) {
    raiseException_returnNULL(PyExc_TypeError, "Second argument should be a cartesian product");
  }

  scan = (PolarScan*)pyscan;
  cartesian = (Cartesian*)pycartesian;

  if (!Transform_ppi(self->transform, scan->scan, cartesian->cartesian)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to transform volume into a ppi");
  }

  Py_RETURN_NONE;
}

/**
 * Creates a cappi from a polar volume
 * @param[in] self the transformer
 * @param[in] args arguments for generating the cappi (polarvolume, cartesian, height in meters)
 * @return Py_None on success, otherwise NULL
 */
static PyObject* _transform_cappi(Transform* self, PyObject* args)
{
  Cartesian* cartesian = NULL;
  PyObject* pycartesian = NULL;
  PolarVolume* pvol = NULL;
  PyObject* pypvol = NULL;
  double height = 0.0L;

  if(!PyArg_ParseTuple(args, "OOd", &pypvol, &pycartesian, &height)) {
    return NULL;
  }

  if (!PolarVolume_Check(pypvol)) {
    raiseException_returnNULL(PyExc_TypeError, "First argument should be a polar volume")
  }

  if (!Cartesian_Check(pycartesian)) {
    raiseException_returnNULL(PyExc_TypeError, "Second argument should be a cartesian product");
  }

  pvol = (PolarVolume*)pypvol;
  cartesian = (Cartesian*)pycartesian;

  if (!Transform_cappi(self->transform, pvol->pvol, cartesian->cartesian, height)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to transform volume into a cappi");
  }

  Py_RETURN_NONE;
}

/**
 * Creates a pseudo-cappi from a polar volume
 * @param[in] self the transformer
 * @param[in] args arguments for generating the pseudo-cappi (polarvolume, cartesian)
 * @return Py_None on success, otherwise NULL
 */
static PyObject* _transform_pcappi(Transform* self, PyObject* args)
{
  Cartesian* cartesian = NULL;
  PyObject* pycartesian = NULL;
  PolarVolume* pvol = NULL;
  PyObject* pypvol = NULL;
  double height = 0.0L;

  if(!PyArg_ParseTuple(args, "OOd", &pypvol, &pycartesian,&height)) {
    return NULL;
  }

  if (!PolarVolume_Check(pypvol)) {
    raiseException_returnNULL(PyExc_TypeError, "First argument should be a polar volume")
  }

  if (!Cartesian_Check(pycartesian)) {
    raiseException_returnNULL(PyExc_TypeError, "Second argument should be a cartesian product");
  }

  pvol = (PolarVolume*)pypvol;
  cartesian = (Cartesian*)pycartesian;

  if (!Transform_pcappi(self->transform, pvol->pvol, cartesian->cartesian, height)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to transform volume into a cappi");
  }

  Py_RETURN_NONE;
}

/**
 * All methods a transformator can have
 */
static struct PyMethodDef _transform_methods[] =
{
  {"ppi", (PyCFunction) _transform_ppi, 1},
  {"cappi", (PyCFunction) _transform_cappi, 1},
  {"pcappi", (PyCFunction) _transform_pcappi, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the transformator
 * @param[in] self - the cartesian product
 */
static PyObject* _transform_getattr(Transform* self, char* name)
{
  PyObject* res = NULL;

  if (strcmp("method", name) == 0) {
    return PyInt_FromLong(Transform_getMethod(self->transform));
  }

  res = Py_FindMethod(_transform_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the transformator
 */
static int _transform_setattr(Transform* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("method", name)==0) {
    if (PyInt_Check(val)) {
      if (!Transform_setMethod(self->transform, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "method must be in valid range");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"method must be a valid RaveTransformMethod");
    }
  }

  result = 0;
done:
  return result;
}

/*@} End of Transform */

/// --------------------------------------------------------------------
/// Projection
/// --------------------------------------------------------------------
/*@{ Projection */

/**
 * Deallocates the projection
 * @param[in] obj the object to deallocate.
 */
static void _projection_dealloc(Projection* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  RAVE_OBJECT_UNBIND(obj->projection, obj);
  RAVE_OBJECT_RELEASE(obj->projection);
  PyObject_Del(obj);
  Projection_dealloc++;
}

static Projection* _projection_createPyObject(Projection_t* projection)
{
  Projection* result = NULL;
  if (projection == NULL) {
    RAVE_CRITICAL0("Trying to create a python projection without the projection");
    raiseException_returnNULL(PyExc_MemoryError, "Trying to create a python projection without the projection");
  }
  result = PyObject_NEW(Projection, &Projection_Type);
  if (result == NULL) {
    RAVE_CRITICAL0("Failed to allocate memory for projection");
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for projection");
  }
  result->projection = RAVE_OBJECT_COPY(projection);
  RAVE_OBJECT_BIND(result->projection, result);
  Projection_alloc++;
  return result;
}

/**
 * Creates a new projection instance.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (id, description, definition).
 * @return the object on success, otherwise NULL
 */
static PyObject* _projection_new(PyObject* self, PyObject* args)
{
  Projection_t* projection = NULL;
  Projection* result = NULL;
  char* id = NULL;
  char* description = NULL;
  char* definition = NULL;

  if (!PyArg_ParseTuple(args, "sss", &id, &description, &definition)) {
    return NULL;
  }

  projection = RAVE_OBJECT_NEW(&Projection_TYPE);
  if (projection == NULL) {
    RAVE_CRITICAL0("Failed to allocate projection");
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for projection");
  }

  if(!Projection_init(projection, id, description, definition)) {
    RAVE_ERROR0("Could not initialize projection");
    RAVE_OBJECT_RELEASE(projection);
    raiseException_returnNULL(PyExc_ValueError, "Failed to initialize projection");
  }

  result = _projection_createPyObject(projection);
  if (result == NULL) {
    RAVE_CRITICAL0("Failed to allocate py projection");
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for py projection");
  }

  RAVE_OBJECT_RELEASE(projection);
  return (PyObject*)result;
}

/**
 * Projects a coordinate pair into the new projection coordinate system
 * @param[in] self the source projection
 * @param[in] args arguments for projecting)
 * @return Py_None on success, otherwise NULL
 */
static PyObject* _projection_transform(Projection* self, PyObject* args)
{
  Projection* tgtproj = NULL;
  PyObject* pytgtproj = NULL;
  PyObject* pycoord = NULL;
  PyObject* result = NULL;

  double x=0.0,y=0.0,z=0.0;
  int coordlen = 0;

  if(!PyArg_ParseTuple(args, "OO", &pytgtproj,&pycoord)) {
    return NULL;
  }

  if (!Projection_Check(pytgtproj)) {
    raiseException_returnNULL(PyExc_TypeError, "First argument should be the target projection")
  }

  if (!PyTuple_Check(pycoord)) {
    raiseException_returnNULL(PyExc_TypeError, "Second argument should be a tuple with either 2 or 3 floats");
  }
  coordlen = PyTuple_Size(pycoord);
  if (coordlen == 2) {
    if(!PyArg_ParseTuple(pycoord, "dd", &x,&y)) {
      return NULL;
    }
  } else if (coordlen == 3) {
    if(!PyArg_ParseTuple(pycoord, "ddd", &x,&y,&z)) {
      return NULL;
    }
  } else {
    raiseException_returnNULL(PyExc_TypeError, "Second argument should be a tuple with either 2 or 3 floats");
  }

  tgtproj = (Projection*)pytgtproj;

  if (coordlen == 2) {
    if (!Projection_transform(self->projection, tgtproj->projection, &x, &y, NULL)) {
      raiseException_returnNULL(PyExc_IOError, "Failed to transform to target projection");
    }
    result = Py_BuildValue("(dd)", x, y);
  } else {
    if (!Projection_transform(self->projection, tgtproj->projection, &x, &y, &z)) {
      raiseException_returnNULL(PyExc_IOError, "Failed to transform to target projection");
    }
    result = Py_BuildValue("(ddd)", x, y, z);
  }

  return result;
}

/**
 * Translates surface coordinate into lon/lat.
 * @param[in] self - the projection
 * @param[in] args - the (x,y) coordinate as a tuple of two doubles.
 * @returns a tuple of two doubles representing the lon/lat coordinate in radians or NULL on failure
 */

static PyObject* _projection_inv(Projection* self, PyObject* args)
{
  double lon=0.0L, lat=0.0L;
  double x=0.0L, y=0.0L;

  if (!PyArg_ParseTuple(args, "(dd)", &x, &y)) {
    return NULL;
  }

  if (!Projection_inv(self->projection, x, y, &lon, &lat)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to project surface coordinates into lon/lat");
  }

  return Py_BuildValue("(dd)", lon, lat);
}

/**
 * Translates lon/lat into surface coordinates.
 * @param[in] self - the projection
 * @param[in] args - the (lon,lat) coordinate as a tuple of two doubles.
 * @returns a tuple of two doubles representing the xy coordinate or NULL on failure
 */

static PyObject* _projection_fwd(Projection* self, PyObject* args)
{
  double lon=0.0L, lat=0.0L;
  double x=0.0L, y=0.0L;

  if (!PyArg_ParseTuple(args, "(dd)", &lon, &lat)) {
    return NULL;
  }

  if (!Projection_fwd(self->projection, lon, lat, &x, &y)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to project surface coordinates into xy");
  }

  return Py_BuildValue("(dd)", x, y);
}

/**
 * All methods a projection can have
 */
static struct PyMethodDef _projection_methods[] =
{
  { "transform", (PyCFunction) _projection_transform, 1},
  { "inv", (PyCFunction) _projection_inv, 1},
  { "fwd", (PyCFunction) _projection_fwd, 1},
  { NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the transformator
 * @param[in] self - the cartesian product
 */
static PyObject* _projection_getattr(Projection* self, char* name)
{
  PyObject* res = NULL;

  if (strcmp("id", name) == 0) {
    return PyString_FromString(Projection_getID(self->projection));
  } else if (strcmp("description", name) == 0) {
    return PyString_FromString(Projection_getDescription(self->projection));
  } else if (strcmp("definition", name) == 0) {
    return PyString_FromString(Projection_getDefinition(self->projection));
  }

  res = Py_FindMethod(_projection_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Sets the specified attribute in the projection
 */
static int _projection_setattr(Projection* self, char* name, PyObject* val)
{
  return -1;
}

/*@} End of Projection */

/// --------------------------------------------------------------------
/// RaveIO
/// --------------------------------------------------------------------
/*@{ RaveIO */

/**
 * Deallocates the RaveIO
 * @param[in] obj the object to deallocate.
 */
static void _raveio_dealloc(RaveIO* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  RAVE_OBJECT_RELEASE(obj->raveio);
  PyObject_Del(obj);
  RaveIO_dealloc++;
}

/**
 * Creates a new RaveIO instance.
 * @param[in] self this instance.
 * @param[in] args arguments for creation.
 * @return the object on success, otherwise NULL
 */
static PyObject* _raveio_new(PyObject* self, PyObject* args)
{
  RaveIO* result = NULL;

  result = PyObject_NEW(RaveIO, &RaveIO_Type);
  if (result == NULL) {
    return NULL;
  }
  result->raveio = RAVE_OBJECT_NEW(&RaveIO_TYPE);
  if (result->raveio == NULL) {
    RAVE_ERROR0("Could not create RaveIO");
    PyObject_Del(result);
    raiseException_returnNULL(PyExc_ValueError, "Failed to create RaveIO");
  }
  RaveIO_alloc++;
  return (PyObject*)result;
}

static PyObject* _raveio_open(PyObject* self, PyObject* args)
{
  RaveIO_t* raveio = NULL;
  RaveIO* result = NULL;

  char* filename = NULL;
  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }
  raveio = RAVE_OBJECT_NEW(&RaveIO_TYPE);
  if (raveio == NULL) {
    raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate RaveIO instance");
  }
  if (!RaveIO_open(raveio, filename)) {
    raiseException_gotoTag(done, PyExc_IOError, "Failed to open file");
  }

  result = PyObject_NEW(RaveIO, &RaveIO_Type);
  if (result != NULL) {
    result->raveio = RAVE_OBJECT_COPY(raveio);
    RaveIO_alloc++;
  }

done:
  RAVE_OBJECT_RELEASE(raveio);
  return (PyObject*)result;
}

/**
 * Returns true or false depending on if a HDF5 nodelist is loaded
 * or not.
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns Py_None on success, otherwise NULL
 */
static PyObject* _raveio_isOpen(RaveIO* self, PyObject* args)
{
  return PyBool_FromLong(RaveIO_isOpen(self->raveio));
}

/**
 * Closes the currently open nodelist.
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns Py_None on success, otherwise NULL
 */
static PyObject* _raveio_close(RaveIO* self, PyObject* args)
{
  RaveIO_close(self->raveio);
  Py_RETURN_NONE;
}

/**
 * Closes the currently open nodelist.
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns Py_None on success, otherwise NULL
 */
static PyObject* _raveio_openFile(RaveIO* self, PyObject* args)
{
  char* filename = NULL;
  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }
  if (!RaveIO_open(self->raveio, filename)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to open file");
  }
  Py_RETURN_NONE;
}

/**
 * Returns the currently opened files object type (/what/object).
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns the object type on success, otherwise -1
 */
static PyObject* _raveio_getObjectType(RaveIO* self, PyObject* args)
{
  return PyInt_FromLong(RaveIO_getObjectType(self->raveio));
}

/**
 * Returns if the currently opened file is supported or not.
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns True if the file structure is supported, otherwise False
 */
static PyObject* _raveio_isSupported(RaveIO* self, PyObject* args)
{
  return PyBool_FromLong(RaveIO_isSupported(self->raveio));
}

static PyObject* _raveio_load(RaveIO* self, PyObject* args)
{
  PyObject* result = NULL;
  switch (RaveIO_getObjectType(self->raveio)) {
  case RaveIO_ObjectType_PVOL: {
    PolarVolume_t* pvol = RaveIO_loadVolume(self->raveio);
    if (pvol != NULL) {
      result = (PyObject*)_polarvolume_createPyObject(pvol);
    }
    RAVE_OBJECT_RELEASE(pvol);
    break;
  }
  default:
    RAVE_DEBUG0("Load: Unsupported object type");
    break;
  }
  if (result == NULL) {
    fprintf(stderr, "Returning NULL\n");
  }
  return result;
}

/**
 * Returns the current files version.
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns The current files ODIM version
 */
static PyObject* _raveio_getOdimVersion(RaveIO* self, PyObject* args)
{
  return PyInt_FromLong(RaveIO_getOdimVersion(self->raveio));
}

/**
 * All methods a RaveIO can have
 */
static struct PyMethodDef _raveio_methods[] =
{
  {"isOpen", (PyCFunction) _raveio_isOpen, 1},
  {"close", (PyCFunction) _raveio_close, 1},
  {"open", (PyCFunction) _raveio_openFile, 1},
  {"getObjectType", (PyCFunction) _raveio_getObjectType, 1},
  {"isSupported", (PyCFunction) _raveio_isSupported, 1},
  {"getOdimVersion", (PyCFunction) _raveio_getOdimVersion, 1},
  {"load", (PyCFunction) _raveio_load, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the RaveIO
 * @param[in] self - the RaveIO instance
 */
static PyObject* _raveio_getattr(RaveIO* self, char* name)
{
  PyObject* res = NULL;

  res = Py_FindMethod(_raveio_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Sets the specified attribute in the raveio
 */
static int _raveio_setattr(RaveIO* self, char* name, PyObject* val)
{
  return -1;
}

/*@} End of RaveIO */


/// --------------------------------------------------------------------
/// RaveList
/// --------------------------------------------------------------------
/*@{ RaveList */

/**
 * Deallocates the RaveList
 * @param[in] obj the object to deallocate.
 */
static void _ravelist_dealloc(RaveList* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  RAVE_OBJECT_RELEASE(obj->list);
  PyObject_Del(obj);
  RaveList_dealloc++;
}

/**
 * Creates a new RaveList instance.
 * @param[in] self this instance.
 * @param[in] args arguments for creation.
 * @return the object on success, otherwise NULL
 */
static PyObject* _ravelist_new(PyObject* self, PyObject* args)
{
  RaveList* result = NULL;

  result = PyObject_NEW(RaveList, &RaveList_Type);
  if (result == NULL) {
    return NULL;
  }
  result->list = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (result->list == NULL) {
    RAVE_ERROR0("Could not create RaveList");
    PyObject_Del(result);
    raiseException_returnNULL(PyExc_ValueError, "Failed to create RaveList");
  }
  RaveList_alloc++;
  return (PyObject*)result;
}

/**
 * Adds a rave object to the list
 * @param[in] self - this instance
 * @param[in] args - a rave object
 * @returns Py_None on success, otherwise NULL
 */
static PyObject* _ravelist_add(RaveList* self, PyObject* args)
{
  PyObject* obj = NULL;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }
  if (PolarScan_Check(obj)) {
    if (!RaveObjectList_add(self->list, (RaveCoreObject*)((PolarScan*)obj)->scan)) {
      raiseException_gotoTag(error, PyExc_MemoryError, "Could not add scan to list");
    }
  } else if (PolarVolume_Check(obj)) {
    if (!RaveObjectList_add(self->list, (RaveCoreObject*)((PolarVolume*)obj)->pvol)) {
      raiseException_gotoTag(error, PyExc_MemoryError, "Could not add volume to list");
    }
  } else {
    raiseException_gotoTag(error, PyExc_AttributeError, "only supports scans, volumes");
  }

  Py_RETURN_NONE;
error:
  return NULL;
}

/**
 * Inserts a rave object at the specified index
 * @param[in] self - this instance
 * @param[in] args - a index and a rave object
 * @returns Py_None on success, otherwise NULL
 */
static PyObject* _ravelist_insert(RaveList* self, PyObject* args)
{
  PyObject* obj = NULL;
  int index = -1;
  if (!PyArg_ParseTuple(args, "iO", &index, &obj)) {
    return NULL;
  }
  if (PolarScan_Check(obj)) {
    if (!RaveObjectList_insert(self->list, index, (RaveCoreObject*)((PolarScan*)obj)->scan)) {
      raiseException_gotoTag(error, PyExc_MemoryError, "Could not add object to list");
    }
  } else if (PolarVolume_Check(obj)) {
    if (!RaveObjectList_insert(self->list, index, (RaveCoreObject*)((PolarVolume*)obj)->pvol)) {
      raiseException_gotoTag(error, PyExc_MemoryError, "Could not add object to list");
    }
  } else {
    raiseException_gotoTag(error, PyExc_AttributeError, "only supports scans, volumes");
  }

  Py_RETURN_NONE;
error:
  return NULL;
}

/**
 * Returns the size of a list object
 * @param[in] self - this instance
 * @param[in] args - None
 * @returns the size of the list
 */
static PyObject* _ravelist_size(RaveList* self, PyObject* args)
{
  return PyInt_FromLong((int)RaveObjectList_size(self->list));
}

/**
 * All methods a RaveList can have
 */
static struct PyMethodDef _ravelist_methods[] =
{
  {"add", (PyCFunction) _ravelist_add, 1},
  {"insert", (PyCFunction) _ravelist_insert, 1},
  {"size", (PyCFunction) _ravelist_size, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the RaveList
 * @param[in] self - the RaveList instance
 */
static PyObject* _ravelist_getattr(RaveList* self, char* name)
{
  PyObject* res = NULL;

  res = Py_FindMethod(_ravelist_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Sets the specified attribute in the ravelist
 */
static int _ravelist_setattr(RaveIO* self, char* name, PyObject* val)
{
  return -1;
}

/*@} End of RaveList */


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

statichere PyTypeObject Transform_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "TransformCore", /*tp_name*/
  sizeof(Transform), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_transform_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_transform_getattr, /*tp_getattr*/
  (setattrfunc)_transform_setattr, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0,
  0, /*tp_as_mapping */
  0 /*tp_hash*/
};

statichere PyTypeObject Projection_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "ProjectionCore", /*tp_name*/
  sizeof(Projection), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_projection_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_projection_getattr, /*tp_getattr*/
  (setattrfunc)_projection_setattr, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0,
  0, /*tp_as_mapping */
  0 /*tp_hash*/
};

statichere PyTypeObject RaveIO_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "RaveIOCore", /*tp_name*/
  sizeof(RaveIO), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_raveio_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_raveio_getattr, /*tp_getattr*/
  (setattrfunc)_raveio_setattr, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0,
  0, /*tp_as_mapping */
  0 /*tp_hash*/
};

statichere PyTypeObject RaveList_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "RaveListCore", /*tp_name*/
  sizeof(RaveList), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_ravelist_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_ravelist_getattr, /*tp_getattr*/
  (setattrfunc)_ravelist_setattr, /*tp_setattr*/
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
  {"transform", (PyCFunction)_transform_new, 1},
  {"projection", (PyCFunction)_projection_new, 1},
  {"io", (PyCFunction)_raveio_new, 1},
  {"open", (PyCFunction)_raveio_open, 1},
  {"list", (PyCFunction)_ravelist_new, 1},
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

static void RavePyModule_statistics(void)
{
  if (((PolarScan_alloc-PolarScan_dealloc) != 0) ||
      ((PolarVolume_alloc-PolarVolume_dealloc) != 0) ||
      ((Cartesian_alloc-Cartesian_dealloc) != 0) ||
      ((Transform_alloc-Transform_dealloc) != 0) ||
      ((Projection_alloc-Projection_dealloc) != 0) ||
      ((RaveIO_alloc-RaveIO_dealloc) != 0) ||
      ((RaveList_alloc-RaveList_dealloc) != 0)) {
    fprintf(stderr, "rave py object statistics\n");
    fprintf(stderr, "Module Name\t alloc\t dealloc\t lost\n");
    fprintf(stderr, "PolarScans \t %05d\t %05d  \t %05d\n", PolarScan_alloc, PolarScan_dealloc, (PolarScan_alloc-PolarScan_dealloc));
    fprintf(stderr, "PolarVolum \t %05d\t %05d  \t %05d\n", PolarVolume_alloc, PolarVolume_dealloc, (PolarVolume_alloc-PolarVolume_dealloc));
    fprintf(stderr, "Cartesian  \t %05d\t %05d  \t %05d\n", Cartesian_alloc, Cartesian_dealloc, (Cartesian_alloc-Cartesian_dealloc));
    fprintf(stderr, "Transform  \t %05d\t %05d  \t %05d\n", Transform_alloc, Transform_dealloc, (Transform_alloc-Transform_dealloc));
    fprintf(stderr, "Projection \t %05d\t %05d  \t %05d\n", Projection_alloc, Projection_dealloc, (Projection_alloc-Projection_dealloc));
    fprintf(stderr, "RaveIO     \t %05d\t %05d  \t %05d\n", RaveIO_alloc, RaveIO_dealloc, (RaveIO_alloc-RaveIO_dealloc));
    fprintf(stderr, "RaveList   \t %05d\t %05d  \t %05d\n", RaveList_alloc, RaveList_dealloc, (RaveList_alloc-RaveList_dealloc));
  }
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
  Transform_Type.ob_type = &PyType_Type;
  Projection_Type.ob_type = &PyType_Type;
  RaveIO_Type.ob_type = &PyType_Type;
  RaveList_Type.ob_type = &PyType_Type;

  HL_init();
  HL_disableErrorReporting();
  HL_disableHdf5ErrorReporting();
  HL_setDebugLevel(HLHDF_SILENT);

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

  if (atexit(RaveCoreObject_printStatistics) != 0) {
    fprintf(stderr, "Could not set atexit function");
  }

  if (atexit(RavePyModule_statistics) != 0) {
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

  add_long_constant(dictionary, "NEAREST", NEAREST);
  add_long_constant(dictionary, "BILINEAR", BILINEAR);
  add_long_constant(dictionary, "CUBIC", CUBIC);
  add_long_constant(dictionary, "CRESSMAN", CRESSMAN);
  add_long_constant(dictionary, "UNIFORM", UNIFORM);
  add_long_constant(dictionary, "INVERSE", INVERSE);

  add_long_constant(dictionary, "RaveValueType_UNDETECT", RaveValueType_UNDETECT);
  add_long_constant(dictionary, "RaveValueType_NODATA", RaveValueType_NODATA);
  add_long_constant(dictionary, "RaveValueType_DATA", RaveValueType_DATA);

  add_long_constant(dictionary, "RaveIO_ObjectType_UNDEFINED", RaveIO_ObjectType_UNDEFINED);
  add_long_constant(dictionary, "RaveIO_ObjectType_PVOL", RaveIO_ObjectType_PVOL);
  add_long_constant(dictionary, "RaveIO_ObjectType_CVOL", RaveIO_ObjectType_CVOL);
  add_long_constant(dictionary, "RaveIO_ObjectType_SCAN", RaveIO_ObjectType_SCAN);
  add_long_constant(dictionary, "RaveIO_ObjectType_RAY", RaveIO_ObjectType_RAY);
  add_long_constant(dictionary, "RaveIO_ObjectType_AZIM", RaveIO_ObjectType_AZIM);
  add_long_constant(dictionary, "RaveIO_ObjectType_IMAGE", RaveIO_ObjectType_IMAGE);
  add_long_constant(dictionary, "RaveIO_ObjectType_COMP", RaveIO_ObjectType_COMP);
  add_long_constant(dictionary, "RaveIO_ObjectType_XSEC", RaveIO_ObjectType_XSEC);
  add_long_constant(dictionary, "RaveIO_ObjectType_VP", RaveIO_ObjectType_VP);
  add_long_constant(dictionary, "RaveIO_ObjectType_PIC", RaveIO_ObjectType_PIC);

  add_long_constant(dictionary, "RaveIO_ODIM_Version_UNDEFINED", RaveIO_ODIM_Version_UNDEFINED);
  add_long_constant(dictionary, "RaveIO_ODIM_Version_2_0", RaveIO_ODIM_Version_2_0);

  import_array(); /*To make sure I get access to Numeric*/
}
/*@} End of Module setup */
