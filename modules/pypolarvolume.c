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
 * Python version of the PolarVolume API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-08
 */
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "Python.h"

#define PYPOLARVOLUME_MODULE
#include "pypolarvolume.h"

#include "pypolarscan.h"
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

/// --------------------------------------------------------------------
/// Polar Volumes
/// --------------------------------------------------------------------
/*@{ Polar Volumes */
/**
 * Returns the native PolarVolume_t instance.
 * @param[in] pypolarvolume - the python polar volume instance
 * @returns the native polar volume instance.
 */
static PolarVolume_t*
PyPolarVolume_GetNative(PyPolarVolume* pypolarvolume)
{
  RAVE_ASSERT((pypolarvolume != NULL), "pypolarvolume == NULL");
  return RAVE_OBJECT_COPY(pypolarvolume->pvol);
}

/**
 * Creates a python polar volume from a native polar volume or will create an
 * initial native PolarVolume if p is NULL.
 * @param[in] p - the native polar volume (or NULL)
 * @returns the python polar volume.
 */
static PyPolarVolume*
PyPolarVolume_New(PolarVolume_t* p)
{
  PyPolarVolume* result = NULL;
  PolarVolume_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&PolarVolume_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for polar volume.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for polar volume.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
  }

  result = PyObject_NEW(PyPolarVolume, &PyPolarVolume_Type);
  if (result == NULL) {
    RAVE_CRITICAL0("Failed to create PyPolarVolume instance");
    raiseException_gotoTag(error, PyExc_MemoryError, "Failed to allocate memory for polar volume.");
  }

  result->pvol = RAVE_OBJECT_COPY(cp);
  RAVE_OBJECT_BIND(result->pvol, result);

error:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the polar volume
 * @param[in] obj the object to deallocate.
 */
static void _pypolarvolume_dealloc(PyPolarVolume* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  RAVE_OBJECT_UNBIND(obj->pvol, obj);
  RAVE_OBJECT_RELEASE(obj->pvol);
  PyObject_Del(obj);
}
#ifdef KALLE
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
#endif
/**
 * Creates a new instance of the polar volume.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pypolarvolume_new(PyObject* self, PyObject* args)
{
  PyPolarVolume* result = PyPolarVolume_New(NULL);
  return (PyObject*)result;
}

/**
 * Adds one scan to a volume.
 * @param[in] self - the polar volume
 * @param[in] args - the scan, must be of type PolarScanCore
 * @return NULL on failure
 */
static PyObject* _pypolarvolume_addScan(PyPolarVolume* self, PyObject* args)
{
  PyObject* inptr = NULL;
  PyPolarScan* polarScan = NULL;

  if (!PyArg_ParseTuple(args, "O", &inptr)) {
    return NULL;
  }

  if (!PyPolarScan_Check(inptr)) {
    raiseException_returnNULL(PyExc_TypeError,"Added object must be of type PolarScanCore");
  }

  polarScan = (PyPolarScan*)inptr;

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
static PyObject* _pypolarvolume_getScan(PyPolarVolume* self, PyObject* args)
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
      result = (PyObject*)PyPolarScan_New(scan);
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
static PyObject* _pypolarvolume_getNumberOfScans(PyPolarVolume* self, PyObject* args)
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
static PyObject* _pypolarvolume_isAscendingScans(PyPolarVolume* self, PyObject* args)
{
  return PyBool_FromLong(PolarVolume_isAscendingScans(self->pvol));
}

/**
 * Returns 1 if is is possible to perform transform operations on this volume or 0 if it isn't.
 * @param[in] self - the polar volume
 * @param[in] args - not used
 * @returns NULL on failure or a PyInteger where 1 indicates that it is possible to perform a transformation.
 */
static PyObject* _pypolarvolume_isTransformable(PyPolarVolume* self, PyObject* args)
{
  return PyBool_FromLong(PolarVolume_isTransformable(self->pvol));
}

/**
 * Sorts the scans by comparing elevations in either ascending or descending order
 * @param[in] self - the polar volume
 * @param[in] args - 1 if soring should be done in ascending order, otherwise descending
 * @return NULL on failure or a Py_None
 */
static PyObject* _pypolarvolume_sortByElevations(PyPolarVolume* self, PyObject* args)
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
static PyObject* _pypolarvolume_getScanClosestToElevation(PyPolarVolume* self, PyObject* args)
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
      result = (PyObject*)PyPolarScan_New(scan);
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
static PyObject* _pypolarvolume_getNearest(PyPolarVolume* self, PyObject* args)
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
static struct PyMethodDef _pypolarvolume_methods[] =
{
  {"addScan", (PyCFunction) _pypolarvolume_addScan, 1},
  {"getScan", (PyCFunction) _pypolarvolume_getScan, 1},
  {"getNumberOfScans", (PyCFunction) _pypolarvolume_getNumberOfScans, 1},
  {"isAscendingScans", (PyCFunction) _pypolarvolume_isAscendingScans, 1},
  {"isTransformable", (PyCFunction) _pypolarvolume_isTransformable, 1},
  {"sortByElevations", (PyCFunction) _pypolarvolume_sortByElevations, 1},
  {"getScanClosestToElevation", (PyCFunction) _pypolarvolume_getScanClosestToElevation, 1},
  {"getNearest", (PyCFunction) _pypolarvolume_getNearest, 1},
  {NULL, NULL} /* sentinel */
};

/**
 * Returns the specified attribute in the polar volume
 */
static PyObject* _pypolarvolume_getattr(PyPolarVolume* self, char* name)
{
  PyObject* res = NULL;
  if (strcmp("longitude", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getLongitude(self->pvol));
  } else if (strcmp("latitude", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getLatitude(self->pvol));
  } else if (strcmp("height", name) == 0) {
    return PyFloat_FromDouble(PolarVolume_getHeight(self->pvol));
  }

  res = Py_FindMethod(_pypolarvolume_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pypolarvolume_setattr(PyPolarVolume* self, char* name, PyObject* val)
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
/// Type definitions
/// --------------------------------------------------------------------
/*@{ Type definitions */
PyTypeObject PyPolarVolume_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "PolarVolumeCore", /*tp_name*/
  sizeof(PyPolarVolume), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pypolarvolume_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pypolarvolume_getattr, /*tp_getattr*/
  (setattrfunc)_pypolarvolume_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_pypolarvolume_new, 1},
  {NULL,NULL} /*Sentinel*/
};

/**
 * Initializes polar volume.
 */
void init_polarvolume(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyPolarVolume_API[PyPolarVolume_API_pointers];
  PyObject *c_api_object = NULL;
  PyPolarVolume_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_polarvolume", functions);
  if (module == NULL) {
    return;
  }
  PyPolarVolume_API[PyPolarVolume_Type_NUM] = (void*)&PyPolarVolume_Type;
  PyPolarVolume_API[PyPolarVolume_GetNative_NUM] = (void *)PyPolarVolume_GetNative;
  PyPolarVolume_API[PyPolarVolume_New_NUM] = (void*)PyPolarVolume_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyPolarVolume_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_polarvolume.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _polarvolume.error");
  }

  import_pypolarscan();
  Rave_initializeDebugger();
}
/*@} End of Module setup */
