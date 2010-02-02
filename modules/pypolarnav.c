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
 * Python module for performing basic polar navigation.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-21
 */
#include "Python.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYPOLARNAV_MODULE        /**< to get correct part of pypolarnav.h */
#include "pypolarnav.h"

#include "pyrave_debug.h"
#include "rave_alloc.h"

/**
 * Debug this module.
 */
PYRAVE_DEBUG_MODULE("_polarnav");

/**
 * Sets a python exception and goto tag
 */
#define raiseException_gotoTag(tag, type, msg) \
{PyErr_SetString(type, msg); goto tag;}

/**
 * Sets a python exception and returns NULL
 */
#define raiseException_returnNULL(type, msg) \
{PyErr_SetString(type, msg); return NULL;}

/**
 * Error object for reporting errors to the python interpreeter
 */
static PyObject *ErrorObject;

/// --------------------------------------------------------------------
/// Polar Navigator
/// --------------------------------------------------------------------
/*@{ Polar Navigator */
/**
 * Returns the native PolarNavigator_t instance.
 * @param[in] pypolarnav - the python polar navigator instance
 * @returns the native polar navigator instance.
 */
static PolarNavigator_t*
PyPolarNavigator_GetNative(PyPolarNavigator* pypolarnav)
{
  RAVE_ASSERT((pypolarnav != NULL), "pypolarnav == NULL");
  return RAVE_OBJECT_COPY(pypolarnav->navigator);
}

/**
 * Creates a python polar volume from a native polar volume or will create an
 * initial native PolarVolume if p is NULL.
 * @param[in] p - the native polar volume (or NULL)
 * @returns the python polar volume.
 */
static PyPolarNavigator*
PyPolarNavigator_New(PolarNavigator_t* p)
{
  PyPolarNavigator* result = NULL;
  PolarNavigator_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&PolarNavigator_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for polar navigator.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for polar navigator.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyPolarNavigator, &PyPolarNavigator_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->navigator = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->navigator, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyPolarNavigator instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyPolarNavigator.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the polar navigator
 * @param[in] obj the object to deallocate.
 */
static void _pypolarnavigator_dealloc(PyPolarNavigator* obj)
{
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->navigator, obj);
  RAVE_OBJECT_RELEASE(obj->navigator);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the polar navigator.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pypolarnavigator_new(PyObject* self, PyObject* args)
{
  PyPolarNavigator* result = PyPolarNavigator_New(NULL);
  return (PyObject*)result;
}

/**
 * Returns the earth radius at the specified latitude.
 * @param[in] self - this instance
 * @param[in] args - the latitude as a double (in radians)
 * @returns a float representing the radius in meters or NULL on failure
 */
static PyObject* _pypolarnavigator_getEarthRadius(PyPolarNavigator* self, PyObject* args)
{
  double lat = 0.0L;
  double radius = 0.0L;

  if (!PyArg_ParseTuple(args, "d", &lat)) {
    return NULL;
  }

  radius = PolarNavigator_getEarthRadius(self->navigator, lat);

  return PyFloat_FromDouble(radius);
}

/**
 * Returns the earth radius at the origin of this navigator.
 * @param[in] self - this instance
 * @param[in] args - Not used
 * @returns a float representing the radius in meters or NULL on failure
 */
static PyObject* _pypolarnavigator_getEarthRadiusOrigin(PyPolarNavigator* self, PyObject* args)
{
  double radius = 0.0L;

  radius = PolarNavigator_getEarthRadiusOrigin(self->navigator);

  return PyFloat_FromDouble(radius);
}

/**
 * Calculates the distance from the lon0/lat0 to the provided lon/lat.
 * @param[in] self - self
 * @param[in] args - a tuple containing (lat, lon) in radians
 * @returns the distance in meters or NULL on failure.
 */
static PyObject* _pypolarnavigator_getDistance(PyPolarNavigator* self, PyObject* args)
{
  double lon = 0.0L, lat = 0.0L;
  if (!PyArg_ParseTuple(args, "(dd)", &lat, &lon)) {
    return NULL;
  }
  return PyFloat_FromDouble(PolarNavigator_getDistance(self->navigator, lat, lon));
}

/**
 * Calculates the distance and azimuth from this navigators origin to the specified lon/lat
 * @param[in] self - this instance
 * @param[in] args - a tuple of doubles representing (latitude, longitude) in radians
 * @returns a tuple of double (distance, azimuth in radians)
 */
static PyObject* _pypolarnavigator_llToDa(PyPolarNavigator* self, PyObject* args)
{
  double lon = 0.0L, lat = 0.0L;
  double d = 0.0L, a = 0.0L;

  if (!PyArg_ParseTuple(args, "(dd)", &lat, &lon)) {
    return NULL;
  }

  PolarNavigator_llToDa(self->navigator, lat, lon, &d, &a);

  return Py_BuildValue("(dd)", d, a);
}

/**
 * Calculates the longitude and latitude at the point that is distance meters and azimuth from the origin.
 * @param[in] self - this instance
 * @param[in] args - two doubles (distance, azimuth in radians)
 * @returns a tuple of double (latitude, longitude) in radians or NULL on failure
 */
static PyObject* _pypolarnavigator_daToLl(PyPolarNavigator* self, PyObject* args)
{
  double lon = 0.0L, lat = 0.0L;
  double d = 0.0L, a = 0.0L;

  if (!PyArg_ParseTuple(args, "dd", &d, &a)) {
    return NULL;
  }

  PolarNavigator_daToLl(self->navigator, d, a, &lat, &lon);

  return Py_BuildValue("(dd)", lat, lon);
}

/**
 * Calculates the range and elevation that is reached from the origin to the distance and height.
 * @param[in] self - this instance
 * @param[in] args - two doubles (distance, height)
 * @returns a tuple of double (range, elevation in radians) or NULL on failure
 */
static PyObject* _pypolarnavigator_dhToRe(PyPolarNavigator* self, PyObject* args)
{
  double d = 0.0L, h = 0.0L;
  double r = 0.0L, e = 0.0L;

  if (!PyArg_ParseTuple(args, "dd", &d, &h)) {
    return NULL;
  }

  PolarNavigator_dhToRe(self->navigator, d, h, &r, &e);

  return Py_BuildValue("(dd)", r, e);
}

/**
 * Calculates the range and height that is reached from the origin to the distance and elevation.
 * @param[in] self - this instance
 * @param[in] args - two doubles (distance, elevation in radians)
 * @returns a tuple of double (range, height) or NULL on failure
 */
static PyObject* _pypolarnavigator_deToRh(PyPolarNavigator* self, PyObject* args)
{
  double d = 0.0L, e = 0.0L;
  double r = 0.0L, h = 0.0L;

  if (!PyArg_ParseTuple(args, "dd", &d, &e)) {
    return NULL;
  }

  PolarNavigator_deToRh(self->navigator, d, e, &r, &h);

  return Py_BuildValue("(dd)", r, h);
}

/**
 * Calculates the distance and height from origin to the specified range and elevation.
 * @param[in] self - this instance
 * @param[in] args - two doubles (range, elevation in radians)
 * @returns a tuple of double (distance, height) or NULL on failure
 */
static PyObject* _pypolarnavigator_reToDh(PyPolarNavigator* self, PyObject* args)
{
  double d = 0.0L, e = 0.0L;
  double r = 0.0L, h = 0.0L;

  if (!PyArg_ParseTuple(args, "dd", &r, &e)) {
    return NULL;
  }

  PolarNavigator_reToDh(self->navigator, r, e, &d, &h);

  return Py_BuildValue("(dd)", d, h);
}

/**
 * All methods a polar navigator can have
 */
static struct PyMethodDef _pypolarnavigator_methods[] =
{
  {"getEarthRadius", (PyCFunction) _pypolarnavigator_getEarthRadius, 1},
  {"getEarthRadiusOrigin", (PyCFunction) _pypolarnavigator_getEarthRadiusOrigin, 1},
  {"getDistance", (PyCFunction) _pypolarnavigator_getDistance, 1},
  {"llToDa", (PyCFunction) _pypolarnavigator_llToDa, 1},
  {"daToLl", (PyCFunction) _pypolarnavigator_daToLl, 1},
  {"dhToRe", (PyCFunction) _pypolarnavigator_dhToRe, 1},
  {"deToRh", (PyCFunction) _pypolarnavigator_deToRh, 1},
  {"reToDh", (PyCFunction) _pypolarnavigator_reToDh, 1},
  { NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the polar navigator
 */
static PyObject* _pypolarnavigator_getattr(PyPolarNavigator* self, char* name)
{
  PyObject* res;
  if (strcmp("poleradius", name) == 0) {
    return PyFloat_FromDouble(PolarNavigator_getPoleRadius(self->navigator));
  } else if (strcmp("equatorradius", name) == 0) {
    return PyInt_FromLong(PolarNavigator_getEquatorRadius(self->navigator));
  } else if (strcmp("lon0", name) == 0) {
    return PyFloat_FromDouble(PolarNavigator_getLon0(self->navigator));
  } else if (strcmp("lat0", name) == 0) {
    return PyInt_FromLong(PolarNavigator_getLat0(self->navigator));
  } else if (strcmp("alt0", name) == 0) {
    return PyFloat_FromDouble(PolarNavigator_getAlt0(self->navigator));
  } else if (strcmp("dndh", name) == 0) {
    return PyInt_FromLong(PolarNavigator_getDndh(self->navigator));
  }

  res = Py_FindMethod(_pypolarnavigator_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar navigator
 */
static int _pypolarnavigator_setattr(PyPolarNavigator* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("poleradius", name)==0) {
    if (PyFloat_Check(val)) {
      PolarNavigator_setPoleRadius(self->navigator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"poleradius must be of type float");
    }
  } else if (strcmp("equatorradius", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarNavigator_setEquatorRadius(self->navigator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "equatorradius must be of type float");
    }
  } else if (strcmp("lon0", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarNavigator_setLon0(self->navigator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "lon0 must be of type float");
    }
  } else if (strcmp("lat0", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarNavigator_setLat0(self->navigator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "lat0 must be of type float");
    }
  } else if (strcmp("alt0", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarNavigator_setAlt0(self->navigator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "alt0 must be of type float");
    }
  } else if (strcmp("dndh", name) == 0) {
    if (PyFloat_Check(val)) {
      PolarNavigator_setDndh(self->navigator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "dndh must be of type float");
    }
  }

  result = 0;
done:
  return result;
}
/*@} End of Polar Scans */

/// --------------------------------------------------------------------
/// Type definitions
/// --------------------------------------------------------------------
/*@{ Type definitions */
PyTypeObject PyPolarNavigator_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "PolarNavigatorCore", /*tp_name*/
  sizeof(PyPolarNavigator), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pypolarnavigator_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pypolarnavigator_getattr, /*tp_getattr*/
  (setattrfunc)_pypolarnavigator_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_pypolarnavigator_new, 1},
  {NULL,NULL} /*Sentinel*/
};

/**
 * Initializes polar navigator.
 */
void init_polarnav(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyPolarNavigator_API[PyPolarNavigator_API_pointers];
  PyObject *c_api_object = NULL;
  PyPolarNavigator_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_polarnav", functions);
  if (module == NULL) {
    return;
  }
  PyPolarNavigator_API[PyPolarNavigator_Type_NUM] = (void*)&PyPolarNavigator_Type;
  PyPolarNavigator_API[PyPolarNavigator_GetNative_NUM] = (void *)PyPolarNavigator_GetNative;
  PyPolarNavigator_API[PyPolarNavigator_New_NUM] = (void*)PyPolarNavigator_New;

  c_api_object = PyCObject_FromVoidPtr((void *)PyPolarNavigator_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_polarnav.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _polarnav.error");
  }

  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
