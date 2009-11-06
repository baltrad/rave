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
#include <Python.h>
#include <arrayobject.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "polarnav.h"
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

/**
 * A polar navigator
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  PolarNavigator_t* navigator;
} PolarNavigator;

/**
 * PolarNavigator represents one polar navigator
 */
staticforward PyTypeObject PolarNavigator_Type;

/**
 * Checks if the object is a PolarNavigator type
 */
#define PolarNavigator_Check(op) ((op)->ob_type == &PolarNavigator_Type)

/// --------------------------------------------------------------------
/// Polar Navigator
/// --------------------------------------------------------------------
/*@{ Polar Navigator */

/**
 * Deallocates the polar navigator
 * @param[in] obj the object to deallocate.
 */
static void _polarnavigator_dealloc(PolarNavigator* obj)
{
  if (obj == NULL) {
    return;
  }
  PolarNavigator_release(obj->navigator);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the polar navigator.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _polarnavigator_new(PyObject* self, PyObject* args)
{
  PolarNavigator* result = NULL;
  result = PyObject_NEW(PolarNavigator, &PolarNavigator_Type);
  if (result == NULL) {
    return NULL;
  }
  result->navigator = PolarNavigator_new();
  if (result->navigator == NULL) {
    RAVE_CRITICAL0("Could not allocate polar navigator");
    PyObject_Del(result);
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate polar navigator");
  }
  return (PyObject*)result;
}

/**
 * Returns the earth radius at the specified latitude.
 * @param[in] self - this instance
 * @param[in] args - the latitude as a double (in radians)
 * @returns a float representing the radius in meters or NULL on failure
 */
static PyObject* _polarnavigator_getEarthRadius(PolarNavigator* self, PyObject* args)
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
static PyObject* _polarnavigator_getEarthRadiusOrigin(PolarNavigator* self, PyObject* args)
{
  double radius = 0.0L;

  radius = PolarNavigator_getEarthRadiusOrigin(self->navigator);

  return PyFloat_FromDouble(radius);
}

/**
 * Calculates the distance and azimuth from this navigators origin to the specified lon/lat
 * @param[in] self - this instance
 * @param[in] args - a tuple of doubles representing (latitude, longitude) in radians
 * @returns a tuple of double (distance, azimuth in radians)
 */
static PyObject* _polarnavigator_llToDa(PolarNavigator* self, PyObject* args)
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
static PyObject* _polarnavigator_daToLl(PolarNavigator* self, PyObject* args)
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
static PyObject* _polarnavigator_dhToRe(PolarNavigator* self, PyObject* args)
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
static PyObject* _polarnavigator_deToRh(PolarNavigator* self, PyObject* args)
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
static PyObject* _polarnavigator_reToDh(PolarNavigator* self, PyObject* args)
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
static struct PyMethodDef _polarnavigator_methods[] =
{
  {"getEarthRadius", (PyCFunction) _polarnavigator_getEarthRadius, 1},
  {"getEarthRadiusOrigin", (PyCFunction) _polarnavigator_getEarthRadiusOrigin, 1},
  {"llToDa", (PyCFunction) _polarnavigator_llToDa, 1},
  {"daToLl", (PyCFunction) _polarnavigator_daToLl, 1},
  {"dhToRe", (PyCFunction) _polarnavigator_dhToRe, 1},
  {"deToRh", (PyCFunction) _polarnavigator_deToRh, 1},
  {"reToDh", (PyCFunction) _polarnavigator_reToDh, 1},
  { NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the polar navigator
 */
static PyObject* _polarnavigator_getattr(PolarNavigator* self, char* name)
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

  res = Py_FindMethod(_polarnavigator_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar navigator
 */
static int _polarnavigator_setattr(PolarNavigator* self, char* name, PyObject* val)
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
statichere PyTypeObject PolarNavigator_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "PolarNavigatorCore", /*tp_name*/
  sizeof(PolarNavigator), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_polarnavigator_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_polarnavigator_getattr, /*tp_getattr*/
  (setattrfunc)_polarnavigator_setattr, /*tp_setattr*/
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
  {"new", (PyCFunction)_polarnavigator_new, 1},
  {NULL,NULL} /*Sentinel*/
};

/**
 * Initializes polar navigator.
 */
void init_polarnav(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  PolarNavigator_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_polarnav", functions);
  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_polarnav.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _polarnav.error");
  }

  Rave_initializeDebugger();

  import_array(); /*To make sure I get access to Numeric*/
}
/*@} End of Module setup */
