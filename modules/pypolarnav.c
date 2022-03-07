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
#include "pyravecompat.h"
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
  {"poleradius", NULL, METH_VARARGS},
  {"equatorradius", NULL, METH_VARARGS},
  {"lon0", NULL, METH_VARARGS},
  {"lat0", NULL, METH_VARARGS},
  {"alt0", NULL, METH_VARARGS},
  {"dndh", NULL, METH_VARARGS},
  {"getEarthRadius", (PyCFunction) _pypolarnavigator_getEarthRadius, 1,
    "getEarthRadius(lat) -> earth radius in meters as float\n\n"
    "Returns the earth radius at the specified latitude in meters\n\n"
    "lat - latitude where earth radius should be calculated"
  },
  {"getEarthRadiusOrigin", (PyCFunction) _pypolarnavigator_getEarthRadiusOrigin, 1,
    "getEarthRadiusOrigin() -> earth radius in meters as float at the origin of this navigator (lat0)\n\n"
    "Returns the earth radius at the origin of this navigator in meters\n\n"
  },
  {"getDistance", (PyCFunction) _pypolarnavigator_getDistance, 1,
    "getDistance((lat,lon)) -> distance between lat0/lon0 and lat/lon as float in meters\n\n"
    "Calculates the distance from the lon0/lat0 to the provided lon/lat in meters.\n\n"
    "lat,lon - a tuple latitude,longitude in radians to which distance should be calculated"
  },
  {"llToDa", (PyCFunction) _pypolarnavigator_llToDa, 1,
    "llToDa((lat,lon)) -> a tuple (distance, azimuth)\n\n"
    "Calculates the distance and azimuth from this navigators origin to the specified lon/lat.\n\n"
    "lat,lon - a tuple latitude,longitude in radians"
  },
  {"daToLl", (PyCFunction) _pypolarnavigator_daToLl, 1,
    "llToDa((distance,azimuth) -> a tuple (lat,lon) in radians\n\n"
    "Calculates the longitude and latitude at the point that is distance meters and azimuth from the origin.\n\n"
    "distance,azimuth - a tuple  where first value is distance in meters and second is the azimuth in radians"
  },
  {"dhToRe", (PyCFunction) _pypolarnavigator_dhToRe, 1,
    "dhToRe((distance,height) -> a tuple (range in meters following the ray,elevation in radians)\n\n"
    "Calculates the range and elevation that is reached from the origin to the distance and height. This calculation takes into account the refraction coefficient dndh.\n\n"
    "distance,height - a tuple  where first value is distance in meters following the surface and second is the height above sea level."
  },
  {"deToRh", (PyCFunction) _pypolarnavigator_deToRh, 1,
      "deToRh((distance,elevation) -> a tuple (range in meters following the ray, height above sea level)\n\n"
      "Calculates the range and height above sea level that is reached from the origin to the distance and the elevation angle. This calculation takes into account the refraction coefficient dndh.\n\n"
      "distance,elevation - a tuple  where first value is distance in meters following the surface and second is the elevation angle."
  },
  {"reToDh", (PyCFunction) _pypolarnavigator_reToDh, 1,
    "reToDh((range,elevation) -> a tuple (distance along surface, elevation angle in radians)\n\n"
    "Calculates the distance and height from origin to the specified range and elevation.\n\n"
    "range,elevation - a tuple  where first value is range in meters following the ray and second is the elevation angle."
  },
  { NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the polar navigator
 */

static PyObject* _pypolarnavigator_getattro(PyPolarNavigator* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "poleradius") == 0) {
    return PyFloat_FromDouble(PolarNavigator_getPoleRadius(self->navigator));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "equatorradius") == 0) {
    return PyFloat_FromDouble(PolarNavigator_getEquatorRadius(self->navigator));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "lon0") == 0) {
    return PyFloat_FromDouble(PolarNavigator_getLon0(self->navigator));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "lat0") == 0) {
    return PyFloat_FromDouble(PolarNavigator_getLat0(self->navigator));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "alt0") == 0) {
    return PyFloat_FromDouble(PolarNavigator_getAlt0(self->navigator));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "dndh") == 0) {
    return PyFloat_FromDouble(PolarNavigator_getDndh(self->navigator));
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the attribute value
 */
static int _pypolarnavigator_setattro(PyPolarNavigator *self, PyObject *name, PyObject *val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "poleradius") == 0) {
    if (PyFloat_Check(val)) {
      PolarNavigator_setPoleRadius(self->navigator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "poleradius must be of type float");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "equatorradius") == 0) {
    if (PyFloat_Check(val)) {
      PolarNavigator_setEquatorRadius(self->navigator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "equatorradius must be of type float");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "lon0") == 0) {
    if (PyFloat_Check(val)) {
      PolarNavigator_setLon0(self->navigator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "lon0 must be of type float");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "lat0") == 0) {
    if (PyFloat_Check(val)) {
      PolarNavigator_setLat0(self->navigator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "lat0 must be of type float");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "alt0") == 0) {
    if (PyFloat_Check(val)) {
      PolarNavigator_setAlt0(self->navigator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "alt0 must be of type float");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "dndh") == 0) {
    if (PyFloat_Check(val)) {
      PolarNavigator_setDndh(self->navigator, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "dndh must be of type float");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError,
        PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }
  result = 0;
done:
  return result;

}
/*@} End of Polar Scans */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pypolarnavigator_module_doc,
    "Routines for navigating in polar space taking into account, earths sphere, radar rays and longitude/latitude.\n"
    "Another aspect is that it also takes into account the refraction, i.e. dndh.\n\n"
    "One very commonly used approach is to calculate where on earth you are from a specific lon/lat position, following the radar ray x meters outward and determine height and position\n\n"
    "The member attributes configurable are:\n"
    " poleradius    - the radius to the poles (in meters).\n"
    " equatorradius - the radius at the equator (in meters).\n"
    " lon0          - the origin longitude (in radians)\n"
    " lat0          - the origin latitude (in radians)\n"
    " alt0          - the origin altitude (in meters)\n"
    " dndh          - dndh (deflection)\n"
    "Usage:\n"
    "import _polarnav, math\n"
    "nav = _polarnav.new()\n"
    "nav.lon0 = 14.0*math.pi/180.0\n"
    "nav.lat0 = 60.0*math.pi/180.0\n"
    "nav.alt0 = 110.0\n"
    "(lat,lon) = nav.daToLl(30000.0, 90.0*math.pi/180.0)\n"
    "print(\"Longitude/latitude 30km east from origin is %f / %f\"%(lat*180.0/math.pi, lon*180.0/math.pi))\n"
    "# Gives: Lon/lat 30km east of origin is 60.000000/14.540345"
);
/*@} End of Documentation about the type */


/// --------------------------------------------------------------------
/// Type definitions
/// --------------------------------------------------------------------
/*@{ Type definitions */
PyTypeObject PyPolarNavigator_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "PolarNavigatorCore", /*tp_name*/
  sizeof(PyPolarNavigator), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pypolarnavigator_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)0, /*tp_getattr*/
  (setattrfunc)0, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0,
  0, /*tp_as_mapping */
  0,                            /*tp_hash*/
  (ternaryfunc)0,               /*tp_call*/
  (reprfunc)0,                  /*tp_str*/
  (getattrofunc)_pypolarnavigator_getattro, /*tp_getattro*/
  (setattrofunc)_pypolarnavigator_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pypolarnavigator_module_doc, /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pypolarnavigator_methods,    /*tp_methods*/
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

/// --------------------------------------------------------------------
/// Module setup
/// --------------------------------------------------------------------
/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pypolarnavigator_new, 1,
    "new() -> new instance of the GraCore object\n\n"
    "Creates a new instance of the GraCore object"
  },
  {NULL,NULL} /*Sentinel*/
};

/**
 * Initializes polar navigator.
 */
MOD_INIT(_polarnav)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyPolarNavigator_API[PyPolarNavigator_API_pointers];
  PyObject *c_api_object = NULL;
  MOD_INIT_SETUP_TYPE(PyPolarNavigator_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyPolarNavigator_Type);

  MOD_INIT_DEF(module, "_polarnav", _pypolarnavigator_module_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyPolarNavigator_API[PyPolarNavigator_Type_NUM] = (void*)&PyPolarNavigator_Type;
  PyPolarNavigator_API[PyPolarNavigator_GetNative_NUM] = (void *)PyPolarNavigator_GetNative;
  PyPolarNavigator_API[PyPolarNavigator_New_NUM] = (void*)PyPolarNavigator_New;

  c_api_object = PyCapsule_New(PyPolarNavigator_API, PyPolarNavigator_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_polarnav.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _polarnav.error");
  }

  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
