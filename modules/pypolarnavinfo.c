/* --------------------------------------------------------------------
Copyright (C) 2024 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the PolarNavigationInfo API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-05-21
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"
#include "rave_types.h"

#define PYPOLARNAVIGATIONINFO_MODULE    /**< to get correct part in pypolarnavinfo.h */
#include "pypolarnavinfo.h"
#include "rave_alloc.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_pypolarnavinfo");

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

/*@{ Area */
/**
 * Returns the native Area_t instance.
 * @param[in] pyarea - the python area instance
 * @returns the native area instance.
 */
static PolarNavigationInfo
PyPolarNavigationInfo_GetNative(PyPolarNavigationInfo* navinfo)
{
  RAVE_ASSERT((navinfo != NULL), "navinfo == NULL");
  return navinfo->info;
}

/**
 * Creates a python area from a native area or will create an
 * initial native area if p is NULL.
 * @param[in] p - the native area (or NULL)
 * @returns the python area product.
 */
static PyPolarNavigationInfo*
PyPolarNavigationInfo_New(PolarNavigationInfo p)
{
  PyPolarNavigationInfo* result = NULL;
  result = PyObject_NEW(PyPolarNavigationInfo, &PyPolarNavigationInfo_Type);
  if (result != NULL) {
    result->info = p;
  }
  return result;
}

/**
 * Deallocates the area
 * @param[in] obj the object to deallocate.
 */
static void _pypolarnavinfo_dealloc(PyPolarNavigationInfo* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the area.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pypolarnavinfo_new(PyObject* self, PyObject* args)
{
  PolarNavigationInfo info;
  memset(&info, 0, sizeof(PolarNavigationInfo));
  info.ai = info.ei = info.ri = -1;
  PyPolarNavigationInfo* result = PyPolarNavigationInfo_New(info);
  return (PyObject*)result;
}

/**
 * All methods a area can have
 */
static struct PyMethodDef _pypolarnavinfo_methods[] =
{
  {"lon", NULL, METH_VARARGS},
  {"lat", NULL, METH_VARARGS},
  {"height", NULL, METH_VARARGS},
  {"actual_height", NULL, METH_VARARGS},
  {"distance", NULL, METH_VARARGS},
  {"range", NULL, METH_VARARGS},
  {"actual_range", NULL, METH_VARARGS},
  {"azimuth", NULL, METH_VARARGS},
  {"actual_azimuth", NULL, METH_VARARGS},
  {"elevation", NULL, METH_VARARGS},
  {"beamwH", NULL, METH_VARARGS},
  {"beamwV", NULL, METH_VARARGS},
  {"otype", NULL, METH_VARARGS},
  {"ei", NULL, METH_VARARGS},
  {"ri", NULL, METH_VARARGS},
  {"ai", NULL, METH_VARARGS},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the structure
 * @param[in] self - the structure
 */
static PyObject* _pypolarnavinfo_getattro(PyPolarNavigationInfo* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "lon") == 0) {
    return PyFloat_FromDouble(self->info.lon);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "lat") == 0) {
    return PyFloat_FromDouble(self->info.lat);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "height") == 0) {
    return PyFloat_FromDouble(self->info.height);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "actual_height") == 0) {
    return PyFloat_FromDouble(self->info.actual_height);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "distance") == 0) {
    return PyFloat_FromDouble(self->info.distance);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "range") == 0) {
    return PyFloat_FromDouble(self->info.range);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "actual_range") == 0) {
    return PyFloat_FromDouble(self->info.actual_range);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "azimuth") == 0) {
    return PyFloat_FromDouble(self->info.azimuth);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "actual_azimuth") == 0) {
    return PyFloat_FromDouble(self->info.actual_azimuth);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "elevation") == 0) {
    return PyFloat_FromDouble(self->info.elevation);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "beamwH") == 0) {
    return PyFloat_FromDouble(self->info.beamwH);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "beamwV") == 0) {
    return PyFloat_FromDouble(self->info.beamwV);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "otype") == 0) {
    return PyLong_FromLong(self->info.otype);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "ei") == 0) {
    return PyLong_FromLong(self->info.ei);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "ri") == 0) {
    return PyLong_FromLong(self->info.ri);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "ai") == 0) {
    return PyLong_FromLong(self->info.ai);
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

static int _pypolarnavinfo_pyobjAsDouble(PyObject* obj, double* v)
{
  *v = 0.0;
  if (PyInt_Check(obj)) {
    *v = (double)PyInt_AsLong(obj);
  } else if (PyLong_Check(obj)) {
    *v = PyLong_AsDouble(obj);
  } else if (PyFloat_Check(obj)) {
    *v = PyFloat_AsDouble(obj);
  } else {
    return 0;
  }
  return 1;
}

/**
 * Sets the attribute value
 */
static int _pypolarnavinfo_setattro(PyPolarNavigationInfo *self, PyObject *name, PyObject *val)
{
  int result = -1;
  double v = 0.0;

  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("lon", name) == 0) {
    if (_pypolarnavinfo_pyobjAsDouble(val, &v)) {
      self->info.lon = v;
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "lon must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("lat", name) == 0) {
    if (_pypolarnavinfo_pyobjAsDouble(val, &v)) {
      self->info.lat = v;
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "lat must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("height", name) == 0) {
    if (_pypolarnavinfo_pyobjAsDouble(val, &v)) {
      self->info.height = v;
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "height must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("actual_height", name) == 0) {
    if (_pypolarnavinfo_pyobjAsDouble(val, &v)) {
      self->info.actual_height = v;
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "actual_height must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("distance", name) == 0) {
    if (_pypolarnavinfo_pyobjAsDouble(val, &v)) {
      self->info.distance = v;
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "distance must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("range", name) == 0) {
    if (_pypolarnavinfo_pyobjAsDouble(val, &v)) {
      self->info.range = v;
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "range must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("actual_range", name) == 0) {
    if (_pypolarnavinfo_pyobjAsDouble(val, &v)) {
      self->info.actual_range = v;
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "actual_range must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("azimuth", name) == 0) {
    if (_pypolarnavinfo_pyobjAsDouble(val, &v)) {
      self->info.azimuth = v;
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "azimuth must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("actual_azimuth", name) == 0) {
    if (_pypolarnavinfo_pyobjAsDouble(val, &v)) {
      self->info.actual_azimuth = v;
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "actual_azimuth must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("elevation", name) == 0) {
    if (_pypolarnavinfo_pyobjAsDouble(val, &v)) {
      self->info.elevation = v;
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "elevation must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("beamwH", name) == 0) {
    if (_pypolarnavinfo_pyobjAsDouble(val, &v)) {
      self->info.beamwH = v;
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "beamwH must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("beamwV", name) == 0) {
    if (_pypolarnavinfo_pyobjAsDouble(val, &v)) {
      self->info.beamwV = v;
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "beamwV must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("otype", name) == 0) {
    if (PyInt_Check(val)) {
      self->info.otype = (int)PyInt_AsLong(val);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "otype must be a int");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("ei", name) == 0) {
    if (PyInt_Check(val)) {
      self->info.ei = (int)PyInt_AsLong(val);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "ei must be a int");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("ri", name) == 0) {
    if (PyInt_Check(val)) {
      self->info.ri = (int)PyInt_AsLong(val);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "ri must be a int");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("ai", name) == 0) {
    if (PyInt_Check(val)) {
      self->info.ai = (int)PyInt_AsLong(val);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "ai must be a int");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, "Unknown attribute");
  }

  result = 0;
done:
  return result;
}

/*@} End of PyPolarNavigationInfo */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pypolarnavinfo_doc,
    "This class contains navigation information associated with a particular polar object like scan or volume.\n"
    "\n"
    "The instance is used as a container for a number of attributes that are relevant when defining navigating in a polar object.\n"
    "Members are:\n\n"
    " * lon                - a double representing the longitude in radians.\n\n"
    " * lat                - a double representing the latitude in radians.\n\n"
    " * height             - a double representing the height in meters.\n\n"
    " * actual_height      - a double representing the actual height in meters.\n\n"
    " * distance           - a double representing the distance to the radar in meters.\n\n"
    " * range              - a double representing the range along the ray in meters.\n\n"
    " * actual_range       - a double representing the actual range along the ray in meters.\n\n"
    " * azimuth            - a double representing the azimuth in radians\n\n"
    " * actual_azimuth     - a double representing the actual azimuth in radians\n\n"
    " * elevation          - a double representing the elevation in radians\n\n"
    " * otype              - an integer representing the type of object this information comes from, either scan or pvol\n\n"
    " * ei                 - an integer representing the elevation index (if applicable)\n\n"
    " * ri                 - an integer representing the range index\n\n"
    " * ai                 - an integer representing the ray (azimuth) index\n\n"
    "\n"
    );
/*@} End of Documentation about the type */


/*@{ Type definitions */
PyTypeObject PyPolarNavigationInfo_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "PolarNavigationInfoCore", /*tp_name*/
  sizeof(PyPolarNavigationInfo), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pypolarnavinfo_dealloc,  /*tp_dealloc*/
  0,                            /*tp_print*/
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
  (getattrofunc)_pypolarnavinfo_getattro, /*tp_getattro*/
  (setattrofunc)_pypolarnavinfo_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pypolarnavinfo_doc,                  /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pypolarnavinfo_methods,              /*tp_methods*/
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

/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pypolarnavinfo_new, 1,
      "new() -> new instance of the PyPolarNavigationInfoCore object\n\n"
      "Creates a new instance of the PyPolarNavigationInfoCore object"},
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_pypolarnavinfo)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyPolarNavigationInfo_API[PyPolarNavigationInfo_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyPolarNavigationInfo_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyPolarNavigationInfo_Type);

  MOD_INIT_DEF(module, "_pypolarnavinfo", _pypolarnavinfo_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyPolarNavigationInfo_API[PyPolarNavigationInfo_Type_NUM] = (void*)&PyPolarNavigationInfo_Type;
  PyPolarNavigationInfo_API[PyPolarNavigationInfo_GetNative_NUM] = (void *)PyPolarNavigationInfo_GetNative;
  PyPolarNavigationInfo_API[PyPolarNavigationInfo_New_NUM] = (void*)PyPolarNavigationInfo_New;

  c_api_object = PyCapsule_New(PyPolarNavigationInfo_API, PyPolarNavigationInfo_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_pypolarnavinfo.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _pypolarnavinfo.error");
    return MOD_INIT_ERROR;
  }

  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
