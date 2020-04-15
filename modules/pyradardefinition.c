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
 * Python version of the RadarDefinition API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-08-31
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYRADARDEFINITION_MODULE    /**< to get correct part in pyarea.h */
#include "pyradardefinition.h"
#include "pyprojection.h"
#include "rave_alloc.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_radardef");

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

/*@{ RadarDefinition */
/**
 * Returns the native RadarDefinition_t instance.
 * @param[in] radar - the python radar definition
 * @returns the native radar definition instance.
 */
static RadarDefinition_t*
PyRadarDefinition_GetNative(PyRadarDefinition* radar)
{
  RAVE_ASSERT((radar != NULL), "radar == NULL");
  return RAVE_OBJECT_COPY(radar->def);
}

/**
 * Creates a python radar definition from a native radar definition or will create an
 * initial radar definition if p is NULL.
 * @param[in] p - the native radar definition (or NULL)
 * @returns the python radar definition.
 */
static PyRadarDefinition*
PyRadarDefinition_New(RadarDefinition_t* p)
{
  PyRadarDefinition* result = NULL;
  RadarDefinition_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RadarDefinition_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for radar definition.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for radar definition.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyRadarDefinition, &PyRadarDefinition_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->def = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->def, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyRadarDefinition instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyRadarDefinition.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the radar definition
 * @param[in] obj the object to deallocate.
 */
static void _pyradardefinition_dealloc(PyRadarDefinition* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->def, obj);
  RAVE_OBJECT_RELEASE(obj->def);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the radar definition.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyradardefinition_new(PyObject* self, PyObject* args)
{
  PyRadarDefinition* result = PyRadarDefinition_New(NULL);
  return (PyObject*)result;
}

/**
 * All methods a area can have
 */
static struct PyMethodDef _pyradardefinition_methods[] =
{
  {"id", NULL},
  {"description", NULL},
  {"longitude", NULL},
  {"latitude", NULL},
  {"height", NULL},
  {"elangles", NULL},
  {"nrays", NULL},
  {"nbins", NULL},
  {"scale", NULL},
  {"beamwidth", NULL},
  {"beamwH", NULL},
  {"beamwV", NULL},
  {"wavelength", NULL},
  {"projection", NULL},
  {NULL, NULL } /* sentinel */
};

static PyObject* _pyradardefinition_getattro(PyRadarDefinition* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "id") == 0) {
    if (RadarDefinition_getID(self->def) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(RadarDefinition_getID(self->def));
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "description") == 0) {
    if (RadarDefinition_getDescription(self->def) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(RadarDefinition_getDescription(self->def));
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "longitude") == 0) {
    return PyFloat_FromDouble(RadarDefinition_getLongitude(self->def));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "latitude") == 0) {
    return PyFloat_FromDouble(RadarDefinition_getLatitude(self->def));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "height") == 0) {
    return PyFloat_FromDouble(RadarDefinition_getHeight(self->def));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "scale") == 0) {
    return PyFloat_FromDouble(RadarDefinition_getScale(self->def));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "beamwidth") == 0) {
    return PyFloat_FromDouble(RadarDefinition_getBeamwidth(self->def));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "beamwH") == 0) {
    return PyFloat_FromDouble(RadarDefinition_getBeamwH(self->def));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "beamwV") == 0) {
    return PyFloat_FromDouble(RadarDefinition_getBeamwV(self->def));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "wavelength") == 0) {
    return PyFloat_FromDouble(RadarDefinition_getWavelength(self->def));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "nbins") == 0) {
    return PyLong_FromLong(RadarDefinition_getNbins(self->def));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "nrays") == 0) {
    return PyLong_FromLong(RadarDefinition_getNrays(self->def));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "elangles") == 0) {
    double* angles = NULL;
    unsigned int i = 0;
    unsigned int n = 0;
    PyObject* ret = PyList_New(0);
    if (ret == NULL) {
      return NULL;
    }
    if (!RadarDefinition_getElangles(self->def, &n, &angles)) {
      Py_DECREF(ret);
      return NULL;
    }
    for (i = 0; i < n; i++) {
      if (PyList_Append(ret, PyFloat_FromDouble(angles[i])) < 0) {
        Py_DECREF(ret);
        RAVE_FREE(angles);
        return NULL;
      }
    }
    RAVE_FREE(angles);
    return ret;
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "projection") == 0) {
    Projection_t* projection = RadarDefinition_getProjection(self->def);
    if (projection != NULL) {
      PyProjection* result = PyProjection_New(projection);
      RAVE_OBJECT_RELEASE(projection);
      return (PyObject*)result;
    } else {
      Py_RETURN_NONE;
    }
  }

  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the radar definition
 */
static int _pyradardefinition_setattro(PyRadarDefinition* self, PyObject *name, PyObject *val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "id") == 0) {
    if (PyString_Check(val)) {
      RadarDefinition_setID(self->def, PyString_AsString(val));
    } else if (val == Py_None) {
      RadarDefinition_setID(self->def, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "id must be a string");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "description") == 0) {
    if (PyString_Check(val)) {
      RadarDefinition_setDescription(self->def, PyString_AsString(val));
    } else if (val == Py_None) {
      RadarDefinition_setDescription(self->def, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "description must be a string");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "longitude") == 0) {
    if (PyFloat_Check(val)) {
      RadarDefinition_setLongitude(self->def, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      RadarDefinition_setLongitude(self->def, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      RadarDefinition_setLongitude(self->def, (double) PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "longitude must be a number");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "latitude") == 0) {
    if (PyFloat_Check(val)) {
      RadarDefinition_setLatitude(self->def, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      RadarDefinition_setLatitude(self->def, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      RadarDefinition_setLatitude(self->def, (double) PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "longitude must be a number");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "height") == 0) {
    if (PyFloat_Check(val)) {
      RadarDefinition_setHeight(self->def, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      RadarDefinition_setHeight(self->def, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      RadarDefinition_setHeight(self->def, (double) PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "height must be a number");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "scale") == 0) {
    if (PyFloat_Check(val)) {
      RadarDefinition_setScale(self->def, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      RadarDefinition_setScale(self->def, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      RadarDefinition_setScale(self->def, (double) PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "scale must be a number");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "beamwidth") == 0) {
    if (PyFloat_Check(val)) {
      RadarDefinition_setBeamwidth(self->def, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      RadarDefinition_setBeamwidth(self->def, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      RadarDefinition_setBeamwidth(self->def, (double) PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "beamwidth must be a number");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "beamwH") == 0) {
    if (PyFloat_Check(val)) {
      RadarDefinition_setBeamwH(self->def, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      RadarDefinition_setBeamwH(self->def, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      RadarDefinition_setBeamwH(self->def, (double) PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "beamwH must be a number");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "beamwV") == 0) {
    if (PyFloat_Check(val)) {
      RadarDefinition_setBeamwV(self->def, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      RadarDefinition_setBeamwV(self->def, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      RadarDefinition_setBeamwV(self->def, (double) PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "beamwV must be a number");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "wavelength") == 0) {
    if (PyFloat_Check(val)) {
      RadarDefinition_setWavelength(self->def, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      RadarDefinition_setWavelength(self->def, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      RadarDefinition_setWavelength(self->def, (double) PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "wavelength must be a number");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "nrays") == 0) {
    if (PyLong_Check(val)) {
      RadarDefinition_setNrays(self->def, PyLong_AsLong(val));
    } else if (PyInt_Check(val)) {
      RadarDefinition_setNrays(self->def, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "nrays must be a integer");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "nbins") == 0) {
    if (PyLong_Check(val)) {
      RadarDefinition_setNbins(self->def, PyLong_AsLong(val));
    } else if (PyInt_Check(val)) {
      RadarDefinition_setNbins(self->def, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "nbins must be a integer");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "elangles") == 0) {
    if (PySequence_Check(val)) {
      PyObject* v = NULL;
      Py_ssize_t n = PySequence_Size(val);
      Py_ssize_t i = 0;
      double* angles = RAVE_MALLOC(n * sizeof(double));
      if (angles == NULL) {
        raiseException_gotoTag(done, PyExc_MemoryError,
            "Could not allocate memory for angles");
      }
      for (i = 0; i < n; i++) {
        v = PySequence_GetItem(val, i);
        if (v != NULL) {
          if (PyFloat_Check(v)) {
            angles[i] = PyFloat_AsDouble(v);
          } else if (PyLong_Check(v)) {
            angles[i] = PyLong_AsDouble(v);
          } else if (PyInt_Check(v)) {
            angles[i] = (double) PyInt_AsLong(v);
          } else {
            Py_XDECREF(v);
            raiseException_gotoTag(done, PyExc_TypeError,
                "height must be a number");
          }
        }
        Py_XDECREF(v);
      }
      if (!RadarDefinition_setElangles(self->def, (unsigned int) n, angles)) {
        raiseException_gotoTag(done, PyExc_MemoryError, "Could not set angles");
      }
      RAVE_FREE(angles);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "elangles must be a sequence of numbers");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "projection") == 0) {
    if (PyProjection_Check(val)) {
      RadarDefinition_setProjection(self->def,
          ((PyProjection*) val)->projection);
    } else if (val == Py_None) {
      RadarDefinition_setProjection(self->def, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "projection must be of ProjectionCore type");
    }
  }
  result = 0;
done:
  return result;
}

/*@} End of RadarDefinition */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyradardefinition_type_doc,
    "Defines the characteristics of a radar\n"
    "It keeps track of some relevant information for a radar:\n"
    "id           - ID for this definition.\n"
    "description  - Description for this definition.\n"
    "longitude    - Longitude in radians\n"
    "latitude     - Latitude in radians\n"
    "height       - Height above sea level\n"
    "elangles     - Array of elevation angles as float in radians\n"
    "nrays        - Number of rays\n"
    "nbins        - Number of bins\n"
    "scale        - the length of the bins in meters\n"
    "beamwidth    - the horizontal beam width in radians\n"
    "beamwH       - the horizontal beam width in radians\n"
    "beamwV       - the vertical beam width in radians\n"
    "wavelength   - the wavelength\n"
    "projection   - the projection to use for this radar. Most likely a lon/lat projection\n"
    "\n"
    "Usage:\n"
    " import _radardef, _projection\n"
    " def = _radardef.new()\n"
    " def.projection =  _projection.new(\"x\", \"y\", \"+proj=latlong +ellps=WGS84 +datum=WGS84\")\n"
    " def.id = 'someid'\n"
    " ....\n"
    " def.elangles=[0.5*math.pi/180.0, 1.0*math.pi/180.0]\n"
    " ..."
    );
/*@} End of Documentation about the module */


/*@{ Type definitions */
PyTypeObject PyRadarDefinition_Type =
{
    PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "RadarDefinitionCore", /*tp_name*/
  sizeof(PyRadarDefinition), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyradardefinition_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pyradardefinition_getattro, /*tp_getattro*/
  (setattrofunc)_pyradardefinition_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyradardefinition_type_doc,  /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyradardefinition_methods,    /*tp_methods*/
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
  {"new", (PyCFunction)_pyradardefinition_new, 1,
    "new() -> new instance of the RadarDefinitionCore object\n\n"
    "Creates a new instance of the RadarDefinitionCore object"
  },
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_radardef)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyRadarDefinition_API[PyRadarDefinition_API_pointers];
  PyObject *c_api_object = NULL;
  MOD_INIT_SETUP_TYPE(PyRadarDefinition_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyRadarDefinition_Type);

  MOD_INIT_DEF(module, "_radardef", _pyradardefinition_type_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyRadarDefinition_API[PyRadarDefinition_Type_NUM] = (void*)&PyRadarDefinition_Type;
  PyRadarDefinition_API[PyRadarDefinition_GetNative_NUM] = (void *)PyRadarDefinition_GetNative;
  PyRadarDefinition_API[PyRadarDefinition_New_NUM] = (void*)PyRadarDefinition_New;

  c_api_object = PyCapsule_New(PyRadarDefinition_API, PyRadarDefinition_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_radardef.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _radardef.error");
    return MOD_INIT_ERROR;
  }

  import_pyprojection();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
