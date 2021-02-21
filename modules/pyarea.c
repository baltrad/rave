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
 * Python version of the Area API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYAREA_MODULE    /**< to get correct part in pyarea.h */
#include "pyarea.h"
#include "pyprojection.h"
#include "rave_alloc.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_area");

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
static Area_t*
PyArea_GetNative(PyArea* pyarea)
{
  RAVE_ASSERT((pyarea != NULL), "pyarea == NULL");
  return RAVE_OBJECT_COPY(pyarea->area);
}

/**
 * Creates a python area from a native area or will create an
 * initial native area if p is NULL.
 * @param[in] p - the native area (or NULL)
 * @returns the python area product.
 */
static PyArea*
PyArea_New(Area_t* p)
{
  PyArea* result = NULL;
  Area_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&Area_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for area.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for area.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyArea, &PyArea_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->area = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->area, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyArea instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyArea.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the area
 * @param[in] obj the object to deallocate.
 */
static void _pyarea_dealloc(PyArea* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->area, obj);
  RAVE_OBJECT_RELEASE(obj->area);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the area.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyarea_new(PyObject* self, PyObject* args)
{
  PyArea* result = PyArea_New(NULL);
  return (PyObject*)result;
}

/**
 * All methods a area can have
 */
static struct PyMethodDef _pyarea_methods[] =
{
  {"id", NULL},
  {"description", NULL},
  {"xsize", NULL},
  {"ysize", NULL},
  {"xscale", NULL},
  {"yscale", NULL},
  {"extent", NULL},
  {"projection", NULL},
  {"pcsid", NULL},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the rave field
 * @param[in] self - the rave field
 */
static PyObject* _pyarea_getattro(PyArea* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "id") == 0) {
    if (Area_getID(self->area) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(Area_getID(self->area));
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "description") == 0) {
    if (Area_getDescription(self->area) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(Area_getDescription(self->area));
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "xsize") == 0) {
    return PyLong_FromLong(Area_getXSize(self->area));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "ysize") == 0) {
    return PyLong_FromLong(Area_getYSize(self->area));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "xscale") == 0) {
    return PyFloat_FromDouble(Area_getXScale(self->area));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "yscale") == 0) {
    return PyFloat_FromDouble(Area_getYScale(self->area));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "extent") == 0) {
    double llX, llY, urX, urY;
    Area_getExtent(self->area, &llX, &llY, &urX, &urY);
    return Py_BuildValue("(dddd)", llX, llY, urX, urY);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "projection") == 0) {
    Projection_t* projection = Area_getProjection(self->area);
    if (projection != NULL) {
      PyProjection* result = PyProjection_New(projection);
      RAVE_OBJECT_RELEASE(projection);
      return (PyObject*)result;
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "pcsid") == 0) {
    if (Area_getPcsid(self->area) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(Area_getPcsid(self->area));
    }
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the attribute value
 */
static int _pyarea_setattro(PyArea *self, PyObject *name, PyObject *val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "id") == 0) {
    if (PyString_Check(val)) {
      Area_setID(self->area, PyString_AsString(val));
    } else if (val == Py_None) {
      Area_setID(self->area, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "id must be a string");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "description") == 0) {
    if (PyString_Check(val)) {
      Area_setDescription(self->area, PyString_AsString(val));
    } else if (val == Py_None) {
      Area_setDescription(self->area, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "description must be a string");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "xsize")==0) {
    if (PyInt_Check(val)) {
      Area_setXSize(self->area, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "xsize must be of type int");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "ysize")==0) {
    if (PyInt_Check(val)) {
      Area_setYSize(self->area, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "ysize must be of type int");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "xscale")==0) {
    if (PyFloat_Check(val)) {
      Area_setXScale(self->area, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      Area_setXScale(self->area, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      Area_setXScale(self->area, (double) PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "xscale must be of type float");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "yscale")==0) {
    if (PyFloat_Check(val)) {
      Area_setYScale(self->area, PyFloat_AsDouble(val));
    } else if (PyLong_Check(val)) {
      Area_setYScale(self->area, PyLong_AsDouble(val));
    } else if (PyInt_Check(val)) {
      Area_setYScale(self->area, (double) PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "yscale must be of type float");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "extent")==0) {
    double llX = 0.0L, llY = 0.0L, urX = 0.0L, urY = 0.0L;
    if (!PyArg_ParseTuple(val, "dddd", &llX, &llY, &urX, &urY)) {
      raiseException_gotoTag(done, PyExc_TypeError,
          "extent must be a tuple containing 4 doubles representing llX,llY,urX,urY");
    }
    Area_setExtent(self->area, llX, llY, urX, urY);
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "projection")==0) {
    if (PyProjection_Check(val)) {
      Area_setProjection(self->area, ((PyProjection*) val)->projection);
    } else if (val == Py_None) {
      Area_setProjection(self->area, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,
          "projection must be of ProjectionCore type");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "pcsid") == 0) {
    if (PyString_Check(val)) {
      Area_setPcsid(self->area, PyString_AsString(val));
    } else if (val == Py_None) {
      Area_setPcsid(self->area, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "pcsid must be a string");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError,
        PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }
  result = 0;
done:
  return result;

}

static PyObject* _pyarea_isArea(PyObject* self, PyObject* args)
{
  PyObject* inobj = NULL;
  if (!PyArg_ParseTuple(args,"O", &inobj)) {
    return NULL;
  }
  if (PyArea_Check(inobj)) {
    return PyBool_FromLong(1);
  }
  return PyBool_FromLong(0);
}
/*@} End of Area */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyarea_doc,
    "This class provides functionality for defining an area used in for example cartesian products.\n"
    "\n"
    "The area instance is used as a container for a number of attributes that are relevant when defining an area.\n"
    "Since this instance is used for defining areas it doesn't contain any methods. Instead there are only a number\n"
    "of members which are:\n\n"
    " * id            - a string identifying this area definition.\n\n"
    " * description   - a string description of this area.\n\n"
    " * xsize         - a integer defining the xsize.\n\n"
    " * ysize         - a integer defining the ysize.\n\n"
    " * xscale        - a float defining the xscale in meters.\n\n"
    " * yscale        - a float defining the yscale in meters.\n\n"
    " * extent        - a tuple of four floats defining the extent of this area (lower left X, lower left Y, upper right X, upper right Y).\n\n"
    " * projection    - the projection definition of type ProjectionCore. When setting this, the pcsid will be reset.\n\n"
    " * pcsid         - the projection id string. When setting this, the projection will be reset.\n\n"
    "\n"
    "Usage us quite straight forward when using this class. However, usually, the area registry is used when creating areas.\n"
    " import _area, _projection, math\n"
    " a = _area.new()\n"
    " a.id = \"myid\"\n"
    " a.description = \"this is an area\"\n"
    " a.xsize = 100\n"
    " a.ysize = 100\n"
    " a.xscale = 1000.0\n"
    " a.yscale = 1000.0\n"
    " a.projection = _rave.projection(\"gnom\",\"gnom\",\"+proj=gnom +R=6371000.0 +lat_0=56.3675 +lon_0=12.8544\")\n"
    " xy = a.projection.fwd((12.8544*math.pi/180.0, 56.3675*math.pi/180.0)))\n"
    " a.extent = (xy[0] - 50*a.xscale, xy[1] - 50*a.yscale, xy[2] + 50*a.xscale, xy[3] + 50*a.yscale)\n"
    );
/*@} End of Documentation about the type */


/*@{ Type definitions */
PyTypeObject PyArea_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "AreaCore", /*tp_name*/
  sizeof(PyArea), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyarea_dealloc,  /*tp_dealloc*/
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
  (getattrofunc)_pyarea_getattro, /*tp_getattro*/
  (setattrofunc)_pyarea_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyarea_doc,                  /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyarea_methods,              /*tp_methods*/
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
  {"new", (PyCFunction)_pyarea_new, 1,
      "new() -> new instance of the AcrrCore object\n\n"
      "Creates a new instance of the AcrrCore object"},
  {"isArea", (PyCFunction)_pyarea_isArea, 1,
      "isArea(obj) -> True if object is an area, otherwise False\n\n"
      "Checks if the provided object is a python area object or not.\n\n"
      "obj - the object to check."},
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_area)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyArea_API[PyArea_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyArea_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyArea_Type);

  MOD_INIT_DEF(module, "_area", _pyarea_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyArea_API[PyArea_Type_NUM] = (void*)&PyArea_Type;
  PyArea_API[PyArea_GetNative_NUM] = (void *)PyArea_GetNative;
  PyArea_API[PyArea_New_NUM] = (void*)PyArea_New;

  c_api_object = PyCapsule_New(PyArea_API, PyArea_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_area.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _area.error");
    return MOD_INIT_ERROR;
  }

  import_pyprojection();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
