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
 * Python version of the projection API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-07
 */
#include "pyravecompat.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYPROJECTION_MODULE   /**< to get correct part of pyprojection.h */
#include "pyprojection.h"

#include "pyrave_debug.h"
#include "rave_alloc.h"
#include "pyravecompat.h"

/**
 * Debug this module.
 */
PYRAVE_DEBUG_MODULE("_projection");

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
/// Projection
/// --------------------------------------------------------------------
/*@{ Projection */
/**
 * Returns the native Projection_t instance.
 * @param[in] pyprojection - the python projection instance
 * @returns the native projection instance.
 */
static Projection_t*
PyProjection_GetNative(PyProjection* pyprojection)
{
  RAVE_ASSERT((pyprojection != NULL), "pyprojection == NULL");
  return RAVE_OBJECT_COPY(pyprojection->projection);
}

/**
 * Creates a python projection from a native projection.
 * @param[in] p - the native projection
 * @returns the python projection.
 */
static PyProjection*
PyProjection_New(Projection_t* p)
{
  PyProjection* result = NULL;
  Projection_t* cp = NULL;

  if (p == NULL) {
    RAVE_CRITICAL0("Trying to create a python projection without the projection");
    raiseException_returnNULL(PyExc_AttributeError, "Trying to create a python projection without the projection");
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyProjection, &PyProjection_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->projection = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->projection, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyProjection instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to create PyProjection instance");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Creates a python projection from a definition.
 * @param[in] id - the projection id
 * @param[in] description - the description of this projection
 * @param[in] definition - the definition for this projection
 * @returns the python projection instance.
 */
static PyProjection*
PyProjection_NewFromDef(const char* id, const char* description, const char* definition)
{
  Projection_t* projection = NULL;
  PyProjection* result = NULL;

  RAVE_ASSERT((id != NULL), "id == NULL");
  RAVE_ASSERT((definition != NULL), "definition == NULL");
  RAVE_ASSERT((description != NULL), "description == NULL");

  projection = RAVE_OBJECT_NEW(&Projection_TYPE);
  if (projection == NULL) {
    RAVE_CRITICAL0("Failed to create projection");
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for projection");
  }

  if(!Projection_init(projection, id, description, definition)) {
    RAVE_ERROR0("Could not initialize projection");
    RAVE_OBJECT_RELEASE(projection);
    raiseException_returnNULL(PyExc_ValueError, "Failed to initialize projection");
  }

  result = PyProjection_New(projection);

  RAVE_OBJECT_RELEASE(projection);

  return result;
}

/**
 * Deallocates the projection
 * @param[in] obj - the object to deallocate.
 */
static void _pyprojection_dealloc(PyProjection* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->projection, obj);
  RAVE_OBJECT_RELEASE(obj->projection);
  PyObject_Del(obj);
}

/**
 * Creates a new projection instance.
 * @param[in] self - this instance.
 * @param[in] args - arguments for creation (id, description, definition).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyprojection_new(PyObject* self, PyObject* args)
{
  PyProjection* result = NULL;
  char* id = NULL;
  char* description = NULL;
  char* definition = NULL;

  if (!PyArg_ParseTuple(args, "sss", &id, &description, &definition)) {
    return NULL;
  }

  result = PyProjection_NewFromDef(id, description, definition);

  return (PyObject*)result;
}

static PyObject* _pyprojection_setDebugLevel(PyObject* self, PyObject* args)
{
  int debugPj = 0;
  if (!PyArg_ParseTuple(args, "i", &debugPj)) {
    return NULL;
  }
  Projection_setDebugLevel(debugPj);
  Py_RETURN_NONE;
}

static PyObject* _pyprojection_getDebugLevel(PyObject* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyInt_FromLong(Projection_getDebugLevel());
}

static PyObject* _pyprojection_getProjVersion(PyObject* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyString_FromString(Projection_getProjVersion());
}


static PyObject* _pyprojection_setDefaultLonLatProjDef(PyObject* self, PyObject* args)
{
  char* defaultProjDef=NULL;
  if (!PyArg_ParseTuple(args, "s", &defaultProjDef)) {
    return NULL;
  }
  if (defaultProjDef == NULL || strlen(defaultProjDef) >= 1024) {
    raiseException_returnNULL(PyExc_AttributeError, "projdef must be less than 1024 characters");
  }
  Projection_setDefaultLonLatProjDef(defaultProjDef);
  Py_RETURN_NONE;
}

static PyObject* _pyprojection_getDefaultLonLatProjDef(PyObject* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyString_FromString(Projection_getDefaultLonLatProjDef());
}

static PyObject* _pyprojection_createDefaultLonLatProjection(PyObject* self, PyObject* args)
{
  Projection_t* proj = NULL;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  proj = Projection_createDefaultLonLatProjection();
  if (proj != NULL) {
    result = (PyObject*)PyProjection_New(proj);
  } else {
    raiseException_returnNULL(PyExc_RuntimeError, "Could not create default lon/lat projection");
  }
  RAVE_OBJECT_RELEASE(proj);
  return result;
}

/**
 * Projects a coordinate pair into the new projection coordinate system
 * @param[in] self - the source projection
 * @param[in] args - arguments for projecting)
 * @return Py_None on success, otherwise NULL
 */
static PyObject* _pyprojection_transform(PyProjection* self, PyObject* args)
{
  PyProjection* tgtproj = NULL;
  PyObject* pytgtproj = NULL;
  PyObject* pycoord = NULL;
  PyObject* result = NULL;

  double x=0.0,y=0.0,z=0.0;
  int coordlen = 0;

  if(!PyArg_ParseTuple(args, "OO", &pytgtproj,&pycoord)) {
    return NULL;
  }

  if (!PyProjection_Check(pytgtproj)) {
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

  tgtproj = (PyProjection*)pytgtproj;

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
 * Projects a coordinate pair into the new projection coordinate system
 * @param[in] self - the source projection
 * @param[in] args - arguments for projecting)
 * @return Py_None on success, otherwise NULL
 */
static PyObject* _pyprojection_transformx(PyProjection* self, PyObject* args)
{
  PyProjection* tgtproj = NULL;
  PyObject* pytgtproj = NULL;
  PyObject* pycoord = NULL;
  PyObject* result = NULL;

  double x=0.0,y=0.0,z=0.0;
  double ox=0.0, oy=0.0, oz=0.0;
  int coordlen = 0;

  if(!PyArg_ParseTuple(args, "OO", &pytgtproj,&pycoord)) {
    return NULL;
  }

  if (!PyProjection_Check(pytgtproj)) {
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

  tgtproj = (PyProjection*)pytgtproj;

  if (coordlen == 2) {
    if (!Projection_transformx(self->projection, tgtproj->projection, x, y, 0.0, &ox, &oy, NULL)) {
      raiseException_returnNULL(PyExc_IOError, "Failed to transform to target projection");
    }
    result = Py_BuildValue("(dd)", ox, oy);
  } else {
    if (!Projection_transformx(self->projection, tgtproj->projection, x, y, z, &ox, &oy, &oz)) {
      raiseException_returnNULL(PyExc_IOError, "Failed to transform to target projection");
    }
    result = Py_BuildValue("(ddd)", ox, oy, oz);
  }

  return result;
}

/**
 * Translates surface coordinate into lon/lat.
 * @param[in] self - the projection
 * @param[in] args - the (x,y) coordinate as a tuple of two doubles.
 * @returns a tuple of two doubles representing the lon/lat coordinate in radians or NULL on failure
 */
static PyObject* _pyprojection_inv(PyProjection* self, PyObject* args)
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

static PyObject* _pyprojection_fwd(PyProjection* self, PyObject* args)
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

static PyObject* _pyprojection_isLatLong(PyProjection* self, PyObject* args)
{
  int v = 0;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  v = Projection_isLatLong(self->projection);
  return PyBool_FromLong(v);
}

MOD_DIR_FORWARD_DECLARE(PyProjection);

/**
 * All methods a projection can have
 */
static struct PyMethodDef _pyprojection_methods[] =
{
  {"id", NULL},
  {"description", NULL},
  {"definition", NULL},
  {"transform", (PyCFunction) _pyprojection_transform, 1,
    "transform(tgtproj, (x,y[,z]) -> (x,y[,z])\n\n"
    "Projects a coordinate pair into the new projection coordinate system. In some projections, z will also be needed and in those cases, the returned value will also contain a z (height)."
    "tgtproj    - The target projection into which we want to navigate\n"
    "x          - x coordinate (or lon)\n"
    "y          - y coordinate (or lat)\n"
    "z          - z coordinate (or height)"
  },
  {"transformx", (PyCFunction) _pyprojection_transformx, 1,
    "transform(tgtproj, (x,y[,z]) -> (x,y[,z])\n\n"
    "This is an alternate version of transform() . This function behaves similar to transform() and the user should be able\n"
    "to use either of them. The c-version is different though.\n"
    "tgtproj    - The target projection into which we want to navigate\n"
    "x          - x coordinate (or lon)\n"
    "y          - y coordinate (or lat)\n"
    "z          - z coordinate (or height)"
  },
  {"inv", (PyCFunction) _pyprojection_inv, 1,
    "inv((x,y)) -> lon/lat\n\n"
    "Translates surface coordinate into lon/lat.\n\n"
    "x - x coordinate\n"
    "y - y coordinate"
  },
  {"fwd", (PyCFunction) _pyprojection_fwd, 1,
    "fwd((lon,lat)) -> x/y\n\n"
    "Translates lon/lat into surface coordinates.\n\n"
    "lon - longitude in radians\n"
    "lat - latitude in radians"
  },
  {"isLatLong", (PyCFunction) _pyprojection_isLatLong, 1,
    "isLatLong() -> boolean\n\n"
    "Returns if this projection is defined as a lat long projection or not.\n\n"
  },

  {"__dir__", (PyCFunction) MOD_DIR_REFERENCE(PyProjection), METH_NOARGS},
  {NULL, NULL } /* sentinel */
};

MOD_DIR_FUNCTION(PyProjection, _pyprojection_methods)

/**
 * Returns the specified attribute in the rave field
 * @param[in] self - the rave field
 */
static PyObject* _pyprojection_getattro(PyProjection* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "id") == 0) {
    return PyString_FromString(Projection_getID(self->projection));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "description") == 0) {
    return PyString_FromString(Projection_getDescription(self->projection));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "definition") == 0) {
    return PyString_FromString(Projection_getDefinition(self->projection));
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the attribute value
 */
static int _pyprojection_setattro(PyProjection *self, PyObject *name, PyObject *value)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }

  raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));

  result = 0;
done:
  return result;
}
/*@} End of Projection */

/*@{ Documentation about the module */
PyDoc_STRVAR(_pyprojection_type_doc,
    "Wrapper around the USGS PROJ.4 library\n"
    "There are 3 readonly attributes that are set when creating the actual instance:\n"
    "id          - identifier of this projection, like ps14e60n\n"
    "description - description of this projection\n"
    "definition  - the USGS PROJ.4 definition string"
    "\n"
    "Usage:\n"
    " import _projection\n"
    " proj = _projection.new(\"merc_proj\", \"mercator projection\", \"+proj=merc +lat_ts=0 +lon_0=0 +k=1.0 +R=6378137.0 +nadgrids=@null +no_defs\")\n"
    " xy = proj.fwd(deg2rad((12.8544, 56.3675)))\n"
    );
/*@} End of Documentation about the module */

/// --------------------------------------------------------------------
/// Module setup
/// --------------------------------------------------------------------
/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pyprojection_new, 1,
    "new(id, description, definition) -> ProjectionCore\n\n"
    "Creates a new projection instance with identifier, description and the USGS PROJ.4 definition\n\n"
    "id          - identifier of this projection, like ps14e60n\n"
    "description - description of this projection\n"
    "definition  - the USGS PROJ.4 definition string"
  },
  {"setDebugLevel", (PyCFunction)_pyprojection_setDebugLevel, 1,
    "setDebugLevel(debugPj)\n\n"
    "Sets the debug level when using proj API\n\n"
    "debugPj          - Value between 0 (NONE) to 3 (FULL) and 4 (TELL?)\n"
  },
  {"getDebugLevel", (PyCFunction)_pyprojection_getDebugLevel, 1,
    "getDebugLevel() -> debug level\n\n"
    "Returns the debug level when using proj API, value between 0 (NONE) to 3 (FULL) and 4 (TELL?)\n\n"
  },
  {"getProjVersion", (PyCFunction)_pyprojection_getProjVersion, 1,
    "getProjVersion() -> proj version\n\n"
    "Returns the Proj version or unknown if it couldn't be identified\n\n"
  },
  {"setDefaultLonLatProjDef", (PyCFunction)_pyprojection_setDefaultLonLatProjDef, 1,
    "setDefaultLonLatProjDef(str)\n\n"
    "Sets the default lon/lat proj definition to use when creating lon/lat projection internally\n\n"
    "str - the pcs definition (max 1023 char long)"
  },
  {"getDefaultLonLatProjDef", (PyCFunction)_pyprojection_getDefaultLonLatProjDef, 1,
    "getDefaultLonLatProjDef(str) -> proj definition str\n\n"
    "Returns the default lon/lat proj definition to use when creating lon/lat projection internally\n\n"
  },
  {"createDefaultLonLatProjection", (PyCFunction)_pyprojection_createDefaultLonLatProjection, 1,
    "createDefaultLonLatProjection() -> default lon/lat projection\n\n"
    "Returns the default lon/lat projection\n\n"
  },
  {NULL,NULL} /*Sentinel*/
};

PyTypeObject PyProjection_Type =
{
   PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "ProjectionCore", /*tp_name*/
  sizeof(PyProjection), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyprojection_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)0/*_pyprojection_getattr*/, /*tp_getattr*/
  (setattrfunc)0/*_pyprojection_setattr*/, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0,
  0, /*tp_as_mapping */
  0, /*tp_hash*/
  (ternaryfunc)0, /*tp_call*/
  (reprfunc)0, /*tp_str*/
  (getattrofunc)_pyprojection_getattro, /*tp_getattro*/
  (setattrofunc)_pyprojection_setattro, /*tp_setattro*/
  0, /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyprojection_type_doc, /*tp_doc*/
  (traverseproc)0, /*tp_traverse*/
  (inquiry)0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  _pyprojection_methods, /*tp_methods*/
  0,                    /*tp_members*/
  0,                      /*tp_getset*/
  0,                      /*tp_base*/
  0,                      /*tp_dict*/
  0,                      /*tp_descr_get*/
  0,                      /*tp_descr_set*/
  0,                      /*tp_dictoffset*/
  0,                      /*tp_init*/
  0,                      /*tp_alloc*/
  0,                      /*tp_new*/
  0,                      /*tp_free*/
  0,                      /*tp_is_gc*/
};

MOD_INIT(_projection)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyProjection_API[PyProjection_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyProjection_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyProjection_Type);

  MOD_INIT_DEF(module, "_projection", _pyprojection_type_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyProjection_API[PyProjection_Type_NUM] = (void*)&PyProjection_Type;
  PyProjection_API[PyProjection_GetNative_NUM] = (void *)PyProjection_GetNative;
  PyProjection_API[PyProjection_New_NUM] = (void*)PyProjection_New;
  PyProjection_API[PyProjection_NewFromDef_NUM] = (void*)PyProjection_NewFromDef;

  c_api_object = PyCapsule_New(PyProjection_API, PyProjection_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_projection.error", NULL, NULL); //PyString_FromString("_projection.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _projection.error");
    return MOD_INIT_ERROR;
  }

  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
