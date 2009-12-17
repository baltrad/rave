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
#include "Python.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYPROJECTION_MODULE   /**< to get correct part of pyprojection.h */
#include "pyprojection.h"

#include "pyrave_debug.h"
#include "rave_alloc.h"

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

/**
 * All methods a projection can have
 */
static struct PyMethodDef _pyprojection_methods[] =
{
  { "transform", (PyCFunction) _pyprojection_transform, 1},
  { "inv", (PyCFunction) _pyprojection_inv, 1},
  { "fwd", (PyCFunction) _pyprojection_fwd, 1},
  { NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the transformator
 * @param[in] self - the cartesian product
 */
static PyObject* _pyprojection_getattr(PyProjection* self, char* name)
{
  PyObject* res = NULL;

  if (strcmp("id", name) == 0) {
    return PyString_FromString(Projection_getID(self->projection));
  } else if (strcmp("description", name) == 0) {
    return PyString_FromString(Projection_getDescription(self->projection));
  } else if (strcmp("definition", name) == 0) {
    return PyString_FromString(Projection_getDefinition(self->projection));
  }

  res = Py_FindMethod(_pyprojection_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Sets the specified attribute in the projection
 */
static int _pyprojection_setattr(PyProjection* self, char* name, PyObject* val)
{
  return -1;
}

/*@} End of Projection */

/// --------------------------------------------------------------------
/// Module setup
/// --------------------------------------------------------------------
/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pyprojection_new, 1},
  {NULL,NULL} /*Sentinel*/
};

PyTypeObject PyProjection_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "ProjectionCore", /*tp_name*/
  sizeof(PyProjection), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyprojection_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pyprojection_getattr, /*tp_getattr*/
  (setattrfunc)_pyprojection_setattr, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0,
  0, /*tp_as_mapping */
  0 /*tp_hash*/
};

PyMODINIT_FUNC
init_projection(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyProjection_API[PyProjection_API_pointers];
  PyObject *c_api_object = NULL;
  PyProjection_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_projection", functions);
  if (module == NULL) {
    return;
  }
  PyProjection_API[PyProjection_Type_NUM] = (void*)&PyProjection_Type;
  PyProjection_API[PyProjection_GetNative_NUM] = (void *)PyProjection_GetNative;
  PyProjection_API[PyProjection_New_NUM] = (void*)PyProjection_New;
  PyProjection_API[PyProjection_NewFromDef_NUM] = (void*)PyProjection_NewFromDef;

  c_api_object = PyCObject_FromVoidPtr((void *)PyProjection_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_projection.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _projection.error");
  }

  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
