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
 * @date 2021-10-14
 */
#include "pyravecompat.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYPROJECTIONPIPELINE_MODULE   /**< to get correct part of pyprojectionpipeline.h */
#include "pyprojectionpipeline.h"

#include "pyprojection.h"

#include "pyrave_debug.h"
#include "rave_alloc.h"
#include "pyravecompat.h"

/**
 * Debug this module.
 */
PYRAVE_DEBUG_MODULE("_projectionpipeline");

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
/// ProjectionPipeline
/// --------------------------------------------------------------------
/*@{ ProjectionPipeline */
/**
 * Returns the native ProjectionPipeline_t instance.
 * @param[in] pyprojectionpipeline - the python projection pipeline instance
 * @returns the native projection pipeline instance.
 */
static ProjectionPipeline_t*
PyProjectionPipeline_GetNative(PyProjectionPipeline* pyprojectionpipeline)
{
  RAVE_ASSERT((pyprojectionpipeline != NULL), "pyprojectionpipeline == NULL");
  return RAVE_OBJECT_COPY(pyprojectionpipeline->pipeline);
}

/**
 * Creates a python projection pipeline from a native projection pipeline.
 * @param[in] p - the native projection pipeline
 * @returns the python projection pipeline.
 */
static PyProjectionPipeline*
PyProjectionPipeline_New(ProjectionPipeline_t* p)
{
  PyProjectionPipeline* result = NULL;
  ProjectionPipeline_t* cp = NULL;

  if (p == NULL) {
    RAVE_CRITICAL0("Trying to create a python projection pipeline without the projection pipeline");
    raiseException_returnNULL(PyExc_AttributeError, "Trying to create a python projection pipeline without the projection pipeline ");
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyProjectionPipeline, &PyProjectionPipeline_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->pipeline = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->pipeline, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyProjectionPipeline instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to create PyProjectionPipeline instance");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the projection pipeline
 * @param[in] obj - the object to deallocate.
 */
static void _pyprojectionpipeline_dealloc(PyProjectionPipeline* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->pipeline, obj);
  RAVE_OBJECT_RELEASE(obj->pipeline);
  PyObject_Del(obj);
}

/**
 * Creates a new projection pipeline instance.
 * @param[in] self - this instance.
 * @param[in] args - arguments for creation (projection, projection).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyprojectionpipeline_new(PyObject* self, PyObject* args)
{
  PyObject *first=NULL, *second=NULL;
  PyProjectionPipeline* result = NULL;
  ProjectionPipeline_t* pipeline = NULL;
  PyProjection *firstPyProjection = NULL, *secondPyProjection = NULL;
  if (!PyArg_ParseTuple(args, "OO", &first, &second)) {
    return NULL;
  }
  if (!PyProjection_Check(first) || !PyProjection_Check(second)) {
    raiseException_returnNULL(PyExc_AttributeError, "Must provide 2 projections when creating a pipeline");
  }

  firstPyProjection = (PyProjection*)first;
  secondPyProjection = (PyProjection*)second;

  pipeline = RAVE_OBJECT_NEW(&ProjectionPipeline_TYPE);
  if (pipeline == NULL) {
    raiseException_returnNULL(PyExc_RuntimeError, "Could not allocate memory");
  }
  if (!ProjectionPipeline_init(pipeline, firstPyProjection->projection, secondPyProjection->projection)) {
    raiseException_gotoTag(done, PyExc_RuntimeError, "Could not initialize pipeline");
  }

  result = PyProjectionPipeline_New(pipeline);

done:
  RAVE_OBJECT_RELEASE(pipeline);
  return (PyObject*)result;
}

/**
 * Creates a new projection pipeline instance between default lon/lat projection and the other projection.
 * @param[in] self - this instance.
 * @param[in] args - arguments for creation (projection).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyprojectionpipeline_createDefaultLonLatPipeline(PyObject* self, PyObject* args)
{
  PyObject *other=NULL;
  PyProjectionPipeline* result = NULL;
  ProjectionPipeline_t* pipeline = NULL;
  PyProjection *pyProjection = NULL;
  if (!PyArg_ParseTuple(args, "O", &other)) {
    return NULL;
  }
  if (!PyProjection_Check(other)) {
    raiseException_returnNULL(PyExc_AttributeError, "Must provide 1 projection when creating the default lon/lat pipeline");
  }

  pyProjection = (PyProjection*)other;

  pipeline = ProjectionPipeline_createDefaultLonLatPipeline(pyProjection->projection);
  if (pipeline != NULL) {
    result = PyProjectionPipeline_New(pipeline);
  } else {
    raiseException_returnNULL(PyExc_RuntimeError, "Could not create pipeline");
  }

  RAVE_OBJECT_RELEASE(pipeline);
  return (PyObject*)result;
}

/**
 * Translates coordinate from first projection to second projection.
 * @param[in] self - the projection
 * @param[in] args - first coordinate pair as a tuple of doubles
 * @returns a tuple of two doubles
 */

static PyObject* _pyprojectionpipeline_fwd(PyProjectionPipeline* self, PyObject* args)
{
  double inu=0.0L, inv=0.0L;
  double outu=0.0L, outv=0.0L;

  if (!PyArg_ParseTuple(args, "(dd)", &inu, &inv)) {
    return NULL;
  }

  if (!ProjectionPipeline_fwd(self->pipeline, inu, inv, &outu, &outv)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to forward coordinate pair");
  }

  return Py_BuildValue("(dd)", outu, outv);
}


/**
 * Translates coordinate from second projection to first projection.
 * @param[in] self - the projection
 * @param[in] args - the first coordinate pair as a tuple of two doubles.
 * @returns a tuple of two doubles
 */
static PyObject* _pyprojection_inv(PyProjectionPipeline* self, PyObject* args)
{
  double inu=0.0L, inv=0.0L;
  double outu=0.0L, outv=0.0L;

  if (!PyArg_ParseTuple(args, "(dd)", &inu, &inv)) {
    return NULL;
  }

  if (!ProjectionPipeline_inv(self->pipeline, inu, inv, &outu, &outv)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to inv coordinate pair");
  }

  return Py_BuildValue("(dd)", outu, outv);
}

MOD_DIR_FORWARD_DECLARE(PyProjectionPipeline);

/**
 * All methods a projection can have
 */
static struct PyMethodDef _pyprojectionpipeline_methods[] =
{
  {"first", NULL},
  {"second", NULL},
  {"fwd", (PyCFunction) _pyprojectionpipeline_fwd, 1,
    "fwd((u,v)) -> u/v\n\n"
    "Translates a coordinate pair from first projection to second.\n\n"
  },
  {"inv", (PyCFunction) _pyprojection_inv, 1,
    "inv((u,v)) -> u/v\n\n"
    "Translates a coordinate pair from second projection to first.\n\n"
  },
  {"__dir__", (PyCFunction) MOD_DIR_REFERENCE(PyProjectionPipeline), METH_NOARGS},
  {NULL, NULL } /* sentinel */
};

MOD_DIR_FUNCTION(PyProjectionPipeline, _pyprojectionpipeline_methods)

/**
 * Returns the specified attribute in the rave field
 * @param[in] self - the rave field
 */
static PyObject* _pyprojectionpipeline_getattro(PyProjectionPipeline* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("first", name) == 0) {
    Projection_t* projection = ProjectionPipeline_getFirstProjection(self->pipeline);
    if (projection != NULL) {
      PyProjection* result = PyProjection_New(projection);
      RAVE_OBJECT_RELEASE(projection);
      return (PyObject*)result;
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("second", name) == 0) {
    Projection_t* projection = ProjectionPipeline_getSecondProjection(self->pipeline);
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
 * Sets the attribute value
 */
static int _pyprojectionpipeline_setattro(PyProjectionPipeline *self, PyObject *name, PyObject *value)
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
PyDoc_STRVAR(_pyprojectionpipeline_type_doc,
    "Helper for forwarding from one projection to another regardless if it is PROJ.4 or > PROJ.4\n"
    "A list of avilable member attributes are described below. For information about member functions, check each functions doc.\n"
    "\n"
    "first          - First projection.\n"
    "second         - Second projection.\n"
    "\n"
    "Usage:\n"
    " import _projectionpipeline\n"
    " pipeline = _projectionpipeline.new(proj1, proj2)\n"
    " xy = pipeline.fwd(deg2rad((12.8544, 56.3675)))\n"
    );
/*@} End of Documentation about the module */

/// --------------------------------------------------------------------
/// Module setup
/// --------------------------------------------------------------------
/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pyprojectionpipeline_new, 1,
    "new(first, second) -> ProjectionPipelineCore\n\n"
    "Creates a new projection pipeline instance with two ProjectionCore objects\n\n"
    "first          - First projection object\n"
    "second         - Second projection object\n"
  },
  {"createDefaultLonLatPipeline", (PyCFunction)_pyprojectionpipeline_createDefaultLonLatPipeline, 1,
    "createDefaultLonLatPipeline(other) -> default lon/lat projection pipeline\n\n"
    "other - the other ProjectionCore instance"
    "Returns the default lon/lat projection pipeline\n\n"
  },
  {NULL,NULL} /*Sentinel*/
};

PyTypeObject PyProjectionPipeline_Type =
{
   PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "ProjectionPipelineCore", /*tp_name*/
  sizeof(PyProjectionPipeline), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyprojectionpipeline_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pyprojectionpipeline_getattro, /*tp_getattro*/
  (setattrofunc)_pyprojectionpipeline_setattro, /*tp_setattro*/
  0, /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyprojectionpipeline_type_doc, /*tp_doc*/
  (traverseproc)0, /*tp_traverse*/
  (inquiry)0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  _pyprojectionpipeline_methods, /*tp_methods*/
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

MOD_INIT(_projectionpipeline)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyProjectionPipeline_API[PyProjectionPipeline_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyProjectionPipeline_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyProjectionPipeline_Type);

  MOD_INIT_DEF(module, "_projectionpipeline", _pyprojectionpipeline_type_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyProjectionPipeline_API[PyProjectionPipeline_Type_NUM] = (void*)&PyProjectionPipeline_Type;
  PyProjectionPipeline_API[PyProjectionPipeline_GetNative_NUM] = (void *)PyProjectionPipeline_GetNative;
  PyProjectionPipeline_API[PyProjectionPipeline_New_NUM] = (void*)PyProjectionPipeline_New;

  c_api_object = PyCapsule_New(PyProjectionPipeline_API, PyProjectionPipeline_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_projectionpipeline.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _projectionpipeline.error");
    return MOD_INIT_ERROR;
  }

  import_pyprojection();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
