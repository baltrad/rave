/* --------------------------------------------------------------------
Copyright (C) 2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the Projection registry
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-12-15
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"
#include "pyprojection.h"

#define PYPROJECTIONREGISTRY_MODULE    /**< to get correct part in pyprojectionregistry.h */
#include "pyprojectionregistry.h"
#include "rave_alloc.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_projectionregistry");

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

/*@{ ProjectionRegistry */
/**
 * Returns the native ProjectionRegistry_t instance.
 * @param[in] pyregistry - the python registry instance
 * @returns the native registry instance.
 */
static ProjectionRegistry_t*
PyProjectionRegistry_GetNative(PyProjectionRegistry* pyregistry)
{
  RAVE_ASSERT((pyregistry != NULL), "pyregistry == NULL");
  return RAVE_OBJECT_COPY(pyregistry->registry);
}

/**
 * Creates a python registry from a native registry or will create an
 * initial native projection registry if p is NULL.
 * @param[in] p - the native projection registry (or NULL)
 * @returns the python projection registry.
 */
static PyProjectionRegistry*
PyProjectionRegistry_New(ProjectionRegistry_t* p)
{
  PyProjectionRegistry* result = NULL;
  ProjectionRegistry_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&ProjectionRegistry_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for projection registry.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for projection registry.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyProjectionRegistry, &PyProjectionRegistry_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->registry = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->registry, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyProjectionRegistry instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyProjectionRegistry.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Opens a file that is supported by projection registry.
 * @param[in] self this instance.
 * @param[in] args arguments for creation. (A string identifying the file)
 * @return the object on success, otherwise NULL
 */
static PyProjectionRegistry*
PyProjectionRegistry_Load(const char* filename)
{
  ProjectionRegistry_t* registry = NULL;
  PyProjectionRegistry* result = NULL;

  if (filename == NULL) {
    raiseException_returnNULL(PyExc_ValueError, "providing a filename that is NULL");
  }

  registry = ProjectionRegistry_load(filename);
  if (registry == NULL) {
    raiseException_gotoTag(done, PyExc_IOError, "Failed to open file");
  }
  result = PyProjectionRegistry_New(registry);

done:
  RAVE_OBJECT_RELEASE(registry);
  return result;
}

/**
 * Deallocates the registry
 * @param[in] obj the object to deallocate.
 */
static void _pyprojectionregistry_dealloc(PyProjectionRegistry* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->registry, obj);
  RAVE_OBJECT_RELEASE(obj->registry);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the registry.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyprojectionregistry_new(PyObject* self, PyObject* args)
{
  PyProjectionRegistry* result = PyProjectionRegistry_New(NULL);
  return (PyObject*)result;
}

static PyObject* _pyprojectionregistry_add(PyProjectionRegistry* self, PyObject* args)
{
  PyObject* inptr = NULL;
  PyProjection* projection = NULL;

  if (!PyArg_ParseTuple(args, "O", &inptr)) {
    return NULL;
  }

  if (!PyProjection_Check(inptr)) {
    raiseException_returnNULL(PyExc_TypeError,"Added object must be of type ProjectionCore");
  }

  projection = (PyProjection*)inptr;

  if (!ProjectionRegistry_add(self->registry, projection->projection)) {
    raiseException_returnNULL(PyExc_MemoryError, "Failed to add projection to registry");
  }

  Py_RETURN_NONE;
}

/**
 * Returns the number of projections in this registry
 * @parma[in] self - this instance
 * @param[in] args - NA
 * @returns the number of projections
 */
static PyObject* _pyprojectionregistry_size(PyProjectionRegistry* self, PyObject* args)
{
  return PyLong_FromLong(ProjectionRegistry_size(self->registry));
}

/**
 * Returns the projection at specified index
 * @parma[in] self - this instance
 * @param[in] args - index
 * @returns the projection
 */
static PyObject* _pyprojectionregistry_get(PyProjectionRegistry* self, PyObject* args)
{
  Projection_t* projection = NULL;
  PyObject* result = NULL;
  int index = 0;

  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  if (index < 0 || index >= ProjectionRegistry_size(self->registry)) {
    raiseException_returnNULL(PyExc_IndexError, "Index out of range");
  }

  if((projection = ProjectionRegistry_get(self->registry, index)) == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "Could not aquire projection");
  }

  result = (PyObject*)PyProjection_New(projection);

  RAVE_OBJECT_RELEASE(projection);

  return result;
}

/**
 * Returns the projection with the specified id
 * @parma[in] self - this instance
 * @param[in] args - pcs id
 * @returns the projection
 */
static PyObject* _pyprojectionregistry_getByName(PyProjectionRegistry* self, PyObject* args)
{
  Projection_t* projection = NULL;
  PyObject* result = NULL;
  char* pcsid = NULL;

  if (!PyArg_ParseTuple(args, "s", &pcsid)) {
    return NULL;
  }

  if((projection = ProjectionRegistry_getByName(self->registry, pcsid)) == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "Could not aquire projection");
  }

  result = (PyObject*)PyProjection_New(projection);

  RAVE_OBJECT_RELEASE(projection);

  return result;
}

/**
 * Removes the projection at specified index
 * @parma[in] self - this instance
 * @param[in] args - index
 * @returns None
 */
static PyObject* _pyprojectionregistry_remove(PyProjectionRegistry* self, PyObject* args)
{
  int index = 0;

  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  ProjectionRegistry_remove(self->registry, index);

  Py_RETURN_NONE;
}

/**
 * Removes the projection with the specified id
 * @parma[in] self - this instance
 * @param[in] args - pcs id
 * @returns None
 */
static PyObject* _pyprojectionregistry_removeByName(PyProjectionRegistry* self, PyObject* args)
{
  char* pcsid = NULL;

  if (!PyArg_ParseTuple(args, "s", &pcsid)) {
    return NULL;
  }

  ProjectionRegistry_removeByName(self->registry, pcsid);

  Py_RETURN_NONE;
}

static PyObject* _pyprojectionregistry_write(PyProjectionRegistry* self, PyObject* args)
{
  char* filename = NULL;
  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }

  if (!ProjectionRegistry_write(self->registry, filename)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to write file");
  }

  Py_RETURN_NONE;
}

/**
 * Loads a registry from an xml file
 * @param[in] self - this instance
 * @param[in] args - a string pointing at the projections registry xml file
 * @return the read registry or NULL on failure
 */
static PyObject* _pyprojectionregistry_load(PyObject* self, PyObject* args)
{
  PyProjectionRegistry* result = NULL;
  char* filename = NULL;
  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }
  result = PyProjectionRegistry_Load(filename);
  return (PyObject*)result;
}

/**
 * All methods a registry can have
 */
static struct PyMethodDef _pyprojectionregistry_methods[] =
{
  {"add", (PyCFunction) _pyprojectionregistry_add, 1},
  {"size", (PyCFunction) _pyprojectionregistry_size, 1},
  {"get", (PyCFunction) _pyprojectionregistry_get, 1},
  {"getByName", (PyCFunction) _pyprojectionregistry_getByName, 1},
  {"remove", (PyCFunction) _pyprojectionregistry_remove, 1},
  {"removeByName", (PyCFunction) _pyprojectionregistry_removeByName, 1},
  {"write", (PyCFunction) _pyprojectionregistry_write, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the registry
 * @param[in] self - the registry
 */
static PyObject* _pyprojectionregistry_getattro(PyProjectionRegistry* self, PyObject* name)
{
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the specified attribute in the registry
 */
static int _pyprojectionregistry_setattro(PyProjectionRegistry* self, PyObject* name, PyObject* val)
{
  int result = -1;
  return result;
}

/*@} End of ProjectionRegistry */

/*@{ Type definitions */
PyTypeObject PyProjectionRegistry_Type =
{
    PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "ProjectionRegistryCore", /*tp_name*/
  sizeof(PyProjectionRegistry), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyprojectionregistry_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
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
  (getattrofunc)_pyprojectionregistry_getattro, /*tp_getattro*/
  (setattrofunc)_pyprojectionregistry_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  0,                            /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyprojectionregistry_methods,/*tp_methods*/
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
  0,                            /*tp_is_gc*/};
/*@} End of Type definitions */

/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pyprojectionregistry_new, 1},
  {"load", (PyCFunction)_pyprojectionregistry_load, 1},
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_projectionregistry)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyProjectionRegistry_API[PyProjectionRegistry_API_pointers];
  PyObject *c_api_object = NULL;
  MOD_INIT_SETUP_TYPE(PyProjectionRegistry_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyProjectionRegistry_Type);

  MOD_INIT_DEF(module, "_projectionregistry", NULL/*doc*/, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyProjectionRegistry_API[PyProjectionRegistry_Type_NUM] = (void*)&PyProjectionRegistry_Type;
  PyProjectionRegistry_API[PyProjectionRegistry_GetNative_NUM] = (void *)PyProjectionRegistry_GetNative;
  PyProjectionRegistry_API[PyProjectionRegistry_New_NUM] = (void*)PyProjectionRegistry_New;
  PyProjectionRegistry_API[PyProjectionRegistry_Load_NUM] = (void*)PyProjectionRegistry_Load;

  c_api_object = PyCapsule_New(PyProjectionRegistry_API, PyProjectionRegistry_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_projectionregistry.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _projectionregistry.error");
    return MOD_INIT_ERROR;
  }


  import_pyprojection();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
