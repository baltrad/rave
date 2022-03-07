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
 * Python version of the Area registry
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
#include "pyprojectionregistry.h"
#include "pyarea.h"

#define PYAREAREGISTRY_MODULE    /**< to get correct part in pyarearegistry.h */
#include "pyarearegistry.h"
#include "rave_alloc.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_arearegistry");

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

/*@{ AreaRegistry */
/**
 * Returns the native AreaRegistry_t instance.
 * @param[in] pyregistry - the python registry instance
 * @returns the native registry instance.
 */
static AreaRegistry_t*
PyAreaRegistry_GetNative(PyAreaRegistry* pyregistry)
{
  RAVE_ASSERT((pyregistry != NULL), "pyregistry == NULL");
  return RAVE_OBJECT_COPY(pyregistry->registry);
}

/**
 * Creates a python registry from a native registry or will create an
 * initial native area registry if p is NULL.
 * @param[in] p - the native area registry (or NULL)
 * @returns the python area registry.
 */
static PyAreaRegistry*
PyAreaRegistry_New(AreaRegistry_t* p)
{
  PyAreaRegistry* result = NULL;
  AreaRegistry_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&AreaRegistry_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for area registry.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for area registry.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyAreaRegistry, &PyAreaRegistry_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->registry = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->registry, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyAreaRegistry instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyAreaRegistry.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Opens a file that is supported by area registry
 * @param[in] filename - the area registry file to load
 * @param[in] pyprojregistry - the projection registry to be used in conjunction with the area registry
 * @return the py area registry on success.
 */
static PyAreaRegistry*
PyAreaRegistry_Load(const char* filename, PyProjectionRegistry* pyprojregistry)
{
  AreaRegistry_t* registry = NULL;
  PyAreaRegistry* result = NULL;
  ProjectionRegistry_t* projregistry = NULL;

  if (filename == NULL) {
    raiseException_returnNULL(PyExc_ValueError, "providing a filename that is NULL");
  }

  if (pyprojregistry != NULL) {
    projregistry = pyprojregistry->registry;
  }

  registry = AreaRegistry_load(filename, projregistry);
  if (registry == NULL) {
    raiseException_gotoTag(done, PyExc_IOError, "Failed to open file");
  }
  result = PyAreaRegistry_New(registry);

done:
  RAVE_OBJECT_RELEASE(registry);
  return result;
}

/**
 * Deallocates the registry
 * @param[in] obj the object to deallocate.
 */
static void _pyarearegistry_dealloc(PyAreaRegistry* obj)
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
static PyObject* _pyarearegistry_new(PyObject* self, PyObject* args)
{
  PyAreaRegistry* result = PyAreaRegistry_New(NULL);
  return (PyObject*)result;
}

/**
 * Adds an area to the registry
 * @param[in] self - self
 * @param[in] args - the area object
 * @returns None on success or NULL on failure
 */
static PyObject* _pyarearegistry_add(PyAreaRegistry* self, PyObject* args)
{
  PyObject* inptr = NULL;
  PyArea* area = NULL;

  if (!PyArg_ParseTuple(args, "O", &inptr)) {
    return NULL;
  }

  if (!PyArea_Check(inptr)) {
    raiseException_returnNULL(PyExc_TypeError,"Added object must be of type AreaCore");
  }

  area = (PyArea*)inptr;

  if (!AreaRegistry_add(self->registry, area->area)) {
    raiseException_returnNULL(PyExc_MemoryError, "Failed to add area to registry");
  }

  Py_RETURN_NONE;
}

/**
 * Returns the number of projections in this registry
 * @parma[in] self - this instance
 * @param[in] args - NA
 * @returns the number of projections
 */
static PyObject* _pyarearegistry_size(PyAreaRegistry* self, PyObject* args)
{
  return PyLong_FromLong(AreaRegistry_size(self->registry));
}

/**
 * Returns the area at specified index
 * @parma[in] self - this instance
 * @param[in] args - index
 * @returns the projection
 */
static PyObject* _pyarearegistry_get(PyAreaRegistry* self, PyObject* args)
{
  Area_t* area = NULL;
  PyObject* result = NULL;
  int index = 0;

  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  if (index < 0 || index >= AreaRegistry_size(self->registry)) {
    raiseException_returnNULL(PyExc_IndexError, "Index out of range");
  }

  if((area = AreaRegistry_get(self->registry, index)) == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "Could not aquire area");
  }

  result = (PyObject*)PyArea_New(area);

  RAVE_OBJECT_RELEASE(area);

  return result;
}

/**
 * Returns the area with the specified id
 * @parma[in] self - this instance
 * @param[in] args - id
 * @returns the area
 */
static PyObject* _pyarearegistry_getByName(PyAreaRegistry* self, PyObject* args)
{
  Area_t* area = NULL;
  PyObject* result = NULL;
  char* id = NULL;

  if (!PyArg_ParseTuple(args, "s", &id)) {
    return NULL;
  }

  if((area = AreaRegistry_getByName(self->registry, id)) == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "Could not aquire area");
  }

  result = (PyObject*)PyArea_New(area);

  RAVE_OBJECT_RELEASE(area);

  return result;
}

/**
 * Removes the area at specified index
 * @parma[in] self - this instance
 * @param[in] args - index
 * @returns None
 */
static PyObject* _pyarearegistry_remove(PyAreaRegistry* self, PyObject* args)
{
  int index = 0;

  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  AreaRegistry_remove(self->registry, index);

  Py_RETURN_NONE;
}

/**
 * Removes the area with the specified id
 * @parma[in] self - this instance
 * @param[in] args - id
 * @returns None
 */
static PyObject* _pyarearegistry_removeByName(PyAreaRegistry* self, PyObject* args)
{
  char* id = NULL;

  if (!PyArg_ParseTuple(args, "s", &id)) {
    return NULL;
  }

  AreaRegistry_removeByName(self->registry, id);

  Py_RETURN_NONE;
}

/**
 * Writes a area registry to file
 */
static PyObject* _pyarearegistry_write(PyAreaRegistry* self, PyObject* args)
{
  char* filename = NULL;
  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }

  if (!AreaRegistry_write(self->registry, filename)) {
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
static PyObject* _pyarearegistry_load(PyObject* self, PyObject* args)
{
  PyAreaRegistry* result = NULL;
  PyObject* pyobject = NULL;
  PyProjectionRegistry* pyprojregistry = NULL;

  char* filename = NULL;
  if (!PyArg_ParseTuple(args, "s|O", &filename, &pyobject)) {
    return NULL;
  }
  if (pyobject != NULL) {
    if (!PyProjectionRegistry_Check(pyobject)) {
      raiseException_returnNULL(PyExc_AttributeError, "Object must be of ProjectionCoreRegistry type");
    }
    pyprojregistry = (PyProjectionRegistry*)pyobject;
  }
  result = PyAreaRegistry_Load(filename, pyprojregistry);
  return (PyObject*)result;
}

/**
 * All methods a registry can have
 */
static struct PyMethodDef _pyarearegistry_methods[] =
{
  {"pregistry", NULL, METH_VARARGS},
  {"add", (PyCFunction) _pyarearegistry_add, 1,
       "add(area)\n\n"
       "Adds an area definition of type AreaCore to the registry\n\n"
       "area  - The area definition of type AreaCore."},
  {"size", (PyCFunction) _pyarearegistry_size, 1,
       "size() -> integer with number of area definitions\n\n"
       "Returns the number of area definitions within the area registry."},
  {"get", (PyCFunction) _pyarearegistry_get, 1,
       "get(i) -> AreaCore\n\n"
       "Returns the area definition at position i in the registry.\n\n"
       "i - index in the registry >= 0 and < size()."},
  {"getByName", (PyCFunction) _pyarearegistry_getByName, 1,
       "getByName(name) -> AreaCore\n\n"
       "Returns the area definition with the specified name.\n\n"
       "name - the name of the area definition"},
  {"remove", (PyCFunction) _pyarearegistry_remove, 1,
       "remove(i)\n\n"
       "Removes the area definition at specified index.\n\n"
       "i - index in the registry >= 0 and < size()."},
  {"removeByName", (PyCFunction) _pyarearegistry_removeByName, 1,
       "removeByName(name)\n\n"
       "Removes the area definition with the specified name.\n\n"
       "name - the name of the area definition to remove."},
  {"write", (PyCFunction)_pyarearegistry_write, 1,
       "write(filename)\n\n"
       "Writes the area registry to the xml file as specified by filename.\n\n"
       "filename - path to the place where the xml area registry file should be stored"},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the registry
 * @param[in] self - the registry
 */
static PyObject* _pyarearegistry_getattro(PyAreaRegistry* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("pregistry", name) == 0) {
    ProjectionRegistry_t* pregistry = AreaRegistry_getProjectionRegistry(self->registry);
    if (pregistry != NULL) {
      PyProjectionRegistry* result = PyProjectionRegistry_New(pregistry);
      RAVE_OBJECT_RELEASE(pregistry);
      return (PyObject*)result;
    } else {
      Py_RETURN_NONE;
    }
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the specified attribute in the registry
 */
static int _pyarearegistry_setattro(PyAreaRegistry* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("projection", name)==0) {
    if (PyProjectionRegistry_Check(val)) {
      AreaRegistry_setProjectionRegistry(self->registry, ((PyProjectionRegistry*)val)->registry);
    } else if (val == Py_None) {
      AreaRegistry_setProjectionRegistry(self->registry, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"projection registry must be of ProjectionRegistryCore type");
    }
  }

  result = 0;
done:
  return result;
}
/*@} End of AreaRegistry */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyarearegistry_type_doc,
    "The area registry provides the user with area definitions that are usable when for example creating cartesian products.\n"
    "\n"
    "After the registry has been created, you are able to add, remove and get area definitions as well as writing to the registry file.\n"
    "You are also able to change the projection registry to use by changing the member variable:\n"
    " * pregistry    - The projection registry of type ProjectionRegistryCore\n\n"
    "A simple example on how this area registry can be used is:\n"
    " import _arearegistry, _projectionregistry\n"
    " reg = _arearegistry.load(\"area_regitry.xml\", _projectionregistry.load(\"projection_registry.xml\"))\n"
    " area = reg.getByName(\"swegmaps_2000\")"
    );
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyAreaRegistry_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "AreaRegistryCore", /*tp_name*/
  sizeof(PyAreaRegistry), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyarearegistry_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pyarearegistry_getattro, /*tp_getattro*/
  (setattrofunc)_pyarearegistry_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyarearegistry_type_doc,     /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyarearegistry_methods,      /*tp_methods*/
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
  {"new", (PyCFunction)_pyarearegistry_new, 1,
      "new() -> new instance of the AreaRegistryCore object\n\n"
      "Creates a new instance of the AreaRegistryCore object"},
  {"load", (PyCFunction)_pyarearegistry_load, 1,
      "load(filename [,projregistry]) -> area registry\n\n"
      "Loads an area registry xml file. \n\n"
      "filename     - the path to the file containing the area registry xml definition\n"
      "projregistry - the projection registry is optional but will be helpful if getting areas from the registry\n\n"
      "The format of the area registry file should be in the format:\n"
      "<?xml version='1.0' encoding='iso-8859-1'?>\n"
      "<areas>\n"
      "  <area id=\"nrd2km\">\n"
      "    <description>Nordic, all radars, 2 km</description>\n"
      "    <areadef>\n"
      "      <arg id=\"pcs\">ps14e60n</arg>\n"
      "      <arg id=\"xsize\" type=\"int\">848</arg>\n"
      "      <arg id=\"ysize\" type=\"int\">1104</arg>\n"
      "      <arg id=\"scale\" type=\"float\">2000</arg>\n"
      "      <!-- can also be xscale / yscale -->\n"
      "      <arg id=\"extent\" type=\"sequence\">-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616</arg>\n"
      "    </areadef>\n"
      "  </area>\n"
      "  ...\n"
      "</areas>"
  },
  {NULL,NULL} /*Sentinel*/
};

/*@{ Documentation about the module */
PyDoc_STRVAR(_pyarearegistry_module_doc,
    "This class provides functionality for managing an area registry. It can also read and write the registry.\n"
    "\n"
    "See the load function for information about format of the xml file.\n"
    "\n"
    "Usage:\n"
    " import _arearegistry\n"
    " reg = _arearegistry.load(\"/tmp/area_registry.xml\", proj_registry)\n"
    " area = reg.getByName(\"swegmaps_2000\")\n"
    );
/*@} End of Documentation about the module */


MOD_INIT(_arearegistry)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyAreaRegistry_API[PyAreaRegistry_API_pointers];
  PyObject *c_api_object = NULL;
  MOD_INIT_SETUP_TYPE(PyAreaRegistry_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyAreaRegistry_Type);

  MOD_INIT_DEF(module, "_arearegistry", _pyarearegistry_module_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyAreaRegistry_API[PyAreaRegistry_Type_NUM] = (void*)&PyAreaRegistry_Type;
  PyAreaRegistry_API[PyAreaRegistry_GetNative_NUM] = (void *)PyAreaRegistry_GetNative;
  PyAreaRegistry_API[PyAreaRegistry_New_NUM] = (void*)PyAreaRegistry_New;
  PyAreaRegistry_API[PyAreaRegistry_Load_NUM] = (void*)PyAreaRegistry_Load;

  c_api_object = PyCapsule_New(PyAreaRegistry_API, PyAreaRegistry_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_arearegistry.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _arearegistry.error");
    return MOD_INIT_ERROR;
  }

  import_pyprojectionregistry();
  import_pyarea();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
