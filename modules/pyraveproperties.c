/* --------------------------------------------------------------------
Copyright (C) 2025 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the rave properties
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-02-14
 */
#include "pyravecompat.h"
#include "pyraveapi.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"
#include "rave_properties.h"

#define PYRAVEPROPERTIES_MODULE
#include "pyraveproperties.h"
#include "pyravevalue.h"
#include "rave_alloc.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_raveproperties");

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

/*@{ RaveProperties */
/**
 * Returns the native instance.
 * @param[in] pyobj - the python instance
 * @returns the native instance.
 */
static RaveProperties_t*
PyRaveProperties_GetNative(PyRaveProperties* pyobj)
{
  RAVE_ASSERT((pyobj != NULL), "pyobj == NULL");
  return RAVE_OBJECT_COPY(pyobj->properties);
}

/**
 * Creates a python instance from a native instnace or will create an
 * initial instance if p is NULL.
 * @param[in] p - the native instance (or NULL)
 * @returns the python instance.
 */
static PyRaveProperties*
PyRaveProperties_New(RaveProperties_t* p)
{
  PyRaveProperties* result = NULL;
  RaveProperties_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RaveProperties_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for properties.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for properties.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyRaveProperties, &PyRaveProperties_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->properties = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->properties, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyRaveProperties instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyRaveProperties.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

// /**
//  * Opens a file that is supported by area registry
//  * @param[in] filename - the area registry file to load
//  * @param[in] pyprojregistry - the projection registry to be used in conjunction with the area registry
//  * @return the py area registry on success.
//  */
// static PyAreaRegistry*
// PyAreaRegistry_Load(const char* filename, PyProjectionRegistry* pyprojregistry)
// {
//   AreaRegistry_t* registry = NULL;
//   PyAreaRegistry* result = NULL;
//   ProjectionRegistry_t* projregistry = NULL;

//   if (filename == NULL) {
//     raiseException_returnNULL(PyExc_ValueError, "providing a filename that is NULL");
//   }

//   if (pyprojregistry != NULL) {
//     projregistry = pyprojregistry->registry;
//   }

//   registry = AreaRegistry_load(filename, projregistry);
//   if (registry == NULL) {
//     raiseException_gotoTag(done, PyExc_IOError, "Failed to open file");
//   }
//   result = PyAreaRegistry_New(registry);

// done:
//   RAVE_OBJECT_RELEASE(registry);
//   return result;
// }

/**
 * Deallocates the instance
 * @param[in] obj the object to deallocate.
 */
static void _pyraveproperties_dealloc(PyRaveProperties* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->properties, obj);
  RAVE_OBJECT_RELEASE(obj->properties);
  PyObject_Del(obj);
}

/**
 * Creates a new instance.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyraveproperties_new(PyObject* self, PyObject* args)
{
  PyRaveProperties* result = PyRaveProperties_New(NULL);
  return (PyObject*)result;
}

/**
 * Associates a key with a value in the properties instance
 * @parma[in] self - this instance
 * @param[in] args - key, value
 * @returns None
 */
static PyObject* _pyraveproperties_set(PyRaveProperties* self, PyObject* args)
{
  PyObject* pyvalue = NULL;
  RaveValue_t* value = NULL;

  char* name = NULL;

  if (!PyArg_ParseTuple(args, "sO", &name, &pyvalue)) {
    return NULL;
  }
  if (PyRaveValue_Check(pyvalue)) {
    if (!RaveProperties_set(self->properties, name, ((PyRaveValue*)pyvalue)->value)) {
      raiseException_gotoTag(fail, PyExc_MemoryError, "Could not set value");
    }
  } else {
    value = PyRaveApi_RaveValueFromObject(pyvalue);
    if (value != NULL) {
      if (!RaveProperties_set(self->properties, name, value)) {
        raiseException_gotoTag(fail, PyExc_MemoryError, "Could not set property");
      }
    }
  }

  RAVE_OBJECT_RELEASE(value);
  Py_RETURN_NONE;
fail:
  RAVE_OBJECT_RELEASE(value);
  return NULL;
}

/**
 * Returns the value for the provided key
 * @param[in] self - self
 * @param[in] args - the key
 * @returns The RaveValueCore
 */
static PyObject* _pyraveproperties_get(PyRaveProperties* self, PyObject* args)
{
  RaveValue_t* value = NULL;
  PyObject* result = NULL;

  char* name = NULL;

  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  value = RaveProperties_get(self->properties, name);
  if (value != NULL) {
    result = PyRaveApi_RaveValueToObject(value);
  } else {
    raiseException_returnNULL(PyExc_ValueError, "Could not find property");
  }
  RAVE_OBJECT_RELEASE(value);

  return result;
}

/**
 * Returns if the property exists or not
 * @param[in] self - self
 * @param[in] args - the key
 * @returns a boolean
 */
 static PyObject* _pyraveproperties_hasProperty(PyRaveProperties* self, PyObject* args)
 {
   char* name = NULL;
 
   if (!PyArg_ParseTuple(args, "s", &name)) {
     return NULL;
   }
   return PyBool_FromLong(RaveProperties_hasProperty(self->properties, name));
 }

 /**
 * Removes the property from the set
 * @param[in] self - self
 * @param[in] args - the key
 * @returns None
 */
 static PyObject* _pyraveproperties_remove(PyRaveProperties* self, PyObject* args)
 {
   char* name = NULL;
 
   if (!PyArg_ParseTuple(args, "s", &name)) {
     return NULL;
   }
   RaveProperties_remove(self->properties, name);
   Py_RETURN_NONE;
 }

/**
 * Returns the number of properties
 * @param[in] self - self
 * @param[in] args - N/A
 * @returns The size
 */
 static PyObject* _pyraveproperties_size(PyRaveProperties* self, PyObject* args)
 {
   if (!PyArg_ParseTuple(args, "")) {
     return NULL;
   }
   return PyLong_FromLong(RaveProperties_size(self->properties));
 }
 
/**
 * All methods a registry can have
 */
static struct PyMethodDef _pyraveproperties_methods[] =
{
  {"set", (PyCFunction) _pyraveproperties_set, 1,
       "set(key, value)\n\n"
       "Sets a key-value in the properties.\n\n"
       "key  - the identifying key.\n"
       "value - the RaveValueCore instance."},
  {"get", (PyCFunction) _pyraveproperties_get, 1,
       "get(key) -> RaveValueCore\n\n"
       "Returns the value for the specified key.\n"},
  {"hasProperty", (PyCFunction) _pyraveproperties_hasProperty, 1,
       "hasProperty(key) -> Boolean\n\n"
       "Returns if property exists or not.\n"},
  {"remove", (PyCFunction) _pyraveproperties_remove, 1,
       "remove(key) \n\n"
       "Removes the property if it exists.\n"},
    {"size", (PyCFunction) _pyraveproperties_size, 1,
      "size() -> int\n\n"
      "Returns the number of properties.\n"},
   {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the properties
 * @param[in] self - the registry
 */
static PyObject* _pyraveproperties_getattro(PyRaveProperties* self, PyObject* name)
{
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the specified attribute in the properties
 */
static int _pyraveproperties_setattro(PyRaveProperties* self, PyObject* name, PyObject* val)
{
  int result = -1;
  return result;
}
/*@} End of RaveProperties */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyraveproperties_type_doc,
    "Property handling"
    );
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyRaveProperties_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "RavePropertiesCore", /*tp_name*/
  sizeof(PyRaveProperties), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyraveproperties_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pyraveproperties_getattro, /*tp_getattro*/
  (setattrofunc)_pyraveproperties_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyraveproperties_type_doc,     /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyraveproperties_methods,      /*tp_methods*/
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
  {"new", (PyCFunction)_pyraveproperties_new, 1,
      "new() -> new instance of the RavePropertiesCore object\n\n"
      "Creates a new instance of the RavePropertiesCore object"},
  // {"load", (PyCFunction)_pyarearegistry_load, 1,
  //     "load(filename [,projregistry]) -> area registry\n\n"
  //     "Loads an area registry xml file. \n\n"
  //     "filename     - the path to the file containing the area registry xml definition\n"
  //     "projregistry - the projection registry is optional but will be helpful if getting areas from the registry\n\n"
  //     "The format of the area registry file should be in the format:\n"
  //     "<?xml version='1.0' encoding='iso-8859-1'?>\n"
  //     "<areas>\n"
  //     "  <area id=\"nrd2km\">\n"
  //     "    <description>Nordic, all radars, 2 km</description>\n"
  //     "    <areadef>\n"
  //     "      <arg id=\"pcs\">ps14e60n</arg>\n"
  //     "      <arg id=\"xsize\" type=\"int\">848</arg>\n"
  //     "      <arg id=\"ysize\" type=\"int\">1104</arg>\n"
  //     "      <arg id=\"scale\" type=\"float\">2000</arg>\n"
  //     "      <!-- can also be xscale / yscale -->\n"
  //     "      <arg id=\"extent\" type=\"sequence\">-738816.513333,-3995515.596160,955183.48666699999,-1787515.59616</arg>\n"
  //     "    </areadef>\n"
  //     "  </area>\n"
  //     "  ...\n"
  //     "</areas>"
  // },
  {NULL,NULL} /*Sentinel*/
};

/*@{ Documentation about the module */
PyDoc_STRVAR(_pyraveproperties_module_doc,
    "This class provides functionality for managing properties.\n"
    );
/*@} End of Documentation about the module */


MOD_INIT(_raveproperties)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyRaveProperties_API[PyRaveProperties_API_pointers];
  PyObject *c_api_object = NULL;
  MOD_INIT_SETUP_TYPE(PyRaveProperties_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyRaveProperties_Type);

  MOD_INIT_DEF(module, "_raveproperties", _pyraveproperties_module_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyRaveProperties_API[PyRaveProperties_Type_NUM] = (void*)&PyRaveProperties_Type;
  PyRaveProperties_API[PyRaveProperties_GetNative_NUM] = (void *)PyRaveProperties_GetNative;
  PyRaveProperties_API[PyRaveProperties_New_NUM] = (void*)PyRaveProperties_New;
  //PyAreaRegistry_API[PyAreaRegistry_Load_NUM] = (void*)PyAreaRegistry_Load;

  c_api_object = PyCapsule_New(PyRaveProperties_API, PyRaveProperties_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_raveproperties.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _raveproperties.error");
    return MOD_INIT_ERROR;
  }

  import_ravevalue();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
