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
 * Python version of the OdimSource API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-16
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYODIMSOURCE_MODULE    /**< to get correct part in pyodimsource.h */
#include "pyodimsource.h"
#include "rave_alloc.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_odimsource");

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
 * Returns the native OdimSource_t instance.
 * @param[in] pysource - the python odim source instance
 * @returns the native odim source instance.
 */
static OdimSource_t*
PyOdimSource_GetNative(PyOdimSource* pysource)
{
  RAVE_ASSERT((pysource != NULL), "pysource == NULL");
  return RAVE_OBJECT_COPY(pysource->source);
}

/**
 * Creates a python odim source from a native source or will create an
 * initial native odim source if p is NULL.
 * @param[in] p - the native odim source (or NULL)
 * @returns the python odim source product.
 */
static PyOdimSource*
PyOdimSource_New(OdimSource_t* p)
{
  PyOdimSource* result = NULL;
  OdimSource_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&OdimSource_TYPE);
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
    result = PyObject_NEW(PyOdimSource, &PyOdimSource_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->source = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->source, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyOdimSource instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyOdimSource.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the odim source
 * @param[in] obj the object to deallocate.
 */
static void _pyodimsource_dealloc(PyOdimSource* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->source, obj);
  RAVE_OBJECT_RELEASE(obj->source);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the area.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyodimsource_new(PyObject* self, PyObject* args)
{
  char *nod = NULL, *wmo = NULL, *wigos = NULL, *plc = NULL, *rad = NULL, *cccc = NULL, *org = NULL;
  OdimSource_t* source = NULL;
  PyOdimSource* result = NULL;
  if (!PyArg_ParseTuple(args, "z|zzzzzz", &nod, &wmo, &wigos, &plc, &rad, &cccc, &org)) {
    return NULL;
  }
  if (nod != NULL) {
    source = OdimSource_create(nod, wmo, wigos, plc, rad, cccc, org);
    result = PyOdimSource_New(source);
  } else {
    PyErr_SetString(PyExc_AttributeError, "Must provide NOD");
  }
  RAVE_OBJECT_RELEASE(source);
  return (PyObject*)result;
}

/**
 * All methods a area can have
 */
static struct PyMethodDef _pyodimsource_methods[] =
{
  {"nod", NULL, METH_VARARGS},
  {"wmo", NULL, METH_VARARGS},
  {"wigos", NULL, METH_VARARGS},
  {"plc", NULL, METH_VARARGS},
  {"rad", NULL, METH_VARARGS},
  {"cccc", NULL, METH_VARARGS},
  {"org", NULL, METH_VARARGS},
  {"source", NULL, METH_VARARGS},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the rave field
 * @param[in] self - the rave field
 */
static PyObject* _pyodimsource_getattro(PyOdimSource* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "nod") == 0) {
    if (OdimSource_getNod(self->source) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(OdimSource_getNod(self->source));
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "wmo") == 0) {
    if (OdimSource_getWmo(self->source) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(OdimSource_getWmo(self->source));
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "wigos") == 0) {
    if (OdimSource_getWigos(self->source) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(OdimSource_getWigos(self->source));
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "plc") == 0) {
    if (OdimSource_getPlc(self->source) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(OdimSource_getPlc(self->source));
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "rad") == 0) {
    if (OdimSource_getRad(self->source) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(OdimSource_getRad(self->source));
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "cccc") == 0) {
    if (OdimSource_getCccc(self->source) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(OdimSource_getCccc(self->source));
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "org") == 0) {
    if (OdimSource_getOrg(self->source) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(OdimSource_getOrg(self->source));
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "source") == 0) {
    if (OdimSource_getSource(self->source) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(OdimSource_getSource(self->source));
    }
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the attribute value
 */
static int _pyodimsource_setattro(PyOdimSource *self, PyObject *name, PyObject *val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }

  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "nod") == 0) {
    raiseException_gotoTag(done, PyExc_TypeError, "Can not set NOD");
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "wmo") == 0) {
    if (PyString_Check(val)) {
      OdimSource_setWmo(self->source, PyString_AsString(val));
    } else if (val == Py_None) {
      OdimSource_setWmo(self->source, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "wmo must be a string");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "wigos") == 0) {
    if (PyString_Check(val)) {
      OdimSource_setWigos(self->source, PyString_AsString(val));
    } else if (val == Py_None) {
      OdimSource_setWigos(self->source, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "wigos must be a string");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "plc") == 0) {
    if (PyString_Check(val)) {
      OdimSource_setPlc(self->source, PyString_AsString(val));
    } else if (val == Py_None) {
      OdimSource_setPlc(self->source, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "plc must be a string");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "rad") == 0) {
    if (PyString_Check(val)) {
      OdimSource_setRad(self->source, PyString_AsString(val));
    } else if (val == Py_None) {
      OdimSource_setRad(self->source, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "rad must be a string");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "cccc") == 0) {
    if (PyString_Check(val)) {
      OdimSource_setCccc(self->source, PyString_AsString(val));
    } else if (val == Py_None) {
      OdimSource_setCccc(self->source, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "cccc must be a string");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "org") == 0) {
    if (PyString_Check(val)) {
      OdimSource_setOrg(self->source, PyString_AsString(val));
    } else if (val == Py_None) {
      OdimSource_setOrg(self->source, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "org must be a string");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError,
        PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }
  result = 0;
done:
  return result;

}

static PyObject* _pyodimsource_isOdimSource(PyObject* self, PyObject* args)
{
  PyObject* inobj = NULL;
  if (!PyArg_ParseTuple(args,"O", &inobj)) {
    return NULL;
  }
  if (PyOdimSource_Check(inobj)) {
    return PyBool_FromLong(1);
  }
  return PyBool_FromLong(0);
}
/*@} End of Odim Source */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyodimsource_doc,
    "This class provides functionality for defining an odim source.\n"
    "\n"
    "The odim source instance is used as a container for the different identification of a source.\n"
    "Since this instance is used for defining a source it doesn't contain any methods. Instead there are only a number\n"
    "of members which are:\n\n"
    " * nod            - a string identifying the NOD.\n\n"
    " * wmo            - a string identifying the WMO.\n\n"
    " * wigos          - a string identifying the WIGOS.\n\n"
    " * plc            - a string identifying the PLC.\n\n"
    " * rad            - a string identifying the RAD.\n\n"
    " * cccc           - a string identifying the CCCC.\n\n"
    " * org            - a string identifying the ORG.\n\n"
    "\n"
    );
/*@} End of Documentation about the type */


/*@{ Type definitions */
PyTypeObject PyOdimSource_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "OdimSourceCore", /*tp_name*/
  sizeof(PyOdimSource), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyodimsource_dealloc,  /*tp_dealloc*/
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
  (getattrofunc)_pyodimsource_getattro, /*tp_getattro*/
  (setattrofunc)_pyodimsource_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyodimsource_doc,                  /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyodimsource_methods,              /*tp_methods*/
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
  {"new", (PyCFunction)_pyodimsource_new, 1,
      "new() -> new instance of the OdimSourceCore object\n\n"
      "Creates a new instance of the OdimSourceCore object"},
  {"isOdimSource", (PyCFunction)_pyodimsource_isOdimSource, 1,
      "isOdimSource(obj) -> True if object is an odim source, otherwise False\n\n"
      "Checks if the provided object is a python odim source object or not.\n\n"
      "obj - the object to check."},
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_odimsource)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyOdimSource_API[PyOdimSource_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyOdimSource_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyOdimSource_Type);

  MOD_INIT_DEF(module, "_odimsource", _pyodimsource_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyOdimSource_API[PyOdimSource_Type_NUM] = (void*)&PyOdimSource_Type;
  PyOdimSource_API[PyOdimSource_GetNative_NUM] = (void *)PyOdimSource_GetNative;
  PyOdimSource_API[PyOdimSource_New_NUM] = (void*)PyOdimSource_New;

  c_api_object = PyCapsule_New(PyOdimSource_API, PyOdimSource_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_odimsource.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _odimsource.error");
    return MOD_INIT_ERROR;
  }

  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
