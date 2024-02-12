/* --------------------------------------------------------------------
Copyright (C) 2009-2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the RaveLegend API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-02-10
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"
#include "rave_legend.h"

#define PYRAVELEGEND_MODULE        /**< to get correct part of pyravelegend.h */
#include "pyravelegend.h"

//#include <arrayobject.h>
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_ravelegend");

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

/*@{ Rave Legend */
/**
 * Returns the native RaveLegend_t instance.
 * @param[in] pylegend - self
 * @returns the native legend instance.
 */
static RaveLegend_t* PyRaveLegend_GetNative(PyRaveLegend* pylegend)
{
  RAVE_ASSERT((pylegend != NULL), "pylegend == NULL");
  return RAVE_OBJECT_COPY(pylegend->legend);
}

/**
 * Creates a python rave legend from a native rave legend or will create an
 * initial native RaveLegend if p is NULL.
 * @param[in] p - the native rave legend (or NULL)
 * @returns the python rave legend.
 */
static PyRaveLegend* PyRaveLegend_New(RaveLegend_t* p)
{
  PyRaveLegend* result = NULL;
  RaveLegend_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RaveLegend_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for rave legend.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for rave legend.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyRaveLegend, &PyRaveLegend_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->legend = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->legend, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyRaveLegend instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for rave legend.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the rave legend
 * @param[in] obj the object to deallocate.
 */
static void _pyravelegend_dealloc(PyRaveLegend* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->legend, obj);
  RAVE_OBJECT_RELEASE(obj->legend);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the rave legend.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyravelegend_new(PyObject* self, PyObject* args)
{
  PyRaveLegend* result = PyRaveLegend_New(NULL);
  return (PyObject*)result;
}

static PyObject* _pyravelegend_addValue(PyRaveLegend* self, PyObject* args)
{
  PyObject* pyval = NULL;
  char* name = NULL;

  if (!PyArg_ParseTuple(args, "sO", &name, &pyval)) {
    return NULL;
  }
  if (!PyString_Check(pyval)) {
    raiseException_returnNULL(PyExc_TypeError, "value must be string");
  }
  if (!RaveLegend_addValue(self->legend, name, PyString_AsString(pyval))) {
    raiseException_returnNULL(PyExc_TypeError, "Could not add entry to legend");
  }
  Py_RETURN_NONE;
}

static PyObject* _pyravelegend_size(PyRaveLegend* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyLong_FromLong(RaveLegend_size(self->legend));
}

static PyObject* _pyravelegend_exists(PyRaveLegend* self, PyObject* args)
{
  char* name = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  return PyBool_FromLong(RaveLegend_exists(self->legend, name));
}


static PyObject* _pyravelegend_getValue(PyRaveLegend* self, PyObject* args)
{
  char* name = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  if (RaveLegend_exists(self->legend, name)) {
    return PyString_FromString(RaveLegend_getValue(self->legend, name));
  }

  raiseException_returnNULL(PyExc_KeyError, "Invalid key name");
}

static PyObject* _pyravelegend_getValueAt(PyRaveLegend* self, PyObject* args)
{
  int index;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }
  if (index < 0 || index >= RaveLegend_size(self->legend)) {
    raiseException_returnNULL(PyExc_IndexError, "Invalid index");  
  }
  return PyString_FromString(RaveLegend_getValueAt(self->legend, index));
}

static PyObject* _pyravelegend_getNameAt(PyRaveLegend* self, PyObject* args)
{
  int index;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }
  if (index < 0 || index >= RaveLegend_size(self->legend)) {
    raiseException_returnNULL(PyExc_IndexError, "Invalid index");  
  }
  return PyString_FromString(RaveLegend_getNameAt(self->legend, index));
}

static PyObject* _pyravelegend_clear(PyRaveLegend* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  RaveLegend_clear(self->legend);
  Py_RETURN_NONE;
}


static PyObject* _pyravelegend_remove(PyRaveLegend* self, PyObject* args)
{
  char* name = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  if (!RaveLegend_remove(self->legend, name)) {
    raiseException_returnNULL(PyExc_KeyError, "Invalid key name");
  }
  Py_RETURN_NONE;
}

static PyObject* _pyravelegend_removeAt(PyRaveLegend* self, PyObject* args)
{
  int index = 0;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }
  if (!RaveLegend_removeAt(self->legend, index)) {
    raiseException_returnNULL(PyExc_IndexError, "Invalid index");  
  }
  Py_RETURN_NONE;
}

/*

int RaveLegend_removeValueAt(RaveLegend_t* self, int index);
*/

MOD_DIR_FORWARD_DECLARE(PyRaveLegend);

/**
 * All methods a legend product can have
 */
static struct PyMethodDef _pyravelegend_methods[] =
{
  {"legend", NULL, METH_VARARGS},
  {"addValue", (PyCFunction) _pyravelegend_addValue, 1,
    "addValue(key,value)\n\n"
    "Adds a value to the legend.\n\n"
    "key   - the key name\n"
    "value - the value"
  },
  {"size", (PyCFunction) _pyravelegend_size, 1,
    "size() -> int\n\n"
    "Returns number of entries in legend.\n\n"
  },
  {"exists", (PyCFunction) _pyravelegend_exists, 1,
    "exists(name) -> bool\n\n"
    "Returns if specified key exists or not.\n\n"
  },
  {"getValue", (PyCFunction) _pyravelegend_getValue, 1,
    "getValue(name) -> value\n\n"
    "Returns value for specified key.\n\n"
    "raises KeyError if name doesnt exist.\n"
  },
  {"getValueAt", (PyCFunction) _pyravelegend_getValueAt, 1,
    "getValueAt(index) -> value\n\n"
    "Returns value at specified index.\n\n"
    "raises IndexError if index < 0 or index >= size.\n"
  },
  {"getNameAt", (PyCFunction) _pyravelegend_getNameAt, 1,
    "getNameAt(index) -> name\n\n"
    "Returns name at specified index.\n\n"
    "raises IndexError if index < 0 or index >= size.\n"
  },
  {"clear", (PyCFunction) _pyravelegend_clear, 1,
    "clear()\n\n"
    "Clears all entries from the legend.\n\n"
  },
  {"remove", (PyCFunction) _pyravelegend_remove, 1,
    "remove(name)\n\n"
    "Removes the entry with specified name.\n\n"
    "raises KeyError if name doesnt exist.\n"
  },
  {"removeAt", (PyCFunction) _pyravelegend_removeAt, 1,
    "removeAt(index)\n\n"
    "Removes entry at specified index.\n\n"
    "raises IndexError if index < 0 or index >= size.\n"
  },
  {"__dir__", (PyCFunction) MOD_DIR_REFERENCE(PyRaveLegend), METH_NOARGS},
  {NULL, NULL } /* sentinel */
};

MOD_DIR_FUNCTION(PyRaveLegend, _pyravelegend_methods)

/**
 * Returns the specified attribute in the rave field
 * @param[in] self - the rave field
 */
static PyObject* _pyravelegend_getattro(PyRaveLegend* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "legend") == 0) {
    PyObject* result = PyList_New(0);
    int nentries = 0, i = 0;
    if (result == NULL) {
      raiseException_returnNULL(PyExc_MemoryError, "Failed to create list");
    }
    nentries = RaveLegend_size(self->legend);
    for (i = 0; i < nentries; i++) {
      const char* name = RaveLegend_getNameAt(self->legend, i);
      const char* value = RaveLegend_getValueAt(self->legend, i);
      PyObject* pyentry = (PyObject*)Py_BuildValue("(ss)", name, value);
      if (pyentry == NULL) {
        Py_XDECREF(result);
        raiseException_returnNULL(PyExc_MemoryError, "Failed to create list entry");
      }
      if (PyList_Append(result, pyentry) < 0) {
        Py_XDECREF(result);
        Py_XDECREF(pyentry);
        raiseException_returnNULL(PyExc_MemoryError, "Failed to add list entry");
      }
      Py_XDECREF(pyentry);
    }
    return result;
  }

  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the attribute value
 */
static int _pyravelegend_setattro(PyRaveLegend *self, PyObject *name, PyObject *value)
{
  int result = -1;

  if (name == NULL) {
    goto done;
  }

  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("legend", name)==0) {
    if (!PyList_Check(value)) {
      raiseException_gotoTag(done, PyExc_TypeError,"legend must be a list of tuples with (name, value)");
    }
    Py_ssize_t nentries = PyObject_Length(value);
    Py_ssize_t i = 0;
    char* ename;
    PyObject* evalue;
    for (i = 0; i < nentries; i++) {
      PyObject* pyentry = PyList_GetItem(value, i);
      if (!PyArg_ParseTuple(pyentry, "sO", &ename, &evalue)) {
        goto done;
      }

      if (!PyString_Check(evalue)) {
        raiseException_gotoTag(done, PyExc_TypeError, "Value in entry tuple must be a string");  
      }
    }
    RaveLegend_clear(self->legend);
    for (i = 0; i < nentries; i++) {
      PyObject* pyentry = PyList_GetItem(value, i);
      if (!PyArg_ParseTuple(pyentry, "sO", &ename, &evalue)) {
        goto done;
      }
      RaveLegend_addValue(self->legend, ename, PyString_AsString(evalue));
      if (!PyString_Check(evalue)) {
        raiseException_gotoTag(done, PyExc_TypeError, "Value in entry tuple must be a string");  
      }
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }


  result = 0;
done:

  return result;
}

/*@} End of rave field */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyravelegend_type_doc,
    "A data container that is used as for defining legends that should be associated to various rave objects.\n\n"
    "\n"
    "Usage:\n"
    " import _ravelegend\n"
    " legend = _ravelegend.new()\n"
    " legend.addValue(\"KEY_1\", \"V1\")\n"
    );
/*@} End of Documentation about the module */


/*@{ Type definitions */
PyTypeObject PyRaveLegend_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0)
  "RaveLegendCore",                  /*tp_name*/
  sizeof(PyRaveLegend),              /*tp_size*/
  0,                                /*tp_itemsize*/
  /* methods */
  (destructor)_pyravelegend_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)0,                   /*tp_getattr*/
  (setattrfunc)0,                   /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0, /*tp_as_sequence */
  0, /*tp_as_mapping */
  (hashfunc)0, /*tp_hash*/
  (ternaryfunc)0, /*tp_call*/
  (reprfunc)0, /*tp_str*/
  (getattrofunc)_pyravelegend_getattro, /*tp_getattro*/
  (setattrofunc)_pyravelegend_setattro, /*tp_setattro*/
  0, /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyravelegend_type_doc, /*tp_doc*/
  (traverseproc)0, /*tp_traverse*/
  (inquiry)0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  _pyravelegend_methods, /*tp_methods*/
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
/*@} End of Type definitions */

/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pyravelegend_new, 1,
    "new() -> new instance of the RaveLegendCore object\n\n"
    "Creates a new instance of the RaveLegendCore object"
  },
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_ravelegend)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyRaveLegend_API[PyRaveLegend_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyRaveLegend_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyRaveLegend_Type);

  MOD_INIT_DEF(module, "_ravelegend", _pyravelegend_type_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyRaveLegend_API[PyRaveLegend_Type_NUM] = (void*)&PyRaveLegend_Type;
  PyRaveLegend_API[PyRaveLegend_GetNative_NUM] = (void *)PyRaveLegend_GetNative;
  PyRaveLegend_API[PyRaveLegend_New_NUM] = (void*)PyRaveLegend_New;

  c_api_object = PyCapsule_New(PyRaveLegend_API, PyRaveLegend_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_ravelegend.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _ravelegend.error");
    return MOD_INIT_ERROR;
  }

  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */

