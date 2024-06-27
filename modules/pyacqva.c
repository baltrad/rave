/* --------------------------------------------------------------------
Copyright (C) 2024 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the Acqva API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-01-18
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYACQVA_MODULE        /**< to get correct part of pyacqva.h */
#include "pyacqva.h"

#include "pypolarvolume.h"
#include "pypolarscan.h"
#include "pycartesian.h"
#include "pyarea.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_acqva");

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

/*@{ Acqva generator */
/**
 * Returns the native Acqva_t instance.
 * @param[in] pyacqva - the python acqva instance
 * @returns the native acqva instance.
 */
static Acqva_t*
PyAcqva_GetNative(PyAcqva* pyacqva)
{
  RAVE_ASSERT((pyacqva != NULL), "pyacqva == NULL");
  return RAVE_OBJECT_COPY(pyacqva->acqva);
}

/**
 * Creates a python acqva from a native acqva or will create an
 * initial native Acqva if p is NULL.
 * @param[in] p - the native acqva (or NULL)
 * @returns the python acqva product generator.
 */
static PyAcqva*
PyAcqva_New(Acqva_t* p)
{
  PyAcqva* result = NULL;
  Acqva_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&Acqva_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for acqva.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for acqva.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyAcqva, &PyAcqva_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->acqva = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->acqva, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyAcqva instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for acqva.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the cartesian product
 * @param[in] obj the object to deallocate.
 */
static void _pyacqva_dealloc(PyAcqva* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->acqva, obj);
  RAVE_OBJECT_RELEASE(obj->acqva);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the acqva.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyacqva_new(PyObject* self, PyObject* args)
{
  PyAcqva* result = PyAcqva_New(NULL);
  return (PyObject*)result;
}

/**
 * Adds a parameter to the acqva generator
 * @param[in] self - self
 * @param[in] args - <quantity as a string>, <gain as double>, <offset as double>
 * @return None on success otherwise NULL
 */
static PyObject* _pyacqva_addParameter(PyAcqva* self, PyObject* args)
{
  char* quantity = NULL;
  double gain = 1.0, offset = 0.0;

  if (!PyArg_ParseTuple(args, "sdd", &quantity, &gain, &offset)) {
    return NULL;
  }
  if (!Acqva_addParameter(self->acqva, quantity, gain, offset)) {
    raiseException_returnNULL(PyExc_AttributeError, "Could not add parameter");
  }

  Py_RETURN_NONE;
}

/**
 * Returns if the composite generator will acqva specified parameter
 * @param[in] self - self
 * @param[in] args - <quantity as a string>
 * @return True or False
 */
static PyObject* _pyacqva_hasParameter(PyAcqva* self, PyObject* args)
{
  char* quantity = NULL;
  if (!PyArg_ParseTuple(args, "s", &quantity)) {
    return NULL;
  }
  return PyBool_FromLong(Acqva_hasParameter(self->acqva, quantity));
}

/**
 * Returns the number of parameters this generator will process
 * @param[in] self - self
 * @param[in] args - N/A
 * @return The number of parameters
 */
static PyObject* _pyacqva_getParameterCount(PyAcqva* self, PyObject* args)
{
  return PyLong_FromLong(Acqva_getParameterCount(self->acqva));
}

/**
 * Returns the parameter at specified index.
 * @param[in] self - self
 * @param[in] args - <index as int>
 * @return A tuple containing (<quantity as string>,<gain as double>,<offset as double>)
 */
static PyObject* _pyacqva_getParameter(PyAcqva* self, PyObject* args)
{
  int i = 0;
  const char* quantity;
  double gain = 1.0, offset = 0.0;

  if (!PyArg_ParseTuple(args, "i", &i)) {
    return NULL;
  }
  quantity = Acqva_getParameter(self->acqva, i, &gain, &offset);
  if (quantity == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "No parameter at specified index");
  }

  return Py_BuildValue("(sdd)", quantity, gain, offset);
}

/**
 * Adds a transformable rave object to the composite generator. Currently,
 * only volumes are supported.
 * @param[in] self - self
 * @param[in] args - a rave object (currently only polar volumes)
 * @returns None on success, otherwise NULL
 */
static PyObject* _pyacqva_add(PyAcqva* self, PyObject* args)
{
  PyObject* obj = NULL;
  RaveCoreObject* rcobject = NULL;

  if(!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }

  if (PyPolarVolume_Check(obj)) {
    rcobject = (RaveCoreObject*)((PyPolarVolume*)obj)->pvol;
  } else if (PyPolarScan_Check(obj)) {
    rcobject = (RaveCoreObject*)((PyPolarScan*)obj)->scan;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "only supported objects are volumes and scans");
  }

  if (!Acqva_add(self->acqva, rcobject)) {
    raiseException_returnNULL(PyExc_MemoryError, "failed to add object to acqva composite generator");
  }

  Py_RETURN_NONE;
}

static PyObject* _pyacqva_applyRadarIndexMapping(PyAcqva* self, PyObject* args)
{
  PyObject* obj = NULL;
  PyObject* keys = NULL;
  PyObject* ko = NULL;
  PyObject* val = NULL;
  RaveObjectHashTable_t* mapping = NULL;
  RaveAttribute_t* attr = NULL;
  Py_ssize_t len = 0, i = 0;

  if (!PyArg_ParseTuple(args, "O", &obj)) {
    raiseException_returnNULL(PyExc_AttributeError, "Takes a mapping between source (string) and a radar index value");
  }

  if (!PyMapping_Check(obj)) {
    raiseException_returnNULL(PyExc_AttributeError, "Takes a mapping between source (string) and a radar index value");
  }
  keys = PyMapping_Keys(obj);
  if (keys == NULL) {
    raiseException_returnNULL(PyExc_AttributeError, "Could not get keys from mapping");
  }

  mapping = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  if (mapping == NULL) {
    raiseException_gotoTag(done, PyExc_MemoryError, "Failed to create native mapping");
  }

  len = PyList_Size(keys);
  for (i = 0; i < len; i++) {
    const char* key;
    ko = PyList_GetItem(keys, i); /* borrowed */
    key = PyString_AsString(ko);
    if (key != NULL) {
      val = PyMapping_GetItemString(obj, key);
      if (val != NULL && PyInt_Check(val)) {
        attr = RaveAttributeHelp_createLong(key, PyInt_AsLong(val));
        if (attr == NULL || !RaveObjectHashTable_put(mapping, key, (RaveCoreObject*)attr)) {
          raiseException_gotoTag(done, PyExc_MemoryError, "Failed to add native map value to mapping");
        }
        RAVE_OBJECT_RELEASE(attr);
      } else {
        raiseException_gotoTag(done, PyExc_AttributeError, "Takes a mapping between source (string) and a radar index value");
      }
      Py_DECREF(val);
    }
  }

  if (!Acqva_applyRadarIndexMapping(self->acqva, mapping)) {
    raiseException_gotoTag(done, PyExc_RuntimeError, "Failed to apply radar index mapping");
  }

  Py_DECREF(keys);
  RAVE_OBJECT_RELEASE(mapping);
  Py_RETURN_NONE;
done:
  Py_DECREF(keys);
  Py_DECREF(val);
  RAVE_OBJECT_RELEASE(attr);
  RAVE_OBJECT_RELEASE(mapping);
  return NULL;
}

/**
 * Generates a acqva composite according to the principles defined in the Acqva object.
 * @param[in] self - self
 * @param[in] args - an area object followed by a height
 * @returns a cartesian product on success, otherwise NULL
 */
static PyObject* _pyacqva_generate(PyAcqva* self, PyObject* args)
{
  PyObject* obj = NULL;
  Cartesian_t* result = NULL;
  PyObject* pyresult = NULL;
  PyObject* pyqualitynames = NULL;
  RaveList_t* qualitynames = NULL;

  if (!PyArg_ParseTuple(args, "O|O", &obj, &pyqualitynames)) {
    return NULL;
  }
  if (!PyArea_Check(obj)) {
    raiseException_returnNULL(PyExc_AttributeError, "argument should be an area");
  }
  if (pyqualitynames != NULL && !PyList_Check(pyqualitynames)) {
    raiseException_returnNULL(PyExc_AttributeError, "second argument should be a list of quality (how/task) names");
  }
  if (pyqualitynames != NULL && PyObject_Length(pyqualitynames) > 0) {
    Py_ssize_t nnames = PyObject_Length(pyqualitynames);
    Py_ssize_t i = 0;
    qualitynames = RAVE_OBJECT_NEW(&RaveList_TYPE);
    if (qualitynames == NULL) {
      raiseException_gotoTag(done, PyExc_MemoryError, "Could not allocate memory");
    }
    for (i = 0; i < nnames; i++) {
      PyObject* pystr = PyList_GetItem(pyqualitynames, i);
      char* dupstr = NULL;
      if (pystr == NULL || !PyString_Check(pystr)) {
        raiseException_gotoTag(done, PyExc_AttributeError, "second argument should be a list of quality (how/task) names (strings)");
      }
      dupstr = RAVE_STRDUP(PyString_AsString(pystr));
      if (dupstr != NULL) {
        if (!RaveList_add(qualitynames, dupstr)) {
          raiseException_gotoTag(done, PyExc_MemoryError, "Could not allocate memory");
          RAVE_FREE(dupstr);
        }
      } else {
        raiseException_gotoTag(done, PyExc_MemoryError, "Could not allocate memory");
      }
      dupstr = NULL; // We have handed it over to the rave list.
    }
  }

  result = Acqva_generate(self->acqva, ((PyArea*)obj)->area, qualitynames);
  if (result == NULL) {
    raiseException_gotoTag(done, PyExc_AttributeError, "failed to generate acqva");
  }

  pyresult = (PyObject*)PyCartesian_New(result);
done:
  RAVE_OBJECT_RELEASE(result);
  RaveList_freeAndDestroy(&qualitynames);
  return pyresult;
}

/**
 * All methods a cartesian product can have
 */
static struct PyMethodDef _pyacqva_methods[] =
{
  {"date", NULL, METH_VARARGS},
  {"time", NULL, METH_VARARGS},
  {"addParameter", (PyCFunction)_pyacqva_addParameter, 1,
    "addParameter(quantity, gain, offset)\n\n" // "sddd", &
    "Adds one parameter (quantity) that should be processed in the run.\n\n"
    "quantity   - the parameter quantity\n"
    "gain       - the gain to be used for the parameter\n"
    "offset     - the offset to be used for the parameter\n"
  },
  {"hasParameter", (PyCFunction)_pyacqva_hasParameter, 1,
    "hasParameter(quantity) -> boolean\n\n"
    "Returns if this acqva generator is going to process specified parameter\n\n"
    "quantity   - the parameter quantity"
  },
  {"getParameterCount", (PyCFunction)_pyacqva_getParameterCount, 1,
    "getParameterCount() -> integer\n\n"
    "Returns the number of parameters that are going to be processed."
  },
  {"getParameter", (PyCFunction)_pyacqva_getParameter, 1,
    "getParameter(index) -> (quantity, gain, offset)\n\n"
    "Returns information about the parameter at index. Returned value will be a tuple of quantity, gain and offset."
  },
  {"add", (PyCFunction) _pyacqva_add, 1,
    "add(object)\n\n"
    "Adds a polar scan or polar volume to the generator.\n\n"
    "object - A polar scan or a polar volume"
  },
  {"applyRadarIndexMapping", (PyCFunction)_pyacqva_applyRadarIndexMapping, 1,
    "applyRadarIndexMapping(mapping)\n\n"
    "If you want the objects included in the composite to have a specific index value when generating the quality\n"
    "field se.smhi.composite.index.radar, then you can provide a hash table that maps source with a RaveAttribute_t\n"
    "containing a long value. The source should be the full source as defined in the added objects. The indexes must\n"
    "be unique values, preferrably starting from 1. If there is a mapping missing, the default behaviour is to take\n"
    "first available integer closest to 1.\n"
    "Note, that in order to the mapping to take, this call must be performed after all the objects has been added to\n"
    "the generator and before calling \ref Composite_generate.\n\n"
    "mapping - A mapping between a source identifier and a radar index. For example:\n"
    "{\"WMO:1234\":1,\"NOD:sesome\":2}\n"
  },
  {"generate", (PyCFunction) _pyacqva_generate, 1,
    "generate(area[,qualityfields]) -> CartesianCore\n\n"
    "Generates a composite according to the configured parameters in the composite structure.\n\n"
    "area          - The AreaCore defining the area to be generated.\n"
    "qualityfields - An optional list of strings identifying how/task values in the quality fields of the polar data.\n"
    "                Each entry in this list will result in the atempt to generate a corresponding quality field\n"
    "                in the resulting cartesian product.\n"
    "Example:\n"
    " result = generator.generate(myarea, [\"se.smhi.composite.distance.radar\",\"pl.imgw.radvolqc.spike\"])"
  },
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the cartesian
 * @param[in] self - the cartesian product
 */
static PyObject* _pyacqva_getattro(PyAcqva* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("date", name) == 0) {
    if (Acqva_getDate(self->acqva) != NULL) {
      return PyString_FromString(Acqva_getDate(self->acqva));
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("time", name) == 0) {
    if (Acqva_getTime(self->acqva) != NULL) {
      return PyString_FromString(Acqva_getTime(self->acqva));
    } else {
      Py_RETURN_NONE;
    }
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pyacqva_setattro(PyAcqva* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("time", name) == 0) {
    if (PyString_Check(val)) {
      if (!Acqva_setTime(self->acqva, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "time must be in the format HHmmss");
      }
    } else if (val == Py_None) {
      Acqva_setTime(self->acqva, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"time must be of type string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("date", name) == 0) {
    if (PyString_Check(val)) {
      if (!Acqva_setDate(self->acqva, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "date must be in the format YYYYMMSS");
      }
    } else if (val == Py_None) {
      Acqva_setDate(self->acqva, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"date must be of type string");
    }
  }

  result = 0;
done:
  return result;
}

/*@} End of Acqva product generator */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyacqva_type_doc,
    "The acqva type provides the possibility to create cartesian composites from a number of polar objects using the ACQVA methodology.\n"
    "To generate the composite, one or many polar scans or polar volumes has to be added to the generator. Then generate should be called with the expected area and an optional list of how/task quality field names.\n"
    "There are a few attributes that can be set besides the functions.\n"
    " date                         - The nominal date as a string in format YYYYMMDD\n"
    " time                         - The nominal time as a string in format HHmmss\n"
    "\n"
    "Usage:\n"
    " import _pyacqva\n"
    " generator = _pyacqva.new()\n"
    " generator.date = \"20200201\"\n"
    " generator.date = \"100000\"\n"
    " generator.addParameter(\"DBZH\", 2.0, 3.0)\n"
    " generator.add(_rave.open(\"se1_pvol_20200201100000.h5\").object)\n"
    " generator.add(_rave.open(\"se2_pvol_20200201100000.h5\").object)\n"
    " generator.add(_rave.open(\"se3_pvol_20200201100000.h5\").object)\n"
    " result = generator.generate(myarea, [\"se.smhi.composite.distance.radar\",\"pl.imgw.radvolqc.spike\"])\n"
    );
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyAcqva_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "AcqvaCore", /*tp_name*/
  sizeof(PyAcqva), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyacqva_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pyacqva_getattro, /*tp_getattro*/
  (setattrofunc)_pyacqva_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyacqva_type_doc,        /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyacqva_methods,             /*tp_methods*/
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
  {"new", (PyCFunction)_pyacqva_new, 1,
    "new() -> new instance of the AcqvaCore object\n\n"
    "Creates a new instance of the AcqvaCore object"
  },
  {NULL,NULL} /*Sentinel*/
};

/**
 * Adds constants to the dictionary (probably the modules dictionary).
 * @param[in] dictionary - the dictionary the long should be added to
 * @param[in] name - the name of the constant
 * @param[in] value - the value
 */
 /* TEMP DISABLED
static void add_long_constant(PyObject* dictionary, const char* name, long value)
{
  PyObject* tmp = NULL;
  tmp = PyInt_FromLong(value);
  if (tmp != NULL) {
    PyDict_SetItemString(dictionary, name, tmp);
  }
  Py_XDECREF(tmp);
}
*/

MOD_INIT(_acqva)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyAcqva_API[PyAcqva_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyAcqva_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyAcqva_Type);

  MOD_INIT_DEF(module, "_acqva", _pyacqva_type_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyAcqva_API[PyAcqva_Type_NUM] = (void*)&PyAcqva_Type;
  PyAcqva_API[PyAcqva_GetNative_NUM] = (void *)PyAcqva_GetNative;
  PyAcqva_API[PyAcqva_New_NUM] = (void*)PyAcqva_New;

  c_api_object = PyCapsule_New(PyAcqva_API, PyAcqva_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_pyacqva.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _pyacqva.error");
    return MOD_INIT_ERROR;
  }


  import_pypolarvolume();
  import_pypolarscan();
  import_pycartesian();
  import_pyarea();
  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
