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
 * Python version of the RaveIOCache API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-02-05
 */
#include "pyravecompat.h"
#include "rave_iocache.h"
#include "rave_object.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYRAVEIOCACHE_MODULE   /**< include correct part of pyiocache.h */
#include "pyiocache.h"

#include "pyravefield.h"
#include "pyrave_debug.h"
#include "rave_alloc.h"
#include "hlhdf.h"
#include "hlhdf_debug.h"

/**
 * Name of the module debugged.
 */
PYRAVE_DEBUG_MODULE("_iocache");

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

/*@{ RaveIOCache products */
/**
 * Returns the native RaveIOCache_t instance.
 * @param[in] iocache - the python iocache instance
 * @returns the native iocache instance.
 */
static RaveIOCache_t*
PyRaveIOCache_GetNative(PyRaveIOCache* iocache)
{
  RAVE_ASSERT((iocache != NULL), "iocache == NULL");
  return RAVE_OBJECT_COPY(iocache->iocache);
}

/**
 * Creates a python raveiocache from a native raveiocache or will create an
 * initial native raveiocache if p is NULL.
 * @param[in] p - the native raveiocache (or NULL)
 * @returns the python raveiocache product.
 */
static PyRaveIOCache*
PyRaveIOCache_New(RaveIOCache_t* p)
{
  PyRaveIOCache* result = NULL;
  RaveIOCache_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RaveIOCache_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for raveiocache.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for raveiocache.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyRaveIOCache, &PyRaveIOCache_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->iocache = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->iocache, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyRaveIOCache instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for raveiocache.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the RaveIO
 * @param[in] obj the object to deallocate.
 */
static void _pyraveiocache_dealloc(PyRaveIOCache* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->iocache, obj);
  RAVE_OBJECT_RELEASE(obj->iocache);
  PyObject_Del(obj);
}

/**
 * Creates a new RaveIO instance.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (not USED)
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyraveiocache_new(PyObject* self, PyObject* args)
{
  PyRaveIOCache* result = PyRaveIOCache_New(NULL);
  return (PyObject*)result;
}

/**
 * Loads a field from a HDF5 file. Assumes that the format is
 * /fieldX/data
 * /fieldX/how, /fieldX/what, /fieldX/where
 * @param[in] self this instance.
 * @param[in] args arguments for creation (not USED)
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyraveiocache_loadField(PyObject* self, PyObject* args)
{
  char* filename = NULL;
  char* fieldname = NULL;
  RaveIOCache_t* iocache = NULL;
  RaveField_t* result = NULL;
  PyRaveField* pyresult = NULL;

  if (!PyArg_ParseTuple(args, "s|s", &filename, &fieldname)) {
    return NULL;
  }

  iocache = RAVE_OBJECT_NEW(&RaveIOCache_TYPE);
  if (iocache == NULL) {
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for io cache");
  }

  result = RaveIOCache_loadField(iocache, filename, fieldname);
  if (result == NULL) {
    raiseException_gotoTag(done, PyExc_IOError, "Could not read file");
  }

  pyresult = PyRaveField_New(result);

done:
  RAVE_OBJECT_RELEASE(iocache);
  RAVE_OBJECT_RELEASE(result);
  return (PyObject*)pyresult;
}

/**
 * Loads a field from a HDF5 file. Assumes that the format is
 * /fieldX/data
 * /fieldX/how, /fieldX/what, /fieldX/where
 * @param[in] self this instance.
 * @param[in] args arguments for creation (not USED)
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyraveiocache_saveField(PyObject* self, PyObject* args)
{
  char* filename = NULL;
  RaveIOCache_t* iocache = NULL;
  PyRaveField* pyfield = NULL;
  PyObject* pyobj = NULL;

  int result = 0;

  if (!PyArg_ParseTuple(args, "Os", &pyobj, &filename)) {
    return NULL;
  }

  if (!PyRaveField_Check(pyobj)) {
    raiseException_returnNULL(PyExc_TypeError,"Added object must be of type RaveFieldCore");
  }

  pyfield = (PyRaveField*)pyobj;

  iocache = RAVE_OBJECT_NEW(&RaveIOCache_TYPE);
  if (iocache == NULL) {
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for io cache");
  }

  result = RaveIOCache_saveField(iocache, pyfield->field, filename);

  if (result == 0) {
    raiseException_gotoTag(done, PyExc_IOError, "Could not save file");
  }

done:
  RAVE_OBJECT_RELEASE(iocache);
  Py_RETURN_NONE;
}

/**
 * All methods a RaveIOCache can have
 */
static struct PyMethodDef _pyraveiocache_methods[] =
{
  {"compression_level", NULL, METH_VARARGS},
  {"fcp_userblock", NULL, METH_VARARGS},
  {"fcp_sizes", NULL, METH_VARARGS},
  {"fcp_symk", NULL, METH_VARARGS},
  {"fcp_istorek", NULL, METH_VARARGS},
  {"fcp_metablocksize", NULL, METH_VARARGS},
  {"file_format", NULL, METH_VARARGS},
  {"error_message", NULL, METH_VARARGS},
  // {"load", (PyCFunction) _pyraveio_load, 1,   "load()\n\n"
  //                                             "Atempts to load the file that is defined by filename\n"},
  // {"save", (PyCFunction) _pyraveio_save, 1,   "save([filename)]\n\n"
  //                                             "Saves the current object (with current settings).\n\n"
  //                                             "filename - is optional. If not specified, the objects filename is used\n"},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the PyRaveIO
 * @param[in] self - the RaveIO instance
 */
static PyObject* _pyraveiocache_getattro(PyRaveIOCache* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("compression_level", name) == 0) {
    return PyInt_FromLong(RaveIOCache_getCompressionLevel(self->iocache));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_userblock", name) == 0) {
    return PyInt_FromLong(RaveIOCache_getUserBlock(self->iocache));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_sizes", name) == 0) {
    size_t sz = 0, addr = 0;
    RaveIOCache_getSizes(self->iocache, &sz, &addr);
    return Py_BuildValue("(ii)", sz, addr);
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_symk", name) == 0) {
    int ik = 0, lk = 0;
    RaveIOCache_getSymk(self->iocache, &ik, &lk);
    return Py_BuildValue("(ii)", ik, lk);
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_istorek", name) == 0) {
    return PyInt_FromLong(RaveIOCache_getIStoreK(self->iocache));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_metablocksize", name) == 0) {
    return PyInt_FromLong(RaveIOCache_getMetaBlockSize(self->iocache));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("error_message", name) == 0) {
    if (RaveIOCache_getErrorMessage(self->iocache) != NULL) {
      return PyString_FromString(RaveIOCache_getErrorMessage(self->iocache));
    } else {
      Py_RETURN_NONE;
    }
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the specified attribute in the raveio
 */
static int _pyraveiocache_setattro(PyRaveIOCache* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("compression_level", name) == 0) {
    if (PyInt_Check(val)) {
      RaveIOCache_setCompressionLevel(self->iocache, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "Compression level should be integer value between 0..9");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_userblock", name) == 0) {
    if (PyInt_Check(val)) {
      RaveIOCache_setUserBlock(self->iocache, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "User block should be integer value");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_sizes", name) == 0) {
    int sz = 0, addr = 0;
    if (!PyArg_ParseTuple(val, "ii", &sz, &addr)) {
      raiseException_gotoTag(done, PyExc_TypeError ,"sizes must be a tuple containing 2 integers representing (size, addr)");
    }
    RaveIOCache_setSizes(self->iocache, sz, addr);
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_symk", name) == 0) {
    int ik = 0, lk = 0;
    if (!PyArg_ParseTuple(val, "ii", &ik, &lk)) {
      raiseException_gotoTag(done, PyExc_TypeError ,"symk must be a tuple containing 2 integers representing (ik, lk)");
    }
    RaveIOCache_setSymk(self->iocache, ik, lk);
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_istorek", name) == 0) {
    if (PyInt_Check(val)) {
      RaveIOCache_setIStoreK(self->iocache, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError ,"istorek must be a integer");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_metablocksize", name) == 0) {
    if (PyInt_Check(val)) {
      RaveIOCache_setMetaBlockSize(self->iocache, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError ,"meta block size must be a integer");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }

  result = 0;
done:
  return result;
}

/*@} End of RaveIO */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyraveiocache_doc,
    "This instance wraps the IO-routines used when writing and/or reading static fields using RAVE.\n"
    "\n"
    " * compression_level- The compression level beeing used. Range between 0 and 9 where 0 means no compression and 9 means highest compression.\n"
    "                      Compression level 1 is lowest compression ratio but fastest and level 9 is highest compression ratio but slowest.\n "
    "\n"
    "The below fcp_<members> are all used for optimizing the file storage. Please refer to HDF5 documentation for more information.\n"
    " * fcp_userblock    - Integer value."
    "\n"
    " * fcp_sizes        - Sizes must be a tuple containing 2 integers representing (size, addr).\n"
    "\n"
    " * fcp_symk         - Symk must be a tuple containing 2 integers representing (ik, lk).\n"
    "\n"
    " * fcp_istorek      - Integer value.\n"
    "\n"
    " * fcp_metablocksize- Integer value.\n"
    "\n"
    );
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyRaveIOCache_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "RaveIOCacheCore", /*tp_name*/
  sizeof(PyRaveIOCache), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyraveiocache_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pyraveiocache_getattro, /*tp_getattro*/
  (setattrofunc)_pyraveiocache_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyraveiocache_doc,                /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyraveiocache_methods,              /*tp_methods*/
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
  {"new", (PyCFunction)_pyraveiocache_new, 1,
      "new() -> new instance of the RaveIOCacheCore object\n\n"
      "Creates a new instance of the RaveIOCacheCore object"},
  {"loadField", (PyCFunction)_pyraveiocache_loadField, 1,
      "loadField(filename[,fieldname]) -> loads a rave field from a hdf5 file\n\n"},
  {"saveField", (PyCFunction)_pyraveiocache_saveField, 1,
      "saveField(filename) -> saves a rave field in file under /fieldX\n\n"},
  {NULL,NULL} /*Sentinel*/
};
/**
 * Adds constants to the dictionary (probably the modules dictionary).
 * @param[in] dictionary - the dictionary the long should be added to
 * @param[in] name - the name of the constant
 * @param[in] value - the value
 */
 /* Temporary disabled
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

PyDoc_STRVAR(_pyraveiocache_module_doc,
    "This class provides functionality for reading and writing cache files used within RAVE.\n"
    "\n"
    "This documentation will only provide information about H5 since this is the format used for static data fields in rave.\n"
    "\n"
    "To read a cache-file:\n"
    ">>> import _iocache\n"
    ">>> obj = _iocache.loadField(\"/var/cache/baltrad/clutter/seang.h5\")\n"
    "\n"
    );

MOD_INIT(_iocache)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyRaveIOCache_API[PyRaveIOCache_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyRaveIOCache_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyRaveIOCache_Type);

  MOD_INIT_DEF(module, "_iocache", _pyraveiocache_module_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyRaveIOCache_API[PyRaveIOCache_Type_NUM] = (void*)&PyRaveIOCache_Type;
  PyRaveIOCache_API[PyRaveIOCache_GetNative_NUM] = (void *)PyRaveIOCache_GetNative;
  PyRaveIOCache_API[PyRaveIOCache_New_NUM] = (void*)PyRaveIOCache_New;

  c_api_object = PyCapsule_New(PyRaveIOCache_API, PyRaveIOCache_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_iocache.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _iocache.error");
    return MOD_INIT_ERROR;
  }


  HL_init();
  HL_disableErrorReporting();
  HL_disableHdf5ErrorReporting();
  HL_setDebugLevel(HLHDF_SILENT);
  import_pyravefield();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
