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
 * Python version of the FileObject API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-12-19
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"
#include "rave.h"
#define PYFILEOBJECT_MODULE    /**< to get correct part in pyfileobject.h */
#include "pyfileobject.h"
#include <arrayobject.h>
#include "pyravevalue.h"
#include "rave_alloc.h"
#include "pyravedata2d.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_fileobject");

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

/*@{ Gra */
/**
 * Returns the native FileObject_t instance.
 * @param[in] pyfobj - the python file object instance
 * @returns the native pia instance.
 */
static FileObject_t*
PyFileObject_GetNative(PyFileObject* pyfobj)
{
  RAVE_ASSERT((pyfobj != NULL), "pyfobj == NULL");
  return RAVE_OBJECT_COPY(pyfobj->fobj);
}

/**
 * Creates a python pia from a native pia or will create an
 * initial native pia if p is NULL.
 * @param[in] p - the native pia (or NULL)
 * @returns the python pia product.
 */
static PyFileObject*
PyFileObject_New(FileObject_t* p)
{
  PyFileObject* result = NULL;
  FileObject_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&FileObject_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for file object.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for file object.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyFileObject, &PyFileObject_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->fobj = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->fobj, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyFileObject instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyFileObject.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the pia
 * @param[in] obj the object to deallocate.
 */
static void _pyfileobject_dealloc(PyFileObject* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->fobj, obj);
  RAVE_OBJECT_RELEASE(obj->fobj);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the gra.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyfileobject_new(PyObject* self, PyObject* args)
{
  PyFileObject* result = PyFileObject_New(NULL);
  return (PyObject*)result;
}

/**
 * Creates the specified group / dataset in the file object
 * @param[in] self - self
 * @param[in] args - a name, either group name or a name specified as a directory structure (/.../.../...)
 * @return the file object
 */
static PyObject* _pyfileobject_create(PyFileObject* self, PyObject* args)
{
  PyObject *result = NULL;
  char* name = NULL;
  int otype = 0;
  FileObject_t* foundobj = NULL;

  if (!PyArg_ParseTuple(args, "s|i", &name, &otype)) {
    return NULL;
  }

  foundobj = FileObject_create(self->fobj, name);
  if (foundobj != NULL) {
    result = (PyObject*)PyFileObject_New(foundobj);
  } else {
    raiseException_returnNULL(PyExc_IOError, "Failure when executing create");
  }

  RAVE_OBJECT_RELEASE(foundobj);
  return result;
}

/**
 * Gets the specified group in the file object
 * @param[in] self - self
 * @param[in] args - a name, either group name or a name specified as a directory structure (/.../.../...)
 * @return the file object
 */
static PyObject* _pyfileobject_get(PyFileObject* self, PyObject* args)
{
  PyObject *result = NULL;
  PyObject* pyobj = NULL;
  FileObject_t* foundobj = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyobj)) {
    return NULL;
  }

  if (PyString_Check(pyobj)) {
    const char* name = PyString_AsString(pyobj);  
    foundobj = FileObject_get(self->fobj, name);
  } else if (PyInt_Check(pyobj) || PyLong_Check(pyobj)) {
    int index = (int)PyLong_AsLong(pyobj);
    if (index < 0 || index >= FileObject_numberOfGroups(self->fobj)) {
      raiseException_returnNULL(PyExc_IndexError, "Index out of bounds");  
    }
    foundobj = FileObject_getByIndex(self->fobj, index);
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "Unsupported key type");
  }

  if (foundobj != NULL) {
    result = (PyObject*)PyFileObject_New(foundobj);
  } else {
    raiseException_returnNULL(PyExc_IOError, "Failure when executing get");
  }

  RAVE_OBJECT_RELEASE(foundobj);
  return result;
}

/**
 * Returns if all names in the file object structure has valid names
 * @param[in] self - self
 * @param[in] args - N/A
 * @return a boolean
 */
static PyObject* _pyfileobject_areNamesSet(PyFileObject* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyBool_FromLong(FileObject_areNamesSet(self->fobj));
}

/**
 * Returns if this group is a dataset or not
 * @param[in] self - self
 * @param[in] args - N/A
 * @return a boolean
 */
static PyObject* _pyfileobject_isDataset(PyFileObject* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyBool_FromLong(FileObject_isDataset(self->fobj));
}

/**
 * Returns if this dataset is loaded or not, typically only relevant if using lazy loading
 * @param[in] self - self
 * @param[in] args - N/A
 * @return a boolean
 */
static PyObject* _pyfileobject_isDatasetLoaded(PyFileObject* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyBool_FromLong(FileObject_isDatasetLoaded(self->fobj));
}

/**
 * Implements the getitem so that it is possible to get items by using [...].
 */
static PyObject* _pyfileobject__getitem__(PyObject* _self, PyObject* args)
{
  PyFileObject* self = (PyFileObject*)_self;
  FileObject_t* foundobj = NULL;
  PyObject* result = NULL;
  if (PyString_Check(args)) {
    const char* name = PyString_AsString(args);  
    foundobj = FileObject_create(self->fobj, name);
  } else if (PyInt_Check(args) || PyLong_Check(args)) {
    int index = (int)PyLong_AsLong(args);
    foundobj = FileObject_getByIndex(self->fobj, index);
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "Unsupported key type\n");
  }
  if (foundobj != NULL) {
    result = (PyObject*)PyFileObject_New(foundobj);
  } else {
    raiseException_returnNULL(PyExc_IOError, "Failure when executing __getitem__");
  }

  RAVE_OBJECT_RELEASE(foundobj);
  return result;
}

/**
 * Returns number of sub-groups this object has, if for example getting size
 */
static long _pyfileobject_len(PyObject *self) {
  return FileObject_numberOfGroups(((PyFileObject*)self)->fobj);
}

/**
 * Returns if the specified group exists
 */
static int _pyfileobject__contains__(PyObject *self, PyObject *key) {
  if (PyString_Check(key)) {
    const char* name = PyString_AsString(key);
    return FileObject_exists(((PyFileObject*)self)->fobj, name);
  }

  return 0;
}

/**
 * String representation of the file object
 */

static PyObject* _pyfileobject__str__(PyObject* _self)
{
  PyFileObject* self = (PyFileObject*)_self;
  char* pstr = FileObject_toString(self->fobj);
  if (pstr != NULL) {
    PyObject* result = PyString_FromString(pstr);
    RAVE_FREE(pstr);
    return result;
  }
  Py_RETURN_NONE;
}

/**
 * Mapping methods required to handle tp_as_mapping
 */
static PyMappingMethods _pyfileobject_mappingmethods = {
    .mp_length = (lenfunc)_pyfileobject_len,
    .mp_subscript = _pyfileobject__getitem__
};

/* Hack to implement "key in dict" */
static PySequenceMethods _pyfileobject_sequencemethods = {
    .sq_length = _pyfileobject_len,
    .sq_concat = 0,
    .sq_repeat = 0,
    .sq_item = 0,
    .sq_ass_item = 0,
    .sq_contains = _pyfileobject__contains__,
    .sq_inplace_concat = 0,
    .sq_inplace_repeat = 0,
};

/**
 * All methods a file object can have
 */
static struct PyMethodDef _pyfileobject_methods[] =
{
  {"__getitem__", (PyCFunction) _pyfileobject__getitem__, METH_O | METH_COEXIST},
  {"name", NULL, METH_VARARGS},
  {"restriction_mode", NULL, METH_VARARGS},
  {"attributes", NULL, METH_VARARGS},
  {"groups", NULL, METH_VARARGS},
  {"numberOfGroups", NULL, METH_VARARGS},
  {"data", NULL, METH_VARARGS},
  {"xsize", NULL, METH_VARARGS},
  {"ysize", NULL, METH_VARARGS},
  {"datatype", NULL, METH_VARARGS},
  {"create", (PyCFunction) _pyfileobject_create, 1,
    "create(name) -> file object\n\n"
  },
  {"get", (PyCFunction) _pyfileobject_get, 1,
    "get(name) -> file object\n\n"
  },
  {"areNamesSet", (PyCFunction) _pyfileobject_areNamesSet, 1,
    "areNamesSet() -> boolean\n\n"
  },
  {"isDataset", (PyCFunction) _pyfileobject_isDataset, 1,
    "isDataset() -> boolean\n\n"
  },
  {"isDatasetLoaded", (PyCFunction) _pyfileobject_isDatasetLoaded, 1,
    "isDatasetLoaded() -> boolean\n\n"
  },

  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the gra
 * @param[in] self - the gra
 */
static PyObject* _pyfileobject_getattro(PyFileObject* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("name", name) == 0) {
    return PyString_FromString(FileObject_getName(self->fobj));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("restriction_mode", name) == 0) {
    return PyLong_FromLong(FileObject_getRestrictionMode(self->fobj));
    return PyString_FromString(FileObject_getName(self->fobj));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("attributes", name) == 0) {
    RaveValue_t* attrs = FileObject_attributes(self->fobj);
    PyObject* result = (PyObject*)PyRaveValue_New(attrs);
    RAVE_OBJECT_RELEASE(attrs);
    return result;
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("groups", name) == 0) {
    RaveObjectList_t* groups = FileObject_groups(self->fobj);
    PyObject* result = NULL;
    int i = 0, nobjs = 0;
    if (groups != NULL) {
      nobjs = RaveObjectList_size(groups);
      result = PyList_New(0);
      for (i = 0; result != NULL && i < nobjs; i++) {
        FileObject_t* fobj = (FileObject_t*)RaveObjectList_get(groups, i);
        if (fobj != NULL) {
          PyObject* pyval = (PyObject*)PyFileObject_New(fobj);
          if (pyval == NULL || PyList_Append(result, pyval) != 0) {
            Py_XDECREF(result);
            result = NULL;
          }
          Py_XDECREF(pyval);
        }
        RAVE_OBJECT_RELEASE(fobj);
      }
    }
    RAVE_OBJECT_RELEASE(groups);
    return result;
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("numberOfGroups", name) == 0) {
    return PyLong_FromLong(FileObject_numberOfGroups(self->fobj));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("data", name) == 0) {
    RaveData2D_t* d2d = FileObject_getData(self->fobj);
    if (d2d != NULL) {
      PyObject* result = (PyObject*)PyRaveData2D_New(d2d);
      RAVE_OBJECT_RELEASE(d2d);
      return result;
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("xsize", name) == 0) {
    if (FileObject_isDataset(self->fobj)) {
      return PyLong_FromLong(FileObject_getDatasetX(self->fobj));
    } else {
      raiseException_returnNULL(PyExc_AttributeError, "File object is not a dataset");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("ysize", name) == 0) {
    if (FileObject_isDataset(self->fobj)) {
      return PyLong_FromLong(FileObject_getDatasetY(self->fobj));
    } else {
      raiseException_returnNULL(PyExc_AttributeError, "File object is not a dataset");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("datatype", name) == 0) {
    if (FileObject_isDataset(self->fobj)) {
      return PyLong_FromLong(FileObject_getDatasetType(self->fobj));
    } else {
      raiseException_returnNULL(PyExc_AttributeError, "File object is not a dataset");
    }
  }

  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the gra
 */
static int _pyfileobject_setattro(PyFileObject* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("name", name) == 0) {
    if (PyString_Check(val)) {
      if (!FileObject_setName(self->fobj, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_RuntimeError, "Could not set name");
      }
    } else {
      raiseException_gotoTag(done, PyExc_ValueError, "Names can only be strings");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("data", name) == 0) {
    if (PyArray_Check(val)) {
      PyArrayObject* arraydata = (PyArrayObject*)val;
      RaveDataType datatype = RaveDataType_UNDEFINED;
      RaveData2D_t* d2d = NULL;
      long xsize = 0;
      long ysize = 0;
      unsigned char* data = NULL;
      if (PyArray_NDIM(arraydata) != 2) {
        raiseException_gotoTag(done, PyExc_ValueError, "A dataset must be of rank 2");
      }
      datatype = translate_pyarraytype_to_ravetype(PyArray_TYPE(arraydata));

      if (PyArray_ITEMSIZE(arraydata) != get_ravetype_size(datatype)) {
        raiseException_gotoTag(done, PyExc_TypeError, "numpy and rave does not have same data sizes");
      }

      xsize  = PyArray_DIM(arraydata, 1);
      ysize  = PyArray_DIM(arraydata, 0);
      data   = PyArray_DATA(arraydata);

      d2d = RAVE_OBJECT_NEW(&RaveData2D_TYPE);
      if (d2d == NULL || !RaveData2D_setData(d2d, xsize, ysize, data, datatype) || !FileObject_setData(self->fobj, d2d)) {
        RAVE_OBJECT_RELEASE(d2d);
        raiseException_gotoTag(done, PyExc_RuntimeError, "Failed to create data2d");
      }
      RAVE_OBJECT_RELEASE(d2d);
    } else if (PyRaveData2D_Check(val)) {
      if (!FileObject_setData(self->fobj, ((PyRaveData2D*)val)->field)) {
        raiseException_gotoTag(done, PyExc_RuntimeError, "Failed to set data");
      }
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("restriction_mode", name) == 0) {
    FileObjectRestrictionMode rmode = FileObjectRestrictionMode_NONE;
    if (PyInt_Check(val)) {
      rmode = PyInt_AsLong(val);
    } else if (PyLong_Check(val)) {
      rmode = PyLong_AsLong(val);
    } else {
      raiseException_gotoTag(done, PyExc_RuntimeError, "Restriction_mode must be inside RestrictionMode range");
    }
    if (rmode >= FileObjectRestrictionMode_NONE && rmode <= FileObjectRestrictionMode_ODIM) {
      FileObject_setRestrictionMode(self->fobj, rmode);
    } else {
      raiseException_gotoTag(done, PyExc_RuntimeError, "Restriction_mode must be inside RestrictionMode range");
    }
  }

  result = 0;
done:
  return result;
}

/*@} End of Gra */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyfileobject_module_doc,
    "Performs the path integrated attenuation according to the Hitchfeld-Bordan method\n\n"
  "Usage:\n"
  " import _pia\n"
  " pia = _pia.new()\n"
  " scan = _raveio.open(\"qcvol.h5\").object.getScan(0)\n"
  " pia.process(scan, \"DBZH\", True, True, True) # Booleans are addparam, reprocessquality, apply\n\n"
);
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyFileObject_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "FileObjectCore", /*tp_name*/
  sizeof(PyFileObject), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyfileobject_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)0,               /*tp_getattr*/
  (setattrfunc)0,               /*tp_setattr*/
  0,                            /*tp_compare*/
  0,                            /*tp_repr*/
  0,                            /*tp_as_number */
  &_pyfileobject_sequencemethods,/*tp_sequence */
  &_pyfileobject_mappingmethods,/*tp_as_mapping */
  0,                            /*tp_hash*/
  (ternaryfunc)0,               /*tp_call*/
  (reprfunc)&_pyfileobject__str__, /*tp_str*/
  (getattrofunc)_pyfileobject_getattro, /*tp_getattro*/
  (setattrofunc)_pyfileobject_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyfileobject_module_doc,            /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyfileobject_methods,              /*tp_methods*/
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
  {"new", (PyCFunction)_pyfileobject_new, 1,
    "new() -> new instance of the FileObjectCore object\n\n"
    "Creates a new instance of the FileObjectCore object"
  },
  {NULL,NULL} /*Sentinel*/
};

/**
 * Adds constants to the dictionary (probably the modules dictionary).
 * @param[in] dictionary - the dictionary the long should be added to
 * @param[in] name - the name of the constant
 * @param[in] value - the value
 */
static void add_long_constant(PyObject* dictionary, const char* name, long value)
{
  PyObject* tmp = NULL;
  tmp = PyInt_FromLong(value);
  if (tmp != NULL) {
    PyDict_SetItemString(dictionary, name, tmp);
  }
  Py_XDECREF(tmp);
}

MOD_INIT(_fileobject)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyFileObject_API[PyFileObject_API_pointers];
  PyObject *c_api_object = NULL;
  MOD_INIT_SETUP_TYPE(PyFileObject_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyFileObject_Type);

  MOD_INIT_DEF(module, "_fileobject", _pyfileobject_module_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyFileObject_API[PyFileObject_Type_NUM] = (void*)&PyFileObject_Type;
  PyFileObject_API[PyFileObject_GetNative_NUM] = (void *)PyFileObject_GetNative;
  PyFileObject_API[PyFileObject_New_NUM] = (void*)PyFileObject_New;

  c_api_object = PyCapsule_New(PyFileObject_API, PyFileObject_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_fileobject.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _fileobject.error");
    return MOD_INIT_ERROR;
  }

  add_long_constant(dictionary, "NONE", FileObjectRestrictionMode_NONE);
  add_long_constant(dictionary, "ODIM", FileObjectRestrictionMode_ODIM);

  import_array(); /*To make sure I get access to Numeric*/
  import_ravevalue();
  import_ravedata2d();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
