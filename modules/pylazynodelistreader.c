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
 * Python version of the RaveIO API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYLAZYNODELISTREADER_MODULE   /**< include correct part of pylazynodelistio.h */
#include <pylazynodelistreader.h>

#include <arrayobject.h>
#include "pyravedata2d.h"
#include "pyrave_debug.h"
#include "rave_alloc.h"
#include "hlhdf.h"
#include "hlhdf_debug.h"

/**
 * Name of the module debugged.
 */
PYRAVE_DEBUG_MODULE("_lazynodelistreader");

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

/*@{ LazyNodeListReader */
/**
 * Returns the native LazyNodeListReader_t instance.
 * @param[in] pylazynodelistreader - the python lazynodelist reader instance
 * @returns the native lazynodelist reader instance.
 */
static LazyNodeListReader_t*
PyLazyNodeListReader_GetNative(PyLazyNodeListReader* pylazynodelistreader)
{
  RAVE_ASSERT((pylazynodelistreader != NULL), "pylazynodelistreader == NULL");
  return RAVE_OBJECT_COPY(pylazynodelistreader->reader);
}

/**
 * Creates a python lazynodelistreader from a native lazynodelistreader or will create an
 * initial native lazynodelistreader if p is NULL.
 * @param[in] p - the native lazynodelistreader (or NULL)
 * @returns the python lazynodelistreader.
 */
static PyLazyNodeListReader*
PyLazyNodeListReader_New(LazyNodeListReader_t* p)
{
  PyLazyNodeListReader* result = NULL;
  LazyNodeListReader_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&LazyNodeListReader_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for lazynodelistio.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for lazynodelistio.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyLazyNodeListReader, &PyLazyNodeListReader_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->reader = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->reader, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyLazyNodeListReader instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for lazynodelistreader.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Read a file that is supported by LazyNodeListReader.
 * @param[in] self this instance.
 * @param[in] args arguments for creation. (A string identifying the file)
 * @return the object on success, otherwise NULL
 */
static PyLazyNodeListReader*
PyLazyNodeListReader_Read(const char* filename)
{
  LazyNodeListReader_t* reader = NULL;
  PyLazyNodeListReader* result = NULL;

  if (filename == NULL) {
    raiseException_returnNULL(PyExc_ValueError, "providing a filename that is NULL");
  }

  reader = LazyNodeListReader_read(filename);
  if (reader == NULL) {
    raiseException_gotoTag(done, PyExc_IOError, "Failed to read file");
  }
  result = PyLazyNodeListReader_New(reader);

done:
  RAVE_OBJECT_RELEASE(reader);
  return result;
}

/**
 * Deallocates the LazyNodeListReader
 * @param[in] obj the object to deallocate.
 */
static void _pylazynodelistreader_dealloc(PyLazyNodeListReader* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->reader, obj);
  RAVE_OBJECT_RELEASE(obj->reader);
  PyObject_Del(obj);
}

/**
 * Creates a new LazyNodeListReader instance.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (not USED)
 * @return the object on success, otherwise NULL
 */
static PyObject* _pylazynodelistreader_new(PyObject* self, PyObject* args)
{
  PyLazyNodeListReader* result = PyLazyNodeListReader_New(NULL);
  return (PyObject*)result;
}

/**
 * Reads a HDF file into the lazy nodelistreader instance
 * @param[in] self this instance.
 * @param[in] args arguments for creation (filename as a string)
 * @return the object on success, otherwise NULL
 */
static PyObject* _pylazynodelistreader_read(PyObject* self, PyObject* args)
{
  PyLazyNodeListReader* result = NULL;

  char* filename = NULL;
  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }
  result = PyLazyNodeListReader_Read(filename);
  return (PyObject*)result;
}

static PyObject* _pylazynodelistreader_preload(PyLazyNodeListReader* self, PyObject* args)
{
  char* quantities = NULL;
  if (!PyArg_ParseTuple(args, "|s", &quantities)) {
    return NULL;
  }
  if (!LazyNodeListReader_preloadQuantities(self->reader, quantities)) {
    raiseException_returnNULL(PyExc_RuntimeError, "Could not fetch datasets");
  }
  Py_RETURN_NONE;
}

static PyObject* _pylazynodelistreader_getDataset(PyLazyNodeListReader* self, PyObject* args)
{
  PyObject* result = NULL;
  char* nodename = NULL;
  if (!PyArg_ParseTuple(args, "s", &nodename)) {
    return NULL;
  }
  RaveData2D_t* data2d = LazyNodeListReader_getDataset(self->reader, nodename);
  if (data2d != NULL) {
    result = (PyObject*)PyRaveData2D_New(data2d);
  } else {
    raiseException_returnNULL(PyExc_IndexError, "Could not aquire dataset (dataset)");
  }
  RAVE_OBJECT_RELEASE(data2d);
  return result;
}

static PyObject* _pylazynodelistreader_getAttribute(PyLazyNodeListReader* self, PyObject* args)
{
  RaveAttribute_t* attribute = NULL;
  char* name = NULL;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }

  attribute = LazyNodeListReader_getAttribute(self->reader, name);
  if (attribute != NULL) {
    RaveAttribute_Format format = RaveAttribute_getFormat(attribute);
    if (format == RaveAttribute_Format_Long) {
      long value = 0;
      RaveAttribute_getLong(attribute, &value);
      result = PyLong_FromLong(value);
    } else if (format == RaveAttribute_Format_Double) {
      double value = 0.0;
      RaveAttribute_getDouble(attribute, &value);
      result = PyFloat_FromDouble(value);
    } else if (format == RaveAttribute_Format_String) {
      char* value = NULL;
      RaveAttribute_getString(attribute, &value);
      result = PyString_FromString(value);
    } else if (format == RaveAttribute_Format_LongArray) {
      long* value = NULL;
      int len = 0;
      int i = 0;
      npy_intp dims[1];
      RaveAttribute_getLongArray(attribute, &value, &len);
      dims[0] = len;
      result = PyArray_SimpleNew(1, dims, PyArray_LONG);
      for (i = 0; i < len; i++) {
        *((long*) PyArray_GETPTR1(result, i)) = value[i];
      }
    } else if (format == RaveAttribute_Format_DoubleArray) {
      double* value = NULL;
      int len = 0;
      int i = 0;
      npy_intp dims[1];
      RaveAttribute_getDoubleArray(attribute, &value, &len);
      dims[0] = len;
      result = PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
      for (i = 0; i < len; i++) {
        *((double*) PyArray_GETPTR1(result, i)) = value[i];
      }
    } else {
      RAVE_CRITICAL1("Undefined format on requested attribute %s", name);
      raiseException_gotoTag(done, PyExc_AttributeError, "Undefined attribute");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, "No such attribute");
  }
done:
  RAVE_OBJECT_RELEASE(attribute);
  return result;
}


static PyObject* _pylazynodelistreader_getNodeNames(PyLazyNodeListReader* self, PyObject* args)
{
  RaveList_t* nodenames = NULL;
  PyObject* result = NULL;
  int nnames = 0;
  int i = 0;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  nodenames = LazyNodeListReader_getNodeNames(self->reader);
  if (nodenames == NULL) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not get names");
  }

  nnames = RaveList_size(nodenames);
  result = PyList_New(0);
  for (i = 0; result != NULL && i < nnames; i++) {
    char* name = (char*)RaveList_get(nodenames, i);
    if (name != NULL) {
      PyObject* pynamestr = PyString_FromString(name);
      if (pynamestr == NULL) {
        goto fail;
      }
      if (PyList_Append(result, pynamestr) != 0) {
        Py_DECREF(pynamestr);
        goto fail;
      }
      Py_DECREF(pynamestr);
    }
  }
  RaveList_freeAndDestroy(&nodenames);
  return result;
fail:
  RaveList_freeAndDestroy(&nodenames);
  Py_XDECREF(result);
  return NULL;
}

static PyObject* _pylazynodelistreader_isLoaded(PyLazyNodeListReader* self, PyObject* args)
{
  char* nodename = NULL;
  if (!PyArg_ParseTuple(args, "s", &nodename))
    return NULL;
  return PyBool_FromLong(LazyNodeListReader_isLoaded(self->reader, nodename));
}

static PyObject* _pylazynodelistreader_exists(PyLazyNodeListReader* self, PyObject* args)
{
  char* nodename = NULL;
  if (!PyArg_ParseTuple(args, "s", &nodename))
    return NULL;
  return PyBool_FromLong(LazyNodeListReader_exists(self->reader, nodename));
}

/**
 * All methods a RaveIO can have
 */
static struct PyMethodDef _pylazynodelistreader_methods[] =
{
    {"preload", (PyCFunction) _pylazynodelistreader_preload, 1,
        "preload()\n\n"
        "Preloads all datasets immediately. This is useful if a lot of different datasets should be loaded and read.\n"},
  {"getDataset", (PyCFunction) _pylazynodelistreader_getDataset, 1,
      "getRaveData2D(nodename) -> rave data 2d field\n\n"
      "Returns the dataset associated with the nodename\n"},
  {"getAttribute", (PyCFunction) _pylazynodelistreader_getAttribute, 1,
      "getAttribute(nodename) -> value\n\n"
      "Returns the attribute value associated with the nodename\n"},
  {"getNodeNames", (PyCFunction) _pylazynodelistreader_getNodeNames, 1,
      "getNodeNames() -> a list of node names\n\n"},
  {"isLoaded", (PyCFunction) _pylazynodelistreader_isLoaded, 1,
      "isLoaded(nodename) -> a boolean \n\n"
      "if node has been fetched into memory or not\n\n"},
  {"exists", (PyCFunction) _pylazynodelistreader_exists, 1,
      "exists(nodename) -> a boolean \n\n"
      "if node exists in the node list or not\n\n"},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the PyLazyNodeListReader
 * @param[in] self - the PyLazyNodeListReader instance
 */
static PyObject* _pylazynodelistreader_getattro(PyLazyNodeListReader* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("filename", name) == 0) {
    return PyString_FromString("KALLE.txt");
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the specified attribute in the lazy nodelist reader
 */
static int _pylazynodelistreader_setattro(PyLazyNodeListReader* self, PyObject* name, PyObject* val)
{
  return -1;
  /*
  int result = -1;
  if (name == NULL) {
    goto done;
  }
done:
  return result;
  */
}

/*@} End of RaveIO */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pylazynodelistreader_doc,
    "This instance wraps the Reader-routines used when for reading data through the lazy nodelist reader.\n"
    "\n"
    " * read()\n"
    );
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyLazyNodeListReader_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "LazyNodeListReaderCore", /*tp_name*/
  sizeof(PyLazyNodeListReader), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pylazynodelistreader_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pylazynodelistreader_getattro, /*tp_getattro*/
  (setattrofunc)_pylazynodelistreader_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pylazynodelistreader_doc,                /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pylazynodelistreader_methods,              /*tp_methods*/
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
  {"new", (PyCFunction)_pylazynodelistreader_new, 1,
      "new() -> new instance of the RaveIOCore object\n\n"
      "Creates a new instance of the RaveIOCore object"},
  {"read", (PyCFunction)_pylazynodelistreader_read, 1,
      "open(filename) -> a LazyNodeListReaderCore instance with a loaded object.\n\n"
      "Reads a file that is supported by lazynodelistreader and loads the structure.\n\n"
      "filename - a HDF5 file that is readable."},
  {NULL,NULL} /*Sentinel*/
};
/**
 * Adds constants to the dictionary (probably the modules dictionary).
 * @param[in] dictionary - the dictionary the long should be added to
 * @param[in] name - the name of the constant
 * @param[in] value - the value
 */
/*
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

PyDoc_STRVAR(_pylazynodelistio_module_doc,
    "This class provides functionality for reading HDF5 files without reading all datasets.\n"
    "Initially it will only read metadata and upon request the datasets."
    "\n"
    "To read a hdf-file:\n"
    ">>> import _lazynodelistio\n"
    ">>> obj = _lazynodelistio.open(\"seang_202001100000.h5\")\n"
    ">>> v = io.getRaveData2D(\"/dataset1/data1/data\")\n"
    );

MOD_INIT(_lazynodelistreader)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyLazyNodeListReader_API[PyLazyNodeListReader_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyLazyNodeListReader_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyLazyNodeListReader_Type);

  MOD_INIT_DEF(module, "_lazynodelistreader", _pylazynodelistio_module_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyLazyNodeListReader_API[PyLazyNodeListReader_Type_NUM] = (void*)&PyLazyNodeListReader_Type;
  PyLazyNodeListReader_API[PyLazyNodeListReader_GetNative_NUM] = (void *)PyLazyNodeListReader_GetNative;
  PyLazyNodeListReader_API[PyLazyNodeListReader_New_NUM] = (void*)PyLazyNodeListReader_New;

  c_api_object = PyCapsule_New(PyLazyNodeListReader_API, PyLazyNodeListReader_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_lazynodelistreader.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _lazynodelistreader.error");
    return MOD_INIT_ERROR;
  }

  HL_init();
  HL_disableErrorReporting();
  HL_disableHdf5ErrorReporting();
  HL_setDebugLevel(HLHDF_SILENT);

  import_ravedata2d();
  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
