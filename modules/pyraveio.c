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
#include "Python.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYRAVEIO_MODULE
#include "pyraveio.h"

#include "pypolarvolume.h"

#include "rave_debug.h"
#include "rave_alloc.h"
#include "hlhdf.h"
#include "hlhdf_debug.h"

/**
 * Some helpful exception defines.
 */
#define raiseException_gotoTag(tag, type, msg) \
{PyErr_SetString(type, msg); goto tag;}

#define raiseException_returnNULL(type, msg) \
{PyErr_SetString(type, msg); return NULL;}

/**
 * Error object for reporting errors to the python interpreeter
 */
static PyObject *ErrorObject;

/*@{ RaveIO products */
/**
 * Returns the native RaveIO_t instance.
 * @param[in] pyraveio - the python raveio instance
 * @returns the native raveio instance.
 */
static RaveIO_t*
PyRaveIO_GetNative(PyRaveIO* pyraveio)
{
  RAVE_ASSERT((pyraveio != NULL), "pyraveio == NULL");
  return RAVE_OBJECT_COPY(pyraveio->raveio);
}

/**
 * Creates a python raveio from a native raveio or will create an
 * initial native raveio if p is NULL.
 * @param[in] p - the native raveio (or NULL)
 * @returns the python raveio product.
 */
static PyRaveIO*
PyRaveIO_New(RaveIO_t* p)
{
  PyRaveIO* result = NULL;
  RaveIO_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RaveIO_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for raveio.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for raveio.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
  }

  result = PyObject_NEW(PyRaveIO, &PyRaveIO_Type);
  if (result == NULL) {
    RAVE_CRITICAL0("Failed to create PyRaveIO instance");
    raiseException_gotoTag(error, PyExc_MemoryError, "Failed to allocate memory for raveio.");
  }

  result->raveio = RAVE_OBJECT_COPY(cp);
  RAVE_OBJECT_BIND(result->raveio, result);
error:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Opens a file that is supported by RaveIO.
 * @param[in] self this instance.
 * @param[in] args arguments for creation. (A string identifying the file)
 * @return the object on success, otherwise NULL
 */
static PyRaveIO*
PyRaveIO_Open(const char* filename)
{
  RaveIO_t* raveio = NULL;
  PyRaveIO* result = NULL;

  if (filename == NULL) {
    raiseException_returnNULL(PyExc_ValueError, "providing a filename that is NULL");
  }
  raveio = RAVE_OBJECT_NEW(&RaveIO_TYPE);
  if (raveio == NULL) {
    raiseException_returnNULL(PyExc_MemoryError, "Failed to create IO instance");
  }
  if (!RaveIO_open(raveio, filename)) {
    raiseException_gotoTag(done, PyExc_IOError, "Failed to open file");
  }
  result = PyRaveIO_New(raveio);
done:
  RAVE_OBJECT_RELEASE(raveio);
  return result;
}

/**
 * Deallocates the RaveIO
 * @param[in] obj the object to deallocate.
 */
static void _pyraveio_dealloc(PyRaveIO* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  RAVE_OBJECT_UNBIND(obj->raveio, obj);
  RAVE_OBJECT_RELEASE(obj->raveio);
  PyObject_Del(obj);
}

/**
 * Creates a new RaveIO instance.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (not USED)
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyraveio_new(PyObject* self, PyObject* args)
{
  PyRaveIO* result = PyRaveIO_New(NULL);
  return (PyObject*)result;
}

/**
 * Opens a file that is supported by raveio
 * @param[in] self this instance.
 * @param[in] args arguments for creation (filename as a string)
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyraveio_open(PyObject* self, PyObject* args)
{
  PyRaveIO* result = NULL;

  char* filename = NULL;
  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }
  result = PyRaveIO_Open(filename);
  return (PyObject*)result;
}

/**
 * Returns true or false depending on if a HDF5 nodelist is loaded
 * or not.
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns Py_None on success, otherwise NULL
 */
static PyObject* _pyraveio_isOpen(PyRaveIO* self, PyObject* args)
{
  return PyBool_FromLong(RaveIO_isOpen(self->raveio));
}

/**
 * Closes the currently open nodelist.
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns Py_None on success, otherwise NULL
 */
static PyObject* _pyraveio_close(PyRaveIO* self, PyObject* args)
{
  RaveIO_close(self->raveio);
  Py_RETURN_NONE;
}

/**
 * Closes the currently open nodelist.
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns Py_None on success, otherwise NULL
 */
static PyObject* _pyraveio_openFile(PyRaveIO* self, PyObject* args)
{
  char* filename = NULL;
  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }
  if (!RaveIO_open(self->raveio, filename)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to open file");
  }
  Py_RETURN_NONE;
}

/**
 * Returns the currently opened files object type (/what/object).
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns the object type on success, otherwise -1
 */
static PyObject* _pyraveio_getObjectType(PyRaveIO* self, PyObject* args)
{
  return PyInt_FromLong(RaveIO_getObjectType(self->raveio));
}

/**
 * Returns if the currently opened file is supported or not.
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns True if the file structure is supported, otherwise False
 */
static PyObject* _pyraveio_isSupported(PyRaveIO* self, PyObject* args)
{
  return PyBool_FromLong(RaveIO_isSupported(self->raveio));
}

static PyObject* _pyraveio_load(PyRaveIO* self, PyObject* args)
{
  PyObject* result = NULL;
  switch (RaveIO_getObjectType(self->raveio)) {
  case RaveIO_ObjectType_PVOL: {
    PolarVolume_t* pvol = RaveIO_loadVolume(self->raveio);
    if (pvol != NULL) {
      result = (PyObject*)PyPolarVolume_New(pvol);
    }
    RAVE_OBJECT_RELEASE(pvol);
    break;
  }
  default:
    RAVE_DEBUG0("Load: Unsupported object type");
    break;
  }
  if (result == NULL) {
    fprintf(stderr, "Returning NULL\n");
  }
  return result;
}

/**
 * Returns the current files version.
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns The current files ODIM version
 */
static PyObject* _pyraveio_getOdimVersion(PyRaveIO* self, PyObject* args)
{
  return PyInt_FromLong(RaveIO_getOdimVersion(self->raveio));
}

/**
 * All methods a RaveIO can have
 */
static struct PyMethodDef _pyraveio_methods[] =
{
  {"isOpen", (PyCFunction) _pyraveio_isOpen, 1},
  {"close", (PyCFunction) _pyraveio_close, 1},
  {"open", (PyCFunction) _pyraveio_openFile, 1},
  {"getObjectType", (PyCFunction) _pyraveio_getObjectType, 1},
  {"isSupported", (PyCFunction) _pyraveio_isSupported, 1},
  {"getOdimVersion", (PyCFunction) _pyraveio_getOdimVersion, 1},
  {"load", (PyCFunction) _pyraveio_load, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the PyRaveIO
 * @param[in] self - the RaveIO instance
 */
static PyObject* _pyraveio_getattr(PyRaveIO* self, char* name)
{
  PyObject* res = NULL;

  res = Py_FindMethod(_pyraveio_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Sets the specified attribute in the raveio
 */
static int _pyraveio_setattr(PyRaveIO* self, char* name, PyObject* val)
{
  return -1;
}

/*@} End of RaveIO */

/*@{ Type definitions */
PyTypeObject PyRaveIO_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "RaveIOCore", /*tp_name*/
  sizeof(PyRaveIO), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyraveio_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_pyraveio_getattr, /*tp_getattr*/
  (setattrfunc)_pyraveio_setattr, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0,
  0, /*tp_as_mapping */
  0 /*tp_hash*/
};
/*@} End of Type definitions */

/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pyraveio_new, 1},
  {"open", (PyCFunction)_pyraveio_open, 1},
  {NULL,NULL} /*Sentinel*/
};

PyMODINIT_FUNC
init_raveio(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyRaveIO_API[PyRaveIO_API_pointers];
  PyObject *c_api_object = NULL;
  PyRaveIO_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_raveio", functions);
  if (module == NULL) {
    return;
  }
  PyRaveIO_API[PyRaveIO_Type_NUM] = (void*)&PyRaveIO_Type;
  PyRaveIO_API[PyRaveIO_GetNative_NUM] = (void *)PyRaveIO_GetNative;
  PyRaveIO_API[PyRaveIO_New_NUM] = (void*)PyRaveIO_New;
  PyRaveIO_API[PyRaveIO_Open_NUM] = (void*)PyRaveIO_Open;

  c_api_object = PyCObject_FromVoidPtr((void *)PyRaveIO_API, NULL);

  if (c_api_object != NULL) {
    PyModule_AddObject(module, "_C_API", c_api_object);
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_raveio.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _raveio.error");
  }

  HL_init();
  HL_disableErrorReporting();
  HL_disableHdf5ErrorReporting();
  HL_setDebugLevel(HLHDF_SILENT);

  import_pypolarvolume();
  Rave_initializeDebugger();
}
/*@} End of Module setup */
