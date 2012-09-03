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

#define PYRAVEIO_MODULE   /**< include correct part of pyraveio.h */
#include "pyraveio.h"

#include "pypolarvolume.h"
#include "pycartesian.h"
#include "pypolarscan.h"
#include "pycartesianvolume.h"
#include "pyverticalprofile.h"
#include "pyrave_debug.h"
#include "rave_alloc.h"
#include "hlhdf.h"
#include "hlhdf_debug.h"

/**
 * Name of the module debugged.
 */
PYRAVE_DEBUG_MODULE("_raveio");

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
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyRaveIO, &PyRaveIO_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->raveio = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->raveio, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyRaveIO instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for raveio.");
    }
  }

done:
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

  raveio = RaveIO_open(filename);
  if (raveio == NULL) {
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
  PYRAVE_DEBUG_OBJECT_DESTROYED;
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
 * Returns if the raveio supports the requested file format.
 * @param[in] self - self
 * @param[in] args - one integer value defining the ODIM file format.
 * @return a python true or false
 */
static PyObject* _pyraveio_supports(PyObject* self, PyObject* args)
{
  RaveIO_ODIM_FileFormat format = RaveIO_ODIM_FileFormat_UNDEFINED;
  int ival = 0;
  if (!PyArg_ParseTuple(args, "i", &ival)) {
    return NULL;
  }
  format = (RaveIO_ODIM_FileFormat)ival;

  return PyBool_FromLong(RaveIO_supports(format));
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
 * Atempts to load the file that is defined by filename.
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns Py_None on success, otherwise NULL
 */
static PyObject* _pyraveio_load(PyRaveIO* self, PyObject* args)
{
  if (!RaveIO_load(self->raveio)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to load file");
  }
  Py_RETURN_NONE;
}

/**
 * Atempts to save the file.
 * @param[in] self - this instance
 * @param[in] args - N/A
 * @returns Py_None on success, otherwise NULL
 */
static PyObject* _pyraveio_save(PyRaveIO* self, PyObject* args)
{
  char* filename = NULL;
  if (!PyArg_ParseTuple(args, "|s", &filename)) {
    return NULL;
  }

  if (!RaveIO_save(self->raveio, filename)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to save file");
  }

  Py_RETURN_NONE;
}

/**
 * All methods a RaveIO can have
 */
static struct PyMethodDef _pyraveio_methods[] =
{
  {"version", NULL},
  {"h5radversion", NULL},
  {"objectType", NULL},
  {"filename", NULL},
  {"object", NULL},
  {"compression_level", NULL},
  {"fcp_userblock", NULL},
  {"fcp_sizes", NULL},
  {"fcp_symk", NULL},
  {"fcp_istorek", NULL},
  {"fcp_metablocksize", NULL},
  {"file_format", NULL},
  {"bufr_table_dir", NULL},
  {"close", (PyCFunction) _pyraveio_close, 1},
  {"load", (PyCFunction) _pyraveio_load, 1},
  {"save", (PyCFunction) _pyraveio_save, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the PyRaveIO
 * @param[in] self - the RaveIO instance
 */
static PyObject* _pyraveio_getattr(PyRaveIO* self, char* name)
{
  PyObject* res = NULL;
  if (strcmp("version", name) == 0) {
    return PyInt_FromLong(RaveIO_getOdimVersion(self->raveio));
  } else if (strcmp("h5radversion", name) == 0) {
    return PyInt_FromLong(RaveIO_getH5radVersion(self->raveio));
  } else if (strcmp("objectType", name) == 0) {
    return PyInt_FromLong(RaveIO_getObjectType(self->raveio));
  } else if (strcmp("filename", name) == 0) {
    if (RaveIO_getFilename(self->raveio) != NULL) {
      return PyString_FromString(RaveIO_getFilename(self->raveio));
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("object", name) == 0) {
    RaveCoreObject* object = RaveIO_getObject(self->raveio);
    if (object != NULL) {
      if (RAVE_OBJECT_CHECK_TYPE(object, &Cartesian_TYPE)) {
        res = (PyObject*)PyCartesian_New((Cartesian_t*)object);
      } else if (RAVE_OBJECT_CHECK_TYPE(object, &PolarVolume_TYPE)) {
        res = (PyObject*)PyPolarVolume_New((PolarVolume_t*)object);
      } else if (RAVE_OBJECT_CHECK_TYPE(object, &PolarScan_TYPE)) {
        res = (PyObject*)PyPolarScan_New((PolarScan_t*)object);
      } else if (RAVE_OBJECT_CHECK_TYPE(object, &CartesianVolume_TYPE)) {
        res = (PyObject*)PyCartesianVolume_New((CartesianVolume_t*)object);
      } else if (RAVE_OBJECT_CHECK_TYPE(object, &VerticalProfile_TYPE)) {
        res = (PyObject*)PyVerticalProfile_New((VerticalProfile_t*)object);
      } else {
        PyErr_SetString(PyExc_NotImplementedError, "support lacking for object type");
      }
      RAVE_OBJECT_RELEASE(object);
      return res;
    } else {
      Py_RETURN_NONE;
    }
  } else if (strcmp("compression_level", name) == 0) {
    return PyInt_FromLong(RaveIO_getCompressionLevel(self->raveio));
  } else if (strcmp("fcp_userblock", name) == 0) {
    return PyInt_FromLong(RaveIO_getUserBlock(self->raveio));
  } else if (strcmp("fcp_sizes", name) == 0) {
    size_t sz = 0, addr = 0;
    RaveIO_getSizes(self->raveio, &sz, &addr);
    return Py_BuildValue("(ii)", sz, addr);
  } else if (strcmp("fcp_symk", name) == 0) {
    int ik = 0, lk = 0;
    RaveIO_getSymk(self->raveio, &ik, &lk);
    return Py_BuildValue("(ii)", ik, lk);
  } else if (strcmp("fcp_istorek", name) == 0) {
    return PyInt_FromLong(RaveIO_getIStoreK(self->raveio));
  } else if (strcmp("fcp_metablocksize", name) == 0) {
    return PyInt_FromLong(RaveIO_getMetaBlockSize(self->raveio));
  } else if (strcmp("file_format", name) == 0) {
    return PyInt_FromLong(RaveIO_getFileFormat(self->raveio));
  } else if (strcmp("bufr_table_dir", name) == 0) {
    if (RaveIO_getBufrTableDir(self->raveio) != NULL) {
      return PyString_FromString(RaveIO_getBufrTableDir(self->raveio));
    } else {
      Py_RETURN_NONE;
    }
  }
  res = Py_FindMethod(_pyraveio_methods, (PyObject*) self, name);
  if (res != NULL)
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
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("version", name)==0) {
    if (PyInt_Check(val)) {
      if (!RaveIO_setOdimVersion(self->raveio, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "illegal version number");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"version must be a valid odim version");
    }
  } else if (strcmp("h5radversion", name) == 0) {
    if (PyInt_Check(val)) {
      if (!RaveIO_setH5radVersion(self->raveio, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "illegal h5rad version number");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"version must be a valid h5rad version");
    }
  } else if (strcmp("filename", name) == 0) {
    if (PyString_Check(val)) {
      if (!RaveIO_setFilename(self->raveio, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_MemoryError, "failed to set file name");
      }
    } else if (val == Py_None) {
      RaveIO_setFilename(self->raveio, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"filename must be of type string");
    }
  } else if (strcmp("object", name) == 0) {
    if (PyCartesian_Check(val)) {
      RaveIO_setObject(self->raveio, (RaveCoreObject*)((PyCartesian*)val)->cartesian);
    } else if (PyPolarScan_Check(val)) {
      RaveIO_setObject(self->raveio, (RaveCoreObject*)((PyPolarScan*)val)->scan);
    } else if (PyPolarVolume_Check(val)) {
      RaveIO_setObject(self->raveio, (RaveCoreObject*)((PyPolarVolume*)val)->pvol);
    } else if (PyCartesianVolume_Check(val)) {
      RaveIO_setObject(self->raveio, (RaveCoreObject*)((PyCartesianVolume*)val)->cvol);
    } else if (PyVerticalProfile_Check(val)) {
      RaveIO_setObject(self->raveio, (RaveCoreObject*)((PyVerticalProfile*)val)->vp);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "Can only save objects of type : cartesian, polarscan, polarvolume or verticalprofile");
    }
  } else if (strcmp("compression_level", name) == 0) {
    if (PyInt_Check(val)) {
      RaveIO_setCompressionLevel(self->raveio, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "Compression level should be integer value between 0..9");
    }
  } else if (strcmp("fcp_userblock", name) == 0) {
    if (PyInt_Check(val)) {
      RaveIO_setUserBlock(self->raveio, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "User block should be integer value");
    }
  } else if (strcmp("fcp_sizes", name) == 0) {
    int sz = 0, addr = 0;
    if (!PyArg_ParseTuple(val, "ii", &sz, &addr)) {
      raiseException_gotoTag(done, PyExc_TypeError ,"sizes must be a tuple containing 2 integers representing (size, addr)");
    }
    RaveIO_setSizes(self->raveio, sz, addr);
  } else if (strcmp("fcp_symk", name) == 0) {
    int ik = 0, lk = 0;
    if (!PyArg_ParseTuple(val, "ii", &ik, &lk)) {
      raiseException_gotoTag(done, PyExc_TypeError ,"symk must be a tuple containing 2 integers representing (ik, lk)");
    }
    RaveIO_setSymk(self->raveio, ik, lk);
  } else if (strcmp("fcp_istorek", name) == 0) {
    if (PyInt_Check(val)) {
      RaveIO_setIStoreK(self->raveio, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError ,"istorek must be a integer");
    }
  } else if (strcmp("fcp_metablocksize", name) == 0) {
    if (PyInt_Check(val)) {
      RaveIO_setMetaBlockSize(self->raveio, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError ,"meta block size must be a integer");
    }
  } else if (strcmp("file_format", name) == 0) {
    raiseException_gotoTag(done, PyExc_AttributeError, "file_format can only be read");
  } else if (strcmp("bufr_table_dir", name) == 0) {
    if (PyString_Check(val)) {
      if (!RaveIO_setBufrTableDir(self->raveio, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_MemoryError, "failed to set bufr table dir");
      }
    } else if (val == Py_None) {
      RaveIO_setBufrTableDir(self->raveio, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"bufr table dir must be of type string");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, name);
  }

  result = 0;
done:
  return result;
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
  {"supports", (PyCFunction)_pyraveio_supports, 1},
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

  add_long_constant(dictionary, "RaveIO_ODIM_Version_UNDEFINED", RaveIO_ODIM_Version_UNDEFINED);
  add_long_constant(dictionary, "RaveIO_ODIM_Version_2_0", RaveIO_ODIM_Version_2_0);
  add_long_constant(dictionary, "RaveIO_ODIM_Version_2_1", RaveIO_ODIM_Version_2_1);

  add_long_constant(dictionary, "RaveIO_ODIM_H5rad_Version_UNDEFINED", RaveIO_ODIM_H5rad_Version_UNDEFINED);
  add_long_constant(dictionary, "RaveIO_ODIM_H5rad_Version_2_0", RaveIO_ODIM_H5rad_Version_2_0);
  add_long_constant(dictionary, "RaveIO_ODIM_H5rad_Version_2_1", RaveIO_ODIM_H5rad_Version_2_1);

  add_long_constant(dictionary, "RaveIO_ODIM_FileFormat_UNDEFINED", RaveIO_ODIM_FileFormat_UNDEFINED);
  add_long_constant(dictionary, "RaveIO_ODIM_FileFormat_HDF5", RaveIO_ODIM_FileFormat_HDF5);
  add_long_constant(dictionary, "RaveIO_ODIM_FileFormat_BUFR", RaveIO_ODIM_FileFormat_BUFR);

  add_long_constant(dictionary, "Rave_ObjectType_UNDEFINED", Rave_ObjectType_UNDEFINED);
  add_long_constant(dictionary, "Rave_ObjectType_PVOL", Rave_ObjectType_PVOL);
  add_long_constant(dictionary, "Rave_ObjectType_CVOL", Rave_ObjectType_CVOL);
  add_long_constant(dictionary, "Rave_ObjectType_SCAN", Rave_ObjectType_SCAN);
  add_long_constant(dictionary, "Rave_ObjectType_RAY", Rave_ObjectType_RAY);
  add_long_constant(dictionary, "Rave_ObjectType_AZIM", Rave_ObjectType_AZIM);
  add_long_constant(dictionary, "Rave_ObjectType_IMAGE", Rave_ObjectType_IMAGE);
  add_long_constant(dictionary, "Rave_ObjectType_COMP", Rave_ObjectType_COMP);
  add_long_constant(dictionary, "Rave_ObjectType_XSEC", Rave_ObjectType_XSEC);
  add_long_constant(dictionary, "Rave_ObjectType_VP", Rave_ObjectType_VP);
  add_long_constant(dictionary, "Rave_ObjectType_PIC", Rave_ObjectType_PIC);

  HL_init();
  HL_disableErrorReporting();
  HL_disableHdf5ErrorReporting();
  HL_setDebugLevel(HLHDF_SILENT);

  import_pypolarvolume();
  import_pypolarscan();
  import_pycartesian();
  import_pycartesianvolume();
  import_pyverticalprofile();
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
