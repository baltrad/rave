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
#include "pyraveapi.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYRAVEIO_MODULE   /**< include correct part of pyraveio.h */
#include "pyraveio.h"

#include "pycartesian.h"
#include "pypolarvolume.h"
#include "pypolarscan.h"
#include "pycartesianvolume.h"
#include "pyverticalprofile.h"
#include "pyravevalue.h"
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
PyRaveIO_Open(const char* filename, int lazyLoading, const char* preloadQuantities)
{
  RaveIO_t* raveio = NULL;
  PyRaveIO* result = NULL;

  if (filename == NULL) {
    raiseException_returnNULL(PyExc_ValueError, "providing a filename that is NULL");
  }

  raveio = RaveIO_open(filename, lazyLoading, preloadQuantities);
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
  int lazyLoading = 0;
  char* preloadQuantities = NULL;
  if (!PyArg_ParseTuple(args, "s|iz", &filename, &lazyLoading, &preloadQuantities)) {
    return NULL;
  }
  result = PyRaveIO_Open(filename, lazyLoading, preloadQuantities);
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
  int lazyLoading = 0;
  char* preloadQuantities = NULL;
  if (!PyArg_ParseTuple(args, "|iz", &lazyLoading, &preloadQuantities)) {
    return NULL;
  }

  if (!RaveIO_load(self->raveio, lazyLoading, preloadQuantities)) {
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
  {"version", NULL, METH_VARARGS},
  {"read_version", NULL, METH_VARARGS},
  {"h5radversion", NULL, METH_VARARGS},
  {"objectType", NULL, METH_VARARGS},
  {"filename", NULL, METH_VARARGS},
  {"object", NULL, METH_VARARGS},
  {"extras", NULL, METH_VARARGS},
  {"strict", NULL, METH_VARARGS},
  {"compression_level", NULL, METH_VARARGS},
  {"fcp_userblock", NULL, METH_VARARGS},
  {"fcp_sizes", NULL, METH_VARARGS},
  {"fcp_symk", NULL, METH_VARARGS},
  {"fcp_istorek", NULL, METH_VARARGS},
  {"fcp_metablocksize", NULL, METH_VARARGS},
  {"file_format", NULL, METH_VARARGS},
  {"bufr_table_dir", NULL, METH_VARARGS},
  {"error_message", NULL, METH_VARARGS},
  {"close", (PyCFunction) _pyraveio_close, 1, "close()\n\n"
                                              "Resets this instance and closes the opened object.\n"},
  {"load", (PyCFunction) _pyraveio_load, 1,   "load()\n\n"
                                              "Atempts to load the file that is defined by filename\n"},
  {"save", (PyCFunction) _pyraveio_save, 1,   "save([filename)]\n\n"
                                              "Saves the current object (with current settings).\n\n"
                                              "filename - is optional. If not specified, the objects filename is used\n"},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the PyRaveIO
 * @param[in] self - the RaveIO instance
 */
static PyObject* _pyraveio_getattro(PyRaveIO* self, PyObject* name)
{
  PyObject* res = NULL;
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("version", name) == 0) {
    return PyInt_FromLong(RaveIO_getOdimVersion(self->raveio));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("read_version", name) == 0) {
    return PyInt_FromLong(RaveIO_getReadOdimVersion(self->raveio));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("h5radversion", name) == 0) {
    return PyInt_FromLong(RaveIO_getH5radVersion(self->raveio));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("objectType", name) == 0) {
    return PyInt_FromLong(RaveIO_getObjectType(self->raveio));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("filename", name) == 0) {
    if (RaveIO_getFilename(self->raveio) != NULL) {
      return PyString_FromString(RaveIO_getFilename(self->raveio));
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("object", name) == 0) {
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
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("extras", name) == 0) {
    RaveValue_t* value = RaveIO_getExtras(self->raveio);
    if (value != NULL) {
      PyObject* pyvalue = PyRaveApi_RaveValueToObject(value);
      RAVE_OBJECT_RELEASE(value);
      return pyvalue;
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("strict", name) == 0) {
    return PyBool_FromLong(RaveIO_isStrict(self->raveio));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("compression_level", name) == 0) {
    return PyInt_FromLong(RaveIO_getCompressionLevel(self->raveio));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_userblock", name) == 0) {
    return PyInt_FromLong(RaveIO_getUserBlock(self->raveio));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_sizes", name) == 0) {
    size_t sz = 0, addr = 0;
    RaveIO_getSizes(self->raveio, &sz, &addr);
    return Py_BuildValue("(ii)", sz, addr);
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_symk", name) == 0) {
    int ik = 0, lk = 0;
    RaveIO_getSymk(self->raveio, &ik, &lk);
    return Py_BuildValue("(ii)", ik, lk);
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_istorek", name) == 0) {
    return PyInt_FromLong(RaveIO_getIStoreK(self->raveio));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_metablocksize", name) == 0) {
    return PyInt_FromLong(RaveIO_getMetaBlockSize(self->raveio));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("file_format", name) == 0) {
    return PyInt_FromLong(RaveIO_getFileFormat(self->raveio));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("bufr_table_dir", name) == 0) {
    if (RaveIO_getBufrTableDir(self->raveio) != NULL) {
      return PyString_FromString(RaveIO_getBufrTableDir(self->raveio));
    } else {
      Py_RETURN_NONE;
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("error_message", name) == 0) {
    if (RaveIO_getErrorMessage(self->raveio) != NULL) {
      return PyString_FromString(RaveIO_getErrorMessage(self->raveio));
    } else {
      Py_RETURN_NONE;
    }
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the specified attribute in the raveio
 */
static int _pyraveio_setattro(PyRaveIO* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("version", name)==0) {
    if (PyInt_Check(val)) {
      if (!RaveIO_setOdimVersion(self->raveio, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "illegal version number");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"version must be a valid odim version");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("read_version", name)==0) {
    raiseException_gotoTag(done, PyExc_TypeError,"read_version can not be set");
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("h5radversion", name) == 0) {
    if (PyInt_Check(val)) {
      if (!RaveIO_setH5radVersion(self->raveio, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "illegal h5rad version number");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"version must be a valid h5rad version");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("filename", name) == 0) {
    if (PyString_Check(val)) {
      if (!RaveIO_setFilename(self->raveio, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_MemoryError, "failed to set file name");
      }
    } else if (val == Py_None) {
      RaveIO_setFilename(self->raveio, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"filename must be of type string");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("object", name) == 0) {
    if (PyCartesian_Check(val)) {
      RaveIO_setObject(self->raveio, (RaveCoreObject*)((PyCartesian*)val)->cartesian);
    }
    else if (PyPolarScan_Check(val)) {
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
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("extras", name) == 0) {
    if (PyRaveValue_Check(val)) {
      if (!RaveIO_setExtras(self->raveio, ((PyRaveValue*)val)->value)) {
        raiseException_gotoTag(done, PyExc_TypeError, "Rave value must be of hashtable type");  
      }
    } else if (val == Py_None) {
      RaveIO_setExtras(self->raveio, NULL);
    } else {
      RaveValue_t* value = PyRaveApi_RaveValueFromObject(val);
      if (value != NULL) {
        if (!RaveIO_setExtras(self->raveio, value)) {
          RAVE_OBJECT_RELEASE(value);
          raiseException_gotoTag(done, PyExc_TypeError, "Rave value must be of hashtable type");  
        }
        RAVE_OBJECT_RELEASE(value);
      } else {
        raiseException_gotoTag(done, PyExc_TypeError, "Rave value must be of hashtable type");  
      }
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("strict", name) == 0) {
    if (PyBool_Check(val)) {
      if (PyObject_IsTrue(val)) {
        RaveIO_setStrict(self->raveio, 1);
      } else {
        RaveIO_setStrict(self->raveio, 0);
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "strict should be a boolean");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("compression_level", name) == 0) {
    if (PyInt_Check(val)) {
      RaveIO_setCompressionLevel(self->raveio, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "Compression level should be integer value between 0..9");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_userblock", name) == 0) {
    if (PyInt_Check(val)) {
      RaveIO_setUserBlock(self->raveio, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "User block should be integer value");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_sizes", name) == 0) {
    int sz = 0, addr = 0;
    if (!PyArg_ParseTuple(val, "ii", &sz, &addr)) {
      raiseException_gotoTag(done, PyExc_TypeError ,"sizes must be a tuple containing 2 integers representing (size, addr)");
    }
    RaveIO_setSizes(self->raveio, sz, addr);
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_symk", name) == 0) {
    int ik = 0, lk = 0;
    if (!PyArg_ParseTuple(val, "ii", &ik, &lk)) {
      raiseException_gotoTag(done, PyExc_TypeError ,"symk must be a tuple containing 2 integers representing (ik, lk)");
    }
    RaveIO_setSymk(self->raveio, ik, lk);
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_istorek", name) == 0) {
    if (PyInt_Check(val)) {
      RaveIO_setIStoreK(self->raveio, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError ,"istorek must be a integer");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("fcp_metablocksize", name) == 0) {
    if (PyInt_Check(val)) {
      RaveIO_setMetaBlockSize(self->raveio, PyInt_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError ,"meta block size must be a integer");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("file_format", name) == 0) {
    if (PyInt_Check(val)) {
      if (!RaveIO_setFileFormat(self->raveio, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_AttributeError, "Only valid writable formats are ODIM HDF5 and CF");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError ,"meta block size must be a integer");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("bufr_table_dir", name) == 0) {
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
    raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }

  result = 0;
done:
  return result;
}

/*@} End of RaveIO */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyraveio_doc,
    "This instance wraps the IO-routines used when writing and/or reading files through RAVE.\n"
    "\n"
    "The members of each object are:\n"
    " * version          - ODIM version that will be used when writing this object. Can be one of:\n"
    "                      + RaveIO_ODIM_Version_2_0\n"
    "                      + RaveIO_ODIM_Version_2_1\n"
    "                      + RaveIO_ODIM_Version_2_2\n"
    "                      + RaveIO_ODIM_Version_2_3 (default)\n"
    " * read_version     - ODIM version of the read file. Note, this is only reflecting actual read file and not if file is written with different version. Can be one of:\n"
    "                      + RaveIO_ODIM_Version_UNDEFINED\n"
    "                      + RaveIO_ODIM_Version_2_0\n"
    "                      + RaveIO_ODIM_Version_2_1\n"
    "                      + RaveIO_ODIM_Version_2_2\n"
    "                      + RaveIO_ODIM_Version_2_3 (default)\n"
    "\n"
    " * h5radversion     - showing the H5 rad version of the read file. Can be one of:\n"
    "                      + RaveIO_ODIM_H5rad_Version_UNDEFINED\n"
    "                      + RaveIO_ODIM_H5rad_Version_2_0\n"
    "                      + RaveIO_ODIM_H5rad_Version_2_1\n"
    "                      + RaveIO_ODIM_H5rad_Version_2_2\n"
    "                      + RaveIO_ODIM_H5rad_Version_2_3\n"
    "\n"
    " * objectType       - What type of object that has been read. Can be one of the following:\n"
    "                      + Rave_ObjectType_PVOL\n"
    "                      + Rave_ObjectType_CVOL\n"
    "                      + Rave_ObjectType_SCAN\n"
    "                      + Rave_ObjectType_RAY\n"
    "                      + Rave_ObjectType_AZIM\n"
    "                      + Rave_ObjectType_IMAGE\n"
    "                      + Rave_ObjectType_COMP\n"
    "                      + Rave_ObjectType_XSEC\n"
    "                      + Rave_ObjectType_VP\n"
    "                      + Rave_ObjectType_PIC\n"
    "\n"
    " * file_format      - The file format. Either read or the one to use when writing. Can be one of \n"
    "                      + RaveIO_ODIM_FileFormat_UNDEFINED\n"
    "                      + RaveIO_ODIM_FileFormat_HDF5\n"
#ifdef RAVE_BUFR_SUPPORTED
    "                      + RaveIO_ODIM_FileFormat_BUFR (only available for reading)\n"
#endif
#ifdef RAVE_CF_SUPPORTED
    "                      + RaveIO_FileFormat_CF (only available for writing)\n"
#endif
    "\n"
    " * object           - The actual object beeing written or read. Upon successful reading, this object will always be set and when writing\n"
    "                      this object has to be set.\n"
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
    "Besides the above members, there are a few methods that also are of interest and further information about these can"
    "be found by printing the doc about each invidivdual function.\n"
    " * close()\n"
    " * load()\n"
    " * save()\n"
    );
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyRaveIO_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "RaveIOCore", /*tp_name*/
  sizeof(PyRaveIO), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyraveio_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pyraveio_getattro, /*tp_getattro*/
  (setattrofunc)_pyraveio_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyraveio_doc,                /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyraveio_methods,              /*tp_methods*/
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
  {"new", (PyCFunction)_pyraveio_new, 1,
      "new() -> new instance of the RaveIOCore object\n\n"
      "Creates a new instance of the RaveIOCore object"},
  {"open", (PyCFunction)_pyraveio_open, 1,
      "open(filename[,lazy_loading[,preload_quantities]]) -> a RaveIOCore instance with a loaded object.\n\n"
      "Opens a file that is supported by raveio and loads the structure.\n\n"
      "filename - a filename pointing to a file supported by raveio.\n"
      "lazy_loading - a boolean if file should be lazy loaded or not. If True, then only meta data is read.\n"
      "preload_quantities - a comma-separated list of quantities for which data should be loaded immediately. E.g. \"DBZH,TH\".\n\n"},
  {"supports", (PyCFunction)_pyraveio_supports, 1,
      "supports(format) -> True or False depending if format supported or not\n\n"
      "Returns if the raveio supports the requested file format.\n\n"
      "format - The requested format, can be one of:\n"
      "  raveio.RaveIO_ODIM_FileFormat_HDF5\n"
      " _raveio.RaveIO_ODIM_FileFormat_BUFR - if built with support\n"
      " _raveio.RaveIO_FileFormat_CF        - if built with support and currently only supports writing"},
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

PyDoc_STRVAR(_pyraveio_module_doc,
    "This class provides functionality for reading and writing files supported by RAVE.\n"
    "\n"
    "There are few different ways to handle files and also a couple of different protocols that are supported all depending on\n"
    "how rave was configured and built.\n"
    "Currently, there are 3 different formats supported. ODIM H5, BUFR H5 (reading) and CF Conventions (NetCDF) (writing). This build supports\n"
    "ODIM H5 2.2\n"
#ifdef RAVE_BUFR_SUPPORTED
    "ODIM BUFR for reading\n"
#endif
#ifdef RAVE_CF_SUPPORTED
    "CF Conventions (NetCDF) for writing\n"
#endif
    "\n"
    "This documentation will only provide information about ODIM H5 since this is the format mostly used within rave.\n"
    "\n"
    "To read a hdf-file:\n"
    ">>> import _raveio\n"
    ">>> obj = _raveio.open(\"seang_202001100000.h5\")\n"
    "\n"
    "After you have opened the file, you maybe want to know what type of product you have read\n"
    "Either you compare the objects format_type with _raveio:s list of constants:\n"
    " * Rave_ObjectType_PVOL\n"
    " * Rave_ObjectType_CVOL\n"
    " * Rave_ObjectType_SCAN\n"
    " * Rave_ObjectType_RAY\n"
    " * Rave_ObjectType_AZIM\n"
    " * Rave_ObjectType_IMAGE\n"
    " * Rave_ObjectType_COMP\n"
    " * Rave_ObjectType_XSEC\n"
    " * Rave_ObjectType_VP\n"
    " * Rave_ObjectType_PIC\n"
    "\n"
    "Like\n"
    ">>> if obj.format_type == _raveio.Rave_ObjectType_PVOL:\n"
    "and so on\n"
    "\n"
    "There are also the possibility to check odim version and file format and compare these against predefined constants and can be found by typing\n"
    ">>> dir(_raveio)\n"
    );

MOD_INIT(_raveio)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyRaveIO_API[PyRaveIO_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyRaveIO_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyRaveIO_Type);

  MOD_INIT_DEF(module, "_raveio", _pyraveio_module_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyRaveIO_API[PyRaveIO_Type_NUM] = (void*)&PyRaveIO_Type;
  PyRaveIO_API[PyRaveIO_GetNative_NUM] = (void *)PyRaveIO_GetNative;
  PyRaveIO_API[PyRaveIO_New_NUM] = (void*)PyRaveIO_New;

  c_api_object = PyCapsule_New(PyRaveIO_API, PyRaveIO_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_raveio.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _raveio.error");
    return MOD_INIT_ERROR;
  }


  add_long_constant(dictionary, "RaveIO_ODIM_Version_UNDEFINED", RaveIO_ODIM_Version_UNDEFINED);
  add_long_constant(dictionary, "RaveIO_ODIM_Version_2_0", RaveIO_ODIM_Version_2_0);
  add_long_constant(dictionary, "RaveIO_ODIM_Version_2_1", RaveIO_ODIM_Version_2_1);
  add_long_constant(dictionary, "RaveIO_ODIM_Version_2_2", RaveIO_ODIM_Version_2_2);
  add_long_constant(dictionary, "RaveIO_ODIM_Version_2_3", RaveIO_ODIM_Version_2_3);
  add_long_constant(dictionary, "RaveIO_ODIM_Version_2_4", RaveIO_ODIM_Version_2_4);

  add_long_constant(dictionary, "RaveIO_ODIM_H5rad_Version_UNDEFINED", RaveIO_ODIM_H5rad_Version_UNDEFINED);
  add_long_constant(dictionary, "RaveIO_ODIM_H5rad_Version_2_0", RaveIO_ODIM_H5rad_Version_2_0);
  add_long_constant(dictionary, "RaveIO_ODIM_H5rad_Version_2_1", RaveIO_ODIM_H5rad_Version_2_1);
  add_long_constant(dictionary, "RaveIO_ODIM_H5rad_Version_2_2", RaveIO_ODIM_H5rad_Version_2_2);
  add_long_constant(dictionary, "RaveIO_ODIM_H5rad_Version_2_3", RaveIO_ODIM_H5rad_Version_2_3);
  add_long_constant(dictionary, "RaveIO_ODIM_H5rad_Version_2_4", RaveIO_ODIM_H5rad_Version_2_4);

  add_long_constant(dictionary, "RaveIO_ODIM_FileFormat_UNDEFINED", RaveIO_ODIM_FileFormat_UNDEFINED);
  add_long_constant(dictionary, "RaveIO_ODIM_FileFormat_HDF5", RaveIO_ODIM_FileFormat_HDF5);
  add_long_constant(dictionary, "RaveIO_ODIM_FileFormat_BUFR", RaveIO_ODIM_FileFormat_BUFR);
  add_long_constant(dictionary, "RaveIO_FileFormat_CF", RaveIO_FileFormat_CF);

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
  //HL_InitializeDebugger();
  HL_disableErrorReporting();
  HL_disableHdf5ErrorReporting();
  HL_setDebugLevel(HLHDF_SILENT);
  import_pypolarvolume();
  import_pypolarscan();
  import_pyverticalprofile();
  import_pycartesianvolume();
  import_pycartesian();
  import_ravevalue();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
