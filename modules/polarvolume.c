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
 * Defines a polar volume and the operations that can be performed
 * on this object.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-14
 */
#include <Python.h>
#include <arrayobject.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "raveutil.h"
#include "rave.h"
#include "polarvolume.h"


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

typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  PolarScan_t scan; /**< the scan type */
} PolarScan;

/**
 * The polar volume object
 */
typedef struct {
   PyObject_HEAD /*Always have to be on top*/
   //HL_NodeList* nodelist; /**< the node list */
} PolarVolume;

/**
 * PolarScan represents one scan in a pvol
 */
staticforward PyTypeObject PolarScan_Type;


/**
 * PolarVolume represents one pvol
 */
staticforward PyTypeObject PolarVolume_Type;

/**
 * Checks if the object is a PolarScan type
 */
#define PolarScan_Check(op) ((op)->ob_type == &PolarScan_Type)

/**
 * Checks if the object is a PolarVolume type
 */
#define PolarVolume_Check(op) ((op)->ob_type == &PolarVolume_Type)

/**
 * Deallocates the polar scan
 * @param[in] obj the object to deallocate.
 */
static void _polarscan_dealloc(PolarScan* obj)
{
  if (obj == NULL) {
    return;
  }
  PyObject_Del(obj);
}

/**
 * Deallocates the polar volume
 * @param[in] obj the object to deallocate.
 */
static void _polarvolume_dealloc(PolarVolume* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the polar scan.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _polarscan_new(PyObject* self, PyObject* args)
{
  PolarScan* result = NULL;
  result = PyObject_NEW(PolarScan, &PolarScan_Type);
  if (result == NULL) {
    return NULL;
  }
  result->scan.elangle = 0.0;
  result->scan.nbins = 0;
  result->scan.rscale = 0.0;
  result->scan.nrays = 0;
  result->scan.rstart = 0.0;
  result->scan.a1gate = 0;
  strcpy(result->scan.quantity, "");
  result->scan.gain = 0.0;
  result->scan.offset = 0.0;
  result->scan.nodata = 0.0;
  result->scan.undetect = 0.0;
  return (PyObject*)result;
}

/**
 * Creates a new instance of the polar volume.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _polarvolume_new(PyObject* self, PyObject* args)
{
  PolarVolume* result = NULL;
  result = PyObject_NEW(PolarVolume, &PolarVolume_Type);
  if (result == NULL) {
    return NULL;
  }
  return (PyObject*)result;
}

/**
 * Adds one scan to a volume.
 * @param[in] self - the polar volume
 * @param[in] args - the scan, must be of type PolarScanCore
 * @return NULL on failure
 */
static PyObject* _polarvolume_add(PolarVolume* self, PyObject* args)
{
  PyObject* inptr = NULL;
  PolarScan* polarScan = NULL;

  if (!PyArg_ParseTuple(args, "O", &inptr)) {
    return NULL;
  }

  if (!PolarScan_Check(inptr)) {
    raiseException_returnNULL(PyExc_TypeError,"Added object must be of type PolarScanCore");
  }

  polarScan = (PolarScan*)inptr;

  Py_RETURN_NONE;
}

/**
 * Creates a cappi from a polar volume
 * @param[in] self the polar volume
 * @param[in] args arguments for generating the cappi
 * @return the object on success, otherwise NULL
 */
static PyObject* _polarvolume_cappi(PolarVolume* self, PyObject* args)
{
  return NULL;
}

/**
 * All methods a polar scan can have
 */
static struct PyMethodDef _polarscan_methods[] =
{
  { NULL, NULL } /* sentinel */
};


/**
 * All methods a polar volume can have
 */
static struct PyMethodDef _polarvolume_methods[] =
{
  { "add", (PyCFunction) _polarvolume_add, 1},
  { "cappi", (PyCFunction) _polarvolume_cappi, 1 },
  { NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the polar volume
 */
static PyObject* _polarscan_getattr(PolarScan* self, char* name)
{
  PyObject* res;
  if (strcmp("elangle", name) == 0) {
    return PyFloat_FromDouble(self->scan.elangle);
  } else if (strcmp("nbins", name) == 0) {
    return PyInt_FromLong(self->scan.nbins);
  } else if (strcmp("rscale", name) == 0) {
    return PyFloat_FromDouble(self->scan.rscale);
  } else if (strcmp("nrays", name) == 0) {
    return PyInt_FromLong(self->scan.nrays);
  } else if (strcmp("rstart", name) == 0) {
    return PyFloat_FromDouble(self->scan.rstart);
  } else if (strcmp("a1gate", name) == 0) {
    return PyInt_FromLong(self->scan.a1gate);
  } else if (strcmp("quantity", name) == 0) {
    return PyString_FromString(self->scan.quantity);
  } else if (strcmp("gain", name) == 0) {
    return PyInt_FromLong(self->scan.gain);
  } else if (strcmp("offset", name) == 0) {
    return PyInt_FromLong(self->scan.offset);
  } else if (strcmp("nodata", name) == 0) {
    return PyInt_FromLong(self->scan.nodata);
  } else if (strcmp("undetect", name) == 0) {
    return PyInt_FromLong(self->scan.undetect);
  }

  res = Py_FindMethod(_polarscan_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _polarscan_setattr(PolarScan* self, char* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (strcmp("elangle", name)==0) {
    if (PyFloat_Check(val)) {
      self->scan.elangle = PyFloat_AsDouble(val);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"elangle must be of type float");
    }
  } else if (strcmp("nbins", name) == 0) {
    if (PyInt_Check(val)) {
      self->scan.nbins = PyInt_AsLong(val);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"nbins must be of type int");
    }
  } else if (strcmp("rscale", name) == 0) {
    if (PyFloat_Check(val)) {
      self->scan.rscale = PyFloat_AsDouble(val);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "rscale must be of type float");
    }
  } else if (strcmp("nrays", name) == 0) {
    if (PyInt_Check(val)) {
      self->scan.nrays = PyInt_AsLong(val);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"nbins must be of type int");
    }
  } else if (strcmp("rstart", name) == 0) {
    if (PyFloat_Check(val)) {
      self->scan.rstart = PyFloat_AsDouble(val);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "rstart must be of type float");
    }
  } else if (strcmp("a1gate", name) == 0) {
    if (PyInt_Check(val)) {
      self->scan.a1gate = PyInt_AsLong(val);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"a1gate must be of type int");
    }
  } else if (strcmp("quantity", name) == 0) {
    if (PyString_Check(val)) {
      strcpy(self->scan.quantity, PyString_AsString(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"quantity must be of type string");
    }
  } else if (strcmp("gain", name) == 0) {
    if (PyFloat_Check(val)) {
      self->scan.gain = PyFloat_AsDouble(val);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "gain must be of type float");
    }
  } else if (strcmp("offset", name) == 0) {
    if (PyFloat_Check(val)) {
      self->scan.offset = PyFloat_AsDouble(val);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "offset must be of type float");
    }
  } else if (strcmp("nodata", name) == 0) {
    if (PyFloat_Check(val)) {
      self->scan.nodata = PyFloat_AsDouble(val);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "nodata must be of type float");
    }
  } else if (strcmp("undetect", name) == 0) {
    if (PyFloat_Check(val)) {
      self->scan.undetect = PyFloat_AsDouble(val);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "undetect must be of type float");
    }
  }

  result = 0;
done:
  return result;
}

/**
 * Returns the specified attribute in the polar volume
 */
static PyObject* _polarvolume_getattr(PolarVolume* self, char* name)
{
  PyObject* res;

  res = Py_FindMethod(_polarvolume_methods, (PyObject*) self, name);
  if (res)
    return res;

  PyErr_Clear();
  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

statichere PyTypeObject PolarScan_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "PolarScanCore", /*tp_name*/
  sizeof(PolarScan), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_polarscan_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_polarscan_getattr, /*tp_getattr*/
  (setattrfunc)_polarscan_setattr, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0,
  0, /*tp_as_mapping */
  0 /*tp_hash*/
};

statichere PyTypeObject PolarVolume_Type =
{
  PyObject_HEAD_INIT(NULL)0, /*ob_size*/
  "PolarVolumeCore", /*tp_name*/
  sizeof(PolarVolume), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_polarvolume_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_polarvolume_getattr, /*tp_getattr*/
  0, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0,
  0, /*tp_as_mapping */
  0 /*tp_hash*/
};

static PyMethodDef functions[] = {
  {"volume", (PyCFunction)_polarvolume_new, 1},
  {"scan", (PyCFunction)_polarscan_new, 1},
  {NULL,NULL} /*Sentinel*/
};

/**
 * Initializes polar volume.
 */
void init_polarvolume(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  PolarVolume_Type.ob_type = &PyType_Type;
  PolarScan_Type.ob_type = &PyType_Type;

  module = Py_InitModule("_polarvolume", functions);
  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_polarvolume.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _polarvolume.error");
  }

//  PyObject *m, *d;
//  PyObject *tmp;
//  PyhlNodelist_Type.ob_type = &PyType_Type;
//  PyhlNode_Type.ob_type = &PyType_Type;
//
//  m = Py_InitModule("_pyhl",functions);
//  d = PyModule_GetDict(m);
//
//  ErrorObject = PyString_FromString("_pyhl.error");
//  if (ErrorObject == NULL || PyDict_SetItemString(d, "error", ErrorObject) != 0) {
//    Py_FatalError("Can't define _pyhl.error");
//  }
//
//  tmp = PyInt_FromLong(ATTRIBUTE_ID);
//  PyDict_SetItemString(d, "ATTRIBUTE_ID", tmp);
//  Py_XDECREF(tmp);
//
//  tmp = PyInt_FromLong(GROUP_ID);
//  PyDict_SetItemString(d,"GROUP_ID",tmp);
//  Py_XDECREF(tmp);
//
//  tmp = PyInt_FromLong(DATASET_ID);
//  PyDict_SetItemString(d,"DATASET_ID",tmp);
//  Py_XDECREF(tmp);
//
//  tmp = PyInt_FromLong(TYPE_ID);
//  PyDict_SetItemString(d,"TYPE_ID",tmp);
//  Py_XDECREF(tmp);
//
//  tmp = PyInt_FromLong(REFERENCE_ID);
//  PyDict_SetItemString(d,"REFERENCE_ID",tmp);
//  Py_XDECREF(tmp);
//
//  tmp = PyInt_FromLong(CT_ZLIB);
//  PyDict_SetItemString(d,"COMPRESSION_ZLIB",tmp);
//  Py_XDECREF(tmp);
//
//  tmp = PyInt_FromLong(CT_SZLIB);
//  PyDict_SetItemString(d,"COMPRESSION_SZLIB",tmp);
//  Py_XDECREF(tmp);
//
//  import_array(); /*To make sure I get access to Numeric*/
//  /*Always have to do this*/
//  HL_init();
//  /*And this I just do to be able to get debugging info from hdf*/
//  HL_setDebugMode(2);
}
