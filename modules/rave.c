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
 * Python wrapper for the rave product generation framework.
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
#include "pypolarscan.h"
#include "pypolarvolume.h"
#include "pycartesian.h"
#include "pycartesianparam.h"
#include "pytransform.h"
#include "pyprojection.h"
#include "pyraveio.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_utilities.h"
#include "pyrave_debug.h"

/**
 * This modules name
 */
PYRAVE_DEBUG_MODULE("_rave");

/**
 * Sets python exception and goto tag.
 */
#define raiseException_gotoTag(tag, type, msg) \
{PyErr_SetString(type, msg); goto tag;}

/**
 * Sets python exception string and returns NULL
 */
#define raiseException_returnNULL(type, msg) \
{PyErr_SetString(type, msg); return NULL;}

/**
 * Error object for reporting errors to the python interpreeter
 */
static PyObject *ErrorObject;

/*@{ Polar Scans */
/**
 * Creates a new instance of the polar scan.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _polarscan_new(PyObject* self, PyObject* args)
{
  PyPolarScan* result = PyPolarScan_New(NULL);
  return (PyObject*)result;
}
/*@} End of Polar Scans */

/*@{ Polar Volumes */
/**
 * Creates a new instance of the polar volume.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _polarvolume_new(PyObject* self, PyObject* args)
{
  PyPolarVolume* result = PyPolarVolume_New(NULL);
  return (PyObject*)result;
}
/*@} End of Polar Volumes */

/*@{ Cartesian products */
/**
 * Creates a new instance of the cartesian product.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _cartesian_new(PyObject* self, PyObject* args)
{
  PyCartesian* result = PyCartesian_New(NULL);
  return (PyObject*)result;
}

/**
 * Creates a new instance of the cartesian parameter.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _cartesianparam_new(PyObject* self, PyObject* args)
{
  PyCartesianParam* result = PyCartesianParam_New(NULL);
  return (PyObject*)result;
}
/*@} End of Cartesian products */

/*@{ Transform */
/**
 * Creates a new instance of the transformator.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _transform_new(PyObject* self, PyObject* args)
{
  PyTransform* result = PyTransform_New(NULL);
  return (PyObject*)result;
}

/*@{ End of Transform */

/*@{ New projection */
static PyObject* _projection_new(PyObject* self, PyObject* args)
{
  PyProjection* result = NULL;
  char* id = NULL;
  char* description = NULL;
  char* definition = NULL;

  if (!PyArg_ParseTuple(args, "sss", &id, &description, &definition)) {
    return NULL;
  }

  result = PyProjection_NewFromDef(id, description, definition);

  return (PyObject*)result;
}
/*@} End of New projection */

/*@{ RaveIO */
/**
 * Creates a new RaveIO instance.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (Not USED)
 * @return the object on success, otherwise NULL
 */
static PyObject* _raveio_new(PyObject* self, PyObject* args)
{
  PyRaveIO* result = PyRaveIO_New(NULL);
  return (PyObject*)result;
}

/**
 * Opens a file that is supported by RaveIO.
 * @param[in] self this instance.
 * @param[in] args arguments for creation. (A string identifying the file)
 * @return the object on success, otherwise NULL
 */
static PyObject* _raveio_open(PyObject* self, PyObject* args)
{
  PyRaveIO* result = NULL;
  char* filename = NULL;
  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }
  result = PyRaveIO_Open(filename);
  return (PyObject*)result;
}

/*@} End of RaveIO */

/*@{ Rave utilities */
/**
 * Returns if xml is supported or not in this build.
 * @param[in] self - self
 * @param[in] args - N/A
 * @returns true if xml is supported otherwise false
 */
static PyObject* _rave_isxmlsupported(PyObject* self, PyObject* args)
{
  return PyBool_FromLong(RaveUtilities_isXmlSupported());
}

/**
 * Sets a specific debug level
 * @param[in] self - self
 * @param[in] args - the debug level as an integer
 * @return None
 */
static PyObject* _rave_setDebugLevel(PyObject* self, PyObject* args)
{
  int lvl = RAVE_SILENT;
  if (!PyArg_ParseTuple(args, "i", &lvl)) {
    return NULL;
  }
  Rave_setDebugLevel(lvl);
  Py_RETURN_NONE;
}

/*@} End of Rave utilities */

/// --------------------------------------------------------------------
/// Module setup
/// --------------------------------------------------------------------
/*@{ Module setup */
static PyMethodDef functions[] = {
  {"volume", (PyCFunction)_polarvolume_new, 1},
  {"scan", (PyCFunction)_polarscan_new, 1},
  {"cartesian", (PyCFunction)_cartesian_new, 1},
  {"cartesianparam", (PyCFunction)_cartesianparam_new, 1},
  {"transform", (PyCFunction)_transform_new, 1},
  {"projection", (PyCFunction)_projection_new, 1},
  {"io", (PyCFunction)_raveio_new, 1},
  {"open", (PyCFunction)_raveio_open, 1},
  {"isXmlSupported", (PyCFunction)_rave_isxmlsupported, 1},
  {"setDebugLevel", (PyCFunction)_rave_setDebugLevel, 1},
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

/**
 * Initializes polar volume.
 */
void init_rave(void)
{
  PyObject *module=NULL,*dictionary=NULL;
  module = Py_InitModule("_rave", functions);
  dictionary = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_rave.error");
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _rave.error");
  }

  if (atexit(rave_alloc_print_statistics) != 0) {
    fprintf(stderr, "Could not set atexit function");
  }

  if (atexit(RaveCoreObject_printStatistics) != 0) {
    fprintf(stderr, "Could not set atexit function");
  }

  // Initialize some constants
  add_long_constant(dictionary, "RaveDataType_UNDEFINED", RaveDataType_UNDEFINED);
  add_long_constant(dictionary, "RaveDataType_CHAR", RaveDataType_CHAR);
  add_long_constant(dictionary, "RaveDataType_UCHAR", RaveDataType_UCHAR);
  add_long_constant(dictionary, "RaveDataType_SHORT", RaveDataType_SHORT);
  add_long_constant(dictionary, "RaveDataType_INT", RaveDataType_INT);
  add_long_constant(dictionary, "RaveDataType_LONG", RaveDataType_LONG);
  add_long_constant(dictionary, "RaveDataType_FLOAT", RaveDataType_FLOAT);
  add_long_constant(dictionary, "RaveDataType_DOUBLE", RaveDataType_DOUBLE);

  add_long_constant(dictionary, "NEAREST", NEAREST);
  add_long_constant(dictionary, "BILINEAR", BILINEAR);
  add_long_constant(dictionary, "CUBIC", CUBIC);
  add_long_constant(dictionary, "CRESSMAN", CRESSMAN);
  add_long_constant(dictionary, "UNIFORM", UNIFORM);
  add_long_constant(dictionary, "INVERSE", INVERSE);

  add_long_constant(dictionary, "RaveValueType_UNDEFINED", RaveValueType_UNDEFINED);
  add_long_constant(dictionary, "RaveValueType_UNDETECT", RaveValueType_UNDETECT);
  add_long_constant(dictionary, "RaveValueType_NODATA", RaveValueType_NODATA);
  add_long_constant(dictionary, "RaveValueType_DATA", RaveValueType_DATA);

  add_long_constant(dictionary, "RaveIO_ODIM_Version_UNDEFINED", RaveIO_ODIM_Version_UNDEFINED);
  add_long_constant(dictionary, "RaveIO_ODIM_Version_2_0", RaveIO_ODIM_Version_2_0);

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

  add_long_constant(dictionary, "Rave_ProductType_UNDEFINED", Rave_ProductType_UNDEFINED);
  add_long_constant(dictionary, "Rave_ProductType_SCAN", Rave_ProductType_SCAN);
  add_long_constant(dictionary, "Rave_ProductType_PPI", Rave_ProductType_PPI);
  add_long_constant(dictionary, "Rave_ProductType_CAPPI", Rave_ProductType_CAPPI);
  add_long_constant(dictionary, "Rave_ProductType_PCAPPI", Rave_ProductType_PCAPPI);
  add_long_constant(dictionary, "Rave_ProductType_ETOP", Rave_ProductType_ETOP);
  add_long_constant(dictionary, "Rave_ProductType_MAX", Rave_ProductType_MAX);
  add_long_constant(dictionary, "Rave_ProductType_RR", Rave_ProductType_RR);
  add_long_constant(dictionary, "Rave_ProductType_VIL", Rave_ProductType_VIL);
  add_long_constant(dictionary, "Rave_ProductType_COMP", Rave_ProductType_COMP);
  add_long_constant(dictionary, "Rave_ProductType_VP", Rave_ProductType_VP);
  add_long_constant(dictionary, "Rave_ProductType_RHI", Rave_ProductType_RHI);
  add_long_constant(dictionary, "Rave_ProductType_XSEC", Rave_ProductType_XSEC);
  add_long_constant(dictionary, "Rave_ProductType_VSP", Rave_ProductType_VSP);
  add_long_constant(dictionary, "Rave_ProductType_HSP", Rave_ProductType_HSP);
  add_long_constant(dictionary, "Rave_ProductType_RAY", Rave_ProductType_RAY);
  add_long_constant(dictionary, "Rave_ProductType_AZIM", Rave_ProductType_AZIM);
  add_long_constant(dictionary, "Rave_ProductType_QUAL", Rave_ProductType_QUAL);
  add_long_constant(dictionary, "Rave_ProductType_PMAX", Rave_ProductType_PMAX);
  add_long_constant(dictionary, "Rave_ProductType_SURF", Rave_ProductType_SURF);

  add_long_constant(dictionary, "Debug_RAVE_SPEWDEBUG", RAVE_SPEWDEBUG);
  add_long_constant(dictionary, "Debug_RAVE_DEBUG", RAVE_DEBUG);
  add_long_constant(dictionary, "Debug_RAVE_DEPRECATED", RAVE_DEPRECATED);
  add_long_constant(dictionary, "Debug_RAVE_INFO", RAVE_INFO);
  add_long_constant(dictionary, "Debug_RAVE_WARNING", RAVE_WARNING);
  add_long_constant(dictionary, "Debug_RAVE_ERROR", RAVE_ERROR);
  add_long_constant(dictionary, "Debug_RAVE_CRITICAL", RAVE_CRITICAL);
  add_long_constant(dictionary, "Debug_RAVE_SILENT", RAVE_SILENT);

  import_array(); /*To make sure I get access to Numeric*/
  import_pyprojection();
  import_pypolarscan();
  import_pypolarvolume();
  import_pycartesian();
  import_pycartesianparam();
  import_pyraveio();
  import_pytransform();
  PYRAVE_DEBUG_INITIALIZE;
}
/*@} End of Module setup */
