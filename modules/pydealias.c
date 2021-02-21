/* --------------------------------------------------------------------
Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the dealiasing API
 * @file
 * @author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2012-11-15
 */
#include "pyravecompat.h"
#include "arrayobject.h"
#include "rave.h"
#include "rave_debug.h"
#include "dealias.h"
#include "pypolarvolume.h"
#include "pypolarscan.h"
#include "pypolarscanparam.h"
#include "pyrave_debug.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_dealias");

/**
 * Sets a Python exception.
 */
#define Raise(type,msg) {PyErr_SetString(type,msg);}

/**
 * Sets a Python exception and goto tag
 */
#define raiseException_gotoTag(tag, type, msg) \
{PyErr_SetString(type, msg); goto tag;}

/**
 * Sets a Python exception and return NULL
 */
#define raiseException_returnNULL(type, msg) \
{PyErr_SetString(type, msg); return NULL;}

/**
 * Error object for reporting errors to the Python interpreter
 */
static PyObject *ErrorObject;


/**
 * Checks whether a scan is dealiased by looking up its
 * "VRAD" param/how/dealiased attribute.
 * @param[in] PolarScan_t object, hopefully containing a "VRAD" parameter
 * @returns Py_True or Py_False
 */
static PyObject* _dealiased_func(PyObject* self, PyObject* args) {
  PyObject* object = NULL;
  PyPolarScan* pyscan = NULL;
  char* parameter = "VRADH";

  if (!PyArg_ParseTuple(args, "O|s", &object, &parameter)) {
    return NULL;
  }

  if (PyPolarScan_Check(object)) {
    pyscan = (PyPolarScan*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "Dealiased check requires scan as input");
  }

  if (dealiased_by_quantity(pyscan->scan, (const char*)parameter)) {
    return PyBool_FromLong(1); /* Instead of Py_RETURN_TRUE since compiler screams about dereferencing */
  }
  return PyBool_FromLong(0); /* Instead of Py_RETURN_FALSE since compiler screams about dereferencing */
}


/**
 * Dealiasing of polar scan polar volume
 * @param[in]
 * @returns Py_None
 */
static PyObject* _dealias_func(PyObject* self, PyObject* args)
{
  PyObject* object = NULL;
  PyPolarScan* scan = NULL;
  PyPolarVolume* volume = NULL;
  int ret = 0;
  char* parameter = "VRADH";
  double emax = EMAX;

  if (!PyArg_ParseTuple(args, "O|sd", &object, &parameter, &emax)) {
    return NULL;
  }

  if (PyPolarVolume_Check(object)) {
    volume = (PyPolarVolume*)object;
  } else if (PyPolarScan_Check(object)) {
    scan = (PyPolarScan*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "Dealiasing requires scan or volume as input");
  }

  if (PyPolarVolume_Check(object)) {
    ret = dealias_pvol_by_quantity(volume->pvol, (const char*)parameter, emax);
  } else {
    ret = dealias_scan_by_quantity(scan->scan, (const char*)parameter, emax);
  }

  if (ret) {
    return PyBool_FromLong(1); /* Instead of Py_RETURN_TRUE since compiler screams about dereferencing */
  } else {
    return PyBool_FromLong(0); /* Instead of Py_RETURN_FALSE since compiler screams about dereferencing */
  }
}

static PyObject* _create_dealiased_parameter(PyObject* self, PyObject* args)
{
  PyObject* object = NULL;
  PyPolarScan* scan = NULL;
  PolarScanParam_t* param = NULL;
  PyObject* result = NULL;

  char* parameter = NULL;
  char* newquantity = NULL;

  if (!PyArg_ParseTuple(args, "Oss", &object, &parameter, &newquantity)) {
    return NULL;
  }

  if (PyPolarScan_Check(object)) {
    scan = (PyPolarScan*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "create_dealiased_parameter requires scan as input");
  }
  param = create_dealiased_parameter(scan->scan, parameter, newquantity);
  if (param != NULL) {
    result = (PyObject*)PyPolarScanParam_New(param);
  } else {
    PyErr_SetString(PyExc_RuntimeWarning, "Could not create dealiased parameter");
  }
  RAVE_OBJECT_RELEASE(param);
  return result;
}

static struct PyMethodDef _dealias_functions[] =
{
  { "dealiased", (PyCFunction) _dealiased_func, METH_VARARGS,
      "dealiased(scan[,quantity]) -> boolean\n\n"
      "Checks whether a scan is dealiased by looking up its the specified quantities param/how/dealiased attribute.\n\n"
      "scan      - the polar scan\n"
      "quantity  - the quantity that should be queried. If not specified, it defaults to VRADH."
  },
  { "dealias", (PyCFunction) _dealias_func, METH_VARARGS,
      "dealias(object, quantity, emax) -> boolean\n\n"
      "Function for dealiasing polar volume or polar scan data for the specified quantity and max elevation angle\n\n"
      "object   - the polar scan or volume\n"
      "quantity - the parameter that should be dealiased\n"
      "emax     - the max elevation angle in degrees"
  },
  { "create_dealiased_parameter", (PyCFunction) _create_dealiased_parameter, METH_VARARGS,
      "create_dealiased_parameter(scan, quantity, newquantity) -> polar scan parameter\n\n"
      "Creates a dealiased parameter from the scan / quantity. The created dealiased parameter will get quantity newquantity\n"
      "scan        - the polar scan\n"
      "quantity    - the quantity that should be processed\n"
      "newquantity - the quantity that the returned parameter should get"
  },
  { NULL, NULL }
};

/*@{ Documentation about the type */
PyDoc_STRVAR(_dealias_module_doc,
  "Provides functionality for dealiasing radial wind data."
);
/*@} End of Documentation about the type */

/**
 * Initialize the _dealias module
 */
MOD_INIT(_dealias)
{
  PyObject* module = NULL;
  PyObject* dictionary = NULL;

  MOD_INIT_DEF(module, "_dealias", _dealias_module_doc, _dealias_functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyErr_NewException("_dealias.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _dealias.error");
    return MOD_INIT_ERROR;
  }

  import_pypolarvolume();
  import_pypolarscan();
  import_pypolarscanparam();
  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}

/*@} End of Module setup */
