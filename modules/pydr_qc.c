/* --------------------------------------------------------------------
Copyright (C) 2019 The Crown (i.e. Her Majesty the Queen in Right of Canada)

This file is an add-on to RAVE.

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
 * Python wrapper to DR_QC
 * @file
 * @author Daniel Michelson, Environment Canada
 * @date 2015-10-28
 */
#include "pydrqc_compat.h"
#include "arrayobject.h"
#include "rave.h"
#include "rave_debug.h"
#include "pyraveio.h"
#include "pypolarvolume.h"
#include "pypolarscan.h"
#include "pyrave_debug.h"
#include "dr_qc.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_dr_qc");

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
 * Calculates depolarization ratio.
 * @param[in] double - ZDR value on the decibel scale
 * @param[in] double - RHOHV value, should be between 0 and MAX_RHOHV
 * @param[in] double - ZDR offset value on the decibel scale
 * @return double - depolarization ratio value on the decibel scale
 */
static PyObject* _drCalculate_func(PyObject* self, PyObject* args) {
  double ZDR, RHOHV, zdr_offset, drdb;

  if (!PyArg_ParseTuple(args, "ddd", &ZDR, &RHOHV, &zdr_offset)) {
    return Py_None;
  }
  drdb = drCalculate(ZDR, RHOHV, zdr_offset);

  return PyFloat_FromDouble(drdb);
}


/**
 * Derives a parameter containing depolarization ratio
 * @param[in] 
 * @return 
 */
static PyObject* _drDeriveParameter_func(PyObject* self, PyObject* args) {
  PyObject* object = NULL;
  PyPolarScan* pyscan = NULL;
  double zdr_offset;

  if (!PyArg_ParseTuple(args, "Od", &object, &zdr_offset)) {
    return NULL;
  }

  if (PyPolarScan_Check(object)) {
    pyscan = (PyPolarScan*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "Depolarization ratio check requires scan as input");
  }

  if (!drDeriveParameter(pyscan->scan, zdr_offset)) {
    raiseException_returnNULL(PyExc_AttributeError, "Depolarization ratio requires RHOHV and ZDR");
  }

  Py_RETURN_NONE;
}


/**
 * Speckle filter and inverse speckle filter.
 * @param[in] 
 * @return 
 */
static PyObject* _drSpeckleFilter_func(PyObject* self, PyObject* args) {
  PyObject* object = NULL;
  PyPolarScan* pyscan = NULL;
  char *param_name;
  int kernely, kernelx;
  double param_thresh, dr_thresh;

  if (!PyArg_ParseTuple(args, "Osiidd", &object, &param_name, &kernely, &kernelx, &param_thresh, &dr_thresh)) {
    return NULL;
  }

  if (PyPolarScan_Check(object)) {
    pyscan = (PyPolarScan*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "DR speckle filter requires scan as input");
  }

  if (!drSpeckleFilter(pyscan->scan, param_name, kernely, kernelx, param_thresh, dr_thresh)) {
    raiseException_returnNULL(PyExc_AttributeError, "DR speckle filter requires parameter (most likely DBZH) and depolarization ratio (DR)");
  }

  Py_RETURN_NONE;
}


static struct PyMethodDef _dr_qc_functions[] =
{
  { "drCalculate", (PyCFunction) _drCalculate_func, METH_VARARGS },
  { "drDeriveParameter", (PyCFunction) _drDeriveParameter_func, METH_VARARGS },
  { "drSpeckleFilter", (PyCFunction) _drSpeckleFilter_func, METH_VARARGS },
  { NULL, NULL }
};

/**
 * Initialize the _dr_qc module
 */
MOD_INIT(_dr_qc)
{
  PyObject* module = NULL;
  PyObject* dictionary = NULL;
  
  MOD_INIT_DEF(module, "_dr_qc", NULL/*doc*/, _dr_qc_functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyErr_NewException("_dr_qc.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _dr_qc.error");
    return MOD_INIT_ERROR;
  }

  import_pyraveio();
  import_pypolarvolume();
  import_pypolarscan();
  import_array(); /*To make sure I get access to numpy*/
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}

/*@} End of Module setup */
