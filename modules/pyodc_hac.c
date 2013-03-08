/* --------------------------------------------------------------------
Copyright (C) 2013 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the HAC API
 * @file
 * @author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2013-02-23
 */
#include "Python.h"
#include "arrayobject.h"
#include "rave.h"
#include "rave_debug.h"
#include "pyrave_debug.h"
#include "pypolarscan.h"
#include "pyravefield.h"
#include "odc_hac.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_odc_hac");

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
 * Runs the HAC filter
 * @param[in] PolarScan_t object, hopefully containing a "DBZH" parameter
 * @param[in] RaveField_t object, containing HAC data and metadata
 * @returns Py_True or Py_False
 */
static PyObject* _hacFilter_func(PyObject* self, PyObject* args) {
  PyObject* scanobj = NULL;
  PyObject* hacobj = NULL;
  PyPolarScan* pyscan = NULL;
  PyRaveField* pyhac = NULL;
  char* quant;
  

  if (!PyArg_ParseTuple(args, "OOs", &scanobj, &hacobj, &quant)) {
    return NULL;
  }

  if (PyPolarScan_Check(scanobj)) {
    pyscan = (PyPolarScan*)scanobj;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "HAC filter requires scan as input");
  }
  if (PyRaveField_Check(hacobj)) {
    pyhac = (PyRaveField*)hacobj;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "HAC filter requires RaveField as HAC input");
  }

  if (hacFilter(pyscan->scan, pyhac->field, quant)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}


/**
 * Increments the HAC
 * @param[in] PolarScan_t object, hopefully containing a "DBZH" parameter
 * @param[in] RaveField_t object, containing HAC data and metadata
 * @returns Py_True or Py_False
 */
static PyObject* _hacIncrement_func(PyObject* self, PyObject* args) {
  PyObject* scanobj = NULL;
  PyObject* hacobj = NULL;
  PyPolarScan* pyscan = NULL;
  PyRaveField* pyhac = NULL;
  char* quant;

  if (!PyArg_ParseTuple(args, "OOs", &scanobj, &hacobj, &quant)) {
    return NULL;
  }

  if (PyPolarScan_Check(scanobj)) {
    pyscan = (PyPolarScan*)scanobj;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "HAC incrementor requires scan as input");
  }
  if (PyRaveField_Check(hacobj)) {
    pyhac = (PyRaveField*)hacobj;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "HAC incrementor requires RaveField as HAC input");
  }

  if (hacIncrement(pyscan->scan, pyhac->field, quant)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}


static struct PyMethodDef _hac_functions[] =
{
  { "hacFilter", (PyCFunction) _hacFilter_func, METH_VARARGS },
  { "hacIncrement", (PyCFunction) _hacIncrement_func, METH_VARARGS },
  { NULL, NULL }
};

/**
 * Initialize the _odc_hac module
 */
PyMODINIT_FUNC init_odc_hac(void)
{
  PyObject* m;
  m = Py_InitModule("_odc_hac", _hac_functions);
  ErrorObject = PyString_FromString("_odc_hac.error");

  if (ErrorObject == NULL || PyDict_SetItemString(PyModule_GetDict(m),
                                                  "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _odc_hac.error");
  }
  import_pypolarscan();
  import_pyravefield();
  import_array(); /*To make sure I get access to numpy*/
  PYRAVE_DEBUG_INITIALIZE;
}

/*@} End of Module setup */
