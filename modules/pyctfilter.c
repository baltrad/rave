/* --------------------------------------------------------------------
Copyright (C) 2014 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the MSG cloud-type filter
 * @file
 * @author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2014-03-27
 */
#include "pyravecompat.h"
#include "arrayobject.h"
#include "rave.h"
#include "rave_debug.h"
#include "pyrave_debug.h"
#include "pycartesian.h"
#include "ctfilter.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_ctfilter");

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
 * Runs the CT filter
 * @param[in] Cartesian_t object, containing an IMAGE with a selected default parameter
 * @param[in] Cartesian_t object, containing an MSG cloud-type product
 * @returns Py_True or Py_False
 */
static PyObject* _ctFilter_func(PyObject* self, PyObject* args) {
  PyObject* prodobj = NULL;
  PyObject* ctobj = NULL;
  PyCartesian* pyProd = NULL;
  PyCartesian* pyCt = NULL;

  if (!PyArg_ParseTuple(args, "OO", &prodobj, &ctobj)) {
    return NULL;
  }

  if (PyCartesian_Check(prodobj)) {
    pyProd = (PyCartesian*)prodobj;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "CT filter, argument 1 requires Cartesian as input");
  }
  if (PyCartesian_Check(ctobj)) {
    pyCt = (PyCartesian*)ctobj;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "CT filter, argument 2 requires Cartesian as input");
  }

  if (ctFilter(pyProd->cartesian, pyCt->cartesian)) {
    return PyBool_FromLong(1);
    /*Py_RETURN_TRUE;*/
  }
  return PyBool_FromLong(0);
  /*Py_RETURN_FALSE;*/
}


static struct PyMethodDef _ctfilter_functions[] =
{
  { "ctFilter", (PyCFunction) _ctFilter_func, METH_VARARGS },
  { NULL, NULL }
};

/**
 * Initialize the _ctfilter module
 */
MOD_INIT(_ctfilter)
{
  PyObject* module = NULL;
  PyObject* dictionary = NULL;
  MOD_INIT_DEF(module, "_ctfilter", NULL/*doc*/, _ctfilter_functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyErr_NewException("_ctfilter.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _ctfilter.error");
    return MOD_INIT_ERROR;
  }
  import_pycartesian();
  import_array(); /*To make sure I get access to numpy*/
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}

/*@} End of Module setup */
