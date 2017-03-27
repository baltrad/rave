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
 * Python version of the POO Compositing Algorithm API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2011-10-28
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "pyrave_debug.h"
#include "pycompositealgorithm.h"
#include "rave_alloc.h"
#include "poo_composite_algorithm.h"

/**
 * This modules name
 */
PYRAVE_DEBUG_MODULE("_poocompositealgorithm");

/**
 * Sets a python exception and goto tag
 */
#define raiseException_gotoTag(tag, type, msg) \
{PyErr_SetString(type, msg); goto tag;}

/**
 * Sets a python exception and return NULL
 */
#define raiseException_returnNULL(type, msg) \
{PyErr_SetString(type, msg); return NULL;}

/**
 * Error object for reporting errors to the python interpreeter
 */
static PyObject *ErrorObject;

/*@{ PooCompositeAlgorithm */
static PyObject* _poocompositealgorithm_new(PyObject* self, PyObject* args)
{
  PooCompositeAlgorithm_t* algorithm = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  algorithm = RAVE_OBJECT_NEW(&PooCompositeAlgorithm_TYPE);
  if (algorithm != NULL) {
    result = (PyObject*)PyCompositeAlgorithm_New((CompositeAlgorithm_t*)algorithm);
  }
  RAVE_OBJECT_RELEASE(algorithm);
  return result;
}

/*@} End of PooCompositeAlgorithm */

/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_poocompositealgorithm_new, 1},
  {NULL, NULL} /*Sentinel*/
};

MOD_INIT(_poocompositealgorithm)
{
  PyObject *module=NULL,*dictionary=NULL;
  MOD_INIT_DEF(module, "_poocompositealgorithm", NULL/*doc*/, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }
  dictionary = PyModule_GetDict(module);
  ErrorObject = PyErr_NewException("_poocompositealgorithm.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _poocompositealgorithm.error");
    return MOD_INIT_ERROR;
  }

  import_compositealgorithm();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
