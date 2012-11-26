/* --------------------------------------------------------------------
Copyright (C) 2012 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python wrappers for IMGW's RADVOL-QC
 * @file
 * @author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2012-11-23
 */
#include "Python.h"
#include "arrayobject.h"
#include "rave.h"
#include "rave_debug.h"
#include "pypolarvolume.h"
#include "pypolarscan.h"
#include "pyrave_debug.h"
#include "radvolatt.h"
#include "radvolbroad.h"
#include "radvolspeck.h"
#include "radvolspike.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_radvol");

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
 * Attenuation correction on "DBZH"
 * @param[in] PolarVolume_t object
 * @returns Py_True or Py_False
 */
static PyObject* _radvolatt_func(PyObject* self, PyObject* args) {
  PyObject* object = NULL;
  PyPolarVolume* pyvolume = NULL;

  if (!PyArg_ParseTuple(args, "O", &object)) {
    return NULL;
  }

  if (PyPolarVolume_Check(object)) {
    pyvolume = (PyPolarVolume*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "attCorrection requires PVOL as input");
  }

  if (RadvolAtt_attCorrection(pyvolume->pvol, NULL)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}


/**
 * Assessment of distance-to-radar related effects on "DBZH"
 * @param[in] PolarVolume_t object
 * @returns Py_True or Py_False
 */
static PyObject* _radvolbroad_func(PyObject* self, PyObject* args) {
  PyObject* object = NULL;
  PyPolarVolume* pyvolume = NULL;

  if (!PyArg_ParseTuple(args, "O", &object)) {
    return NULL;
  }

  if (PyPolarVolume_Check(object)) {
    pyvolume = (PyPolarVolume*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "broadAssessment requires PVOL as input");
  }

  if (RadvolBroad_broadAssessment(pyvolume->pvol, NULL)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}


/**
 * Speck removal
 * @param[in] PolarVolume_t object
 * @returns Py_True or Py_False
 */
static PyObject* _radvolspeck_func(PyObject* self, PyObject* args) {
  PyObject* object = NULL;
  PyPolarVolume* pyvolume = NULL;

  if (!PyArg_ParseTuple(args, "O", &object)) {
    return NULL;
  }

  if (PyPolarVolume_Check(object)) {
    pyvolume = (PyPolarVolume*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "speckRemoval requires PVOL as input");
  }

  if (RadvolSpeck_speckRemoval(pyvolume->pvol, NULL)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}


/**
 * Spike removal
 * @param[in] PolarVolume_t object
 * @returns Py_True or Py_False
 */
static PyObject* _radvolspike_func(PyObject* self, PyObject* args) {
  PyObject* object = NULL;
  PyPolarVolume* pyvolume = NULL;

  if (!PyArg_ParseTuple(args, "O", &object)) {
    return NULL;
  }

  if (PyPolarVolume_Check(object)) {
    pyvolume = (PyPolarVolume*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "spikeRemoval requires PVOL as input");
  }

  if (RadvolSpike_spikeRemoval(pyvolume->pvol, NULL)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}


static struct PyMethodDef _radvol_functions[] =
{
  { "attCorrection", (PyCFunction) _radvolatt_func, METH_VARARGS },
  { "broadAssessment", (PyCFunction) _radvolbroad_func, METH_VARARGS },
  { "speckRemoval", (PyCFunction) _radvolspeck_func, METH_VARARGS },
  { "spikeRemoval", (PyCFunction) _radvolspike_func, METH_VARARGS },
  { NULL, NULL }
};

/**
 * Initialize the _radvol module
 */
PyMODINIT_FUNC init_radvol(void)
{
  PyObject* m;
  m = Py_InitModule("_radvol", _radvol_functions);
  ErrorObject = PyString_FromString("_radvol.error");

  if (ErrorObject == NULL || PyDict_SetItemString(PyModule_GetDict(m),
                                                  "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _radvol.error");
  }
  import_pypolarvolume();
  import_pypolarscan();
  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
}

/*@} End of Module setup */
