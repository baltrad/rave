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
 * Python interface to RACK
 * @file
 * @author Daniel Mattsson on behalf of Swedish Meteorological and Hydrological Institute (SMHI)
 * @date 2011-04-13
 */

#include <Python.h>
#include "pypolarvolume.h"
#include "pypolarscan.h"
#include "pyraveio.h"
#include <polarvolume.h>
#include <polarscan.h>
#include "RackIf.h"
#include "pyrave_debug.h"


/**
 * Name of the module debugged.
 */
PYRAVE_DEBUG_MODULE("_rack");

static PyObject *ErrorObject;

/**
 * Sets a Python exception.
 */
#define Raise(type,msg) {PyErr_SetString(type,msg);}

/**
 * This function calls RACK with either a PyPolarScan or a PyPolarVolume and a string
 * with arguments matching the input when calling RACK from command line. In other
 * words, the string matches the regular argv. 
 * @param[in] polarObj - PyObject containing a PyPolarVolume or PyPolarScan
 * @param[in] arguments - string of arguments
 * @returns a PyObject containing a PyPolarVolume or PyPolarScan, NULL if failed.
 */
static PyObject* _rack_func(PyObject* self, PyObject* args)
{
  const char* argv;
  PyObject* polarObj = NULL;
  PyObject* retObj = NULL;
  //printf("Starting rack_func!\n");

  if (!PyArg_ParseTuple(args, "Os", &polarObj, &argv)) {
    printf("Failed to Parse tuple in _rack_func.\n");
    return NULL;
  }
  if (PyPolarVolume_Check(polarObj)) {
    PolarVolume_t* pvol = PyPolarVolume_GetNative((PyPolarVolume*)polarObj);
    PolarVolume_t* retVol = (PolarVolume_t*)execRack((RaveCoreObject*)pvol, argv);
    if (retVol != NULL){
      //printf("execRack didn't return NULL ptr.\n");
      retObj = (PyObject*)PyPolarVolume_New(retVol);
    }
    RAVE_OBJECT_RELEASE(pvol);
    RAVE_OBJECT_RELEASE(retVol); 
  }
  else if (PyPolarScan_Check(polarObj)) {
    PolarScan_t* scan = PyPolarScan_GetNative((PyPolarScan*)polarObj);
    PolarScan_t* retScan = (PolarScan_t*)execRack((RaveCoreObject*)scan, argv);
    if (retScan != NULL){
      //printf("execRack didn't return NULL ptr.\n");
      retObj = (PyObject*)PyPolarScan_New(retScan);
    }
    RAVE_OBJECT_RELEASE(scan);
    RAVE_OBJECT_RELEASE(retScan);
  }
  else {
    Py_FatalError("Can't convert from PyObject to polar type _rack.error");
  }
  return retObj;
}

/**
 * This function converts a PyRaveIO to the corresponding Polar type, for instance
 * PyPolarScan or PyPolarVolume.
 * @param[in] polarObj - PyObject containing PyRaveIO.
 * @returns a PyObject containing a PyPolarVolume or PyPolarScan, NULL if failed.
 */
static PyObject* _rack_get_polar_from_raveio_func(PyObject* self, PyObject* args)
{
  PyObject* obj = NULL;
  RaveIO_t* raveio = NULL; 
  PyObject* retObj = NULL;

  if (!PyArg_ParseTuple(args, "O", &obj)) {
    printf("Failed to Parse tuple in _rack_get_polar_from_raveio_func.\n");
    return NULL;
  }
  raveio = ((PyRaveIO*)obj)->raveio;
  Rave_ObjectType raveType = RaveIO_getObjectType(raveio);
  if (raveType == Rave_ObjectType_PVOL) {
    PolarVolume_t* pvol = (PolarVolume_t*)RaveIO_getObject(raveio);
    retObj = (PyObject*)PyPolarVolume_New(pvol);
    RAVE_OBJECT_RELEASE(pvol);
  }
  else if (raveType == Rave_ObjectType_SCAN) {
    PolarScan_t* scan = (PolarScan_t*)RaveIO_getObject(raveio);
    PyPolarScan* pyScan = PyPolarScan_New(scan);
    retObj = (PyObject*)pyScan;
    RAVE_OBJECT_RELEASE(scan);
  }
  else {
    Py_FatalError("Can't convert from PyRaveIO to polar type _rack.error");
  }
  return retObj;
}

static struct PyMethodDef _rack_functions[] =
{
  { "rack", (PyCFunction) _rack_func, METH_VARARGS },
  { "getPolarFromRaveIO", (PyCFunction) _rack_get_polar_from_raveio_func, METH_VARARGS },
  { NULL, NULL }
};

/**
 * Initialize the _rack module
 */
PyMODINIT_FUNC init_rack(void)
{
  PyObject* m;
  m = Py_InitModule("_rack", _rack_functions);
  ErrorObject = PyString_FromString("_rack.error");
  if (ErrorObject == NULL || PyDict_SetItemString(PyModule_GetDict(m),
                                                  "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _rack.error");
  }

  import_pypolarvolume();
  import_pypolarscan();
  PYRAVE_DEBUG_INITIALIZE;
}
