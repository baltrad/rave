/* --------------------------------------------------------------------
Copyright (C) 2026 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the HOOF dealiasing API
 * @file
 * @author Peter Smerkol, Vito Švagelj, ARSO
 * @date 2026-11-13
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "pyravecompat.h"
#include "arrayobject.h"
#include "rave.h"
#include "rave_debug.h"
#include "hoofdealias.h"
#include "pypolarvolume.h"
#include "pypolarscan.h"
#include "pyrave_debug.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_hoofdealias");

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
 * Checks whether a scan is dealiased by looking up the param/how/dealiased attribute
 * of the specified quantity.
 * @param[in] PolarScan_t object and optionally the quantity (default VRADH)
 * @returns Py_True or Py_False
 */
static PyObject* _hoofdealiased_func(PyObject* self, PyObject* args)
{
  PyObject* object = NULL;
  char* quantity = "VRADH";

  if (!PyArg_ParseTuple(args, "O|s", &object, &quantity)) {
    return NULL;
  }
  if (!PyPolarScan_Check(object)) {
    raiseException_returnNULL(PyExc_AttributeError, "Dealiased check requires scan as input");
  }

  return PyBool_FromLong(hoofdealiased_by_quantity(((PyPolarScan*)object)->scan, (const char*)quantity));
}

/**
 * HOOF dealiasing of a polar volume
 * @param[in] PolarVolume_t object and optionally the quantity and the dealiasing parameters
 * @returns Py_True if at least one scan was dealiased, otherwise Py_False
 */
static PyObject* _hoofdealias_func(PyObject* self, PyObject* args, PyObject* kwds)
{
  PyObject* object = NULL;
  char* quantity = "VRADH";
  double zsector = HOOFDEALIAS_ZSECTOR;
  double zmax = HOOFDEALIAS_ZMAX;
  int minpoints = HOOFDEALIAS_MINPOINTS;
  double maxwind = HOOFDEALIAS_MAXWIND;
  static char* kwlist[] = {"object", "quantity", "zsector", "zmax", "minpoints", "maxwind", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|sddid", kwlist, &object, &quantity, &zsector, &zmax, &minpoints, &maxwind)) {
    return NULL;
  }
  if (!PyPolarVolume_Check(object)) {
    raiseException_returnNULL(PyExc_AttributeError, "HOOF dealiasing requires a volume as input, since the wind model is fitted using all scans");
  }
  if (zsector <= 0.0 || maxwind <= 0.0) {
    raiseException_returnNULL(PyExc_ValueError, "zsector and maxwind must be positive");
  }

  return PyBool_FromLong(hoofdealias_pvol_by_quantity(((PyPolarVolume*)object)->pvol, (const char*)quantity,
                                                      zsector, zmax, minpoints, maxwind));
}

static struct PyMethodDef _hoofdealias_functions[] =
{
  { "dealiased", (PyCFunction) _hoofdealiased_func, METH_VARARGS,
      "dealiased(scan[,quantity]) -> boolean\n\n"
      "Checks whether a scan is dealiased by looking up the specified quantity's param/how/dealiased attribute.\n\n"
      "scan      - the polar scan\n"
      "quantity  - the quantity that should be queried. If not specified, it defaults to VRADH."
  },
  { "dealias", (PyCFunction) _hoofdealias_func, METH_VARARGS | METH_KEYWORDS,
      "dealias(volume[,quantity,zsector,zmax,minpoints,maxwind]) -> boolean\n\n"
      "HOOF dealiasing (torus mapping) of the specified quantity in all scans of a polar volume, in place.\n"
      "A wind model is fitted for each height layer using all scans. Bins in height layers without a\n"
      "wind model can not be dealiased and are set to nodata. Returns True if at least one scan was dealiased.\n\n"
      "volume    - the polar volume\n"
      "quantity  - the quantity to dealias, defaults to VRADH (RAVE reads ODIM 2.2 VRAD as VRADH)\n"
      "zsector   - height layer thickness in m, defaults to 100\n"
      "zmax      - maximum height above sea level of the height layers in m, defaults to no limit\n"
      "minpoints - a wind model needs more than minpoints points in a height layer, defaults to 500\n"
      "maxwind   - maximum wind speed of a wind model in m/s, defaults to 60"
  },
  { NULL, NULL }
};

/*@{ Documentation about the type */
PyDoc_STRVAR(_hoofdealias_module_doc,
  "Provides HOOF dealiasing (torus mapping with height layer wind models) of radial wind data."
);
/*@} End of Documentation about the type */

/**
 * Initialize the _hoofdealias module
 */
MOD_INIT(_hoofdealias)
{
  PyObject* module = NULL;
  PyObject* dictionary = NULL;

  MOD_INIT_DEF(module, "_hoofdealias", _hoofdealias_module_doc, _hoofdealias_functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyErr_NewException("_hoofdealias.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _hoofdealias.error");
    return MOD_INIT_ERROR;
  }

  import_pypolarvolume();
  import_pypolarscan();
  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}

/*@} End of Module setup */
