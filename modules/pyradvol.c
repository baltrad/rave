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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION 
#include "pyravecompat.h"
#include "arrayobject.h"
#include "rave.h"
#include "rave_debug.h"
#include "pypolarvolume.h"
#include "pypolarscan.h"
#include "pyrave_debug.h"
#include "radvol.h"
#include "radvolatt.h"
#include "radvolbroad.h"
#include "radvolnmet.h"
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
 * Returns a native int from a Python object attribute
 * @param[in] ino Python object
 * @param[in] astr attribute name
 * @returns int
 */
static int getInt(PyObject* ino, const char* astr) {
  PyObject* o = NULL;
  int i=0;
  o = PyObject_GetAttrString(ino, astr);
  if (o != NULL) {
    if (PyLong_Check(o)) {
      i = (int)PyLong_AsLong(o);
    } else if (PyFloat_Check(o)) {
      i = (int)PyFloat_AsDouble(o);
    }
    Py_CLEAR(o);
  }
  return i;
}

/**
 * Returns a native double from a Python object attribute
 * @param[in] ino Python object
 * @param[in] astr attribute name
 * @returns double
 */
static double getDouble(PyObject* ino, const char* astr) {
  PyObject* o = NULL;
  double d;
  o = PyObject_GetAttrString(ino, astr);
  d = PyFloat_AsDouble(o);
  Py_CLEAR(o);
  return d;
}

/**
 * Maps attributes from a generic Python object to equivalent struct
 * @param[in] ino Python object
 * @param[in] params Radvol_params_t struct pointer
 */
static void mapParams(PyObject* ino, Radvol_params_t* params) {
  params->DBZHtoTH       = getInt(ino,    "DBZHtoTH");
  params->BROAD_QIOn     = getInt(ino,    "BROAD_QIOn");
  params->BROAD_QCOn     = getInt(ino,    "BROAD_QCOn");
  params->BROAD_LhQI1    = getDouble(ino, "BROAD_LhQI1");
  params->BROAD_LhQI0    = getDouble(ino, "BROAD_LhQI0");
  params->BROAD_LvQI1    = getDouble(ino, "BROAD_LvQI1");
  params->BROAD_LvQI0    = getDouble(ino, "BROAD_LvQI0");
  params->BROAD_Pulse    = getDouble(ino, "BROAD_Pulse");
  params->SPIKE_QIOn     = getInt(ino,    "SPIKE_QIOn");
  params->SPIKE_QCOn     = getInt(ino,    "SPIKE_QCOn");
  params->SPIKE_QI       = getDouble(ino, "SPIKE_QI");
  params->SPIKE_QIUn     = getDouble(ino, "SPIKE_QIUn");
  params->SPIKE_ACovFrac = getDouble(ino, "SPIKE_ACovFrac");
  params->SPIKE_AAzim    = getInt(ino,    "SPIKE_AAzim");
  params->SPIKE_AVarAzim = getInt(ino,    "SPIKE_AVarAzim");
  params->SPIKE_ABeam    = getInt(ino,    "SPIKE_ABeam");
  params->SPIKE_AVarBeam = getInt(ino,    "SPIKE_AVarBeam");
  params->SPIKE_AFrac    = getDouble(ino, "SPIKE_AFrac");
  params->SPIKE_BDiff    = getDouble(ino, "SPIKE_BDiff");
  params->SPIKE_BAzim    = getInt(ino,    "SPIKE_BAzim");
  params->SPIKE_BFrac    = getDouble(ino, "SPIKE_BFrac");
  params->NMET_QIOn      = getInt(ino,    "NMET_QIOn");
  params->NMET_QCOn      = getInt(ino,    "NMET_QCOn");
  params->NMET_QI        = getDouble(ino, "NMET_QI");
  params->NMET_QIUn      = getDouble(ino, "NMET_QIUn");
  params->NMET_AReflMin  = getDouble(ino, "NMET_AReflMin");
  params->NMET_AReflMax  = getDouble(ino, "NMET_AReflMax");
  params->NMET_AAltMin   = getDouble(ino, "NMET_AAltMin");
  params->NMET_AAltMax   = getDouble(ino, "NMET_AAltMax");
  params->NMET_ADet      = getDouble(ino, "NMET_ADet");
  params->NMET_BAlt      = getDouble(ino, "NMET_BAlt");
  params->SPECK_QIOn     = getInt(ino,    "SPECK_QIOn");
  params->SPECK_QCOn     = getInt(ino,    "SPECK_QCOn");
  params->SPECK_QI       = getDouble(ino, "SPECK_QI");
  params->SPECK_QIUn     = getDouble(ino, "SPECK_QIUn");
  params->SPECK_AGrid    = getDouble(ino, "SPECK_AGrid");
  params->SPECK_ANum     = getDouble(ino, "SPECK_ANum");
  params->SPECK_AStep    = getDouble(ino, "SPECK_AStep");
  params->SPECK_BGrid    = getDouble(ino, "SPECK_BGrid");
  params->SPECK_BNum     = getDouble(ino, "SPECK_BNum");
  params->SPECK_BStep    = getDouble(ino, "SPECK_BStep");
  params->BLOCK_QIOn     = getInt(ino,    "BLOCK_QIOn");
  params->BLOCK_QCOn     = getInt(ino,    "BLOCK_QCOn");
  params->BLOCK_MaxElev  = getDouble(ino, "BLOCK_MaxElev");
  params->BLOCK_dBLim    = getDouble(ino, "BLOCK_dBLim");
  params->BLOCK_GCQI     = getDouble(ino, "BLOCK_GCQI");
  params->BLOCK_GCQIUn   = getDouble(ino, "BLOCK_GCQIUn");
  params->BLOCK_GCMinPbb = getDouble(ino, "BLOCK_GCMinPbb");
  params->BLOCK_PBBQIUn  = getDouble(ino, "BLOCK_PBBQIUn");
  params->BLOCK_PBBMax   = getDouble(ino, "BLOCK_PBBMax");
  params->ATT_QIOn       = getInt(ino,    "ATT_QIOn");
  params->ATT_QCOn       = getInt(ino,    "ATT_QCOn");
  params->ATT_a          = getDouble(ino, "ATT_a");
  params->ATT_b          = getDouble(ino, "ATT_b");
  params->ATT_ZRa        = getDouble(ino, "ATT_ZRa");
  params->ATT_ZRb        = getDouble(ino, "ATT_ZRb");
  params->ATT_QIUn       = getDouble(ino, "ATT_QIUn");
  params->ATT_QI1        = getDouble(ino, "ATT_QI1");
  params->ATT_QI0        = getDouble(ino, "ATT_QI0");
  params->ATT_Refl       = getDouble(ino, "ATT_Refl");
  params->ATT_Last       = getDouble(ino, "ATT_Last");
  params->ATT_Sum        = getDouble(ino, "ATT_Sum");
}


/**
 * Attenuation correction on "DBZH"
 * @param[in] PolarVolume_t or PolarScan_t object
 * @param[in] Generic object containing algorithm parameters
 * @returns Py_True or Py_False
 */
static PyObject* _radvolatt_func(PyObject* self, PyObject* args) {
  PyObject* object = NULL;
  PyObject* params = NULL;
  Radvol_params_t rpars;
  PyPolarVolume* pyvolume = NULL;
  PyPolarScan* pyscan = NULL;
  int ret = 0;

  if (!PyArg_ParseTuple(args, "OO", &object, &params)) {
    return NULL;
  }

  if (PyPolarVolume_Check(object)) {
    pyvolume = (PyPolarVolume*)object;
  } else if (PyPolarScan_Check(object)) {
    pyscan = (PyPolarScan*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "attCorrection requires PVOL or SCAN as input");
  }

  mapParams(params, &rpars);

  if (PyPolarVolume_Check(object)) {
    ret = RadvolAtt_attCorrection_pvol(pyvolume->pvol, &rpars, NULL);
  } else {
    ret = RadvolAtt_attCorrection_scan(pyscan->scan, &rpars, NULL);
  }

  if (ret) {
    return PyBool_FromLong(1);
  } else {
    return PyBool_FromLong(0);
  }
}


/**
 * Assessment of distance-to-radar related effects on "DBZH"
 * @param[in] PolarVolume_t or PolarScan_t object
 * @param[in] Generic object containing algorithm parameters
 * @returns Py_True or Py_False
 */
static PyObject* _radvolbroad_func(PyObject* self, PyObject* args) {
  PyObject* object = NULL;
  PyObject* params = NULL;
  PyPolarVolume* pyvolume = NULL;
  Radvol_params_t rpars;
  PyPolarScan* pyscan = NULL;
  int ret = 0;

  if (!PyArg_ParseTuple(args, "OO", &object, &params)) {
    return NULL;
  }

  if (PyPolarVolume_Check(object)) {
    pyvolume = (PyPolarVolume*)object;
  } else if (PyPolarScan_Check(object)) {
    pyscan = (PyPolarScan*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "broadAssessment requires PVOL or SCAN as input");
  }

  mapParams(params, &rpars);

  if (PyPolarVolume_Check(object)) {
    ret = RadvolBroad_broadAssessment_pvol(pyvolume->pvol, &rpars, NULL);
  } else {
    ret = RadvolBroad_broadAssessment_scan(pyscan->scan, &rpars, NULL);
  }

  if (ret) {
    return PyBool_FromLong(1);
  } else {
    return PyBool_FromLong(0);
  }
}

/**
 * Non-meteorological echoes removal
 * @param[in] PolarVolume_t or PolarScan_t object
 * @param[in] Generic object containing algorithm parameters
 * @returns Py_True or Py_False
 */
static PyObject* _radvolnmet_func(PyObject* self, PyObject* args) {
  PyObject* object = NULL;
  PyObject* params = NULL;
  PyPolarVolume* pyvolume = NULL;
  Radvol_params_t rpars;
  PyPolarScan* pyscan = NULL;
  int ret = 0;

  if (!PyArg_ParseTuple(args, "OO", &object, &params)) {
    return NULL;
  }

  if (PyPolarVolume_Check(object)) {
    pyvolume = (PyPolarVolume*)object;
  } else if (PyPolarScan_Check(object)) {
    pyscan = (PyPolarScan*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "nmetRemoval requires PVOL or SCAN as input");
  }

  mapParams(params, &rpars);

  if (PyPolarVolume_Check(object)) {
    ret = RadvolNmet_nmetRemoval_pvol(pyvolume->pvol, &rpars, NULL);
  } else {
    ret = RadvolNmet_nmetRemoval_scan(pyscan->scan, &rpars, NULL);
  }

  if (ret) {
    return PyBool_FromLong(1);
  } else {
    return PyBool_FromLong(0);
  }
}

/**
 * Speck removal
 * @param[in] PolarVolume_t or PolarScan_t object
 * @param[in] Generic object containing algorithm parameters
 * @returns Py_True or Py_False
 */
static PyObject* _radvolspeck_func(PyObject* self, PyObject* args) {
  PyObject* object = NULL;
  PyObject* params = NULL;
  PyPolarVolume* pyvolume = NULL;
  Radvol_params_t rpars;
  PyPolarScan* pyscan = NULL;
  int ret = 0;

  if (!PyArg_ParseTuple(args, "OO", &object, &params)) {
    return NULL;
  }

   if (PyPolarVolume_Check(object)) {
    pyvolume = (PyPolarVolume*)object;
  } else if (PyPolarScan_Check(object)) {
    pyscan = (PyPolarScan*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "speckRemoval requires PVOL or SCAN as input");
  }

  mapParams(params, &rpars);

  if (PyPolarVolume_Check(object)) {
    ret = RadvolSpeck_speckRemoval_pvol(pyvolume->pvol, &rpars, NULL);
  } else {
    ret = RadvolSpeck_speckRemoval_scan(pyscan->scan, &rpars, NULL);
  }

  if (ret) {
    return PyBool_FromLong(1);
  } else {
    return PyBool_FromLong(0);
  }
}


/**
 * Spike removal
 * @param[in] PolarVolume_t or PolarScan_t object
 * @param[in] Generic object containing algorithm parameters
 * @returns Py_True or Py_False
 */
static PyObject* _radvolspike_func(PyObject* self, PyObject* args) {
  PyObject* object = NULL;
  PyObject* params = NULL;
  PyPolarVolume* pyvolume = NULL;
  Radvol_params_t rpars;
  PyPolarScan* pyscan = NULL;
  int ret = 0;

  if (!PyArg_ParseTuple(args, "OO", &object, &params)) {
    return NULL;
  }

  if (PyPolarVolume_Check(object)) {
    pyvolume = (PyPolarVolume*)object;
  } else if (PyPolarScan_Check(object)) {
    pyscan = (PyPolarScan*)object;
  } else {
    raiseException_returnNULL(PyExc_AttributeError, "spikeRemoval requires PVOL or SCAN as input");
  }

  mapParams(params, &rpars);

  if (PyPolarVolume_Check(object)) {
    ret = RadvolSpike_spikeRemoval_pvol(pyvolume->pvol, &rpars, NULL);
  } else {
    ret = RadvolSpike_spikeRemoval_scan(pyscan->scan, &rpars, NULL);
  }

  if (ret) {
    return PyBool_FromLong(1);
  } else {
    return PyBool_FromLong(0);
  }
}


static struct PyMethodDef _radvol_functions[] =
{
  { "attCorrection", (PyCFunction) _radvolatt_func, METH_VARARGS,
    "attCorrection(object, params) -> boolean\n\n"
    "Radvol-QC algorithms of correction for attenuation in rain.\n\n"
    "object - a polar volume or scan\n"
    "params - the radvol option class. Can be fetched using rave_radvol_realtime.get_options.\n\n"
    "Usage:\n"
    " import _radvol, rave_radvol_realtime, _raveio\n"
    " vol = _raveio.open(\"somevolume.h5\").object\n"
    " pars = rave_radvol_realtime.get_options(vol)\n"
    " _radvol.attCorrection(vol, pars)"
  },
  { "broadAssessment", (PyCFunction) _radvolbroad_func, METH_VARARGS,
    "broadAssessment(object, params) -> boolean\n\n"
    "Radvol-QC algorithms for assessment of distance to radar related effects.\n\n"
    "object - a polar volume or scan\n"
    "params - the radvol option class. Can be fetched using rave_radvol_realtime.get_options.\n\n"
    "Usage:\n"
    " import _radvol, rave_radvol_realtime, _raveio\n"
    " vol = _raveio.open(\"somevolume.h5\").object\n"
    " pars = rave_radvol_realtime.get_options(vol)\n"
    " _radvol.broadAssessment(vol, pars)"
  },
  { "nmetRemoval", (PyCFunction) _radvolnmet_func, METH_VARARGS,
    "nmetRemoval(object, params) -> boolean\n\n"
    "Radvol-QC algorithms for non-meteorological echoes removal.\n\n"
    "object - a polar volume or scan\n"
    "params - the radvol option class. Can be fetched using rave_radvol_realtime.get_options.\n\n"
    "Usage:\n"
    " import _radvol, rave_radvol_realtime, _raveio\n"
    " vol = _raveio.open(\"somevolume.h5\").object\n"
    " pars = rave_radvol_realtime.get_options(vol)\n"
    " _radvol.nmetRemoval(vol, pars)"
  },
  { "speckRemoval", (PyCFunction) _radvolspeck_func, METH_VARARGS,
    "speckRemoval(object, params) -> boolean\n\n"
    "Radvol-QC algorithms for speck removal.\n\n"
    "object - a polar volume or scan\n"
    "params - the radvol option class. Can be fetched using rave_radvol_realtime.get_options.\n\n"
    "Usage:\n"
    " import _radvol, rave_radvol_realtime, _raveio\n"
    " vol = _raveio.open(\"somevolume.h5\").object\n"
    " pars = rave_radvol_realtime.get_options(vol)\n"
    " _radvol.speckRemoval(vol, pars)"
  },
  { "spikeRemoval", (PyCFunction) _radvolspike_func, METH_VARARGS,
    "spikeRemoval(object, params) -> boolean\n\n"
    "Radvol-QC algorithms for spike removal.\n\n"
    "object - a polar volume or scan\n"
    "params - the radvol option class. Can be fetched using rave_radvol_realtime.get_options.\n\n"
    "Usage:\n"
    " import _radvol, rave_radvol_realtime, _raveio\n"
    " vol = _raveio.open(\"somevolume.h5\").object\n"
    " pars = rave_radvol_realtime.get_options(vol)\n"
    " _radvol.spikeRemoval(vol, pars)"
  },
  { NULL, NULL }
};

/*@{ Documentation about the module */
PyDoc_STRVAR(_pyradvol_module_doc,
    "Radvol-QC is software developed in IMGW-PIB (Poland) for corrections\n"
    "and generation of quality information for volumes of weather radar\n"
    "data. The work has been performed in the frame of the BALTRAD Project.\n"
    "\n"
    "At present the following algorithms are included in the Radvol-QC package:\n"
    " - BROAD: Assessment of distance to radar related effects (for quality characterization),\n"
    " - SPIKE: Removal of geometrically shaped non-meteorological echoes (from sun,emitters, etc.) (for data correction and quality characterization),\n"
    " - NMET: Removal of non-meteorological echoes (for data correction and quality characterization),\n"
    " - SPECK: Removal of measurement noise (specks) (for data correction and quality characterization),\n"
    " - [ BLOCK: Beam blockage correction (for data correction and quality characterization) - included into beamb package ],\n"
    " - ATT: Correction for attenuation in rain (for data correction and quality characterization).\n"
    "\n"
    );
/*@} End of Documentation about the module */

/**
 * Initialize the _radvol module
 */
MOD_INIT(_radvol)
{
  PyObject* module = NULL;
  PyObject* dictionary = NULL;

  MOD_INIT_DEF(module, "_radvol", _pyradvol_module_doc, _radvol_functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyErr_NewException("_radvol.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _radvol.error");
    return MOD_INIT_ERROR;
  }

  import_pypolarvolume();
  import_pypolarscan();
  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}

/*@} End of Module setup */
