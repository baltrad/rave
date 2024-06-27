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
 * Python interface to KNMI's sun scanning functionality
 * @file
 * @author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2011-01-19
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <pyravecompat.h>
#include "rave.h"
#include "rave_debug.h"
#include "pypolarvolume.h"
#include "pypolarscan.h"
#include "pyrave_debug.h"
#include "scansun.h"
#include "pyraveio.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_scansun");

/**
 * Sets a Python exception.
 */
#define Raise(type,msg) {PyErr_SetString(type,msg);}

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
 * This function calculates the solar elevation and azimuth using the
 * geographical position, date, and time. The equations and constants are taken
 * from the WMO guide on Meteorological Instruments and Methods of Observations
 * (CIMO, WMO no. 8), annex 7.D. The equations have been slightly modified and
 * extended to include the calculation of both the sine and cosine of the
 * azimuth.
 * Modified slightly further to include the refracted (perceived) elevation angle.
 * @param[in] lon - double containing the longitude position
 * @param[in] lat - double containing the latitude position
 * @param[in] yyyymmdd - year-month-day as a long
 * @param[in] hhmmss - hour-minute-second as a long
 * @param[out] elev - elevation angle above the horizon in degrees, as a pointer to a double
 * @param[out] azim - azimuth angle clockwise from true north, as a pointer to a double
 * @param[out] relev - refracted elevation angle, based on elev, as a pointer to a double
 * @returns a Tuple containing the output values as PyFloats
 */
static PyObject* _solar_elev_azim_func(PyObject* self, PyObject* args)
{
  double lon, lat, elev, azim, relev;
  long yyyymmdd, hhmmss;

  if (!PyArg_ParseTuple(args, "ddll", &lon, &lat, &yyyymmdd, &hhmmss)) {
    return NULL;
  }

  solar_elev_azim(lon, lat, yyyymmdd, hhmmss, &elev, &azim, &relev);

  return Py_BuildValue("ddd", elev, azim, relev);
}

/**
 * Calculation of refraction.
 * @param[in] PyFloat containing the real elevation angle of the sun.
 * @returns PyFloat containing the corrected (perceived) elevation angle.
 */
static PyObject* _refraction_func(PyObject* self, PyObject* args)
{
  double elev, refr;

  if (!PyArg_ParseTuple(args, "d", &elev)) {
    return NULL;
  }

  refr = elev + refraction(&elev);
  return Py_BuildValue("d", refr);
}

/**
 * Performs full sun scan on polar data, either scan or volume, from object in memory.
 * @param[in] Polar data object, either a scan or a volume.
 * @returns a Python list containing tuples, one for each sun hit, or None if no hits.
 */
static PyObject* _scansunFromObject_func(PyObject* self, PyObject* args)
{
	char* source = NULL;
	RaveList_t* list = RAVE_OBJECT_NEW(&RaveList_TYPE);
	RVALS* ret = NULL;
	PyObject* rlist;
	PyObject* reto;
	PyObject* pyobject = NULL;
	PyPolarVolume* pyvolume = NULL;
	PyPolarScan* pyscan = NULL;
	RaveCoreObject* object = NULL;
	Rave_ObjectType ot = Rave_ObjectType_UNDEFINED;

	if (!PyArg_ParseTuple(args, "O", &pyobject)) {
		RAVE_OBJECT_RELEASE(list);
		return NULL;
	}

	if (PyPolarVolume_Check(pyobject)) {
		pyvolume = (PyPolarVolume*)pyobject;
		object = (RaveCoreObject*)pyvolume->pvol;
		ot = Rave_ObjectType_PVOL;
	} else if (PyPolarScan_Check(pyobject)) {
		pyscan = (PyPolarScan*)pyobject;
		object = (RaveCoreObject*)pyscan->scan;
		ot = Rave_ObjectType_SCAN;
	} else {
		RAVE_OBJECT_RELEASE(list);
		raiseException_returnNULL(PyExc_AttributeError, "scansunFromObject requires PVOL or SCAN as input");
	}

	if (!scansunFromObject(object, ot, list, &source)) {
		RAVE_OBJECT_RELEASE(list);
		raiseException_returnNULL(PyExc_RuntimeError, "scansunFromObject could not process file");
	}

	rlist = PyList_New(0);
	if (RaveList_size(list) > 0) {
		while ((ret = RaveList_removeLast(list)) != NULL) {
			PyObject* rtuple = Py_BuildValue("ldddddidddddss", ret->date,
                                                         	   ret->timer,
															   ret->Elev,
															   ret->Azimuth,
															   ret->ElevSun,
															   ret->AzimSun,
															   ret->n,
															   ret->dBSunFlux,
															   ret->SunMean,
															   ret->SunStdd,
															   ret->ZdrMean,
															   ret->ZdrStdd,
															   ret->quant1,
															   ret->quant2);
			PyList_Append(rlist, rtuple);
			Py_DECREF(rtuple);
			RAVE_FREE(ret);
		}
	}

	RAVE_OBJECT_RELEASE(list);
	reto = Py_BuildValue("sO", source, rlist);
	Py_DECREF(rlist);
	RAVE_FREE(source);
	return reto;
}

/**
 * Performs full sun scan on polar data, either scan or volume.
 * @param[in] filename - const char* containing the filename to process
 * @returns a Python list containing tuples, one for each sun hit, or None if no hits.
 */
static PyObject* _scansun_func(PyObject* self, PyObject* args)
{
	char* source = NULL;
	RaveList_t* list = RAVE_OBJECT_NEW(&RaveList_TYPE);
	RVALS* ret = NULL;
	const char* filename;
	PyObject* rlist;
	PyObject* reto;

	if (!PyArg_ParseTuple(args, "s", &filename)) {
		RAVE_OBJECT_RELEASE(list);
		return NULL;
	}

	if (!scansun(filename, list, &source)) {
		RAVE_OBJECT_RELEASE(list);
		raiseException_returnNULL(PyExc_IOError, "Scansun could not process file");
	}

	rlist = PyList_New(0);
	if (RaveList_size(list) > 0) {
		while ((ret = RaveList_removeLast(list)) != NULL) {
			PyObject* rtuple = Py_BuildValue("ldddddidddddss", ret->date,
                                                         	   ret->timer,
															   ret->Elev,
															   ret->Azimuth,
															   ret->ElevSun,
															   ret->AzimSun,
															   ret->n,
															   ret->dBSunFlux,
															   ret->SunMean,
															   ret->SunStdd,
															   ret->ZdrMean,
															   ret->ZdrStdd,
															   ret->quant1,
															   ret->quant2);
			PyList_Append(rlist, rtuple);
			Py_DECREF(rtuple);
			RAVE_FREE(ret);
		}
	}
	RAVE_OBJECT_RELEASE(list);
	reto = Py_BuildValue("sO", source, rlist);
	Py_DECREF(rlist);
	RAVE_FREE(source);
	return reto;
}

static struct PyMethodDef _scansun_functions[] =
{
  { "solar_elev_azim", (PyCFunction) _solar_elev_azim_func, METH_VARARGS,
    "solar_elev_azim(lon, lat, yyyymmdd, hhmmss) -> (elev, azim, relev)\n\n"
    "This function calculates the solar elevation and azimuth using the\n"
    "geographical position, date, and time. The equations and constants are taken\n"
    "from the WMO guide on Meteorological Instruments and Methods of Observations\n"
    "(CIMO, WMO no. 8), annex 7.D. The equations have been slightly modified and\n"
    "extended to include the calculation of both the sine and cosine of the azimuth\n\n"
    "Modified slightly further to include the refracted (perceived) elevation angle.\n\n"
    "lon      - double containing the longitude position in degrees\n"
    "lat      - double containing the longitude position in degrees\n"
    "yyyymmdd - year-month-day as a long\n"
    "hhmmss   - hour-minute-second as a long\n\n"
    "The returned values are:\n"
    "elev - elevation angle above the horizon in degrees, as a pointer to a double\n"
    "azim - azimuth angle clockwise from true north, as a pointer to a double\n"
    "relev - refracted elevation angle, based on elev, as a pointer to a double"
  },
  { "refraction", (PyCFunction) _refraction_func, METH_VARARGS,
    "refraction(elev) -> refr\n\n"
    "Calculation of refraction. Returns the the corrected (perceived) elevation angle.\n\n"
    "elev - the real elevation angle of the sun in degrees\n"
  },
  { "scansunFromObject", (PyCFunction) _scansunFromObject_func, METH_VARARGS,
    "scansunFromObject(object) -> (source, list)\n\n"
    "Performs full sun scan on polar data, either scan or volume, from object in memory.\n\n"
    "object - the inobject, can either be a polar scan or polar volume\n\n"
    "Returns a tuple (source, list) where first value is the what/source, the second is a list of values."
  },
  { "scansun", (PyCFunction) _scansun_func, METH_VARARGS,
    "scansun(filename) -> (source, list)\n\n"
    "Performs full sun scan on polar data, either scan or volume from a file pointed out by filename.\n\n"
    "filename - the file that should be analyzed\n\n"
    "Returns a tuple (source, list) where first value is the what/source, the second is a list of values."
  },
  { NULL, NULL }
};

/*@{ Documentation about the module */
PyDoc_STRVAR(_scansun_module_doc,
    "This module is based on KNMI's sun scanning functionality\n"
    "Original (but updated) preamble\n"
    "This program scans volume reflectivity data in ODIM_H5 format for solar\n"
    "interferences using SMHI's HL-HDF library. Added features:\n"
    "1) Use of minimum elevation and range is changed to use of minimum height and range\n"
    "2) Received solar power is normalized to the band width"
    );
/*@} End of Documentation about the module */

/**
 * Initialize the _scansun module
 */
MOD_INIT(_scansun)
{
  PyObject* module = NULL;
  PyObject* dictionary = NULL;
  MOD_INIT_DEF(module, "_scansun", _scansun_module_doc, _scansun_functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyErr_NewException("_scansun.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _scansun.error");
    return MOD_INIT_ERROR;
  }

  import_pyraveio();
  import_pypolarvolume();
  import_pypolarscan();
  import_array();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}

/*@} End of Module setup */
