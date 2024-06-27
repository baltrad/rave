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
 * Python wrapper for the rave product generation framework.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-14
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION 
#include <pyravecompat.h>
#include <arrayobject.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "rave_object.h"
#include "raveutil.h"
#include "rave.h"
#include "rave_types.h"
#include "rave_io.h"
#include "pypolarvolume.h"
#include "pytransform.h"
#include "pyraveio.h"
#include "pypolarscan.h"
#include "pycartesian.h"
#include "pycartesianparam.h"
#include "pyprojection.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_utilities.h"
#include "pyrave_debug.h"
#include "rave_datetime.h"
#include "proj_wkt_helper.h"

/**
 * This modules name
 */
PYRAVE_DEBUG_MODULE("_rave");

/**
 * Sets python exception and goto tag.
 */
#define raiseException_gotoTag(tag, type, msg) \
{PyErr_SetString(type, msg); goto tag;}

/**
 * Sets python exception string and returns NULL
 */
#define raiseException_returnNULL(type, msg) \
{PyErr_SetString(type, msg); return NULL;}

/**
 * Error object for reporting errors to the python interpreeter
 */
static PyObject *ErrorObject;

/*@{ Polar Scans */
/**
 * Creates a new instance of the polar scan.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _polarscan_new(PyObject* self, PyObject* args)
{
  PyPolarScan* result = PyPolarScan_New(NULL);
  return (PyObject*)result;
}
/*@} End of Polar Scans */

/*@{ Polar Volumes */
/**
 * Creates a new instance of the polar volume.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _polarvolume_new(PyObject* self, PyObject* args)
{
  PyPolarVolume* result = PyPolarVolume_New(NULL);
  return (PyObject*)result;
}
/*@} End of Polar Volumes */

/*@{ Cartesian products */
/**
 * Creates a new instance of the cartesian product.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _cartesian_new(PyObject* self, PyObject* args)
{
  PyCartesian* result = PyCartesian_New(NULL);
  return (PyObject*)result;
}

/**
 * Creates a new instance of the cartesian parameter.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _cartesianparam_new(PyObject* self, PyObject* args)
{
  PyCartesianParam* result = PyCartesianParam_New(NULL);
  return (PyObject*)result;
}
/*@} End of Cartesian products */

/*@{ Transform */
/**
 * Creates a new instance of the transformator.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _transform_new(PyObject* self, PyObject* args)
{
  PyTransform* result = PyTransform_New(NULL);
  return (PyObject*)result;
}

/*@{ End of Transform */

/*@{ New projection */
static PyObject* _projection_new(PyObject* self, PyObject* args)
{
  PyProjection* result = NULL;
  char* id = NULL;
  char* description = NULL;
  char* definition = NULL;

  if (!PyArg_ParseTuple(args, "sss", &id, &description, &definition)) {
    return NULL;
  }

  result = PyProjection_NewFromDef(id, description, definition);

  return (PyObject*)result;
}
/*@} End of New projection */

/*@{ RaveIO */
/**
 * Creates a new RaveIO instance.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (Not USED)
 * @return the object on success, otherwise NULL
 */
static PyObject* _raveio_new(PyObject* self, PyObject* args)
{
  PyRaveIO* result = PyRaveIO_New(NULL);
  return (PyObject*)result;
}

/**
 * Opens a file that is supported by RaveIO.
 * @param[in] self this instance.
 * @param[in] args arguments for creation. (A string identifying the file)
 * @return the object on success, otherwise NULL
 */
static PyObject* _raveio_open(PyObject* self, PyObject* args)
{
  PyRaveIO* result = NULL;
  char* filename = NULL;
  int lazyLoading = 0;
  char* preloadQuantities = NULL;
  if (!PyArg_ParseTuple(args, "s|iz", &filename, &lazyLoading, &preloadQuantities)) {
    return NULL;
  }
  result = PyRaveIO_Open(filename, lazyLoading, preloadQuantities);
  return (PyObject*)result;
}

/*@} End of RaveIO */

/*@{ Rave utilities */
/**
 * Returns if xml is supported or not in this build.
 * @param[in] self - self
 * @param[in] args - N/A
 * @returns true if xml is supported otherwise false
 */
static PyObject* _rave_isxmlsupported(PyObject* self, PyObject* args)
{
  return PyBool_FromLong(RaveUtilities_isXmlSupported());
}

/**
 * Returns if CF conventions is supported or not in this build.
 * @param[in] self - self
 * @param[in] args - N/A
 * @returns true if xml is supported otherwise false
 */
static PyObject* _rave_isCFConventionSupported(PyObject* self, PyObject* args)
{
  return PyBool_FromLong(RaveUtilities_isCFConventionSupported());
}

/**
 * Returns if legacy proj (PROJ.4 and PROJ 5) is supported or not in this build.
 * @param[in] self - self
 * @param[in] args - N/A
 * @returns true if PROJ.4 is enabled otherwise false
 */
static PyObject* _rave_isLegacyProjEnabled(PyObject* self, PyObject* args)
{
  return PyBool_FromLong(RaveUtilities_isLegacyProjEnabled());
}

/**
 * Sets a specific debug level
 * @param[in] self - self
 * @param[in] args - the debug level as an integer
 * @return None
 */
static PyObject* _rave_setDebugLevel(PyObject* self, PyObject* args)
{
  int lvl = RAVE_SILENT;
  if (!PyArg_ParseTuple(args, "i", &lvl)) {
    return NULL;
  }
  Rave_setDebugLevel(lvl);
  Py_RETURN_NONE;
}

static PyObject* _rave_setTrackObjectCreation(PyObject* self, PyObject* args)
{
  int track = 0;
  if (!PyArg_ParseTuple(args, "i", &track)) {
    return NULL;
  }
  RaveCoreObject_setTrackObjects(track);
  Py_RETURN_NONE;  
}

static PyObject* _rave_getTrackObjectCreation(PyObject* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyInt_FromLong(RaveCoreObject_getTrackObjects());
}

/**
 * Simple helper to compare two rave date time pairs.
 * @param[in] self - self
 * @param[in] args - (d1,t1,d2,t2) where date is in format YYYYmmdd and time is in HHMMSS.
 * @return negative if d1/t1 is before d2/t2. 0 if they are equal and a positive value otherwise
 */
static PyObject* _rave_compare_datetime(PyObject* self, PyObject* args)
{
  char *d1=NULL, *t1=NULL, *d2=NULL, *t2=NULL;
  RaveDateTime_t* dt1 = NULL;
  RaveDateTime_t* dt2 = NULL;
  int result = -1;
  if (!PyArg_ParseTuple(args, "ssss", &d1, &t1, &d2, &t2)) {
    return NULL;
  }
  dt1 = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  dt2 = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  if (dt1 != NULL && dt2 != NULL) {
    if (!RaveDateTime_setDate(dt1, d1) ||
        !RaveDateTime_setTime(dt1, t1) ||
        !RaveDateTime_setDate(dt2, d2) ||
        !RaveDateTime_setTime(dt2, t2)) {
      raiseException_gotoTag(done, PyExc_AttributeError, "Could not set date time strings");
    }
    result = RaveDateTime_compare(dt1, dt2);
  }

done:
  RAVE_OBJECT_RELEASE(dt1);
  RAVE_OBJECT_RELEASE(dt2);
  return PyLong_FromLong(result);
}

/**
 * Translates a projection into a list of well known text attributes representing the projection
 * @param[in] self - self
 * @param[in] args - a ProjectionCore instance
 */
static PyObject* _rave_translate_from_projection_to_wkt(PyObject* self, PyObject* args)
{
  PyObject* obj = NULL;
  RaveObjectList_t* attrList = NULL;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "O", &obj))
    return NULL;
  if (!PyProjection_Check(obj)) {
    raiseException_returnNULL(PyExc_AttributeError, "Must be of type ProjectionCore");
  }
  attrList = RaveWkt_translate_from_projection(((PyProjection*)obj)->projection);
  if (attrList != NULL) {
    int i = 0, n = 0;
    result = PyList_New(0);
    n = RaveObjectList_size(attrList);
    for (i = 0; i < n; i++) {
      RaveAttribute_t* attr = (RaveAttribute_t*)RaveObjectList_get(attrList, i);
      if (RaveAttribute_getFormat(attr) == RaveAttribute_Format_String) {
        char* v = NULL;
        PyObject* o = NULL;
        RaveAttribute_getString(attr, &v);
        o = Py_BuildValue("(ss)", RaveAttribute_getName(attr), v);
        PyList_Append(result, o);
        Py_XDECREF(o);
      } else if (RaveAttribute_getFormat(attr) == RaveAttribute_Format_Double) {
        double v = 0.0;
        PyObject* o=NULL;
        RaveAttribute_getDouble(attr, &v);
        o = Py_BuildValue("(sd)", RaveAttribute_getName(attr), v);
        PyList_Append(result, o);
        Py_XDECREF(o);
      } else if (RaveAttribute_getFormat(attr) == RaveAttribute_Format_DoubleArray) {
        double* v = NULL;
        int vn = 0;
        int vi = 0;
        PyObject* o = NULL;
        PyObject* own = NULL;
        RaveAttribute_getDoubleArray(attr, &v, &vn);
        o = PyList_New(0);
        for (vi = 0; vi < vn; vi++) {
          PyObject* ov = PyFloat_FromDouble(v[vi]);
          PyList_Append(o, ov);
          Py_XDECREF(ov);
        }
        own = Py_BuildValue("(sO)", RaveAttribute_getName(attr), o);
        PyList_Append(result, own);
        Py_XDECREF(o);
        Py_XDECREF(own);
      }
      RAVE_OBJECT_RELEASE(attr);
    }
  }
  RAVE_OBJECT_RELEASE(attrList);
  return result;
}


/*@} End of Rave utilities */

/// --------------------------------------------------------------------
/// Module setup
/// --------------------------------------------------------------------
/*@{ Module setup */
static PyMethodDef functions[] = {
  {"volume", (PyCFunction)_polarvolume_new, 1,
    "volume() -> polar volume\n\n"
    "Creates a new instance of PolarVolumeCore"
  },
  {"scan", (PyCFunction)_polarscan_new, 1,
    "scan() -> polar scan\n\n"
    "Creates a new instance of PolarScanCore"
  },
  {"cartesian", (PyCFunction)_cartesian_new, 1,
    "cartesian() -> cartesian\n\n"
    "Creates a new instance of CartesianCore"
  },
  {"cartesianparam", (PyCFunction)_cartesianparam_new, 1,
    "cartesianparam() -> cartesian parameter\n\n"
    "Creates a new instance of CartesianParamCore"
  },
  {"transform", (PyCFunction)_transform_new, 1,
    "transform() -> transform instance\n\n"
    "Creates a new instance of TransformCore"
  },
  {"projection", (PyCFunction)_projection_new, 1,
    "projection(id, description, definition) -> projection\n\n"
    "Creates a projection instance\n\n"
    "id          - Id of this projection instance\n"
    "description - Description of this projection instance\n"
    "definition  - The PROJ.4 definition"
  },
  {"io", (PyCFunction)_raveio_new, 1,
    "io() -> rave io instance\n\n"
    "Creates a RaveIOCore instance"
  },
  {"open", (PyCFunction)_raveio_open, 1,
    "open(filename) -> rave io core instance\n\n"
    "Opens specified file and loads it into a rave io core instance\n\n"
    "filename - the path to the file to be loaded"
  },
  {"isXmlSupported", (PyCFunction)_rave_isxmlsupported, 1,
    "isXmlSupported() -> a boolean\n\n"
    "Returns if this build of rave supports xml loading"
  },
  {"isCFConventionSupported", (PyCFunction)_rave_isCFConventionSupported, 1,
    "isCFConventionSupported() -> a boolean\n\n"
    "Returns if this build has been built with support for writing files in CF convention."
  },
  {"isLegacyProjEnabled", (PyCFunction)_rave_isLegacyProjEnabled, 1,
    "isLegacyProjEnabled() -> a boolean\n\n"
    "Returns if this build has been built with support for PROJ.4 & PROJ 5 (True) or if the PROJ version is >= 5 (False)."
  },
  {"setDebugLevel", (PyCFunction)_rave_setDebugLevel, 1,
    "setDebugLevel(level)\n\n"
    "Sets the debug level to use when running the rave c modules. Can be one of:\n"
    "  + Debug_RAVE_SPEWDEBUG    - This provides a lot of debuginformation, most which probably is not interesting\n"
    "  + Debug_RAVE_DEBUG        - Basic debug information\n"
    "  + Debug_RAVE_DEPRECATED   - Print information from deprecated functions\n"
    "  + Debug_RAVE_INFO         - Information\n"
    "  + Debug_RAVE_WARNING      - Warnings\n"
    "  + Debug_RAVE_ERROR        - Errors\n"
    "  + Debug_RAVE_CRITICAL     - Critical errors, typically if this occur, it probably ends with a crash\n"
    "  + Debug_RAVE_SILENT       - Don't display anything (default)\n"
  },
  {"setTrackObjectCreation", (PyCFunction)_rave_setTrackObjectCreation, 1,
    "setTrackObjectCreation(boolean)\n\n"
    "Sets object creation should be monitored or not\n"
  },
  {"getTrackObjectCreation", (PyCFunction)_rave_getTrackObjectCreation, 1,
    "getTrackObjectCreation()\n\n"
    "Returns if object creation is monitored or not\n"
  },
  {"compare_datetime", (PyCFunction) _rave_compare_datetime, 1,
    "compare_datetime(d1,t1,d2,t2) -> an integer\n\n"
    "Since several date/times used in the rave objects are defined as date + time, this utility function will help comparing two different date+time pairs.\n"
    "The returned value will be if negative if d1+t1 is before d2+t2. 0 if they are equal and a positive value otherwise\n\n"
    "d1 - First date in format YYYYmmdd\n"
    "t1 - First time in format HHMMSS\n"
    "d2 - Second date in format YYYYmmdd\n"
    "t2 - Second time in format HHMMSS\n"
  },
  {"translate_from_projection_to_wkt", (PyCFunction) _rave_translate_from_projection_to_wkt, 1,
    "translate_from_projection_to_wkt(proj) -> list\n\n"
    "Translates a projection into a list of well known text attributes representing the projection.\n"
    "proj - the ProjectionCore instance.\n\n"
    "Example:\n"
    ">>>proj = _rave.projection(\"myid\", \"laea\", \"+proj=laea +lat_0=1 +lon_0=2 +x_0=14 +y_0=60 +R=6378137.0\")\n"
    ">>>_rave.translate_from_projection_to_wkt(proj)\n"
    "[('grid_mapping_name', 'lambert_azimuthal_equal_area'),\n"
    " ('longitude_of_projection_origin', 2.0),\n"
    " ('latitude_of_projection_origin', 1.0),\n"
    " ('false_easting', 14.0),\n"
    " ('false_northing', 60.0),\n"
    " ('earth_radius', 6378137.0)]"
  },
  {NULL,NULL} /*Sentinel*/
};

/*@{ Documentation about the type */
PyDoc_STRVAR(_rave_module_doc,
    "The _rave module provides some utility functions when running the rave software, like setting debug level, provide constants, creating objects and check if some features are enabled.\n"
    "Since the actual functions describes their usage the various constants are first going to be described below:\n"
    "Data types are used when creating data fields or querying data fields.\n"
    "  + RaveDataType_UNDEFINED - If data has not been initialized yet\n"
    "  + RaveDataType_CHAR      - If data is defined as char (8-bit)\n"
    "  + RaveDataType_UCHAR     - If data is defined as unsigned char (8-bit)\n"
    "  + RaveDataType_SHORT     - If data is defined as short integer (16-bit)\n"
    "  + RaveDataType_USHORT    - If data is defined as unsigned short integer (16-bit)\n"
    "  + RaveDataType_INT       - If data is defined as integer (32-bit)\n"
    "  + RaveDataType_UINT      - If data is defined as unsigned integer (32-bit)\n"
    "  + RaveDataType_LONG      - If data is defined as long integer (64-bit)\n"
    "  + RaveDataType_ULONG     - If data is defined as unsigned long integer (64-bit)\n"
    "  + RaveDataType_FLOAT     - If data is defined as float value (32-bit)\n"
    "  + RaveDataType_DOUBLE    - If data is defined as double float value (64-bit)\n"
    "\n"
    "Most values used in rave can be of 3 different types:\n"
    "  + RaveValueType_UNDEFINED - If value type haven't been defined yet or can't be determined.\n"
    "  + RaveValueType_UNDETECT  - There is a value but it doesn't exist any data (like no rain, ...)\n"
    "  + RaveValueType_NODATA    - There is no coverage at the location and hence no data found\n"
    "  + RaveValueType_DATA      - We have data at the location\n"
    "\n"
    "Object type can be found in for example raveio for giving information on what type of object that has been read. More information about the various types can be found in the ODIM H5 specification.\n"
    "  + Rave_ObjectType_UNDEFINED - Object read can not be defined (most likely because it couldn't be read).\n"
    "  + Rave_ObjectType_PVOL      - Polar Volume\n"
    "  + Rave_ObjectType_CVOL      - Cartesian volume\n"
    "  + Rave_ObjectType_SCAN      - Polar scan\n"
    "  + Rave_ObjectType_RAY       - Single polar ray\n"
    "  + Rave_ObjectType_AZIM      - Azimuthal object\n"
    "  + Rave_ObjectType_IMAGE     - 2-D cartesian image\n"
    "  + Rave_ObjectType_COMP      - Cartesian composite image(s)\n"
    "  + Rave_ObjectType_XSEC      - 2-D vertical cross section(s)\n"
    "  + Rave_ObjectType_VP        - 1-D vertical profile\n"
    "  + Rave_ObjectType_PIC       - Embedded graphical image\n"
    "\n"
    "Product types that defines the various products. Usually defined in what/product, More information about the various types can be found in the ODIM H5 specification.\n"
    "  + Rave_ProductType_UNDEFINED - Undefined product type.\n"
    "  + Rave_ProductType_SCAN      - A scan of polar data\n"
    "  + Rave_ProductType_PPI       - Plan position indicator\n"
    "  + Rave_ProductType_CAPPI     - Constant altitude PPI\n"
    "  + Rave_ProductType_PCAPPI    - Pseudo-CAPPI\n"
    "  + Rave_ProductType_ETOP      - Echo top\n"
    "  + Rave_ProductType_MAX       - Maximum\n"
    "  + Rave_ProductType_RR        - Accumulation\n"
    "  + Rave_ProductType_VIL       - Vertically integrated liquid water\n"
    "  + Rave_ProductType_COMP      - Composite\n"
    "  + Rave_ProductType_VP        - Vertical profile\n"
    "  + Rave_ProductType_RHI       - Range height indicator\n"
    "  + Rave_ProductType_XSEC      - Arbitrary vertical slice\n"
    "  + Rave_ProductType_VSP       - Vertical side panel\n"
    "  + Rave_ProductType_HSP       - Horizontal side panel\n"
    "  + Rave_ProductType_RAY       - Ray\n"
    "  + Rave_ProductType_AZIM      - Azimuthal type product\n"
    "  + Rave_ProductType_QUAL      - Quality metric\n"
    "  + Rave_ProductType_PMAX      - Pseudo-MAX\n"
    "  + Rave_ProductType_SURF      - Surface type\n"
    "\n"
    "There are also a number of different version strings that can occur in an ODIM H5 file.\n"
    "  + RaveIO_ODIM_Version_UNDEFINED - Undefined ODIM version\n"
    "  + RaveIO_ODIM_Version_2_0       - ODIM H5 2.0\n"
    "  + RaveIO_ODIM_Version_2_1       - ODIM H5 2.1\n"
    "  + RaveIO_ODIM_Version_2_2       - ODIM H5 2.2\n"
    "and\n"
    "  + RaveIO_ODIM_H5rad_Version_UNDEFINED - Undefined ODIM version\n"
    "  + RaveIO_ODIM_H5rad_Version_2_0       - ODIM H5rad 2.0\n"
    "  + RaveIO_ODIM_H5rad_Version_2_1       - ODIM H5rad 2.1\n"
    "  + RaveIO_ODIM_H5rad_Version_2_2       - ODIM H5rad 2.2\n"
    "\n"
    "The logging within the rave c-modules are done to stderr and due to that, the logging is turned off as default behaviour.\n"
    "In some cases it might be necessary to get some information on what happens and why errors occurs. In those cases it is possible\n"
    "to turn on the debugging with setDebugLevel and specify one of the below levels:\n"
    "  + Debug_RAVE_SPEWDEBUG    - This provides a lot of debuginformation, most which probably is not interesting\n"
    "  + Debug_RAVE_DEBUG        - Basic debug information\n"
    "  + Debug_RAVE_DEPRECATED   - Print information from deprecated functions\n"
    "  + Debug_RAVE_INFO         - Information\n"
    "  + Debug_RAVE_WARNING      - Warnings\n"
    "  + Debug_RAVE_ERROR        - Errors\n"
    "  + Debug_RAVE_CRITICAL     - Critical errors, typically if this occur, it probably ends with a crash\n"
    "  + Debug_RAVE_SILENT       - Don't display anything\n"
    " Logging can be turned on with:\n"
    " import _rave\n"
    " _rave.setDebugLevel(_rave.Debug_RAVE_DEBUG)\n"
    "\n"
    );
/*@} End of Documentation about the type */

/**
 * Adds constants to the dictionary (probably the modules dictionary).
 * @param[in] dictionary - the dictionary the long should be added to
 * @param[in] name - the name of the constant
 * @param[in] value - the value
 */
static void add_long_constant(PyObject* dictionary, const char* name, long value)
{
  PyObject* tmp = NULL;
  tmp = PyInt_FromLong(value);
  if (tmp != NULL) {
    PyDict_SetItemString(dictionary, name, tmp);
  }
  Py_XDECREF(tmp);
}

/**
 * Initializes _rave.
 */
MOD_INIT(_rave)
{
  PyObject *module=NULL,*dictionary=NULL;

  MOD_INIT_DEF(module, "_rave", _rave_module_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyErr_NewException("_rave.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _rave.error");
    return MOD_INIT_ERROR;
  }

  if (atexit(rave_alloc_print_statistics) != 0) {
    fprintf(stderr, "Could not set atexit function");
  }

  if (atexit(RaveCoreObject_printStatistics) != 0) {
    fprintf(stderr, "Could not set atexit function");
  }

  // Initialize some constants
  add_long_constant(dictionary, "RaveDataType_UNDEFINED", RaveDataType_UNDEFINED);
  add_long_constant(dictionary, "RaveDataType_CHAR", RaveDataType_CHAR);
  add_long_constant(dictionary, "RaveDataType_UCHAR", RaveDataType_UCHAR);
  add_long_constant(dictionary, "RaveDataType_SHORT", RaveDataType_SHORT);
  add_long_constant(dictionary, "RaveDataType_USHORT", RaveDataType_USHORT);
  add_long_constant(dictionary, "RaveDataType_INT", RaveDataType_INT);
  add_long_constant(dictionary, "RaveDataType_UINT", RaveDataType_UINT);
  add_long_constant(dictionary, "RaveDataType_LONG", RaveDataType_LONG);
  add_long_constant(dictionary, "RaveDataType_ULONG", RaveDataType_ULONG);
  add_long_constant(dictionary, "RaveDataType_FLOAT", RaveDataType_FLOAT);
  add_long_constant(dictionary, "RaveDataType_DOUBLE", RaveDataType_DOUBLE);

  add_long_constant(dictionary, "NEAREST", NEAREST);
  add_long_constant(dictionary, "BILINEAR", BILINEAR);
  add_long_constant(dictionary, "CUBIC", CUBIC);
  add_long_constant(dictionary, "CRESSMAN", CRESSMAN);
  add_long_constant(dictionary, "UNIFORM", UNIFORM);
  add_long_constant(dictionary, "INVERSE", INVERSE);

  add_long_constant(dictionary, "RaveValueType_UNDEFINED", RaveValueType_UNDEFINED);
  add_long_constant(dictionary, "RaveValueType_UNDETECT", RaveValueType_UNDETECT);
  add_long_constant(dictionary, "RaveValueType_NODATA", RaveValueType_NODATA);
  add_long_constant(dictionary, "RaveValueType_DATA", RaveValueType_DATA);

  add_long_constant(dictionary, "RaveIO_ODIM_Version_UNDEFINED", RaveIO_ODIM_Version_UNDEFINED);
  add_long_constant(dictionary, "RaveIO_ODIM_Version_2_0", RaveIO_ODIM_Version_2_0);
  add_long_constant(dictionary, "RaveIO_ODIM_Version_2_1", RaveIO_ODIM_Version_2_1);
  add_long_constant(dictionary, "RaveIO_ODIM_Version_2_2", RaveIO_ODIM_Version_2_2);
  add_long_constant(dictionary, "RaveIO_ODIM_Version_2_3", RaveIO_ODIM_Version_2_3);
  add_long_constant(dictionary, "RaveIO_ODIM_Version_2_4", RaveIO_ODIM_Version_2_4);

  add_long_constant(dictionary, "RaveIO_ODIM_H5rad_Version_UNDEFINED", RaveIO_ODIM_H5rad_Version_UNDEFINED);
  add_long_constant(dictionary, "RaveIO_ODIM_H5rad_Version_2_0", RaveIO_ODIM_H5rad_Version_2_0);
  add_long_constant(dictionary, "RaveIO_ODIM_H5rad_Version_2_1", RaveIO_ODIM_H5rad_Version_2_1);
  add_long_constant(dictionary, "RaveIO_ODIM_H5rad_Version_2_2", RaveIO_ODIM_H5rad_Version_2_2);
  add_long_constant(dictionary, "RaveIO_ODIM_H5rad_Version_2_3", RaveIO_ODIM_H5rad_Version_2_3);
  add_long_constant(dictionary, "RaveIO_ODIM_H5rad_Version_2_4", RaveIO_ODIM_H5rad_Version_2_4);

  add_long_constant(dictionary, "Rave_ObjectType_UNDEFINED", Rave_ObjectType_UNDEFINED);
  add_long_constant(dictionary, "Rave_ObjectType_PVOL", Rave_ObjectType_PVOL);
  add_long_constant(dictionary, "Rave_ObjectType_CVOL", Rave_ObjectType_CVOL);
  add_long_constant(dictionary, "Rave_ObjectType_SCAN", Rave_ObjectType_SCAN);
  add_long_constant(dictionary, "Rave_ObjectType_RAY", Rave_ObjectType_RAY);
  add_long_constant(dictionary, "Rave_ObjectType_AZIM", Rave_ObjectType_AZIM);
  add_long_constant(dictionary, "Rave_ObjectType_IMAGE", Rave_ObjectType_IMAGE);
  add_long_constant(dictionary, "Rave_ObjectType_COMP", Rave_ObjectType_COMP);
  add_long_constant(dictionary, "Rave_ObjectType_XSEC", Rave_ObjectType_XSEC);
  add_long_constant(dictionary, "Rave_ObjectType_VP", Rave_ObjectType_VP);
  add_long_constant(dictionary, "Rave_ObjectType_PIC", Rave_ObjectType_PIC);

  add_long_constant(dictionary, "Rave_ProductType_UNDEFINED", Rave_ProductType_UNDEFINED);
  add_long_constant(dictionary, "Rave_ProductType_SCAN", Rave_ProductType_SCAN);
  add_long_constant(dictionary, "Rave_ProductType_PPI", Rave_ProductType_PPI);
  add_long_constant(dictionary, "Rave_ProductType_CAPPI", Rave_ProductType_CAPPI);
  add_long_constant(dictionary, "Rave_ProductType_PCAPPI", Rave_ProductType_PCAPPI);
  add_long_constant(dictionary, "Rave_ProductType_ETOP", Rave_ProductType_ETOP);
  add_long_constant(dictionary, "Rave_ProductType_MAX", Rave_ProductType_MAX);
  add_long_constant(dictionary, "Rave_ProductType_RR", Rave_ProductType_RR);
  add_long_constant(dictionary, "Rave_ProductType_VIL", Rave_ProductType_VIL);
  add_long_constant(dictionary, "Rave_ProductType_COMP", Rave_ProductType_COMP);
  add_long_constant(dictionary, "Rave_ProductType_VP", Rave_ProductType_VP);
  add_long_constant(dictionary, "Rave_ProductType_RHI", Rave_ProductType_RHI);
  add_long_constant(dictionary, "Rave_ProductType_XSEC", Rave_ProductType_XSEC);
  add_long_constant(dictionary, "Rave_ProductType_VSP", Rave_ProductType_VSP);
  add_long_constant(dictionary, "Rave_ProductType_HSP", Rave_ProductType_HSP);
  add_long_constant(dictionary, "Rave_ProductType_RAY", Rave_ProductType_RAY);
  add_long_constant(dictionary, "Rave_ProductType_AZIM", Rave_ProductType_AZIM);
  add_long_constant(dictionary, "Rave_ProductType_QUAL", Rave_ProductType_QUAL);
  add_long_constant(dictionary, "Rave_ProductType_PMAX", Rave_ProductType_PMAX);
  add_long_constant(dictionary, "Rave_ProductType_SURF", Rave_ProductType_SURF);
  add_long_constant(dictionary, "Rave_ProductType_EBASE", Rave_ProductType_EBASE);

  add_long_constant(dictionary, "Debug_RAVE_SPEWDEBUG", RAVE_SPEWDEBUG);
  add_long_constant(dictionary, "Debug_RAVE_DEBUG", RAVE_DEBUG);
  add_long_constant(dictionary, "Debug_RAVE_DEPRECATED", RAVE_DEPRECATED);
  add_long_constant(dictionary, "Debug_RAVE_INFO", RAVE_INFO);
  add_long_constant(dictionary, "Debug_RAVE_WARNING", RAVE_WARNING);
  add_long_constant(dictionary, "Debug_RAVE_ERROR", RAVE_ERROR);
  add_long_constant(dictionary, "Debug_RAVE_CRITICAL", RAVE_CRITICAL);
  add_long_constant(dictionary, "Debug_RAVE_SILENT", RAVE_SILENT);

  import_array(); /*To make sure I get access to Numeric*/

  import_pyprojection();
  import_pycartesian();
  import_pycartesianparam();
  import_pypolarscan();
  import_pyraveio();
  import_pypolarvolume();
  import_pytransform();

  PYRAVE_DEBUG_INITIALIZE;

  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
