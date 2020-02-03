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
 * Python version of the Acrr API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2012-06-01
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYACRR_MODULE    /**< to get correct part in pyacrr.h */
#include "pyacrr.h"
#include "rave_alloc.h"
#include "pycartesianparam.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_acrr");

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

/*@{ Acrr */
/**
 * Returns the native RaveAcrr_t instance.
 * @param[in] pyacrr - the python acrr instance
 * @returns the native acrr instance.
 */
static RaveAcrr_t*
PyAcrr_GetNative(PyAcrr* pyacrr)
{
  RAVE_ASSERT((pyacrr != NULL), "pyacrr == NULL");
  return RAVE_OBJECT_COPY(pyacrr->acrr);
}

/**
 * Creates a python acrr from a native acrr or will create an
 * initial native acrr if p is NULL.
 * @param[in] p - the native acrr (or NULL)
 * @returns the python acrr product.
 */
static PyAcrr*
PyAcrr_New(RaveAcrr_t* p)
{
  PyAcrr* result = NULL;
  RaveAcrr_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RaveAcrr_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for acrr.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for acrr.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyAcrr, &PyAcrr_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->acrr = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->acrr, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyAcrr instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyAcrr.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the acrr
 * @param[in] obj the object to deallocate.
 */
static void _pyacrr_dealloc(PyAcrr* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->acrr, obj);
  RAVE_OBJECT_RELEASE(obj->acrr);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the acrr.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyacrr_new(PyObject* self, PyObject* args)
{
  PyAcrr* result = PyAcrr_New(NULL);
  return (PyObject*)result;
}

/**
 * Returns if the acrr has been initialized which occurs after the
 * first call to sum.
 * @param[in] self - self
 * @param[in] args - N/A
 * @return a boolean
 */
static PyObject* _pyacrr_isInitialized(PyAcrr* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyBool_FromLong(RaveAcrr_isInitialized(self->acrr));
}

static PyObject* _pyacrr_getQuantity(PyAcrr* self, PyObject* args)
{
  if(!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  if (RaveAcrr_getQuantity(self->acrr) == NULL) {
    Py_RETURN_NONE;
  }
  return PyString_FromString(RaveAcrr_getQuantity(self->acrr));
}

/**
 * Sums a parameter with the previously calculated values.
 * @param[in] self - self
 * @param[in] args - param (cartesian parameter), zr_a (double), zr_b (double)
 * @return None on success otherwise an exception will be thrown
 */
static PyObject* _pyacrr_sum(PyAcrr* self, PyObject* args)
{
  PyObject* pyo = NULL;
  double zr_a = 0.0, zr_b = 0.0;
  if (!PyArg_ParseTuple(args, "Odd", &pyo, &zr_a, &zr_b)) {
    return NULL;
  }

  if (!PyCartesianParam_Check(pyo)) {
    raiseException_returnNULL(PyExc_ValueError, "First parameter must be a cartesian parameter");
  }

  if (!RaveAcrr_sum(self->acrr, ((PyCartesianParam*)pyo)->param, zr_a, zr_b)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to process parameter");
  }

  Py_RETURN_NONE;
}

/**
 * Generates the result
 * @param[in] self - self
 * @return the cartesian parameter with quantity ACRR and the associated quality field on success otherwise NULL
 */
static PyObject* _pyacrr_accumulate(PyAcrr* self, PyObject* args)
{
  double accept = 0.0;
  long N = 0;
  double hours = 0.0;
  CartesianParam_t* param = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "dld", &accept, &N, &hours)) {
    return NULL;
  }

  param = RaveAcrr_accumulate(self->acrr, accept, N, hours);
  if (param != NULL) {
    result = (PyObject*)PyCartesianParam_New(param);
  } else {
    raiseException_returnNULL(PyExc_IOError, "Failure when accumulating result");
  }

  RAVE_OBJECT_RELEASE(param);
  return result;
}

/**
 * All methods a acrr can have
 */
static struct PyMethodDef _pyacrr_methods[] =
{
  {"nodata", NULL},
  {"undetect", NULL},
  {"quality_field_name", NULL},
  {"isInitialized", (PyCFunction) _pyacrr_isInitialized, 1,
    "isInitialized()\n\n"
    "Checks if this instance has been initialized. Will occur at first call to sum(..)"},
  {"getQuantity", (PyCFunction) _pyacrr_getQuantity, 1,
    "getQuantity() -> quantity\n\n"
    "Returns the quantity that was set to be used during initialization."},
  {"sum", (PyCFunction) _pyacrr_sum, 1,
    "sum(pyo, zr_a, zr_b)\n\n"
    "Adds a cartesian parameter to the acrr accumulation.\n\n"
    "pyo  - is the cartesian parameter. If acrr instance is initialized, quantity, xsize & ysize must be same as previous calls to sum.\n"
    "zr_a - ZR A constant\n"
    "zr_b - ZR b constant"},
  {"accumulate", (PyCFunction) _pyacrr_accumulate, 1,
    "accumulate(accept, N, hours) -> resulting cartesian parameter\n\n"
    "Calculates the resulting product from previous calls to sum.\n\n"
    "accept - the number of many nodata-pixels that are allowed in order for the pixel to be used\n"
    "N      - Number of files that we expect to be used. Which might be greater or equal to number of calls to sum\n"
    "hours  - Number of hours we are covering.\n"},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the acrr
 * @param[in] self - the acrr
 */
static PyObject* _pyacrr_getattro(PyAcrr* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("nodata", name) == 0) {
    return PyFloat_FromDouble(RaveAcrr_getNodata(self->acrr));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("undetect", name) == 0) {
    return PyFloat_FromDouble(RaveAcrr_getUndetect(self->acrr));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("quality_field_name", name) == 0) {
    return PyString_FromString(RaveAcrr_getQualityFieldName(self->acrr));
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the acrr
 */
static int _pyacrr_setattro(PyAcrr* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("nodata", name) == 0) {
    if (PyInt_Check(val)) {
      RaveAcrr_setNodata(self->acrr, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveAcrr_setNodata(self->acrr, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveAcrr_setNodata(self->acrr, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "nodata must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("undetect", name) == 0) {
    if (PyInt_Check(val)) {
      RaveAcrr_setUndetect(self->acrr, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveAcrr_setUndetect(self->acrr, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveAcrr_setUndetect(self->acrr, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "undetect must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("quality_field_name", name) == 0) {
    if (PyString_Check(val)) {
      if (!RaveAcrr_setQualityFieldName(self->acrr, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_MemoryError, "failure to set quality field name");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "quality_field_name must be a string");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError,
        PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }

  result = 0;
done:
  return result;
}

/*@} End of Acrr */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyacrr_doc,
    "This instance provides functionality for generating accumulated precipitation products.\n"
    "\n"
    "Usage is based on the user providing cartesian parameters that should be of same quantity and covered area.\n"
    "There is no check verifying that extent is same, only checks are x&y-size and quantity.\n"
    "Assuming that you have a number of catesian products, the usage is straight forward.\n"
    ">>> import _acrr\n"
    ">>> acrr = _acrr.new()\n"
    ">>> acrr.nodata = -1.0\n"
    ">>> acrr.undetect = 0.0\n"
    ">>> zr_a = 200.0\n"
    ">>> zr_b = 1.6\n"
    ">>> accept = 0 #accept is the required limit for how many nodata-pixels that are allowed in order for the pixel to be used\n"
    ">>> N = 5 # Note, we have 5 files when covering 1 hour with 15-minute intervals\n"
    ">>> hours = 1 # One hour\n"
    ">>> acrr.quality_field_name = \"se.smhi.composite.distance.radar\" # name of quality field\n"
    ">>> for f in [\"gmap_202001010000.h5\", \"gmap_202001010015.h5\", \"gmap_202001010030.h5\", \"gmap_202001010045.h5\", \"gmap_202001010100.h5\"]:\n"
    ">>>   acrr.sum(_raveio.open(f).object.getParameter(\"DBZH\"), zr_a, zr_b)\n"
    ">>> result = acrr.accumulate(accept, N, hours)\n"
    "\n"
    "There is obviously more to the above example. For example, each parameter must contain required quality field as defined in acrr.quality_field_name\n"
    "and you need to verify that the opened files are in fact compoisites so that you don't have to convert them first.\n"
    "\n"
    "As can seen in the example, there are 3 members used.\n"
    " * nodata             - The nodata value that should be set in the resulting product. Default value is 1.0.\n"
    " * undetect           - The undetect value that should be set in the resulting product. Default value is 0.0.\n"
    " * quality_field_name - The distance (quality) field that should be used when processing the cartesian parameters.\n"
    "\n"
    "Then, there are a few methods that are provided. More information about these can be found by printing the doc about them.\n"
    " * isInitialized      - When first call to sum(...) is performed, the acrr instance is initialized with basic information.\n"
    "                        after that, it's not possible to invoke sum with a product with different x/y-size and quality field.\n"
    "\n"
    " * getQuantity        - Set during initialization.\n"
    "\n"
    " * sum                - Sums up the provided cartesian parameter. First call to sum in a sequence will initialize the acrr structure.\n"
    "\n"
    " * accumulate         - Calculates the resulting product from previous calls to sum.\n"
    );
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyAcrr_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "AcrrCore", /*tp_name*/
  sizeof(PyAcrr), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyacrr_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)0,               /*tp_getattr*/
  (setattrfunc)0,               /*tp_setattr*/
  0,                            /*tp_compare*/
  0,                            /*tp_repr*/
  0,                            /*tp_as_number */
  0,
  0,                            /*tp_as_mapping */
  0,                            /*tp_hash*/
  (ternaryfunc)0,               /*tp_call*/
  (reprfunc)0,                  /*tp_str*/
  (getattrofunc)_pyacrr_getattro, /*tp_getattro*/
  (setattrofunc)_pyacrr_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyacrr_doc,                  /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyacrr_methods,              /*tp_methods*/
  0,                            /*tp_members*/
  0,                            /*tp_getset*/
  0,                            /*tp_base*/
  0,                            /*tp_dict*/
  0,                            /*tp_descr_get*/
  0,                            /*tp_descr_set*/
  0,                            /*tp_dictoffset*/
  0,                            /*tp_init*/
  0,                            /*tp_alloc*/
  0,                            /*tp_new*/
  0,                            /*tp_free*/
  0,                            /*tp_is_gc*/
};
/*@} End of Type definitions */

/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pyacrr_new, 1,
      "new() -> new instance of the AcrrCore object\n\n"
      "Creates a new instance of the AcrrCore object"},
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_acrr)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyAcrr_API[PyAcrr_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyAcrr_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyAcrr_Type);

  MOD_INIT_DEF(module, "_acrr", _pyacrr_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyAcrr_API[PyAcrr_Type_NUM] = (void*)&PyAcrr_Type;
  PyAcrr_API[PyAcrr_GetNative_NUM] = (void *)PyAcrr_GetNative;
  PyAcrr_API[PyAcrr_New_NUM] = (void*)PyAcrr_New;

  c_api_object = PyCapsule_New(PyAcrr_API, PyAcrr_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_acrr.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _acrr.error");
    return MOD_INIT_ERROR;
  }

  import_pycartesianparam();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
