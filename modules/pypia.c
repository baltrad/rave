/* --------------------------------------------------------------------
Copyright (C) 2025 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the PIA API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-12-04
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYPIA_MODULE    /**< to get correct part in pypia.h */
#include "pypia.h"
#include "pyravefield.h"
#include "pypolarscan.h"
#include "pypolarscanparam.h"
#include "rave_alloc.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_pia");

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

/*@{ Gra */
/**
 * Returns the native RavePIA_t instance.
 * @param[in] pypia - the python pia instance
 * @returns the native pia instance.
 */
static RavePIA_t*
PyPia_GetNative(PyPia* pypia)
{
  RAVE_ASSERT((pypia != NULL), "pypia == NULL");
  return RAVE_OBJECT_COPY(pypia->pia);
}

/**
 * Creates a python pia from a native pia or will create an
 * initial native pia if p is NULL.
 * @param[in] p - the native pia (or NULL)
 * @returns the python pia product.
 */
static PyPia*
PyPia_New(RavePIA_t* p)
{
  PyPia* result = NULL;
  RavePIA_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RavePIA_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for pia.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for pia.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyPia, &PyPia_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->pia = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->pia, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyPia instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyPia.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the pia
 * @param[in] obj the object to deallocate.
 */
static void _pypia_dealloc(PyPia* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->pia, obj);
  RAVE_OBJECT_RELEASE(obj->pia);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the gra.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pypia_new(PyObject* self, PyObject* args)
{
  PyPia* result = PyPia_New(NULL);
  return (PyObject*)result;
}


/**
 * Returns the how/task name that this module will specify as how/task
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the string
 */
static PyObject* _pypia_getHowTaskName(PyObject* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyString_FromString(RavePIA_getHowTaskName());
}



/**
 * Calculates PIA
 * @param[in] self - self
 * @param[in] args - a scan and a quantity
 * @return the rave field containing the PIA field
 */
static PyObject* _pypia_calculatePIA(PyPia* self, PyObject* args)
{
  PyObject* pyscan = NULL;
  PyObject *result = NULL;
  RaveField_t* field = NULL;
  char* quantity = NULL;

  if (!PyArg_ParseTuple(args, "Os", &pyscan,&quantity)) {
    return NULL;
  }

  if (!PyPolarScan_Check(pyscan)) {
    raiseException_returnNULL(PyExc_AttributeError, "Must provide scan, quantity to calculatePIA");
  }

  field = RavePIA_calculatePIA(self->pia, ((PyPolarScan*)pyscan)->scan, quantity, NULL);

  if (field != NULL) {
    result = (PyObject*)PyRaveField_New(field);
  } else {
    raiseException_returnNULL(PyExc_IOError, "Failure when calculating PIA");
  }

  RAVE_OBJECT_RELEASE(field);
  return result;
}


/**
 * Calculates PIA
 * @param[in] self - self
 * @param[in] args - a scan and a quantity
 * @return the rave field containing the PIA field
 */
static PyObject* _pypia_createPIAParameter(PyPia* self, PyObject* args)
{
  PyObject* pyscan = NULL;
  PyObject *result = NULL;
  PolarScanParam_t* param = NULL;
  char* quantity = NULL;

  if (!PyArg_ParseTuple(args, "Os", &pyscan,&quantity)) {
    return NULL;
  }

  if (!PyPolarScan_Check(pyscan)) {
    raiseException_returnNULL(PyExc_AttributeError, "Must provide scan, quantity to createPIAParameter");
  }

  param = RavePIA_createPIAParameter(self->pia, ((PyPolarScan*)pyscan)->scan, quantity, NULL, NULL);

  if (param != NULL) {
    result = (PyObject*)PyPolarScanParam_New(param);
  } else {
    raiseException_returnNULL(PyExc_IOError, "Failure when creating PIA parameter");
  }

  RAVE_OBJECT_RELEASE(param);
  return result;
}

/**
 * Performs the processing of the PIA
 * @param[in] self - self
 * @param[in] args Osiii (scan, quantity, addparam, reprocessquality, apply)
 * @return None
 */
static PyObject* _pypia_process(PyPia* self, PyObject* args)
{
  PyObject* pyscan = NULL;
  int addparam=0, reprocessquality=0, apply=0;
  char* quantity = NULL;

  if (!PyArg_ParseTuple(args, "Osiii", &pyscan, &quantity, &addparam, &reprocessquality, &apply)) {
    return NULL;
  }

  if (!PyPolarScan_Check(pyscan)) {
    raiseException_returnNULL(PyExc_AttributeError, "Must provide scan, quantity to createPIAParameter");
  }

  if (!RavePIA_process(self->pia, ((PyPolarScan*)pyscan)->scan, quantity, addparam, reprocessquality, apply)) {
    raiseException_returnNULL(PyExc_RuntimeError, "Failure when processing PIA");
  }

  Py_RETURN_NONE;
}

/**
 * All methods a acrr can have
 */
static struct PyMethodDef _pypia_methods[] =
{
  {"coeff_zk_power", NULL, METH_VARARGS},
  {"exp_zk_power", NULL, METH_VARARGS},
  {"max_pia", NULL, METH_VARARGS},
  {"rr", NULL, METH_VARARGS},
  {"calculatePIA", (PyCFunction) _pypia_calculatePIA, 1,
    "calculatePIA(scan, quantity) -> field\n\n"
  },
  {"createPIAParameter", (PyCFunction) _pypia_createPIAParameter, 1,
    "createPIAParameter(scan, quantity) -> param\n\n"
  },
  {"process", (PyCFunction) _pypia_process, 1,
    "process(scan, quantity, addparam, reprocessquality, apply)\n\n"
  },
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the gra
 * @param[in] self - the gra
 */
static PyObject* _pypia_getattro(PyPia* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("coeff_zk_power", name) == 0) {
    return PyFloat_FromDouble(RavePIA_getZkPowerCoefficient(self->pia));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("exp_zk_power", name) == 0) {
    return PyFloat_FromDouble(RavePIA_getZkPowerExponent(self->pia));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("max_pia", name) == 0) {
    return PyFloat_FromDouble(RavePIA_getPiaMax(self->pia));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("rr", name) == 0) {
    return PyFloat_FromDouble(RavePIA_getRangeResolution(self->pia));
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the gra
 */
static int _pypia_setattro(PyPia* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("coeff_zk_power", name) == 0) {
    if (PyInt_Check(val)) {
      RavePIA_setZkPowerCoefficient(self->pia, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RavePIA_setZkPowerCoefficient(self->pia, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RavePIA_setZkPowerCoefficient(self->pia, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "coeff_zk_power must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("exp_zk_power", name) == 0) {
    if (PyInt_Check(val)) {
      RavePIA_setZkPowerExponent(self->pia, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RavePIA_setZkPowerExponent(self->pia, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RavePIA_setZkPowerExponent(self->pia, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "exp_zk_power must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("max_pia", name) == 0) {
    if (PyInt_Check(val)) {
      RavePIA_setPiaMax(self->pia, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RavePIA_setPiaMax(self->pia, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RavePIA_setPiaMax(self->pia, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "max_pia must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("rr", name) == 0) {
    if (PyInt_Check(val)) {
      RavePIA_setRangeResolution(self->pia, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RavePIA_setRangeResolution(self->pia, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RavePIA_setRangeResolution(self->pia, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "rr must be a number");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, "Unknown attribute");
  }

  result = 0;
done:
  return result;
}

/*@} End of Gra */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pypia_module_doc,
    "Performs the path integrated attenuation according to the Hitchfeld-Bordan method\n\n"
  "Usage:\n"
  " import _pia\n"
  " pia = _pia.new()\n"
  " scan = _raveio.open(\"qcvol.h5\").object.getScan(0)\n"
  " pia.process(scan, \"DBZH\", True, True, True) # Booleans are addparam, reprocessquality, apply\n\n"
);
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyPia_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "PiaCore", /*tp_name*/
  sizeof(PyPia), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pypia_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pypia_getattro, /*tp_getattro*/
  (setattrofunc)_pypia_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pypia_module_doc,            /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pypia_methods,              /*tp_methods*/
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
  {"new", (PyCFunction)_pypia_new, 1,
    "new() -> new instance of the PiaCore object\n\n"
    "Creates a new instance of the PiaCore object"
  },
  {"getHowTaskName", (PyCFunction)_pypia_getHowTaskName, 1,
    "getHowTaskName() -> the how/task name\n\n"
  },
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_pia)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyPia_API[PyPia_API_pointers];
  PyObject *c_api_object = NULL;
  MOD_INIT_SETUP_TYPE(PyPia_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyPia_Type);

  MOD_INIT_DEF(module, "_pia", _pypia_module_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyPia_API[PyPia_Type_NUM] = (void*)&PyPia_Type;
  PyPia_API[PyPia_GetNative_NUM] = (void *)PyPia_GetNative;
  PyPia_API[PyPia_New_NUM] = (void*)PyPia_New;

  c_api_object = PyCapsule_New(PyPia_API, PyPia_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_pia.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _pia.error");
    return MOD_INIT_ERROR;
  }

  import_pyravefield();
  import_pypolarscan();
  import_pypolarscanparam();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
