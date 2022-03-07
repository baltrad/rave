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
 * Python version of the Gra API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2014-03-28
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYGRA_MODULE    /**< to get correct part in pygra.h */
#include "pygra.h"
#include "pyravefield.h"
#include "pycartesianparam.h"
#include "rave_alloc.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_gra");

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
 * Returns the native GraAcrr_t instance.
 * @param[in] pygra - the python gra instance
 * @returns the native gra instance.
 */
static RaveGra_t*
PyGra_GetNative(PyGra* pygra)
{
  RAVE_ASSERT((pygra != NULL), "pygra == NULL");
  return RAVE_OBJECT_COPY(pygra->gra);
}

/**
 * Creates a python gra from a native gra or will create an
 * initial native gra if p is NULL.
 * @param[in] p - the native gra (or NULL)
 * @returns the python gra product.
 */
static PyGra*
PyGra_New(RaveGra_t* p)
{
  PyGra* result = NULL;
  RaveGra_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RaveGra_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for gra.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for gra.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyGra, &PyGra_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->gra = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->gra, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyGra instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyGra.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the gra
 * @param[in] obj the object to deallocate.
 */
static void _pygra_dealloc(PyGra* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->gra, obj);
  RAVE_OBJECT_RELEASE(obj->gra);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the gra.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pygra_new(PyObject* self, PyObject* args)
{
  PyGra* result = PyGra_New(NULL);
  return (PyObject*)result;
}

/**
 * Generates the result
 * @param[in] self - self
 * @param[in] args - a rave field containing distances, a cartesian parameter containing the data
 * @return the cartesian parameter with the gra coefficients applied
 */
static PyObject* _pygra_apply(PyGra* self, PyObject* args)
{
  PyObject* pyfield = NULL;
  PyObject* pyparameter = NULL;
  CartesianParam_t* graparam = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "OO", &pyfield, &pyparameter)) {
    return NULL;
  }

  if (!PyRaveField_Check(pyfield) || !PyCartesianParam_Check(pyparameter)) {
    raiseException_returnNULL(PyExc_AttributeError, "Must provide apply with <rave field with distance>, <cartesian parameter with data>");
  }

  graparam = RaveGra_apply(self->gra, ((PyRaveField*)pyfield)->field, ((PyCartesianParam*)pyparameter)->param);

  if (graparam != NULL) {
    result = (PyObject*)PyCartesianParam_New(graparam);
  } else {
    raiseException_returnNULL(PyExc_IOError, "Failure when applying gra coefficients");
  }

  RAVE_OBJECT_RELEASE(graparam);
  return result;
}

/**
 * All methods a acrr can have
 */
static struct PyMethodDef _pygra_methods[] =
{
  {"A", NULL, METH_VARARGS},
  {"B", NULL, METH_VARARGS},
  {"C", NULL, METH_VARARGS},
  {"upperThreshold", NULL, METH_VARARGS},
  {"lowerThreshold", NULL, METH_VARARGS},
  {"zrA", NULL, METH_VARARGS},
  {"zrb", NULL, METH_VARARGS},
  {"apply", (PyCFunction) _pygra_apply, 1,
    "apply(distanceField, cartesian_parameter) -> cartesian parameter\n\n"
    "Applies the coefficients on the parameter field. The distance field dimensions must match the parameter dimensions.\n"
    "If the quantity is ACRR, then no conversion of the value is required. If the quantity is any of DBZH, DBZV, TH or TV, then "
    "the values are converted to MM/H and then back again to reflectivity.\n\n"
    "distanceField       - The distance field\n"
    "cartesian_parameter - The cartesian parameter field that should be adjusted\n\n"
    "Returns an adjusted cartesian parameter field"
  },
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the gra
 * @param[in] self - the gra
 */
static PyObject* _pygra_getattro(PyGra* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("A", name) == 0) {
    return PyFloat_FromDouble(RaveGra_getA(self->gra));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("B", name) == 0) {
    return PyFloat_FromDouble(RaveGra_getB(self->gra));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("C", name) == 0) {
    return PyFloat_FromDouble(RaveGra_getC(self->gra));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("upperThreshold", name) == 0) {
    return PyFloat_FromDouble(RaveGra_getUpperThreshold(self->gra));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("lowerThreshold", name) == 0) {
    return PyFloat_FromDouble(RaveGra_getLowerThreshold(self->gra));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("zrA", name) == 0) {
    return PyFloat_FromDouble(RaveGra_getZRA(self->gra));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("zrb", name) == 0) {
    return PyFloat_FromDouble(RaveGra_getZRB(self->gra));
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the gra
 */
static int _pygra_setattro(PyGra* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("A", name) == 0) {
    if (PyInt_Check(val)) {
      RaveGra_setA(self->gra, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveGra_setA(self->gra, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveGra_setA(self->gra, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "A must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("B", name) == 0) {
    if (PyInt_Check(val)) {
      RaveGra_setB(self->gra, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveGra_setB(self->gra, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveGra_setB(self->gra, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "B must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("C", name) == 0) {
    if (PyInt_Check(val)) {
      RaveGra_setC(self->gra, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveGra_setC(self->gra, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveGra_setC(self->gra, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "C must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("upperThreshold", name) == 0) {
    if (PyInt_Check(val)) {
      RaveGra_setUpperThreshold(self->gra, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveGra_setUpperThreshold(self->gra, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveGra_setUpperThreshold(self->gra, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "upperThreshold must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("lowerThreshold", name) == 0) {
    if (PyInt_Check(val)) {
      RaveGra_setLowerThreshold(self->gra, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveGra_setLowerThreshold(self->gra, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveGra_setLowerThreshold(self->gra, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "lowerThreshold must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("zrA", name) == 0) {
    if (PyInt_Check(val)) {
      RaveGra_setZRA(self->gra, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveGra_setZRA(self->gra, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveGra_setZRA(self->gra, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "zrA must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("zrb", name) == 0) {
    if (PyInt_Check(val)) {
      RaveGra_setZRB(self->gra, (double)PyInt_AsLong(val));
    } else if (PyLong_Check(val)) {
      RaveGra_setZRB(self->gra, PyLong_AsDouble(val));
    } else if (PyFloat_Check(val)) {
      RaveGra_setZRB(self->gra, PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "zrb must be a number");
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
PyDoc_STRVAR(_pygra_module_doc,
    "Gauge radar adjustment (GRA). This class performs the actual gauge adjustment\n"
    "using the derived coefficients. This can be done in two ways: one for ACRR products\n"
    "(most common) and another for reflectivity (any of DBZH, DBZV, TH, TV).\n"
    "In the case of ACRR, the parameter is already mm which is good.\n"
    "In the case of reflectivity, the parameter needs to be converted to R (mm/hr), the correction applied, and\n"
    "then the result converted back to reflectivity. This should be done in C. Functionality exists already in\n"
    "raveutil.c/h: dBZ2R or raw2R and back.\n"
    "Default Z-R coefficients are given in rave_defined.ZR_A and ZR_b. The C could look something like this (from N2):\n"
    "  F = A + B*DIST + C*pow(DIST, 2.0);\n"
    "  F = RAVEMIN(F, 2.0);    upper threshold 20 dBR\n"
    "  F = RAVEMAX(F, -0.25);  lower threshold -2.5 dBR\n"
    "  out = R*pow(10.0, F);\n"
    "\n"
    "final lower threhold on gauge-adjusted result\n"
    "\n"
    "if (out < lt) { out=0.0; }\n"
    "\n"
    "The member variables are:\n"
    " A              - The A coefficient in the formula A + B*DIST + C*pow(DIST, 2.0);\n"
    " B              - The B coefficient in the formula A + B*DIST + C*pow(DIST, 2.0);\n"
    " C              - The C coefficient in the formula A + B*DIST + C*pow(DIST, 2.0);\n"
    " upperThreshold - The upper threshold in 10ths of dBR. Default is 2.0 (20 dBR)\n"
    " lowerThreshold - The lower threshold in 10ths of dBR. Default is -0.25 (-2.5 dBR)\n"
    " zrA            - ZR A coefficient.\n"
    " zrB            - ZR B coefficient when converting from reflectivity to MM/H\n\n"
  "Usage:\n"
  " import _gra\n"
  " gra = _gra.new()\n"
  " swecomp = _raveio.open(\"swecomposite_20200225.h5\").object\n"
  " dt = create_datetime_from(swecomp)\n"
  " dt = dt - datetime.timedelta(seconds=3600*12) # 12 hours back in time\n"
  " (A,B,C) = db.get_gra_coefficient(dt)\n"
  " gra.A = A\n"
  " gra.B = B\n"
  " gra.C = C\n"
  " gra.zrA = ZR_A\n"
  " gra.zrb = ZR_b\n"
  " distanceField = swecomp.findQualityFieldByHowTask(\"se.smhi.composite.distance.radar\")\n"
  " gra_field = gra.apply(distanceField, swecomp.getParameter(\"DBZH\"))\n"
  " gra_field.quantity = \"DBZH_CORR\"\n"
);
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyGra_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "GraCore", /*tp_name*/
  sizeof(PyGra), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pygra_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pygra_getattro, /*tp_getattro*/
  (setattrofunc)_pygra_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pygra_module_doc,            /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pygra_methods,              /*tp_methods*/
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
  {"new", (PyCFunction)_pygra_new, 1,
    "new() -> new instance of the GraCore object\n\n"
    "Creates a new instance of the GraCore object"

  },
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_gra)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyGra_API[PyGra_API_pointers];
  PyObject *c_api_object = NULL;
  MOD_INIT_SETUP_TYPE(PyGra_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyGra_Type);

  MOD_INIT_DEF(module, "_gra", _pygra_module_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyGra_API[PyGra_Type_NUM] = (void*)&PyGra_Type;
  PyGra_API[PyGra_GetNative_NUM] = (void *)PyGra_GetNative;
  PyGra_API[PyGra_New_NUM] = (void*)PyGra_New;

  c_api_object = PyCapsule_New(PyGra_API, PyGra_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_gra.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _gra.error");
    return MOD_INIT_ERROR;
  }

  import_pyravefield();
  import_pycartesianparam();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
