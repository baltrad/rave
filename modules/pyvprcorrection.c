/* --------------------------------------------------------------------
Copyright (C) 2015 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the VPR Correction API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2015-03-23
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYVPRCORRECTION_MODULE /**< include correct part of pyvprcorrection.h */
#include "pyvprcorrection.h"

#include "pypolarscan.h"
#include "pypolarvolume.h"
#include "pyrave_debug.h"
#include "rave_alloc.h"

/**
 * This modules name
 */
PYRAVE_DEBUG_MODULE("_vprcorrection");

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

/*@{ VprCorrection */
/**
 * Returns the native RaveVprCorrection_t instance.
 * @param[in] pyvprcorrection - the python vpr correction instance
 * @returns the native vpr correction instance.
 */
static RaveVprCorrection_t*
PyVprCorrection_GetNative(PyVprCorrection* pyvprcorrection)
{
  RAVE_ASSERT((pyvprcorrection != NULL), "pyvprcorrection == NULL");
  return RAVE_OBJECT_COPY(pyvprcorrection->vpr);
}

/**
 * Creates a python vpr correction from a native vpr correction or will create an
 * initial native vpr correction if p is NULL.
 * @param[in] p - the native vpr correction (or NULL)
 * @returns the python vpr correction product.
 */
static PyVprCorrection*
PyVprCorrection_New(RaveVprCorrection_t* p)
{
  PyVprCorrection* result = NULL;
  RaveVprCorrection_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RaveVprCorrection_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for vpr correction.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for vpr correction.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyVprCorrection, &PyVprCorrection_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->vpr = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->vpr, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyVprCorrection instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyVprCorrection.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the vpr correction
 * @param[in] obj the object to deallocate.
 */
static void _pyvprcorrection_dealloc(PyVprCorrection* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->vpr, obj);
  RAVE_OBJECT_RELEASE(obj->vpr);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the vpr correction instance.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyvprcorrection_new(PyObject* self, PyObject* args)
{
  PyVprCorrection* result = PyVprCorrection_New(NULL);
  return (PyObject*)result;
}

/**
 * Returns the number of height intervals that the current heightLimit / profileHeight setting gives.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the number of height intervals
 */
static PyObject* _pyvprcorrection_getNumberOfHeightIntervals(PyVprCorrection* self, PyObject* val)
{
  return PyInt_FromLong(RaveVprCorrection_getNumberOfHeightIntervals(self->vpr));
}

/**
 * Returns an height array with all actual heights for the current heightLimit / profileHeight setting.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the height array
 */
static PyObject* _pyvprcorrection_getHeights(PyVprCorrection* self, PyObject* val)
{
  int nElem = 0;
  double* elems = NULL;
  PyObject* pylist = NULL;

  elems = RaveVprCorrection_getHeights(self->vpr, &nElem);
  if (elems != NULL && nElem > 0) {
    int i = 0;
    pylist = PyList_New(nElem);
    for (i = 0; i < nElem; i++) {
      PyList_SetItem(pylist, i, PyFloat_FromDouble(elems[i]));
    }
  } else {
    raiseException_gotoTag(done, PyExc_RuntimeError, "Could not generate height array");
  }
done:
  RAVE_FREE(elems);
  return pylist;
}

static PyObject* _pyvprcorrection_getReflectivityArray(PyVprCorrection* self, PyObject* val)
{
  PyObject* pyo = NULL;
  int nElem = 0;
  RaveVprValue_t* elems = NULL;
  PyObject* pylist = NULL;
  if (!PyArg_ParseTuple(val, "O", &pyo)) {
    return NULL;
  }
  if (!PyPolarVolume_Check(pyo)) {
    raiseException_returnNULL(PyExc_TypeError, "Must provide a polar volume to get a relevant reflectivity array");
  }
  elems = RaveVprCorrection_getReflectivityArray(self->vpr, ((PyPolarVolume*)pyo)->pvol, &nElem);
  if (elems != NULL && nElem > 0) {
    int i = 0;
    pylist = PyList_New(nElem);
    for (i = 0; i < nElem; i++) {
      PyList_SetItem(pylist, i, PyFloat_FromDouble(elems[i].value));
    }
  } else {
    raiseException_gotoTag(done, PyExc_RuntimeError, "Could not generate reflectivity array");
  }
done:
  RAVE_FREE(elems);
  return pylist;
}

static PyObject* _pyvprcorrection_getIdealVpr(PyVprCorrection* self, PyObject* val)
{
  PyObject *pyo = NULL, *pyhto = NULL;
  int nElem = 0, i = 0;
  double* elems = NULL;
  RaveHeightTemperature_t* htarray = NULL;
  int htsize = 0;

  PyObject* pylist = NULL;

  if (!PyArg_ParseTuple(val, "OO", &pyo, &pyhto)) {
    return NULL;
  }
  if (!PyPolarVolume_Check(pyo)) {
    raiseException_returnNULL(PyExc_TypeError, "Must provide a polar volume to get a relevant reflectivity array");
  }
  if (!PyList_Check(pyhto)) {
    raiseException_returnNULL(PyExc_TypeError, "Must provide a list with height temperature tuples");
  }
  htsize = PyList_Size(pyhto);
  htarray = RAVE_MALLOC(sizeof(RaveHeightTemperature_t) * htsize);
  if (htarray == NULL) {
    raiseException_gotoTag(done, PyExc_RuntimeError, "Failed to allocate memory");
  }

  for (i = 0; i < htsize; i++) {
    PyObject* pyval = PyList_GetItem(pyhto, (Py_ssize_t)i); /* Borrows reference */
    double height = 0.0, temperature = 0.0;

    if (!PyArg_ParseTuple(pyval, "dd", &height, &temperature)) {
      raiseException_gotoTag(done, PyExc_TypeError, "Height temperature list must contain tuples with height temperature");
    }
    htarray[i].height = height;
    htarray[i].temperature = temperature;
  }

  elems = RaveVprCorrection_getIdealVpr(self->vpr, ((PyPolarVolume*)pyo)->pvol, htsize, htarray, &nElem);
  if (elems != NULL && nElem > 0) {
    int i = 0;
    pylist = PyList_New(nElem);
    for (i = 0; i < nElem; i++) {
      PyList_SetItem(pylist, i, PyFloat_FromDouble(elems[i]));
    }
  } else {
    raiseException_gotoTag(done, PyExc_RuntimeError, "Could not generate Ideal VPR array");
  }

done:
  RAVE_FREE(htarray);
  RAVE_FREE(elems);
  return pylist;
}

PyObject* _pyvprcorrectionhelper_lsqFirstOrder(PyObject* self, PyObject* args)
{
  PyObject* pyobj = NULL;
  PyObject* result = NULL;
  double *x = NULL, *y = NULL, a = 0.0, b = 0.0;
  int nelem = 0.0;
  int i = 0;

  if(!PyArg_ParseTuple(args, "O", &pyobj)) {
    return NULL;
  }
  if (!PyList_Check(pyobj)) {
    raiseException_returnNULL(PyExc_TypeError, "A list of tuples with (x,y) pairs required");
  }
  nelem = PyList_Size(pyobj);
  x = RAVE_MALLOC(sizeof(double) * nelem);
  y = RAVE_MALLOC(sizeof(double) * nelem);
  if (x == NULL || y == NULL) {
    raiseException_gotoTag(done, PyExc_RuntimeError, "Failed to allocate memory");
  }
  for (i = 0; i < nelem; i++) {
    double xval, yval;
    PyObject* pyval = PyList_GetItem(pyobj, (Py_ssize_t)i); /* Borrows reference */
    if (!PyArg_ParseTuple(pyval, "dd", &xval, &yval)) {
      Py_XDECREF(pyval);
      goto done;
    }
    x[i] = xval;
    y[i] = yval;
  }
  if (!RaveVprCorrectionHelper_lsqFirstOrder(nelem, x, y, &a, &b)) {
    raiseException_gotoTag(done, PyExc_FloatingPointError, "Failed to calculate a & b coefficients");
  }

  result = Py_BuildValue("(dd)", a, b);
done:
  RAVE_FREE(x);
  RAVE_FREE(y);
  return result;
}

PyObject* _pyvprcorrectionhelper_readH1D(PyObject* self, PyObject* args)
{
  char* filename;
  int nitems = 0, i = 0;
  RaveHeightTemperature_t* profile = NULL;
  PyObject *pylist = NULL, *result = 0;
  PyObject *pyheight = NULL, *pytemp = NULL;

  if(!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }
  profile = RaveVprCorrectionHelper_readH1D(filename, &nitems);
  if (profile != NULL) {
    pylist = PyList_New(nitems);
    if (pylist != NULL) {
      for (i = 0; i < nitems; i++) {
        pyheight = PyFloat_FromDouble(profile[i].height);
        pytemp = PyFloat_FromDouble(profile[i].temperature);
        if (pyheight != NULL && pytemp != NULL) {
          PyList_SetItem(pylist, i, Py_BuildValue("(OO)", pyheight, pytemp)); /* SET ITEM STEAL REFERENCE */
        } else {
          raiseException_gotoTag(done, PyExc_RuntimeError, "Failed to create height or temperature objects");
        }
        Py_XDECREF(pyheight);
        Py_XDECREF(pytemp);
      }
      result = pylist;
      pylist = NULL;
    } else {
      raiseException_gotoTag(done, PyExc_RuntimeError, "Failed to create list");
    }
  } else {
    raiseException_gotoTag(done, PyExc_RuntimeError, "Failed to read profile");
  }

done:
  Py_XDECREF(pylist);
  return result;
}

/**
 * All methods a transformator can have
 */
static struct PyMethodDef _pyvprcorrection_methods[] =
{
  {"minReflectivity", NULL, METH_VARARGS},
  {"heightLimit", NULL, METH_VARARGS},
  {"profileHeight", NULL, METH_VARARGS},
  {"minDistance", NULL, METH_VARARGS},
  {"maxDistance", NULL, METH_VARARGS},
  {"plusTemperature", NULL, METH_VARARGS},
  {"minusTemperature", NULL, METH_VARARGS},
  {"dzdh", NULL, METH_VARARGS},
  {"getReflectivityArray", (PyCFunction) _pyvprcorrection_getReflectivityArray, 1,
    "getReflectivityArray(pvol) -> list of reflectivities\n\n"
    "Creates a list of reflectivities containing the median reflectivities for each profile height bin.\n\n"
    "pvol - the polar volume to be processed"
  },
  {"getIdealVpr", (PyCFunction) _pyvprcorrection_getIdealVpr, 1,
    "getIdealVpr(pvol, hto) -> ideal vpr array\n\n"
    "Creates the ideal VPR from the volume and height temperature array.\n\n"
    "pvol - the polar volume to be processed\n"
    "hto  - a list of height-temperature tuples"
  },
  {"getNumberOfHeightIntervals", (PyCFunction) _pyvprcorrection_getNumberOfHeightIntervals, 1,
    "getNumberOfHeightIntervals() -> number of height intervals\n\n"
    "Returns the number of height intervals that the current heightLimit / profileHeight setting gives."
  },
  {"getHeights", (PyCFunction) _pyvprcorrection_getHeights, 1,
    "getHeights() -> number of height intervals\n\n"
    "Returns an height array with all actual heights for the current heightLimit / profileHeight setting."
  },
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the vpr correction instance
 * @param[in] self - the vpr correction
 */
static PyObject* _pyvprcorrection_getattro(PyVprCorrection* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("minReflectivity", name) == 0) {
    return PyFloat_FromDouble(RaveVprCorrection_getMinReflectivity(self->vpr));
  }else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("heightLimit", name) == 0) {
    return PyFloat_FromDouble(RaveVprCorrection_getHeightLimit(self->vpr));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("profileHeight", name) == 0) {
    return PyFloat_FromDouble(RaveVprCorrection_getProfileHeight(self->vpr));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("minDistance", name) == 0) {
    return PyFloat_FromDouble(RaveVprCorrection_getMinDistance(self->vpr));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("maxDistance", name) == 0) {
    return PyFloat_FromDouble(RaveVprCorrection_getMaxDistance(self->vpr));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("plusTemperature", name) == 0) {
    return PyFloat_FromDouble(RaveVprCorrection_getPlusTemperature(self->vpr));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("minusTemperature", name) == 0) {
    return PyFloat_FromDouble(RaveVprCorrection_getMinusTemperature(self->vpr));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("dzdh", name) == 0) {
    return PyFloat_FromDouble(RaveVprCorrection_getDzdh(self->vpr));
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the vpr correction
 */
static int _pyvprcorrection_setattro(PyVprCorrection* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }

  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("minReflectivity", name) == 0) {
    if (PyInt_Check(val)) {
      RaveVprCorrection_setMinReflectivity(self->vpr, (double)PyInt_AsLong(val));
    } else if (PyFloat_Check(val)) {
      RaveVprCorrection_setMinReflectivity(self->vpr, (double)PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"Min reflectivity must be a valid float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("heightLimit", name) == 0) {
    if (PyInt_Check(val)) {
      RaveVprCorrection_setHeightLimit(self->vpr, (double)PyInt_AsLong(val));
    } else if (PyFloat_Check(val)) {
      RaveVprCorrection_setHeightLimit(self->vpr, (double)PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"Height must be a valid float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("profileHeight", name) == 0) {
    if (PyInt_Check(val)) {
      RaveVprCorrection_setProfileHeight(self->vpr, (double)PyInt_AsLong(val));
    } else if (PyFloat_Check(val)) {
      RaveVprCorrection_setProfileHeight(self->vpr, (double)PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"Profile Height must be a valid float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("minDistance", name) == 0) {
    if (PyInt_Check(val)) {
      RaveVprCorrection_setMinDistance(self->vpr, (double)PyInt_AsLong(val));
    } else if (PyFloat_Check(val)) {
      RaveVprCorrection_setMinDistance(self->vpr, (double)PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"Min Distance must be a valid float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("maxDistance", name) == 0) {
    if (PyInt_Check(val)) {
      RaveVprCorrection_setMaxDistance(self->vpr, (double)PyInt_AsLong(val));
    } else if (PyFloat_Check(val)) {
      RaveVprCorrection_setMaxDistance(self->vpr, (double)PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"Max Distance must be a valid float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("plusTemperature", name) == 0) {
    if (PyInt_Check(val)) {
      RaveVprCorrection_setPlusTemperature(self->vpr, (double)PyInt_AsLong(val));
    } else if (PyFloat_Check(val)) {
      RaveVprCorrection_setPlusTemperature(self->vpr, (double)PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"Plus temperature must be a valid float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("minusTemperature", name) == 0) {
    if (PyInt_Check(val)) {
      RaveVprCorrection_setMinusTemperature(self->vpr, (double)PyInt_AsLong(val));
    } else if (PyFloat_Check(val)) {
      RaveVprCorrection_setMinusTemperature(self->vpr, (double)PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"Minus temperature must be a valid float");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("dzdh", name) == 0) {
    if (PyInt_Check(val)) {
      RaveVprCorrection_setDzdh(self->vpr, (double)PyInt_AsLong(val));
    } else if (PyFloat_Check(val)) {
      RaveVprCorrection_setDzdh(self->vpr, (double)PyFloat_AsDouble(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError,"Dzdh must be a valid float");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }

  result = 0;
done:
  return result;
}
/*@} End of Transform */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyvprcorrection_type_doc,
    "One type of implementation of the vpr correction functionality. NOTE: This module has not been tested or verified!\n"
    "There are several settings required to generate the vpr profile as well as H1D files to get temperatures and heights for that radar.\n"
    " minReflectivity   - the min reflectivity to be used when generating the vpr profile.\n"
    " heightLimit       - the height limit for what reflectivities should be used in the identification of stratiform and convective rain.\n"
    " profileHeight     - the height of the individual profile bins. The resolution of the reflectivity profile will be heightLimit / profileHeight.\n"
    " minDistance       - the min distance limit for what reflectivities should be used in the identification of stratiform and convective rain.\n"
    " maxDistance       - the max distance limit for what reflectivities should be used in the identification of stratiform and convective rain.\n"
    " plusTemperature   - the plus temperature used for ensuring that the temperatures are in a suitable range.\n"
    "                     Selection is based on that at least one of the temperatures must be lower than the minus temp and one above the plus temperature.\n"
    " minusTemperature  - the minus temperature used for ensuring that the temperatures are in a suitable range.\n"
    "                     Selection is based on that at least one of the temperatures must be lower than the minus temp and one above the plus temperature.\n"
    " dzdh              - the lowest dzdh slope that should be allowed above the bright band.\n"
    "                     The slope also has to be negative. This means that the allowed slope should be dzdh < slope < 0.\n"
    "\n"
    "Usage:\n"
    " import _vprcorrection, _raveio\n"
    " pvol = _raveio.open(\"polar_volume.h5\")\n"
    " vpr = _vprcorrection.new()\n"
    " vpr.minReflectivity = 0.0\n"
    " vpr.minDistance = 1000.0\n"
    " vpr.maxDistance = 25000.0\n"
    " vpr.profileHeight = 100.0\n"
    " vpr.heightLimit = 10000.0\n"
    " htarr = _vprcorrection.readH1D(\"RAD_H1D_201107010100+001H00M.bpm\")\n"
    " result = vpr.getIdealVpr(pvol, htarr)"
    );
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyVprCorrection_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "VprCorrectionCore", /*tp_name*/
  sizeof(PyVprCorrection), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyvprcorrection_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pyvprcorrection_getattro, /*tp_getattro*/
  (setattrofunc)_pyvprcorrection_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyvprcorrection_type_doc,    /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyvprcorrection_methods,     /*tp_methods*/
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
  {"new", (PyCFunction)_pyvprcorrection_new, 1,
    "new() -> new instance of the VprCorrectionCore object\n\n"
    "Creates a new instance of the VprCorrectionCore object"
  },
  {"readH1D", (PyCFunction) _pyvprcorrectionhelper_readH1D, 1,
    "readH1D(filename) -> list of tuples (height, temperature)\n\n"
    "Reads a H1D file to get temperatures and heights in a list.\n\n"
    "filename - The name of the H1D file"
  },
  {"lsqFirstOrder", (PyCFunction) _pyvprcorrectionhelper_lsqFirstOrder, 1,
    "lsqFirstOrder([(x,y),(x1,x1),....]) -> tuple with the a & b coefficient\n\n"
    "Least square fit of a first degree polynomal. Takes a list of tuples with (x, y) and executes a least square fitting.\n"
    "Result will be a & b coefficients in the equation ax + b\n\n"
    "[(x,y),(x1,y1),....] - a list of tuples with x as first value and y as second"
  },
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_vprcorrection)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyVprCorrection_API[PyVprCorrection_API_pointers];
  PyObject *c_api_object = NULL;
  MOD_INIT_SETUP_TYPE(PyVprCorrection_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyVprCorrection_Type);

  MOD_INIT_DEF(module, "_vprcorrection", _pyvprcorrection_type_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyVprCorrection_API[PyVprCorrection_Type_NUM] = (void*)&PyVprCorrection_Type;
  PyVprCorrection_API[PyVprCorrection_GetNative_NUM] = (void *)PyVprCorrection_GetNative;
  PyVprCorrection_API[PyVprCorrection_New_NUM] = (void*)PyVprCorrection_New;

  c_api_object = PyCapsule_New(PyVprCorrection_API, PyVprCorrection_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_vprcorrection.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _vprcorrection.error");
    return MOD_INIT_ERROR;
  }

  import_pypolarvolume();
  import_pypolarscan();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
