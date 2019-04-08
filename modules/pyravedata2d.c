/* --------------------------------------------------------------------
Copyright (C) 2009-2019 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the RaveData2D API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2019-02-18
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYRAVEDATA2D_MODULE        /**< to get correct part of pyravedata2d.h */
#include "pyravedata2d.h"

#include <arrayobject.h>
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_ravedata2d");

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

static PyObject* _pyravedata2d_setData(PyRaveData2D* self, PyObject* args);

/*@{ Rave field */
/**
 * Returns the native RaveField_t instance.
 * @param[in] pyfield - self
 * @returns the native cartesian instance.
 */
static RaveData2D_t* PyRaveData2D_GetNative(PyRaveData2D* pyfield)
{
  RAVE_ASSERT((pyfield != NULL), "pyfield == NULL");
  return RAVE_OBJECT_COPY(pyfield->field);
}

/**
 * Creates a python rave data2d field from a native rave data2d field or will create an
 * initial native RaveData2D field if p is NULL.
 * @param[in] p - the native rave field (or NULL)
 * @returns the python rave field.
 */
static PyRaveData2D* PyRaveData2D_New(RaveData2D_t* p)
{
  PyRaveData2D* result = NULL;
  RaveData2D_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&RaveData2D_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for rave data2d field.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for rave data2d field.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyRaveData2D, &PyRaveData2D_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->field = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->field, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyRaveData2D instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for rave data2d field.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the rave field
 * @param[in] obj the object to deallocate.
 */
static void _pyravedata2d_dealloc(PyRaveData2D* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->field, obj);
  RAVE_OBJECT_RELEASE(obj->field);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the rave data 2d field.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyravedata2d_new(PyObject* self, PyObject* args)
{
  PyObject* pyin = NULL;
  PyRaveData2D* result = PyRaveData2D_New(NULL);
  if (!PyArg_ParseTuple(args, "|O", &pyin))
    return NULL;
  if (pyin != NULL) {
    if (_pyravedata2d_setData(result, args) == NULL) {
      Py_DECREF(result);
      result = NULL;
    }
  }
  return (PyObject*)result;
}

/**
 * Sets the data
 * @param[in] self this instance.
 * @param[in] args arguments for creation
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyravedata2d_setData(PyRaveData2D* self, PyObject* args)
{
  PyObject* inarray = NULL;
  PyArrayObject* arraydata = NULL;
  RaveDataType datatype = RaveDataType_UNDEFINED;
  long xsize = 0;
  long ysize = 0;
  unsigned char* data = NULL;

  if (!PyArg_ParseTuple(args, "O", &inarray)) {
    return NULL;
  }

  if (!PyArray_Check(inarray)) {
    raiseException_returnNULL(PyExc_TypeError, "Data must be of arrayobject type")
  }

  arraydata = (PyArrayObject*)inarray;

  if (PyArray_NDIM(arraydata) != 2) {
    raiseException_returnNULL(PyExc_ValueError, "A cartesian product must be of rank 2");
  }

  datatype = translate_pyarraytype_to_ravetype(PyArray_TYPE(arraydata));

  if (PyArray_ITEMSIZE(arraydata) != get_ravetype_size(datatype)) {
    raiseException_returnNULL(PyExc_TypeError, "numpy and rave does not have same data sizes");
  }

  xsize  = PyArray_DIM(arraydata, 1);
  ysize  = PyArray_DIM(arraydata, 0);
  data   = PyArray_DATA(arraydata);

  if (!RaveData2D_setData(self->field, xsize, ysize, data, datatype)) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not allocate memory");
  }

  Py_RETURN_NONE;
}

static PyObject* _pyravedata2d_getData(PyRaveData2D* self, PyObject* args)
{
  long xsize = 0, ysize = 0;
  RaveDataType type = RaveDataType_UNDEFINED;
  PyObject* result = NULL;
  npy_intp dims[2] = {0,0};
  int arrtype = 0;
  void* data = NULL;

  xsize = RaveData2D_getXsize(self->field);
  ysize = RaveData2D_getYsize(self->field);
  type = RaveData2D_getType(self->field);
  data = RaveData2D_getData(self->field);

  dims[1] = (npy_intp)xsize;
  dims[0] = (npy_intp)ysize;
  arrtype = translate_ravetype_to_pyarraytype(type);

  if (data == NULL) {
    raiseException_returnNULL(PyExc_IOError, "rave data2d field does not have any data");
  }

  if (arrtype == PyArray_NOTYPE) {
    raiseException_returnNULL(PyExc_IOError, "Could not translate data type");
  }
  result = PyArray_SimpleNew(2, dims, arrtype);
  if (result == NULL) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not create resulting array");
  }
  if (result != NULL) {
    int nbytes = xsize*ysize*PyArray_ITEMSIZE(result);
    memcpy(((PyArrayObject*)result)->data, (unsigned char*)RaveData2D_getData(self->field), nbytes);
  }
  return result;
}

static PyObject* _pyravedata2d_setValue(PyRaveData2D* self, PyObject* args)
{
  long x = 0, y = 0;
  double value = 0.0;
  if (!PyArg_ParseTuple(args, "lld", &x, &y, &value)) {
    return NULL;
  }

  if (!RaveData2D_setValue(self->field, x, y, value)) {
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject* _pyravedata2d_getValue(PyRaveData2D* self, PyObject* args)
{
  double value = 0.0L;
  long x = 0, y = 0;
  int result = 0;

  if (!PyArg_ParseTuple(args, "ll", &x, &y)) {
    return NULL;
  }

  result = RaveData2D_getValue(self->field, x, y, &value);

  return Py_BuildValue("(id)", result, value);
}

/**
 * Concatenates two fields x-wise.
 * @param[in] self - self
 * @param[in] args - the other rave field object
 * @return a rave data2d field object on success otherwise NULL
 */
static PyObject* _pyravedata2d_concatx(PyRaveData2D* self, PyObject* args)
{
  PyObject* result = NULL;
  PyObject* pyin = NULL;
  RaveData2D_t *field = NULL;
  if (!PyArg_ParseTuple(args, "O", &pyin)) {
    return NULL;
  }
  if (!PyRaveData2D_Check(pyin)) {
    raiseException_returnNULL(PyExc_ValueError, "Argument must be another rave data2d field");
  }
  field = RaveData2D_concatX(self->field, ((PyRaveData2D*)pyin)->field);
  if (field == NULL) {
    raiseException_gotoTag(done, PyExc_ValueError, "Failed to concatenate fields");
  }

  result = (PyObject*)PyRaveData2D_New(field);
done:
  RAVE_OBJECT_RELEASE(field);
  return result;
}

static PyObject* _pyravedata2d_min(PyRaveData2D* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyFloat_FromDouble(RaveData2D_min(self->field));
}

static PyObject* _pyravedata2d_max(PyRaveData2D* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyFloat_FromDouble(RaveData2D_max(self->field));
}

static PyObject* _pyravedata2d_fill(PyRaveData2D* self, PyObject* args)
{
  double value = 0.0;

  if (!PyArg_ParseTuple(args, "d", &value)) {
    return NULL;
  }

  if (!RaveData2D_fill(self->field, value))
    raiseException_returnNULL(PyExc_ValueError, "Failed to fill field");

  Py_RETURN_NONE;
}

static PyObject* _pyravedata2d_circshift(PyRaveData2D* self, PyObject* args)
{
  long x = 0, y = 0;
  PyObject* result = NULL;
  RaveData2D_t *field = NULL;

  if (!PyArg_ParseTuple(args, "ll", &x, &y)) {
    return NULL;
  }

  field = RaveData2D_circshift(self->field, x, y);
  if (field == NULL) {
    raiseException_gotoTag(done, PyExc_ValueError, "Failed to run circular shift on field");
  }

  result = (PyObject*)PyRaveData2D_New(field);
done:
  RAVE_OBJECT_RELEASE(field);
  return result;
}

static PyObject* _pyravedata2d_add(PyRaveData2D* self, PyObject* args)
{
  PyObject* pyin = NULL;
  PyObject* result = NULL;
  RaveData2D_t* field = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyin)) {
    return NULL;
  }
  if (PyFloat_Check(pyin)) {
    field = RaveData2D_addNumber(self->field, PyFloat_AsDouble(pyin));
  } else if (PyInt_Check(pyin)) {
    field = RaveData2D_addNumber(self->field, (double)PyInt_AsLong(pyin));
  } else if (PyLong_Check(pyin)) {
    field = RaveData2D_addNumber(self->field, (double)PyLong_AsLong(pyin));
  } else if (PyRaveData2D_Check(pyin)) {
    field = RaveData2D_add(self->field, ((PyRaveData2D*)pyin)->field);
  } else {
    raiseException_gotoTag(done, PyExc_ValueError, "Value added must be a number of another RaveData2D object");
  }

  result = (PyObject*)PyRaveData2D_New(field);
done:
  RAVE_OBJECT_RELEASE(field);
  return result;
}

static PyObject* _pyravedata2d_sub(PyRaveData2D* self, PyObject* args)
{
  PyObject* pyin = NULL;
  PyObject* result = NULL;
  RaveData2D_t* field = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyin)) {
    return NULL;
  }
  if (PyFloat_Check(pyin)) {
    field = RaveData2D_subNumber(self->field, PyFloat_AsDouble(pyin));
  } else if (PyInt_Check(pyin)) {
    field = RaveData2D_subNumber(self->field, (double)PyInt_AsLong(pyin));
  } else if (PyLong_Check(pyin)) {
    field = RaveData2D_subNumber(self->field, (double)PyLong_AsLong(pyin));
  } else if (PyRaveData2D_Check(pyin)) {
    field = RaveData2D_sub(self->field, ((PyRaveData2D*)pyin)->field);
  } else {
    raiseException_gotoTag(done, PyExc_ValueError, "Value substracted must be a number of another RaveData2D object");
  }

  result = (PyObject*)PyRaveData2D_New(field);
done:
  RAVE_OBJECT_RELEASE(field);
  return result;
}

static PyObject* _pyravedata2d_emul(PyRaveData2D* self, PyObject* args)
{
  PyObject* pyin = NULL;
  PyObject* result = NULL;
  RaveData2D_t* field = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyin)) {
    return NULL;
  }
  if (PyFloat_Check(pyin)) {
    field = RaveData2D_mulNumber(self->field, PyFloat_AsDouble(pyin));
  } else if (PyInt_Check(pyin)) {
    field = RaveData2D_mulNumber(self->field, (double)PyInt_AsLong(pyin));
  } else if (PyLong_Check(pyin)) {
    field = RaveData2D_mulNumber(self->field, (double)PyLong_AsLong(pyin));
  } else if (PyRaveData2D_Check(pyin)) {
    field = RaveData2D_emul(self->field, ((PyRaveData2D*)pyin)->field);
  } else {
    raiseException_gotoTag(done, PyExc_ValueError, "Value multiplicated with must be a number of another RaveData2D object");
  }

  result = (PyObject*)PyRaveData2D_New(field);
done:
  RAVE_OBJECT_RELEASE(field);
  return result;
}

static PyObject* _pyravedata2d_epow(PyRaveData2D* self, PyObject* args)
{
  PyObject* pyin = NULL;
  PyObject* result = NULL;
  RaveData2D_t* field = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyin)) {
    return NULL;
  }
  if (PyFloat_Check(pyin)) {
    field = RaveData2D_powNumber(self->field, PyFloat_AsDouble(pyin));
  } else if (PyInt_Check(pyin)) {
    field = RaveData2D_powNumber(self->field, (double)PyInt_AsLong(pyin));
  } else if (PyLong_Check(pyin)) {
    field = RaveData2D_powNumber(self->field, (double)PyLong_AsLong(pyin));
  } else if (PyRaveData2D_Check(pyin)) {
    field = RaveData2D_epow(self->field, ((PyRaveData2D*)pyin)->field);
  } else {
    raiseException_gotoTag(done, PyExc_ValueError, "Value pow with must be a number of another RaveData2D object");
  }

  result = (PyObject*)PyRaveData2D_New(field);
done:
  RAVE_OBJECT_RELEASE(field);
  return result;
}

static PyObject* _pyravedata2d_medfilt2(PyRaveData2D* self, PyObject* args)
{
  PyObject* result = NULL;
  RaveData2D_t* field = NULL;
  long winXsize = 0, winYsize = 0;
  if (!PyArg_ParseTuple(args, "ll", &winXsize, &winYsize)) {
    return NULL;
  }
  field = RaveData2D_medfilt2(self->field, winXsize, winYsize);
  if (field == NULL) {
    raiseException_gotoTag(done, PyExc_RuntimeError, "Failed to generate medfilt2 field");
  }
  result = (PyObject*)PyRaveData2D_New(field);
done:
  RAVE_OBJECT_RELEASE(field);
  return result;
}

static PyObject* _pyravedata2d_cumsum(PyRaveData2D* self, PyObject* args)
{
  PyObject* result = NULL;
  RaveData2D_t* field = NULL;
  int dir = 1;
  if (!PyArg_ParseTuple(args, "|i", &dir)) {
    return NULL;
  }
  field = RaveData2D_cumsum(self->field, dir);
  if (field == NULL) {
    raiseException_gotoTag(done, PyExc_RuntimeError, "Failed to generate cumsum field");
  }
  result = (PyObject*)PyRaveData2D_New(field);
done:
  RAVE_OBJECT_RELEASE(field);
  return result;
}

static PyObject* _pyravedata2d_movingstd(PyRaveData2D* self, PyObject* args)
{
  PyObject* result = NULL;
  RaveData2D_t* field = NULL;
  long nx = 0, ny = 0;
  if (!PyArg_ParseTuple(args, "ll", &nx, &ny)) {
    return NULL;
  }

  field = RaveData2D_movingstd(self->field, nx, ny);
  if (field == NULL) {
    raiseException_gotoTag(done, PyExc_RuntimeError, "Failed to generate movingstd field");
  }
  result = (PyObject*)PyRaveData2D_New(field);
done:
  RAVE_OBJECT_RELEASE(field);
  return result;
}

static PyObject* _pyravedata2d_hist(PyRaveData2D* self, PyObject* args)
{
  long* hist = NULL;
  long nbins = 0, i = 0;
  long nnodata = 0;
  PyObject* result = NULL;
  if (!PyArg_ParseTuple(args, "l", &nbins))
    return NULL;
  hist = RaveData2D_hist(self->field, nbins, &nnodata);
  if (hist == NULL) {
    raiseException_returnNULL(PyExc_RuntimeError, "Failed to generate histogram");
  }
  result = PyList_New(nbins);
  if (result != NULL) {
    for (i = 0; i < nbins; i++) {
      PyList_SetItem(result, i, PyInt_FromLong(hist[i]));
    }
  }
  RAVE_FREE(hist);
  return result;
}

static PyObject* _pyravedata2d_entropy(PyRaveData2D* self, PyObject* args)
{
  double entropy = 0.0;
  long nbins = 2;

  if (!PyArg_ParseTuple(args, "|l", &nbins))
    return NULL;
  if (!RaveData2D_entropy(self->field, nbins, &entropy)) {
    raiseException_returnNULL(PyExc_RuntimeError, "Failed to generate entropy");
  }
  return PyFloat_FromDouble(entropy);
}

static PyObject* _pyravedata2d_str(PyRaveData2D* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, ""))
    return NULL;
  return PyString_FromString(RaveData2D_str(self->field));
}

MOD_DIR_FORWARD_DECLARE(PyRaveData2D);

/**
 * All methods a cartesian product can have
 */
static struct PyMethodDef _pyravedata2d_methods[] =
{
  {"xsize", NULL},
  {"ysize", NULL},
  {"datatype", NULL},
  {"nodata", NULL},
  {"useNodata", NULL},
  {"setData", (PyCFunction) _pyravedata2d_setData, 1},
  {"getData", (PyCFunction) _pyravedata2d_getData, 1},
  {"setValue", (PyCFunction) _pyravedata2d_setValue, 1},
  {"getValue", (PyCFunction) _pyravedata2d_getValue, 1},
  {"concatx", (PyCFunction) _pyravedata2d_concatx, 1},
  {"min", (PyCFunction) _pyravedata2d_min, 1},
  {"max", (PyCFunction) _pyravedata2d_max, 1},
  {"fill", (PyCFunction) _pyravedata2d_fill, 1},
  {"circshift", (PyCFunction) _pyravedata2d_circshift, 1},
  {"add", (PyCFunction) _pyravedata2d_add, 1},
  {"sub", (PyCFunction) _pyravedata2d_sub, 1},
  {"emul", (PyCFunction) _pyravedata2d_emul, 1},
  {"epow", (PyCFunction) _pyravedata2d_epow, 1},
  {"medfilt2", (PyCFunction) _pyravedata2d_medfilt2, 1},
  {"cumsum", (PyCFunction) _pyravedata2d_cumsum, 1},
  {"movingstd", (PyCFunction) _pyravedata2d_movingstd, 1},
  {"hist", (PyCFunction) _pyravedata2d_hist, 1},
  {"entropy", (PyCFunction) _pyravedata2d_entropy, 1},
  {"str", (PyCFunction) _pyravedata2d_str, 1},
  {"__dir__", (PyCFunction) MOD_DIR_REFERENCE(PyRaveData2D), METH_NOARGS},
  {NULL, NULL } /* sentinel */
};

MOD_DIR_FUNCTION(PyRaveData2D, _pyravedata2d_methods)

/**
 * Returns the specified attribute in the rave data2d field
 * @param[in] self - the rave data2d field
 */
static PyObject* _pyravedata2d_getattro(PyRaveData2D* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "xsize") == 0) {
    return PyInt_FromLong(RaveData2D_getXsize(self->field));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "ysize") == 0) {
    return PyInt_FromLong(RaveData2D_getYsize(self->field));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "datatype") == 0) {
    return PyInt_FromLong(RaveData2D_getType(self->field));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "nodata") == 0) {
    return PyFloat_FromDouble(RaveData2D_getNodata(self->field));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "useNodata") == 0) {
    return PyBool_FromLong(RaveData2D_usingNodata(self->field));
  }

  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the attribute value
 */
static int _pyravedata2d_setattro(PyRaveData2D *self, PyObject *name, PyObject *value)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "nodata") == 0) {
    if (PyFloat_Check(value)) {
      RaveData2D_setNodata(self->field, PyFloat_AsDouble(value));
    } else if (PyLong_Check(value)) {
      RaveData2D_setNodata(self->field, (double)PyLong_AsLong(value));
    } else if (PyInt_Check(value)) {
      RaveData2D_setNodata(self->field, (double)PyInt_AsLong(value));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "nodata must be of type float");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "useNodata") == 0) {
    if (PyBool_Check(value)) {
      if (PyObject_IsTrue(value))
        RaveData2D_useNodata(self->field, 1);
      else
        RaveData2D_useNodata(self->field, 0);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "useNodata must be of type bool");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError, PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }

  result = 0;
done:
  return result;
}

/*@} End of rave field */

/*@{ Type definitions */
PyTypeObject PyRaveData2D_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0)
  "RaveData2DCore",                  /*tp_name*/
  sizeof(PyRaveData2D),              /*tp_size*/
  0,                                /*tp_itemsize*/
  /* methods */
  (destructor)_pyravedata2d_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)0,                   /*tp_getattr*/
  (setattrfunc)0,                   /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_as_number */
  0, /*tp_as_sequence */
  0, /*tp_as_mapping */
  (hashfunc)0, /*tp_hash*/
  (ternaryfunc)0, /*tp_call*/
  (reprfunc)0, /*tp_str*/
  (getattrofunc)_pyravedata2d_getattro, /*tp_getattro*/
  (setattrofunc)_pyravedata2d_setattro, /*tp_setattro*/
  0, /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  0, /*tp_doc*/
  (traverseproc)0, /*tp_traverse*/
  (inquiry)0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  _pyravedata2d_methods, /*tp_methods*/
  0,                    /*tp_members*/
  0,                      /*tp_getset*/
  0,                      /*tp_base*/
  0,                      /*tp_dict*/
  0,                      /*tp_descr_get*/
  0,                      /*tp_descr_set*/
  0,                      /*tp_dictoffset*/
  0,                      /*tp_init*/
  0,                      /*tp_alloc*/
  0,                      /*tp_new*/
  0,                      /*tp_free*/
  0,                      /*tp_is_gc*/
};
/*@} End of Type definitions */

/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pyravedata2d_new, 1},
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_ravedata2d)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyRaveData2D_API[PyRaveData2D_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyRaveData2D_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyRaveData2D_Type);

  MOD_INIT_DEF(module, "_ravedata2d", NULL/*doc*/, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyRaveData2D_API[PyRaveData2D_Type_NUM] = (void*)&PyRaveData2D_Type;
  PyRaveData2D_API[PyRaveData2D_GetNative_NUM] = (void *)PyRaveData2D_GetNative;
  PyRaveData2D_API[PyRaveData2D_New_NUM] = (void*)PyRaveData2D_New;

  c_api_object = PyCapsule_New(PyRaveData2D_API, PyRaveData2D_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_ravedata2d.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _ravedata2d.error");
    return MOD_INIT_ERROR;
  }

  import_array(); /*To make sure I get access to Numeric*/
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */

