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
 * Python version of the Transform API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define PYBITMAP_GENERATOR_MODULE /**< include correct part of pybitmapgenerator.h */
#include <pybitmapgenerator.h>

#include "pycartesian.h"
#include "pycartesianparam.h"
#include "pyravefield.h"
#include "pyrave_debug.h"
#include "rave_alloc.h"

/**
 * This modules name
 */
PYRAVE_DEBUG_MODULE("_bitmapgenerator");

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

/*@{ SurroundingBitmap */
/**
 * Returns the native BitmapGenerator_t instance.
 * @param[in] pygenerator - the python instance
 * @returns the native bitmap generator instance.
 */
static BitmapGenerator_t*
PyBitmapGenerator_GetNative(PyBitmapGenerator* pygenerator)
{
  RAVE_ASSERT((pygenerator != NULL), "pygenerator == NULL");
  return RAVE_OBJECT_COPY(pygenerator->generator);
}

/**
 * Creates a python bitmap generator from a native bitmap generator or will create an
 * initial native bitmap generator if p is NULL.
 * @param[in] p - the native bitmap generator (or NULL)
 * @returns the python bitmap generator product.
 */
static PyBitmapGenerator*
PyBitmapGenerator_New(BitmapGenerator_t* p)
{
  PyBitmapGenerator* result = NULL;
  BitmapGenerator_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&BitmapGenerator_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for bitmap generator.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for bitmap generator.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyBitmapGenerator, &PyBitmapGenerator_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->generator = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->generator, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyBitmapGenerator instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyBitmapGenerator.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the generator
 * @param[in] obj the object to deallocate.
 */
static void _pybitmapgenerator_dealloc(PyBitmapGenerator* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->generator, obj);
  RAVE_OBJECT_RELEASE(obj->generator);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the generator.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pybitmapgenerator_new(PyObject* self, PyObject* args)
{
  PyBitmapGenerator* result = PyBitmapGenerator_New(NULL);
  return (PyObject*)result;
}

/**
 * Creates a surrounding bitmap
 * @param[in] self the bitmap generator
 * @param[in] args arguments for generating the the cartesian parameter
 * @return a RaveField on success otherwise an exception is thrown
 */
static PyObject* _pybitmapgenerator_create_surrounding(PyBitmapGenerator* self, PyObject* args)
{
  PyObject* pycartesianparam = NULL;
  RaveField_t* result = NULL;
  PyObject* pyresult = NULL;

  if(!PyArg_ParseTuple(args, "O", &pycartesianparam)) {
    return NULL;
  }

  if (!PyCartesianParam_Check(pycartesianparam)) {
    raiseException_returnNULL(PyExc_TypeError, "Argument should be a cartesian parameter");
  }

  result = BitmapGenerator_create_surrounding(self->generator, ((PyCartesianParam*)pycartesianparam)->param);
  if (result == NULL) {
    raiseException_returnNULL(PyExc_IOError, "Failed to cartesian parameter into a bitmap");
  }

  pyresult = (PyObject*)PyRaveField_New(result);

  RAVE_OBJECT_RELEASE(result);
  return pyresult;
}

/**
 * Creates a bitmap with the intersections
 * @param[in] self the bitmap generator
 * @param[in] args arguments for generating the intersection (cartesian parameter, radar index field)
 * @return a RaveField on success otherwise an exception is thrown
 */
static PyObject* _pybitmapgenerator_create_intersect(PyBitmapGenerator* self, PyObject* args)
{
  PyObject* pycartesianparam = NULL;
  char* qualityFlag = NULL;
  RaveField_t* qualityField = NULL;
  RaveField_t* result = NULL;
  PyObject* pyresult = NULL;

  if(!PyArg_ParseTuple(args, "Os", &pycartesianparam, &qualityFlag)) {
    return NULL;
  }

  if (!PyCartesianParam_Check(pycartesianparam)) {
    raiseException_returnNULL(PyExc_TypeError, "Argument 1 should be a cartesian parameter");
  }

  qualityField = CartesianParam_getQualityFieldByHowTask(((PyCartesianParam*)pycartesianparam)->param, qualityFlag);
  if (qualityField == NULL) {
    raiseException_returnNULL(PyExc_TypeError, "Could not find a matching quality field");
  }

  result = BitmapGenerator_create_intersect(self->generator, ((PyCartesianParam*)pycartesianparam)->param, qualityFlag);
  if (result == NULL) {
    raiseException_gotoTag(done, PyExc_IOError, "Failed to create an intersection for the parameter");
  }

  pyresult = (PyObject*)PyRaveField_New(result);
done:
  RAVE_OBJECT_RELEASE(qualityField);
  RAVE_OBJECT_RELEASE(result);
  return pyresult;
}

/**
 * All methods a bitmap generator can have
 */
static struct PyMethodDef _pybitmapgenerator_methods[] =
{
  {"create_surrounding", (PyCFunction) _pybitmapgenerator_create_surrounding, 1},
  {"create_intersect", (PyCFunction) _pybitmapgenerator_create_intersect, 1},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the bitmap generator
 * @param[in] self - the transform
 */
static PyObject* _pybitmapgenerator_getattro(PyBitmapGenerator* self, PyObject* name)
{
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the bitmap generator
 */
static int _pybitmapgenerator_setattro(PyBitmapGenerator* self, PyObject* name, PyObject* val)
{
  int result = -1;
  return result;
}
/*@} End of Bitmap generator */

/*@{ Type definitions */
PyTypeObject PyBitmapGenerator_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "BitmapGeneratorCore", /*tp_name*/
  sizeof(PyBitmapGenerator), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pybitmapgenerator_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pybitmapgenerator_getattro, /*tp_getattro*/
  (setattrofunc)_pybitmapgenerator_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  0,                            /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pybitmapgenerator_methods,   /*tp_methods*/
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
  0,                            /*tp_is_gc*/};

/*@} End of Type definitions */

/*@{ Module setup */
static PyMethodDef functions[] = {
  {"new", (PyCFunction)_pybitmapgenerator_new, 1},
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_bitmapgenerator)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyBitmapGenerator_API[PyBitmapGenerator_API_pointers];
  PyObject *c_api_object = NULL;
  MOD_INIT_SETUP_TYPE(PyBitmapGenerator_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyBitmapGenerator_Type);

  MOD_INIT_DEF(module, "_bitmapgenerator", NULL/*doc*/, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyBitmapGenerator_API[PyBitmapGenerator_Type_NUM] = (void*)&PyBitmapGenerator_Type;
  PyBitmapGenerator_API[PyBitmapGenerator_GetNative_NUM] = (void *)PyBitmapGenerator_GetNative;
  PyBitmapGenerator_API[PyBitmapGenerator_New_NUM] = (void*)PyBitmapGenerator_New;

  c_api_object = PyCapsule_New(PyBitmapGenerator_API, PyBitmapGenerator_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_bitmapgenerator.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _bitmapgenerator.error");
    return MOD_INIT_ERROR;
  }

  import_pycartesian();
  import_pycartesianparam();
  import_pyravefield();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
