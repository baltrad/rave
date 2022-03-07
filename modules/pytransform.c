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

#define PYTRANSFORM_MODULE /**< include correct part of pytransform.h */
#include "pytransform.h"

#include "pyarea.h"
#include "pypolarscan.h"
#include "pypolarvolume.h"
#include "pycartesian.h"
#include "pycartesianparam.h"
#include "pyradardefinition.h"
#include "pyrave_debug.h"
#include "rave_alloc.h"

/**
 * This modules name
 */
PYRAVE_DEBUG_MODULE("_transform");

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

/*@{ Transform */
/**
 * Returns the native Transform_t instance.
 * @param[in] pytransform - the python transform instance
 * @returns the native transform instance.
 */
static Transform_t*
PyTransform_GetNative(PyTransform* pytransform)
{
  RAVE_ASSERT((pytransform != NULL), "pytransform == NULL");
  return RAVE_OBJECT_COPY(pytransform->transform);
}

/**
 * Creates a python transform from a native transform or will create an
 * initial native transform if p is NULL.
 * @param[in] p - the native transform (or NULL)
 * @returns the python transform product.
 */
static PyTransform*
PyTransform_New(Transform_t* p)
{
  PyTransform* result = NULL;
  Transform_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&Transform_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for transform.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for transform.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyTransform, &PyTransform_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->transform = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->transform, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyTransform instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyTransform.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the transformator
 * @param[in] obj the object to deallocate.
 */
static void _pytransform_dealloc(PyTransform* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->transform, obj);
  RAVE_OBJECT_RELEASE(obj->transform);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the transformator.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pytransform_new(PyObject* self, PyObject* args)
{
  PyTransform* result = PyTransform_New(NULL);
  return (PyObject*)result;
}

/**
 * Creates a ppi from a polar volume
 * @param[in] self the transformer
 * @param[in] args arguments for generating the ppi (polarvolume, cartesian)
 * @return Py_None on success, otherwise NULL
 */
static PyObject* _pytransform_ppi(PyTransform* self, PyObject* args)
{
  PyCartesian* cartesian = NULL;
  PyObject* pycartesian = NULL;
  PyPolarScan* scan = NULL;
  PyObject* pyscan = NULL;

  if(!PyArg_ParseTuple(args, "OO", &pyscan, &pycartesian)) {
    return NULL;
  }

  if (!PyPolarScan_Check(pyscan)) {
    raiseException_returnNULL(PyExc_TypeError, "First argument should be a polar scan")
  }

  if (!PyCartesian_Check(pycartesian)) {
    raiseException_returnNULL(PyExc_TypeError, "Second argument should be a cartesian product");
  }

  scan = (PyPolarScan*)pyscan;
  cartesian = (PyCartesian*)pycartesian;

  if (!Transform_ppi(self->transform, scan->scan, cartesian->cartesian)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to transform volume into a ppi");
  }

  Py_RETURN_NONE;
}

/**
 * Creates a cappi from a polar volume
 * @param[in] self the transformer
 * @param[in] args arguments for generating the cappi (polarvolume, cartesian, height in meters)
 * @return Py_None on success, otherwise NULL
 */
static PyObject* _pytransform_cappi(PyTransform* self, PyObject* args)
{
  PyCartesian* cartesian = NULL;
  PyObject* pycartesian = NULL;
  PyPolarVolume* pvol = NULL;
  PyObject* pypvol = NULL;
  double height = 0.0L;

  if(!PyArg_ParseTuple(args, "OOd", &pypvol, &pycartesian, &height)) {
    return NULL;
  }

  if (!PyPolarVolume_Check(pypvol)) {
    raiseException_returnNULL(PyExc_TypeError, "First argument should be a polar volume")
  }

  if (!PyCartesian_Check(pycartesian)) {
    raiseException_returnNULL(PyExc_TypeError, "Second argument should be a cartesian product");
  }

  pvol = (PyPolarVolume*)pypvol;
  cartesian = (PyCartesian*)pycartesian;

  if (!Transform_cappi(self->transform, pvol->pvol, cartesian->cartesian, height)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to transform volume into a cappi");
  }

  Py_RETURN_NONE;
}

/**
 * Creates a pseudo-cappi from a polar volume
 * @param[in] self the transformer
 * @param[in] args arguments for generating the pseudo-cappi (polarvolume, cartesian)
 * @return Py_None on success, otherwise NULL
 */
static PyObject* _pytransform_pcappi(PyTransform* self, PyObject* args)
{
  PyCartesian* cartesian = NULL;
  PyObject* pycartesian = NULL;
  PyPolarVolume* pvol = NULL;
  PyObject* pypvol = NULL;
  double height = 0.0L;

  if(!PyArg_ParseTuple(args, "OOd", &pypvol, &pycartesian,&height)) {
    return NULL;
  }

  if (!PyPolarVolume_Check(pypvol)) {
    raiseException_returnNULL(PyExc_TypeError, "First argument should be a polar volume")
  }

  if (!PyCartesian_Check(pycartesian)) {
    raiseException_returnNULL(PyExc_TypeError, "Second argument should be a cartesian product");
  }

  pvol = (PyPolarVolume*)pypvol;
  cartesian = (PyCartesian*)pycartesian;

  if (!Transform_pcappi(self->transform, pvol->pvol, cartesian->cartesian, height)) {
    raiseException_returnNULL(PyExc_IOError, "Failed to transform volume into a cappi");
  }

  Py_RETURN_NONE;
}

static PyObject* _pytransform_ctoscan(PyTransform* self, PyObject* args)
{
  PyCartesian* cartesian = NULL;
  PyObject* pycartesian = NULL;
  PyRadarDefinition* radardef = NULL;
  PyObject* pyradardef = NULL;
  double elangle = 0.0;
  char* quantity = NULL;
  PolarScan_t* scan = NULL;
  PyPolarScan* pyscan = NULL;

  if(!PyArg_ParseTuple(args, "OOds", &pycartesian, &pyradardef, &elangle, &quantity)) {
    return NULL;
  }

  if (!PyCartesian_Check(pycartesian)) {
    raiseException_returnNULL(PyExc_TypeError, "First argument should be a cartesian product");
  }
  if (!PyRadarDefinition_Check(pyradardef)) {
    raiseException_returnNULL(PyExc_TypeError, "Second argument should be a radar definition");
  }

  cartesian = (PyCartesian*)pycartesian;
  radardef = (PyRadarDefinition*)pyradardef;

  scan = Transform_ctoscan(self->transform, cartesian->cartesian, radardef->def, elangle, quantity);
  if (scan == NULL) {
    raiseException_returnNULL(PyExc_IOError, "Failed to transform cartesian into a scan");
  }

  pyscan = PyPolarScan_New(scan);
  RAVE_OBJECT_RELEASE(scan);
  return (PyObject*)pyscan;
}

static PyObject* _pytransform_ctop(PyTransform* self, PyObject* args)
{
  PyCartesian* cartesian = NULL;
  PyObject* pycartesian = NULL;
  PyRadarDefinition* radardef = NULL;
  PyObject* pyradardef = NULL;
  char* quantity = NULL;
  PolarVolume_t* volume = NULL;
  PyPolarVolume* pyvolume = NULL;

  if(!PyArg_ParseTuple(args, "OOs", &pycartesian, &pyradardef, &quantity)) {
    return NULL;
  }

  if (!PyCartesian_Check(pycartesian)) {
    raiseException_returnNULL(PyExc_TypeError, "First argument should be a cartesian product");
  }
  if (!PyRadarDefinition_Check(pyradardef)) {
    raiseException_returnNULL(PyExc_TypeError, "Second argument should be a radar definition");
  }

  cartesian = (PyCartesian*)pycartesian;
  radardef = (PyRadarDefinition*)pyradardef;

  volume = Transform_ctop(self->transform, cartesian->cartesian, radardef->def, quantity);
  if (volume == NULL) {
    raiseException_returnNULL(PyExc_IOError, "Failed to transform cartesian into a volume");
  }

  pyvolume = PyPolarVolume_New(volume);
  RAVE_OBJECT_RELEASE(volume);
  return (PyObject*)pyvolume;
}

/**
 * If a value is == UNDETECT and the surrounding 4 pixels == DATA, then
 * the value set is the avg for the surrounding 4 pixels.
 * If provided object is a cartesian parameter, only that parameter will be modified
 * and if the provided object is a cartesian product, all parameters will be modified.
 * @param[in] self - self
 * @param[in] args - a object, either a cartesian parameter or a cartesian product.
 * @return a new object with modified data.
 */
static PyObject* _pytransform_fillGap(PyTransform* self, PyObject* args)
{
  PyObject* inobj = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "O", &inobj)) {
    return NULL;
  }
  if (!PyCartesian_Check(inobj) && !PyCartesianParam_Check(inobj)) {
    raiseException_returnNULL(PyExc_AttributeError, "fillGap should be called with a cartesian or cartesian parameter");
  }

  if (PyCartesian_Check(inobj)) {
    Cartesian_t* filled = Transform_fillGap(self->transform, ((PyCartesian*)inobj)->cartesian);
    if (filled != NULL) {
      result = (PyObject*)PyCartesian_New(filled);
    }
    RAVE_OBJECT_RELEASE(filled);
  } else {
    CartesianParam_t* filled = Transform_fillGapOnParameter(self->transform, ((PyCartesianParam*)inobj)->param);
    if (filled != NULL) {
      result = (PyObject*)PyCartesianParam_New(filled);
    }
    RAVE_OBJECT_RELEASE(filled);
  }

  return result;
}

/**
 * Combines a number of cartesian areas into the one specified by the area definition. This should not be confused with the
 * cartesian composite generation. This function instead works like a area-combiner where the individual tiles will result
 * in a full area.
 */
static PyObject* _pytransform_combine_tiles(PyTransform* self, PyObject* args)
{
  PyObject* pyarea = NULL;
  PyObject* pytiles = NULL;
  Cartesian_t* result = NULL;
  PyObject* pyresult = NULL;
  RaveObjectList_t* tiles = NULL;
  Py_ssize_t n = 0, i = 0;

  if (!PyArg_ParseTuple(args, "OO", &pyarea, &pytiles)) {
    return NULL;
  }

  if (!PyArea_Check(pyarea)) {
    raiseException_returnNULL(PyExc_AttributeError, "combine_tiles requires a AreaCore instance as first argument");
  }
  if (!PySequence_Check(pytiles)) {
    raiseException_returnNULL(PyExc_AttributeError, "combine_tiles requires a list of cartesian products as second argument");
  }
  tiles = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (!tiles) {
    raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for tiles");
  }
  n = PySequence_Size(pytiles);
  for (i = 0; i < n; i++) {
    PyObject* v = PySequence_GetItem(pytiles, i);
    if (!PyCartesian_Check(v)) {
      Py_XDECREF(v);
      raiseException_gotoTag(done, PyExc_AttributeError, "Input should be a list of cartesian tiles");
    }
    if (!RaveObjectList_add(tiles, (RaveCoreObject*)((PyCartesian*)v)->cartesian)) {
      Py_XDECREF(v);
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to add item to list");
    }
    Py_XDECREF(v);
  }
  result = Transform_combine_tiles(self->transform, ((PyArea*)pyarea)->area, tiles);
  if (result == NULL) {
    raiseException_gotoTag(done, PyExc_AttributeError, "Failed to combine tiles");
  }
  pyresult = (PyObject*)PyCartesian_New(result);

done:
  RAVE_OBJECT_RELEASE(tiles);
  RAVE_OBJECT_RELEASE(result);
  return pyresult;
}

/**
 * All methods a transformator can have
 */
static struct PyMethodDef _pytransform_methods[] =
{
  {"method", NULL, METH_VARARGS},
  {"ppi", (PyCFunction) _pytransform_ppi, 1,
    "ppi(scan, cartesian)\n\n"
    "DEPRECATED. Use _composite instead.\n"
  },
  {"cappi", (PyCFunction) _pytransform_cappi, 1,
    "cappi(scan, cartesian, height)\n\n"
    "DEPRECATED. Use _composite instead.\n"
  },
  {"pcappi", (PyCFunction) _pytransform_pcappi, 1,
    "pcappi(scan, cartesian, height)\n\n"
    "DEPRECATED. Use _composite instead.\n"
  },
  {"ctoscan", (PyCFunction) _pytransform_ctoscan, 1,
    "ctoscan(cartesian, radardef, elangle, quantity) -> scan\n\n"
    "Creates a scan from a cartesian parameter with specified quantity. Uses radardef to get correct radar information and the elevation angle should be in radians\n"
    "cartesian - the cartesian object\n"
    "radardef  - the radar definition with information about location, geometry, ...\n"
    "elangle   - the elevation of the scan to be created\n"
    "quantity  - the parameter in the cartesian object"
  },
  {"ctop", (PyCFunction) _pytransform_ctop, 1,
    "ctop(cartesian, radardef, quantity) -> scan\n\n"
    "Creates a polar volume from a cartesian parameter with specified quantity. Uses radardef to get correct radar information and the elevation angles.\n"
    "cartesian - the cartesian object\n"
    "radardef  - the radar definition with information about location, geometry, ...\n"
    "quantity  - the parameter in the cartesian object"
  },
  {"fillGap", (PyCFunction) _pytransform_fillGap, 1,
    "fillGap(object) -> cartesian or cartesian parameter\n\n"
    "If a value is == UNDETECT and the surrounding 4 pixels == DATA, then the value set is the avg for the surrounding 4 pixels.\n"
    "If provided object is a cartesian parameter, only that parameter will be modified and if the provided object is a cartesian product, all parameters will be modified.\n\n"
    "object - either a cartesian object or a cartesian parameter\n\n"
    "If object is a cartesian, then result will be a cartesian and if a cartesian parameter is used as input, then the result will be a cartesian parameter"
  },
  {"combine_tiles", (PyCFunction) _pytransform_combine_tiles, 1,
    "combine_tiles(area, tiles) -> cartesian\n\n"
    "Combines a number of cartesian areas into the one specified by the area definition. This should not be confused with the\n"
    "cartesian composite generation. This function instead works like a area-combiner where the individual tiles will result\n"
    "in a full area.\n\n"
    "area - the area that should be created\n"
    "tiles - a list of cartesian objects that will be used to build the resulting cartesian object"
  },
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the transformator
 * @param[in] self - the transform
 */
static PyObject* _pytransform_getattro(PyTransform* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("method", name) == 0) {
    return PyInt_FromLong(Transform_getMethod(self->transform));
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the transformator
 */
static int _pytransform_setattro(PyTransform* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("method", name)==0) {
    if (PyInt_Check(val)) {
      if (!Transform_setMethod(self->transform, PyInt_AsLong(val))) {
        raiseException_gotoTag(done, PyExc_ValueError, "method must be in valid range");
      }
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "method must be a valid RaveTransformMethod");
    }
  }

  result = 0;
done:
  return result;
}
/*@} End of Transform */

/*@{ Documentation about the module */
PyDoc_STRVAR(_transform_type_doc,
    "Provides some useful functions when performing transformations.\n"
    "Usage:\n"
    " import _transform\n"
    " t = _transform.new()\n"
    " # One of provided functions, for example:"
    " a = t.fillGap(_raveio.open(\"cartesian.h5\")"
    );
/*@} End of Documentation about the module */


/*@{ Type definitions */
PyTypeObject PyTransform_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "TransformCore", /*tp_name*/
  sizeof(PyTransform), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pytransform_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pytransform_getattro, /*tp_getattro*/
  (setattrofunc)_pytransform_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _transform_type_doc,          /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pytransform_methods,         /*tp_methods*/
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
  {"new", (PyCFunction)_pytransform_new, 1,
    "new() -> new instance of the TransformCore object\n\n"
    "Creates a new instance of the TransformCore object"
  },
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_transform)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyTransform_API[PyTransform_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyTransform_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyTransform_Type);

  MOD_INIT_DEF(module, "_transform", _transform_type_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyTransform_API[PyTransform_Type_NUM] = (void*)&PyTransform_Type;
  PyTransform_API[PyTransform_GetNative_NUM] = (void *)PyTransform_GetNative;
  PyTransform_API[PyTransform_New_NUM] = (void*)PyTransform_New;

  c_api_object = PyCapsule_New(PyTransform_API, PyTransform_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_transform.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _transform.error");
    return MOD_INIT_ERROR;
  }

  import_pypolarvolume();
  import_pypolarscan();
  import_pycartesian();
  import_pycartesianparam();
  import_pyradardefinition();
  import_pyarea();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
