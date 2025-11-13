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
 * Python version of the Area API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYACQVAFEATUREMAP_MODULE    /**< to get correct part in pyarea.h */
#include "pyacqvafeaturemap.h"
#include <arrayobject.h>

#include "rave_alloc.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_acqvafeaturemap");

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

/*@{ AcqvaFeatureMap */
/**
 * Returns the native AcqvaFeatureMap_t instance.
 * @param[in] pyarea - the python area instance
 * @returns the native area instance.
 */
static AcqvaFeatureMap_t*
PyAcqvaFeatureMap_GetNative(PyAcqvaFeatureMap* pyfeaturemap)
{
  RAVE_ASSERT((pyfeaturemap != NULL), "pyfeaturemap == NULL");
  return RAVE_OBJECT_COPY(pyfeaturemap->featuremap);
}

/**
 * Creates a python area from a native area or will create an
 * initial native area if p is NULL.
 * @param[in] p - the native area (or NULL)
 * @returns the python area product.
 */
static PyAcqvaFeatureMap*
PyAcqvaFeatureMap_New(AcqvaFeatureMap_t* p)
{
  PyAcqvaFeatureMap* result = NULL;
  AcqvaFeatureMap_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&AcqvaFeatureMap_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for feature map.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for feature map.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyAcqvaFeatureMap, &PyAcqvaFeatureMap_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->featuremap = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->featuremap, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyAcqvaFeatureMap instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyAcqvaFeatureMap.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the feature map
 * @param[in] obj the object to deallocate.
 */
static void _pyacqvafeaturemap_dealloc(PyAcqvaFeatureMap* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->featuremap, obj);
  RAVE_OBJECT_RELEASE(obj->featuremap);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the feature map.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyacqvafeaturemap_new(PyObject* self, PyObject* args)
{
  PyAcqvaFeatureMap* result = PyAcqvaFeatureMap_New(NULL);
  return (PyObject*)result;
}

static PyObject* _pyacqvafeaturemap_load(PyObject* self, PyObject* args)
{
  AcqvaFeatureMap_t* featuremap = NULL;
  PyAcqvaFeatureMap* result = NULL;
  char* filename = NULL;
  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }
  featuremap = AcqvaFeatureMap_load(filename);
  if (featuremap != NULL) {
    result = PyAcqvaFeatureMap_New(featuremap);
    RAVE_OBJECT_RELEASE(featuremap);
  }
  if (result == NULL) {
    raiseException_returnNULL(PyExc_IOError, "Could not load feature map");
  }
  return (PyObject*)result;
}

static PyObject* _pyacqvafeaturemap_createField(PyAcqvaFeatureMap* self, PyObject* args)
{
  long nbins = 0, nrays = 0;
  int datatype = 0;
  double elangle = 0.0;
  PyAcqvaFeatureMapField* result = NULL;
  AcqvaFeatureMapField_t* acqvafield = NULL;

  if (!PyArg_ParseTuple(args, "(ll)id", &nbins, &nrays, &datatype, &elangle)) {
    return NULL;
  }
  acqvafield = AcqvaFeatureMap_createField(self->featuremap, nbins, nrays, datatype, elangle);
  if (acqvafield != NULL) {
    result = PyAcqvaFeatureMapField_New(acqvafield);
  }

  RAVE_OBJECT_RELEASE(acqvafield);
  return (PyObject*)result;
}

static PyObject* _pyacqvafeaturemap_findField(PyAcqvaFeatureMap* self, PyObject* args)
{
  long nbins = 0, nrays = 0;
  double elangle = 0.0;
  PyAcqvaFeatureMapField* result = NULL;
  AcqvaFeatureMapField_t* acqvafield = NULL;

  if (!PyArg_ParseTuple(args, "(ll)d", &nbins, &nrays, &elangle)) {
    return NULL;
  }
  acqvafield = AcqvaFeatureMap_findField(self->featuremap, nbins, nrays, elangle);
  if (acqvafield != NULL) {
    result = PyAcqvaFeatureMapField_New(acqvafield);
    RAVE_OBJECT_RELEASE(acqvafield);
    return (PyObject*)result;
  }
  Py_RETURN_NONE;
}

static PyObject* _pyacqvafeaturemap_getNumberOfElevations(PyAcqvaFeatureMap* self, PyObject* args)
{

  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyLong_FromLong(AcqvaFeatureMap_getNumberOfElevations(self->featuremap));
}

static PyObject* _pyacqvafeaturemap_createElevation(PyAcqvaFeatureMap* self, PyObject* args)
{
  AcqvaFeatureMapElevation_t* group;
  PyAcqvaFeatureMapElevation* result = NULL;

  double elangle = 0.0;
  if (!PyArg_ParseTuple(args, "d", &elangle)) {
    return NULL;
  }
  group = AcqvaFeatureMap_createElevation(self->featuremap, elangle);
  if (group != NULL) {
    result = PyAcqvaFeatureMapElevation_New(group);
  }
  RAVE_OBJECT_RELEASE(group);
  return (PyObject*)result;
}

static PyObject* _pyacqvafeaturemap_getElevation(PyAcqvaFeatureMap* self, PyObject* args)
{
  AcqvaFeatureMapElevation_t* group;
  PyAcqvaFeatureMapElevation* result = NULL;
  int index = 0;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }
  group = AcqvaFeatureMap_getElevation(self->featuremap, index);
  if (group != NULL) {
    result = PyAcqvaFeatureMapElevation_New(group);
  } else {
    PyErr_SetString(PyExc_IndexError, "Not found");
  }
  RAVE_OBJECT_RELEASE(group);

  return (PyObject*)result;
}

static PyObject* _pyacqvafeaturemap_removeElevation(PyAcqvaFeatureMap* self, PyObject* args)
{
  int index = 0;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }
  AcqvaFeatureMap_removeElevation(self->featuremap, index);
  Py_RETURN_NONE;
}

static PyObject* _pyacqvafeaturemap_findElevation(PyAcqvaFeatureMap* self, PyObject* args)
{
  AcqvaFeatureMapElevation_t* group=NULL;
  PyAcqvaFeatureMapElevation* result = NULL;
  double elangle = 0;
  if (!PyArg_ParseTuple(args, "d", &elangle)) {
    return NULL;
  }
  group = AcqvaFeatureMap_findElevation(self->featuremap, elangle);
  if (group != NULL) {
    result = PyAcqvaFeatureMapElevation_New(group);
    RAVE_OBJECT_RELEASE(group);
    return (PyObject*)result;
  }
  Py_RETURN_NONE;
}

static PyObject* _pyacqvafeaturemap_save(PyAcqvaFeatureMap* self, PyObject* args)
{
  char* filename = NULL;
  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }
  if (!AcqvaFeatureMap_save(self->featuremap, filename)) {
    raiseException_returnNULL(PyExc_IOError, "Could not save file");
  }
  Py_RETURN_NONE;
}

/**
 * All methods a area can have
 */
static struct PyMethodDef _pyacqvafeaturemap_methods[] =
{
  {"nod", NULL, METH_VARARGS},
  {"longitude", NULL, METH_VARARGS},
  {"latitude", NULL, METH_VARARGS},
  {"height", NULL, METH_VARARGS},
  {"startdate", NULL, METH_VARARGS},
  {"enddate", NULL, METH_VARARGS},
  {"createField", (PyCFunction) _pyacqvafeaturemap_createField, 1,
    "createField((nbins, nrays), datatype, elangle)\n\n"
      "Creates a field with specified nbins/nrays/datatype in the elevation group with elangle.\n"
      "If elevation group doesn't exit, the elevation group is also created."},
  {"findField", (PyCFunction) _pyacqvafeaturemap_findField, 1,
    "findField((nbins, nrays), elangle) -> field\n\n"
      "Locates a field with specified elangle and nbins/nrays.\n"},
  {"createElevation", (PyCFunction) _pyacqvafeaturemap_createElevation, 1,
    "createElevation(elangle)\n\n"
      "Creates an elevation group in the feature map if it doesn't already exist.\n"},
  {"getNumberOfElevations", (PyCFunction) _pyacqvafeaturemap_getNumberOfElevations, 1,
    "getNumberOfElevations()\n\n"
      "Returns number of elevations."},
  {"getElevation", (PyCFunction) _pyacqvafeaturemap_getElevation, 1,
    "getElevation(index) -> elevation group\n\n"
      "Returns elevation group at specified index."},
  {"removeElevation", (PyCFunction) _pyacqvafeaturemap_removeElevation, 1,
    "removeElevation(index)\n\n"
      "Removes the elevation group at specified index."},
  {"findElevation", (PyCFunction) _pyacqvafeaturemap_findElevation, 1,
    "findElevation(elangle)\n\n"
      "Locates an elevation with matching elangle."},
  {"save", (PyCFunction) _pyacqvafeaturemap_save, 1,
    "save(filename)\n\n"
      "Saves the feature map in filename."},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the rave field
 * @param[in] self - the rave field
 */
static PyObject* _pyacqvafeaturemap_getattro(PyAcqvaFeatureMap* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "nod") == 0) {
    if (AcqvaFeatureMap_getNod(self->featuremap) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(AcqvaFeatureMap_getNod(self->featuremap));
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("longitude", name) == 0) {
    return PyFloat_FromDouble(AcqvaFeatureMap_getLongitude(self->featuremap));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("latitude", name) == 0) {
    return PyFloat_FromDouble(AcqvaFeatureMap_getLatitude(self->featuremap));
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("height", name) == 0) {
    return PyFloat_FromDouble(AcqvaFeatureMap_getHeight(self->featuremap));
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "startdate") == 0) {
    if (AcqvaFeatureMap_getStartdate(self->featuremap) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(AcqvaFeatureMap_getStartdate(self->featuremap));
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "enddate") == 0) {
    if (AcqvaFeatureMap_getEnddate(self->featuremap) == NULL) {
      Py_RETURN_NONE;
    } else {
      return PyString_FromString(AcqvaFeatureMap_getEnddate(self->featuremap));
    }
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the attribute value
 */
static int _pyacqvafeaturemap_setattro(PyAcqvaFeatureMap *self, PyObject *name, PyObject *val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "nod") == 0) {
    if (PyString_Check(val)) {
      if (!AcqvaFeatureMap_setNod(self->featuremap, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_TypeError,"could not set nod");
      }
    } else if (val == Py_None) {
      AcqvaFeatureMap_setNod(self->featuremap, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "nod must be a string");
    }

  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("longitude", name) == 0) {
    if (PyFloat_Check(val)) {
      AcqvaFeatureMap_setLongitude(self->featuremap, PyFloat_AsDouble(val));
    } else if (PyInt_Check(val)) {
      AcqvaFeatureMap_setLongitude(self->featuremap, (double)PyInt_AsLong(val));
    } else if (PyLong_AsLong(val)) {
      AcqvaFeatureMap_setLongitude(self->featuremap, (double)PyLong_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "longitude must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("latitude", name) == 0) {
    if (PyFloat_Check(val)) {
      AcqvaFeatureMap_setLatitude(self->featuremap, PyFloat_AsDouble(val));
    } else if (PyInt_Check(val)) {
      AcqvaFeatureMap_setLatitude(self->featuremap, (double)PyInt_AsLong(val));
    } else if (PyLong_AsLong(val)) {
      AcqvaFeatureMap_setLatitude(self->featuremap, (double)PyLong_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "latitude must be a number");
    }
  } else if (PY_COMPARE_STRING_WITH_ATTRO_NAME("height", name) == 0) {
    if (PyFloat_Check(val)) {
      AcqvaFeatureMap_setHeight(self->featuremap, PyFloat_AsDouble(val));
    } else if (PyInt_Check(val)) {
      AcqvaFeatureMap_setHeight(self->featuremap, (double)PyInt_AsLong(val));
    } else if (PyLong_AsLong(val)) {
      AcqvaFeatureMap_setHeight(self->featuremap, (double)PyLong_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "height must be a number");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "startdate") == 0) {
    if (PyString_Check(val)) {
      if (!AcqvaFeatureMap_setStartdate(self->featuremap, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_TypeError,"could not set startdate");
      }
    } else if (val == Py_None) {
      AcqvaFeatureMap_setStartdate(self->featuremap, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "startdate must be a string");
    }
  } else if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "enddate") == 0) {
    if (PyString_Check(val)) {
      if (!AcqvaFeatureMap_setEnddate(self->featuremap, PyString_AsString(val))) {
        raiseException_gotoTag(done, PyExc_TypeError,"could not set enddate");
      }
    } else if (val == Py_None) {
      AcqvaFeatureMap_setEnddate(self->featuremap, NULL);
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "enddate must be a string");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError,
        PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }
  result = 0;
done:
  return result;
}
/*@} End of AcqvaFeatureMap */

/*@{ AcqvaFeatureMapElevation */
/**
 * Returns the native AcqvaFeatureMap_t instance.
 * @param[in] pyarea - the python area instance
 * @returns the native area instance.
 */
static AcqvaFeatureMapElevation_t*
PyAcqvaFeatureMapElevation_GetNative(PyAcqvaFeatureMapElevation* pyfeaturemap)
{
  RAVE_ASSERT((pyfeaturemap != NULL), "pyfeaturemap == NULL");
  return RAVE_OBJECT_COPY(pyfeaturemap->elevation);
}

/**
 * Creates a python area from a native area or will create an
 * initial native area if p is NULL.
 * @param[in] p - the native area (or NULL)
 * @returns the python area product.
 */
static PyAcqvaFeatureMapElevation*
PyAcqvaFeatureMapElevation_New(AcqvaFeatureMapElevation_t* p)
{
  PyAcqvaFeatureMapElevation* result = NULL;
  AcqvaFeatureMapElevation_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&AcqvaFeatureMapElevation_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for feature map elevation group.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for feature map elevation group.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyAcqvaFeatureMapElevation, &PyAcqvaFeatureMapElevation_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->elevation = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->elevation, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyAcqvaFeatureMapElevation instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyAcqvaFeatureMapElevation.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the feature map
 * @param[in] obj the object to deallocate.
 */
static void _pyacqvafeaturemapelevation_dealloc(PyAcqvaFeatureMapElevation* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->elevation, obj);
  RAVE_OBJECT_RELEASE(obj->elevation);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the feature map.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyacqvafeaturemapelevation_new(PyObject* self, PyObject* args)
{
  PyAcqvaFeatureMapElevation* result = PyAcqvaFeatureMapElevation_New(NULL);
  return (PyObject*)result;
}

static PyObject* _pyacqvafeaturemapelevation_add(PyAcqvaFeatureMapElevation* self, PyObject* args)
{
  PyObject* inptr = NULL;
  PyAcqvaFeatureMapField* field = NULL;

  if (!PyArg_ParseTuple(args, "O", &inptr)) {
    return NULL;
  }

  if (!PyAcqvaFeatureMapField_Check(inptr)) {
    raiseException_returnNULL(PyExc_TypeError,"Added object must be of type AcqvaFeatureMapFieldCore");
  }

  field = (PyAcqvaFeatureMapField*)inptr;

  if (AcqvaFeatureMapElevation_has(self->elevation, AcqvaFeatureMapField_getNbins(field->field), AcqvaFeatureMapField_getNrays(field->field))) {
    raiseException_returnNULL(PyExc_AttributeError, "Elevation group already consists of a field with those dimensions");
  }

  if (fabs(AcqvaFeatureMapElevation_getElangle(self->elevation) - AcqvaFeatureMapField_getElangle(field->field)) > 1e-4) {
    raiseException_returnNULL(PyExc_AttributeError, "Different elangle in elevation and field");
  }

  if (!AcqvaFeatureMapElevation_add(self->elevation, field->field)) {
    raiseException_returnNULL(PyExc_AttributeError, "Failed to add field to elevation group");
  }

  Py_RETURN_NONE;
}

static PyObject* _pyacqvafeaturemapelevation_size(PyAcqvaFeatureMapElevation* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }
  return PyLong_FromLong(AcqvaFeatureMapElevation_size(self->elevation));
}

static PyObject* _pyacqvafeaturemapelevation_get(PyAcqvaFeatureMapElevation* self, PyObject* args)
{
  int index = -1;
  AcqvaFeatureMapField_t* field = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }

  if (index < 0 || index >= AcqvaFeatureMapElevation_size(self->elevation)) {
    raiseException_returnNULL(PyExc_IndexError, "Index out of range");
  }

  if((field = AcqvaFeatureMapElevation_get(self->elevation, index)) == NULL) {
    raiseException_returnNULL(PyExc_IndexError, "Could not aquire field");
  }

  if (field != NULL) {
    result = (PyObject*)PyAcqvaFeatureMapField_New(field);
  }

  RAVE_OBJECT_RELEASE(field);

  return result;
}

static PyObject* _pyacqvafeaturemapelevation_remove(PyAcqvaFeatureMapElevation* self, PyObject* args)
{
  int index = -1;
  if (!PyArg_ParseTuple(args, "i", &index)) {
    return NULL;
  }
  AcqvaFeatureMapElevation_remove(self->elevation, index);
  Py_RETURN_NONE;
}

static PyObject* _pyacqvafeaturemapelevation_find(PyAcqvaFeatureMapElevation* self, PyObject* args)
{
  AcqvaFeatureMapField_t* field = NULL;
  long nbins = 0, nrays = 0;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "(ll)", &nbins, &nrays)) {
    return NULL;
  }

  field = AcqvaFeatureMapElevation_find(self->elevation, nbins, nrays);
  if (field != NULL) {
    result = (PyObject*)PyAcqvaFeatureMapField_New(field);
    RAVE_OBJECT_RELEASE(field);
  }

  if (result != NULL) {
    return result;
  }

  Py_RETURN_NONE;
}


/**
 * All methods a featuremap elevation can have
 */
static struct PyMethodDef _pyacqvafeaturemapelevation_methods[] =
{
  {"elangle", NULL, METH_VARARGS},
  {"add", (PyCFunction) _pyacqvafeaturemapelevation_add, 1,
       "add(field)\n\n"
       "Adds a field of type AcqvaFeatureMapFieldCore to the elevation group\n\n"},
  {"size", (PyCFunction) _pyacqvafeaturemapelevation_size, 1,
       "size()\n\n"
       "Returns number of fields in group\n\n"},
  {"get", (PyCFunction) _pyacqvafeaturemapelevation_get, 1,
       "get(index)\n\n"
       "Returns the field at specified index\n\n"},
  {"remove", (PyCFunction) _pyacqvafeaturemapelevation_remove, 1,
       "remove(index)\n\n"
       "Removes the field at specified index\n\n"},
  {"find", (PyCFunction) _pyacqvafeaturemapelevation_find, 1,
       "find(nbins, nrays)\n\n"
       "Returns the field with specified number of bins / rays\n\n"},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the rave field
 * @param[in] self - the rave field
 */
static PyObject* _pyacqvafeaturemapelevation_getattro(PyAcqvaFeatureMapElevation* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "elangle") == 0) {
    return PyFloat_FromDouble(AcqvaFeatureMapElevation_getElangle(self->elevation));
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the attribute value
 */
static int _pyacqvafeaturemapelevation_setattro(PyAcqvaFeatureMapElevation *self, PyObject *name, PyObject *val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "elangle") == 0) {
    if (PyFloat_Check(val)) {
      AcqvaFeatureMapElevation_setElangle(self->elevation, PyFloat_AsDouble(val));
    } else if (PyInt_Check(val)) {
      AcqvaFeatureMapElevation_setElangle(self->elevation, (double)PyInt_AsLong(val));
    } else if (PyLong_AsLong(val)) {
      AcqvaFeatureMapElevation_setElangle(self->elevation, (double)PyLong_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "elangle must be a number");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError,
        PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }
  result = 0;
done:
  return result;
}
/*@} End of AcqvaFeatureMapElevation */

/*@{ AcqvaFeatureMapField */
/**
 * Returns the native AcqvaFeatureMapField_t instance.
 * @param[in] pyarea - the python instance
 * @returns the native area instance.
 */
static AcqvaFeatureMapField_t*
PyAcqvaFeatureMapField_GetNative(PyAcqvaFeatureMapField* pyfeaturemap)
{
  RAVE_ASSERT((pyfeaturemap != NULL), "pyfeaturemap == NULL");
  return RAVE_OBJECT_COPY(pyfeaturemap->field);
}

/**
 * Creates a python area from a native area or will create an
 * initial native area if p is NULL.
 * @param[in] p - the native area (or NULL)
 * @returns the python area product.
 */
static PyAcqvaFeatureMapField*
PyAcqvaFeatureMapField_New(AcqvaFeatureMapField_t* p)
{
  PyAcqvaFeatureMapField* result = NULL;
  AcqvaFeatureMapField_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&AcqvaFeatureMapField_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for feature map field.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for feature map field.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyAcqvaFeatureMapField, &PyAcqvaFeatureMapField_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->field = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->field, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyAcqvaFeatureMapField instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyAcqvaFeatureMapField.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the feature map
 * @param[in] obj the object to deallocate.
 */
static void _pyacqvafeaturemapfield_dealloc(PyAcqvaFeatureMapField* obj)
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
 * Creates a new instance of the feature map field.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyacqvafeaturemapfield_new(PyObject* self, PyObject* args)
{
  long nbins = 0, nrays = 0;
  int datatype = 0;
  double elangle = 0.0;
  PyAcqvaFeatureMapField* result = NULL;

  if (!PyArg_ParseTuple(args, "|(ll)id", &nbins, &nrays, &datatype, &elangle)) {
    return NULL;
  }

  if (nbins != 0 && nrays != 0) {
    AcqvaFeatureMapField_t* field = AcqvaFeatureMapField_createField(nbins, nrays, datatype, elangle);
    if (field != NULL) {
      result = PyAcqvaFeatureMapField_New(field);
      RAVE_OBJECT_RELEASE(field);
    } else {
      raiseException_returnNULL(PyExc_MemoryError, "Could not create feature map field");
    }
  } else {
    result = PyAcqvaFeatureMapField_New(NULL);
  }
  return (PyObject*)result;
}

static PyObject* _pyacqvafeaturemapfield_createData(PyAcqvaFeatureMapField* self, PyObject* args)
{
  long nbins = 0, nrays = 0;
  int datatype = 0;
  if (!PyArg_ParseTuple(args, "(ll)i", &nbins, &nrays, &datatype)) {
    return NULL;
  }
  if (!AcqvaFeatureMapField_createData(self->field, nbins, nrays, datatype)) {
    raiseException_returnNULL(PyExc_AttributeError, "Could not create data field");
  }
  return (PyObject*)PyAcqvaFeatureMapField_New(self->field);
}

static PyObject* _pyacqvafeaturemapfield_setData(PyAcqvaFeatureMapField* self, PyObject* args)
{
  PyObject* inarray = NULL;
  PyArrayObject* arraydata = NULL;
  RaveDataType datatype = RaveDataType_UNDEFINED;
  long nbins = 0;
  long nrays = 0;
  unsigned char* data = NULL;

  if (!PyArg_ParseTuple(args, "O", &inarray)) {
    return NULL;
  }

  if (!PyArray_Check(inarray)) {
    raiseException_returnNULL(PyExc_TypeError, "Data must be of arrayobject type")
  }

  arraydata = (PyArrayObject*)inarray;

  if (PyArray_NDIM(arraydata) != 2) {
    raiseException_returnNULL(PyExc_ValueError, "A feature map field must be of rank 2");
  }

  datatype = translate_pyarraytype_to_ravetype(PyArray_TYPE(arraydata));

  if (PyArray_ITEMSIZE(arraydata) != get_ravetype_size(datatype)) {
    raiseException_returnNULL(PyExc_TypeError, "numpy and rave does not have same data sizes");
  }

  nbins  = PyArray_DIM(arraydata, 1);
  nrays  = PyArray_DIM(arraydata, 0);
  data   = PyArray_DATA(arraydata);

  if (!AcqvaFeatureMapField_setData(self->field, nbins, nrays, data, datatype)) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not allocate memory");
  }

  Py_RETURN_NONE;
}

static PyObject* _pyacqvafeaturemapfield_getData(PyAcqvaFeatureMapField* self, PyObject* args)
{
  long nbins = 0, nrays = 0;
  RaveDataType type = RaveDataType_UNDEFINED;
  PyObject* result = NULL;
  npy_intp dims[2] = {0,0};
  int arrtype = 0;
  void* data = NULL;

  nbins = AcqvaFeatureMapField_getNbins(self->field);
  nrays = AcqvaFeatureMapField_getNrays(self->field);
  type = AcqvaFeatureMapField_getDatatype(self->field);
  data = AcqvaFeatureMapField_getData(self->field);

  dims[0] = (npy_intp)nrays;
  dims[1] = (npy_intp)nbins;
  arrtype = translate_ravetype_to_pyarraytype(type);

  if (data == NULL) {
    raiseException_returnNULL(PyExc_IOError, "acqva feature map field does not have any data");
  }

  if (arrtype == NPY_NOTYPE) {
    raiseException_returnNULL(PyExc_IOError, "Could not translate data type");
  }
  result = PyArray_SimpleNew(2, dims, arrtype);

  if (result == NULL) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not create resulting array");
  }
  if (result != NULL) {
    int nbytes = nbins*nrays*(PyArray_ITEMSIZE((PyArrayObject*)result));
    memcpy(PyArray_DATA((PyArrayObject*)result), data, nbytes);
  }

  return result;
}

/**
 * fills the array with value
 * @param[in] self this instance.
 * @param[in] args - v
 */
static PyObject* _pyacqvafeaturemapfield_fill(PyAcqvaFeatureMapField* self, PyObject* args)
{
  double v = 0.0L;
  if (!PyArg_ParseTuple(args, "d", &v)) {
    return NULL;
  }

  if(!AcqvaFeatureMapField_fill(self->field, v)) {
    raiseException_returnNULL(PyExc_ValueError, "Could not fill cells");
  }

  Py_RETURN_NONE;
}

/**
 * Returns the value at the specified ray and bin index.
 * @param[in] self - this instance
 * @param[in] args - bin index, ray index.
 * @returns a tuple of value type and value
 */
static PyObject* _pyacqvafeaturemapfield_getValue(PyAcqvaFeatureMapField* self, PyObject* args)
{
  double value = 0.0L;
  long ray = 0, bin = 0;
  if (!PyArg_ParseTuple(args, "(ll)", &bin, &ray)) {
    return NULL;
  }

  if (!AcqvaFeatureMapField_getValue(self->field, bin, ray, &value)) {
    raiseException_returnNULL(PyExc_IndexError, "Could not get value");
  }
  return PyFloat_FromDouble(value);
}

/**
 * sets the value at the specified position
 * @param[in] self this instance.
 * @param[in] args - tuple (bin, ray) and v
 * @return 0 on failure, otherwise 1
 */
static PyObject* _pyacqvafeaturemapfield_setValue(PyAcqvaFeatureMapField* self, PyObject* args)
{
  long bin = 0, ray = 0;
  double v = 0.0L;
  if (!PyArg_ParseTuple(args, "(ll)d", &bin, &ray, &v)) {
    return NULL;
  }

  if(!AcqvaFeatureMapField_setValue(self->field, bin, ray, v)) {
    raiseException_returnNULL(PyExc_ValueError, "Could not set value");
  }

  Py_RETURN_NONE;
}

/**
 * All methods a featuremap elevation can have
 */
static struct PyMethodDef _pyacqvafeaturemapfield_methods[] =
{
  {"elangle", NULL, METH_VARARGS},
  {"nbins", NULL, METH_VARARGS},
  {"nrays", NULL, METH_VARARGS},
  {"datatype", NULL, METH_VARARGS},
  {"createData", (PyCFunction) _pyacqvafeaturemapfield_createData, 1,
       "createData((nbins, nrays), datatype)\n\n"
       "Creates the data array in the field\n\n"},
  {"setData", (PyCFunction) _pyacqvafeaturemapfield_setData, 1,
       "setData(numpyarray)\n\n"
       "Sets the numpyarray\n\n"},
  {"getData", (PyCFunction) _pyacqvafeaturemapfield_getData, 1,
       "getData()\n\n"
       "Returns the numpy array\n\n"},
  {"fill", (PyCFunction) _pyacqvafeaturemapfield_fill, 1,
    "fill(value)\n\n"
    "Fills the array with value \n\n"
    "value - the value that should be filled."
  },
  {"getValue", (PyCFunction) _pyacqvafeaturemapfield_getValue, 1,
    "getValue((bin,ray)) -> the value at the specified bin and ray index.\n\n"
    "Returns the value at the specified bin and ray index. \n\n"
    "bin - bin index\n"
    "ray - ray index"
  },
  {"setValue", (PyCFunction) _pyacqvafeaturemapfield_setValue, 1,
    "setValue((bin,ray),value) -> 1 on success otherwise 0\n\n"
    "Sets the value at the specified position. \n\n"
    "bin   - bin index\n"
    "ray   - ray index\n"
    "value - the value that should be set at specified position."
  },
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the rave field
 * @param[in] self - the rave field
 */
static PyObject* _pyacqvafeaturemapfield_getattro(PyAcqvaFeatureMapField* self, PyObject* name)
{
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "elangle") == 0) {
    return PyFloat_FromDouble(AcqvaFeatureMapField_getElangle(self->field));
  } else  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "nbins") == 0) {
    return PyLong_FromLong(AcqvaFeatureMapField_getNbins(self->field));
  } else  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "nrays") == 0) {
    return PyLong_FromLong(AcqvaFeatureMapField_getNrays(self->field));
  } else  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "datatype") == 0) {
    return PyLong_FromLong(AcqvaFeatureMapField_getDatatype(self->field));
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the attribute value
 */
static int _pyacqvafeaturemapfield_setattro(PyAcqvaFeatureMapField *self, PyObject *name, PyObject *val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_ATTRO_NAME_WITH_STRING(name, "elangle") == 0) {
    if (PyFloat_Check(val)) {
      AcqvaFeatureMapField_setElangle(self->field, PyFloat_AsDouble(val));
    } else if (PyInt_Check(val)) {
      AcqvaFeatureMapField_setElangle(self->field, (double)PyInt_AsLong(val));
    } else if (PyLong_AsLong(val)) {
      AcqvaFeatureMapField_setElangle(self->field, (double)PyLong_AsLong(val));
    } else {
      raiseException_gotoTag(done, PyExc_TypeError, "elangle must be a number");
    }
  } else {
    raiseException_gotoTag(done, PyExc_AttributeError,
        PY_RAVE_ATTRO_NAME_TO_STRING(name));
  }
  result = 0;
done:
  return result;
}
/*@} End of AcqvaFeatureMapElevation */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyacqvafeaturemap_doc,
    "This class provides functionality for defining an area used in for example cartesian products.\n"
    "\n"
    );
/*@} End of Documentation about the type */


/*@{ Type definitions */
PyTypeObject PyAcqvaFeatureMap_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "AcqvaFeatureMapCore", /*tp_name*/
  sizeof(PyAcqvaFeatureMap), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyacqvafeaturemap_dealloc,  /*tp_dealloc*/
  0,                            /*tp_print*/
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
  (getattrofunc)_pyacqvafeaturemap_getattro, /*tp_getattro*/
  (setattrofunc)_pyacqvafeaturemap_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyacqvafeaturemap_doc,                  /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyacqvafeaturemap_methods,              /*tp_methods*/
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

PyTypeObject PyAcqvaFeatureMapElevation_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "AcqvaFeatureMapElevationCore", /*tp_name*/
  sizeof(PyAcqvaFeatureMapElevation), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyacqvafeaturemapelevation_dealloc,  /*tp_dealloc*/
  0,                            /*tp_print*/
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
  (getattrofunc)_pyacqvafeaturemapelevation_getattro, /*tp_getattro*/
  (setattrofunc)_pyacqvafeaturemapelevation_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyacqvafeaturemap_doc,                  /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyacqvafeaturemapelevation_methods,              /*tp_methods*/
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

PyTypeObject PyAcqvaFeatureMapField_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "AcqvaFeatureMapFieldCore", /*tp_name*/
  sizeof(PyAcqvaFeatureMapField), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyacqvafeaturemapfield_dealloc,  /*tp_dealloc*/
  0,                            /*tp_print*/
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
  (getattrofunc)_pyacqvafeaturemapfield_getattro, /*tp_getattro*/
  (setattrofunc)_pyacqvafeaturemapfield_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyacqvafeaturemap_doc,                  /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyacqvafeaturemapfield_methods,              /*tp_methods*/
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
  {"map", (PyCFunction)_pyacqvafeaturemap_new, 1,
      "map(|(nbins,nrays),type) -> new instance of the AcqvaFeaturemapCore object\n\n"
      "Creates a new instance of the AcqvaFeaturemapCore object"},
  {"elevation", (PyCFunction)_pyacqvafeaturemapelevation_new, 1,
      "elevation() -> new instance of the AcqvaFeaturemapElevationCore object\n\n"
      "Creates a new instance of the AcqvaFeaturemapElevationCore object"},
  {"field", (PyCFunction)_pyacqvafeaturemapfield_new, 1,
      "field() -> new instance of the AcqvaFeaturemapFieldCore object\n\n"
      "Creates a new instance of the AcqvaFeaturemapFieldCore object"},
  {"load", (PyCFunction)_pyacqvafeaturemap_load, 1,
      "load(filename) -> loads the feature map with specified filename\n\n"
      "The loaded feature map"},

  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_acqvafeaturemap)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyAcqvaFeatureMap_API[PyAcqvaFeatureMap_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyAcqvaFeatureMap_Type, &PyType_Type);
  MOD_INIT_VERIFY_TYPE_READY(&PyAcqvaFeatureMap_Type);

  MOD_INIT_SETUP_TYPE(PyAcqvaFeatureMapElevation_Type, &PyType_Type);
  MOD_INIT_VERIFY_TYPE_READY(&PyAcqvaFeatureMapElevation_Type);

  MOD_INIT_SETUP_TYPE(PyAcqvaFeatureMapField_Type, &PyType_Type);
  MOD_INIT_VERIFY_TYPE_READY(&PyAcqvaFeatureMapField_Type);

  MOD_INIT_DEF(module, "_acqvafeaturemap", _pyacqvafeaturemap_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyAcqvaFeatureMap_API[PyAcqvaFeatureMap_Type_NUM] = (void*)&PyAcqvaFeatureMap_Type;
  PyAcqvaFeatureMap_API[PyAcqvaFeatureMap_GetNative_NUM] = (void *)PyAcqvaFeatureMap_GetNative;
  PyAcqvaFeatureMap_API[PyAcqvaFeatureMap_New_NUM] = (void*)PyAcqvaFeatureMap_New;
  PyAcqvaFeatureMap_API[PyAcqvaFeatureMapElevation_Type_NUM] = (void*)&PyAcqvaFeatureMapElevation_Type;
  PyAcqvaFeatureMap_API[PyAcqvaFeatureMapElevation_GetNative_NUM] = (void*)PyAcqvaFeatureMapElevation_GetNative;
  PyAcqvaFeatureMap_API[PyAcqvaFeatureMapElevation_New_NUM] = (void*)PyAcqvaFeatureMapElevation_New;
  PyAcqvaFeatureMap_API[PyAcqvaFeatureMapField_Type_NUM] = (void*)&PyAcqvaFeatureMapField_Type;
  PyAcqvaFeatureMap_API[PyAcqvaFeatureMapField_GetNative_NUM] = (void*)PyAcqvaFeatureMapField_GetNative;
  PyAcqvaFeatureMap_API[PyAcqvaFeatureMapField_New_NUM] = (void*)PyAcqvaFeatureMapField_New;

  c_api_object = PyCapsule_New(PyAcqvaFeatureMap_API, PyAcqvaFeatureMap_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_acqvafeaturemap.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _acqvafeaturemap.error");
    return MOD_INIT_ERROR;
  }
  import_array(); /*To make sure I get access to Numeric*/

  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
