/* --------------------------------------------------------------------
Copyright (C) 2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Python version of the Area registry
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-12-15
 */
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYODIMSOURCES_MODULE    /**< to get correct part in pyodimsources.h */
#include "pyodimsources.h"
#include "rave_alloc.h"
#include "pyodimsource.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_odimsources");

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

/*@{ OdimSources */
/**
 * Returns the native OdimSources_t instance.
 * @param[in] pysources - the python odim sources instance
 * @returns the native odim sources instance.
 */
static OdimSources_t*
PyOdimSources_GetNative(PyOdimSources* pysources)
{
  RAVE_ASSERT((pysources != NULL), "pysources == NULL");
  return RAVE_OBJECT_COPY(pysources->sources);
}

/**
 * Creates a python registry from a native odim sources or will create an
 * initial native odim sources if p is NULL.
 * @param[in] p - the native odim sources (or NULL)
 * @returns the python odim sources.
 */
static PyOdimSources*
PyOdimSources_New(OdimSources_t* p)
{
  PyOdimSources* result = NULL;
  OdimSources_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&OdimSources_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for odim sources.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for odim sources.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyOdimSources, &PyOdimSources_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->sources = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->sources, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyOdimSources instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for PyOdimSources.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Opens a file that is supported by odim sources
 * @param[in] filename - the odim sources file to load
 * @return the py odim sources on success.
 */
static PyOdimSources*
PyOdimSources_Load(const char* filename)
{
  OdimSources_t* sources = NULL;
  PyOdimSources* result = NULL;

  if (filename == NULL) {
    raiseException_returnNULL(PyExc_ValueError, "providing a filename that is NULL");
  }

  sources = OdimSources_load(filename);
  if (sources == NULL) {
    raiseException_gotoTag(done, PyExc_IOError, "Failed to open file");
  }
  result = PyOdimSources_New(sources);

done:
  RAVE_OBJECT_RELEASE(sources);
  return result;
}

/**
 * Deallocates the registry
 * @param[in] obj the object to deallocate.
 */
static void _pyodimsources_dealloc(PyOdimSources* obj)
{
  /*Nothing yet*/
  if (obj == NULL) {
    return;
  }
  PYRAVE_DEBUG_OBJECT_DESTROYED;
  RAVE_OBJECT_UNBIND(obj->sources, obj);
  RAVE_OBJECT_RELEASE(obj->sources);
  PyObject_Del(obj);
}

/**
 * Creates a new instance of the registry.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pyodimsources_new(PyObject* self, PyObject* args)
{
  PyOdimSources* result = PyOdimSources_New(NULL);
  return (PyObject*)result;
}

/**
 * Adds a source to the registry
 * @param[in] self - self
 * @param[in] args - the odim source object
 * @returns None on success or NULL on failure
 */
static PyObject* _pyodimsources_add(PyOdimSources* self, PyObject* args)
{
  PyObject* inptr = NULL;
  PyOdimSource* source = NULL;

  if (!PyArg_ParseTuple(args, "O", &inptr)) {
    return NULL;
  }

  if (!PyOdimSource_Check(inptr)) {
    raiseException_returnNULL(PyExc_TypeError,"Added object must be of type OdimSourceCore");
  }

  source = (PyOdimSource*)inptr;

  if (!OdimSources_add(self->sources, source->source)) {
    raiseException_returnNULL(PyExc_MemoryError, "Failed to add source to registry");
  }

  Py_RETURN_NONE;
}

/**
 * Returns the number of projections in this registry
 * @parma[in] self - this instance
 * @param[in] args - NA
 * @returns the number of projections
 */
static PyObject* _pyodimsources_size(PyOdimSources* self, PyObject* args)
{
  return PyLong_FromLong(OdimSources_size(self->sources));
}

/**
 * Returns the NODS registered in the registry
 * @parma[in] self - this instance
 * @param[in] args - NA
 * @returns the list of NOD ids
 */
static PyObject* _pyodimsources_nods(PyOdimSources* self, PyObject* args)
{
  RaveList_t* list = NULL;
  PyObject* result = NULL;
  int i = 0, n = 0;
  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }

  list = OdimSources_nods(self->sources);
  if (list == NULL) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not get NOD ids");
  }

  n = RaveList_size(list);
  result = PyList_New(0);
  for (i = 0; result != NULL && i < n; i++) {
    char* name = RaveList_get(list, i);
    if (name != NULL) {
      PyObject* pynamestr = PyString_FromString(name);
      if (pynamestr == NULL) {
        goto fail;
      }
      if (PyList_Append(result, pynamestr) != 0) {
        Py_DECREF(pynamestr);
        goto fail;
      }
      Py_DECREF(pynamestr);
    }
  }
  RaveList_freeAndDestroy(&list);
  return result;
fail:
  RaveList_freeAndDestroy(&list);
  Py_XDECREF(result);
  return result;
}

/**
 * Returns the odim source for the specified NOD
 * @parma[in] self - this instance
 * @param[in] args - the NOD name
 * @returns the odim source
 */
static PyObject* _pyodimsources_get(PyOdimSources* self, PyObject* args)
{
  OdimSource_t* source = NULL;
  PyObject* result = NULL;
  char* nod = NULL;

  if (!PyArg_ParseTuple(args, "s", &nod)) {
    return NULL;
  }

  source = OdimSources_get(self->sources, nod);
  if (source != NULL) {
    result = (PyObject*)PyOdimSource_New(source);
  } else {
    raiseException_returnNULL(PyExc_KeyError, "Could not identify NOD");
  }

  RAVE_OBJECT_RELEASE(source);

  return result;
}

/**
 * Returns the odim source for the specified WMO identifier
 * @parma[in] self - this instance
 * @param[in] args - the WMO identifier
 * @returns the odim source
 */
static PyObject* _pyodimsources_get_wmo(PyOdimSources* self, PyObject* args)
{
  OdimSource_t* source = NULL;
  PyObject* result = NULL;
  char* wmo = NULL;

  if (!PyArg_ParseTuple(args, "s", &wmo)) {
    return NULL;
  }

  source = OdimSources_get_wmo(self->sources, wmo);
  if (source != NULL) {
    result = (PyObject*)PyOdimSource_New(source);
  } else {
    raiseException_returnNULL(PyExc_KeyError, "Could not identify WMO");
  }

  RAVE_OBJECT_RELEASE(source);

  return result;
}

/**
 * Returns the odim source for the specified WIGOS
 * @parma[in] self - this instance
 * @param[in] args - the WIGOS identifier
 * @returns the odim source
 */
static PyObject* _pyodimsources_get_wigos(PyOdimSources* self, PyObject* args)
{
  OdimSource_t* source = NULL;
  PyObject* result = NULL;
  char* wigos = NULL;

  if (!PyArg_ParseTuple(args, "s", &wigos)) {
    return NULL;
  }

  source = OdimSources_get_wigos(self->sources, wigos);
  if (source != NULL) {
    result = (PyObject*)PyOdimSource_New(source);
  } else {
    raiseException_returnNULL(PyExc_KeyError, "Could not identify WIGOS");
  }

  RAVE_OBJECT_RELEASE(source);

  return result;
}

/**
 * Returns the odim source for the specified RAD
 * @parma[in] self - this instance
 * @param[in] args - the RAD identifier
 * @returns the odim source
 */
static PyObject* _pyodimsources_get_rad(PyOdimSources* self, PyObject* args)
{
  OdimSource_t* source = NULL;
  PyObject* result = NULL;
  char* rad = NULL;

  if (!PyArg_ParseTuple(args, "s", &rad)) {
    return NULL;
  }

  source = OdimSources_get_rad(self->sources, rad);
  if (source != NULL) {
    result = (PyObject*)PyOdimSource_New(source);
  } else {
    raiseException_returnNULL(PyExc_KeyError, "Could not identify RAD");
  }

  RAVE_OBJECT_RELEASE(source);

  return result;
}

/**
 * Returns the odim source for the specified PLC
 * @parma[in] self - this instance
 * @param[in] args - the PLC identifier
 * @returns the odim source
 */
static PyObject* _pyodimsources_get_plc(PyOdimSources* self, PyObject* args)
{
  OdimSource_t* source = NULL;
  PyObject* result = NULL;
  char* plc = NULL;

  if (!PyArg_ParseTuple(args, "s", &plc)) {
    return NULL;
  }

  source = OdimSources_get_plc(self->sources, plc);
  if (source != NULL) {
    result = (PyObject*)PyOdimSource_New(source);
  } else {
    raiseException_returnNULL(PyExc_KeyError, "Could not identify PLC");
  }

  RAVE_OBJECT_RELEASE(source);

  return result;
}

/**
 * Returns the odim source that could be identified for the source string
 * @parma[in] self - this instance
 * @param[in] args - the ODIM source string
 * @returns the odim source or None if not found
 */
static PyObject* _pyodimsources_identify(PyOdimSources* self, PyObject* args)
{
  OdimSource_t* source = NULL;
  PyObject* result = NULL;
  char* sourcestr = NULL;

  if (!PyArg_ParseTuple(args, "s", &sourcestr)) {
    return NULL;
  }

  source = OdimSources_identify(self->sources, sourcestr);
  if (source != NULL) {
    result = (PyObject*)PyOdimSource_New(source);
  } 

  RAVE_OBJECT_RELEASE(source);
  if (result != NULL) {
    return result;
  }
  Py_RETURN_NONE;
}

/**
 * Loads a sources registry from an xml file
 * @param[in] self - this instance
 * @param[in] args - a string pointing at the odim sources xml file
 * @return the read registry or NULL on failure
 */
static PyObject* _pyodimsources_load(PyObject* self, PyObject* args)
{
  PyOdimSources* result = NULL;

  char* filename = NULL;
  if (!PyArg_ParseTuple(args, "s", &filename)) {
    return NULL;
  }
  result = PyOdimSources_Load(filename);
  return (PyObject*)result;
}

/**
 * All methods a registry can have
 */
static struct PyMethodDef _pyodimsources_methods[] =
{
  {"add", (PyCFunction) _pyodimsources_add, 1,
       "add(source)\n\n"
       "Adds a source of type OdimSourceCore to the registry\n\n"
       "source  - The source definition of type OdimSourceCore."},
  {"size", (PyCFunction) _pyodimsources_size, 1,
       "size() -> integer with number of sources\n\n"
       "Returns the number of sources."},
  {"nods", (PyCFunction) _pyodimsources_nods, 1,
       "nods() -> list of registered nods.\n\n"},
  {"get", (PyCFunction) _pyodimsources_get, 1,
        "get(nod) -> OdimSourceCore\n\n"
        "Returns the odim source for the specified nod.\n\n"
        "nod - the NOD of the requested source."},
  {"get_wmo", (PyCFunction) _pyodimsources_get_wmo, 1,
        "get_wmo(wmo) -> OdimSourceCore\n\n"
        "Returns the odim source for the specified WMO identifier.\n\n"
        "wmo - the WMO identifier of the requested source."},
  {"get_wigos", (PyCFunction) _pyodimsources_get_wigos, 1,
        "get_wigos(wigos) -> OdimSourceCore\n\n"
        "Returns the odim source for the specified WIGOS identifier.\n\n"
        "wigos - the WIGOS identifier of the requested source."},
  {"get_rad", (PyCFunction) _pyodimsources_get_rad, 1,
        "get_rad(rad) -> OdimSourceCore\n\n"
        "Returns the odim source for the specified RAD identifier.\n\n"
        "rad - the RAD identifier of the requested source."},
  {"get_plc", (PyCFunction) _pyodimsources_get_plc, 1,
        "get_plc(plc) -> OdimSourceCore\n\n"
        "Returns the odim source for the specified PLC identifier.\n\n"
        "plc - the PLC identifier of the requested source."},
  {"identify", (PyCFunction) _pyodimsources_identify, 1,
        "identify(sourcestr) -> OdimSourceCore\n\n"
        "Tries to identify a source from the source string (according to ODIM format)\n\n"
        "The order that will be used to identify the source is NOD,WIGOS,WMO,RAD and PLC. If WMO:00000, then it will be ignored.\n\n"
        "sourcestr - the ODIM source string in format, [NOD:xxxxx,][WMO:....,][WIGOS:...]..."},
  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the registry
 * @param[in] self - the registry
 */
static PyObject* _pyodimsources_getattro(PyOdimSources* self, PyObject* name)
{
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Sets the specified attribute in the registry
 */
static int _pyodimsources_setattro(PyOdimSources* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  result = 0;
done:
  return result;
}
/*@} End of AreaRegistry */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pyodimsources_type_doc,
    "The odim sources registry provides the user with odim sources when working with the data.\n"
    "\n"
    "A simple example on how this odim sources registry can be used is:\n"
    " import _odimsources\n"
    " reg = _odimsources.load(\"odim_sources.xml\"))\n"
    " source = reg.get(\"sekkr\")"
    );
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyOdimSources_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "OdimSourcesCore", /*tp_name*/
  sizeof(PyOdimSources), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pyodimsources_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pyodimsources_getattro, /*tp_getattro*/
  (setattrofunc)_pyodimsources_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pyodimsources_type_doc,     /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pyodimsources_methods,      /*tp_methods*/
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
  {"new", (PyCFunction)_pyodimsources_new, 1,
      "new() -> new instance of the OdimSourcesCore object\n\n"
      "Creates a new instance of the OdimSourcesCore object"},
  {"load", (PyCFunction)_pyodimsources_load, 1,
      "load(filename) -> odim sources registry\n\n"
      "Loads a odim sources xml file. \n\n"
      "filename     - the path to the file containing the area registry xml definition\n"
      "The format of the area registry file should be in the format:\n"
      "<?xml version='1.0' encoding='iso-8859-1'?>\n"
      "<radar-db author=\"\">\n"
      "  <se CCCC=\"ESWI\" org=\"82\">\n"
      "    <sekrn plc=\"Kiruna\" rad=\"SE40\" wmo=\"02032\" wigos=\"0-20000-0-2032\" />\n"
      "    ...\n"
      "  </se>\n"
      "  ....\n"
      "</radar-db>"
  },
  {NULL,NULL} /*Sentinel*/
};

/*@{ Documentation about the module */
PyDoc_STRVAR(_pyodimsources_module_doc,
    "This class provides functionality for querying an odim sources registry.\n"
    "\n"
    "See the load function for information about format of the xml file.\n"
    "\n"
    "Usage:\n"
    " import _odimsources\n"
    " reg = _odimsources.load(\"/tmp/odim_sources.xml\")\n"
    " area = reg.get(\"sekkr\")\n"
    );
/*@} End of Documentation about the module */


MOD_INIT(_odimsources)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyOdimSources_API[PyOdimSources_API_pointers];
  PyObject *c_api_object = NULL;
  MOD_INIT_SETUP_TYPE(PyOdimSources_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyOdimSources_Type);

  MOD_INIT_DEF(module, "_odimsources", _pyodimsources_module_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyOdimSources_API[PyOdimSources_Type_NUM] = (void*)&PyOdimSources_Type;
  PyOdimSources_API[PyOdimSources_GetNative_NUM] = (void *)PyOdimSources_GetNative;
  PyOdimSources_API[PyOdimSources_New_NUM] = (void*)PyOdimSources_New;
  PyOdimSources_API[PyOdimSources_Load_NUM] = (void*)PyOdimSources_Load;

  c_api_object = PyCapsule_New(PyOdimSources_API, PyOdimSources_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);

  ErrorObject = PyErr_NewException("_odimsources.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _odimsources.error");
    return MOD_INIT_ERROR;
  }

  import_odimsource();
  PYRAVE_DEBUG_INITIALIZE;
  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
