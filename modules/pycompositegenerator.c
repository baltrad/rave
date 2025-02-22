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
 * Python version of the Composite generator API.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024"-12-05
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "compositefactorymanager.h"
#include "pyravecompat.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pyrave_debug.h"

#define PYCOMPOSITEGENERATOR_MODULE        /**< to get correct part of pycompositegenerator.h */
#include "pyraveproperties.h"
#include "pycompositegenerator.h"
#include "pycompositegeneratorfactory.h"
#include "pycompositefactorymanager.h"
#include "pycompositearguments.h"
#include "pycompositefilter.h"
#include "pypolarvolume.h"
#include "pypolarscan.h"
#include "pycartesian.h"
#include "pyarea.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include "rave.h"

/**
 * Debug this module
 */
PYRAVE_DEBUG_MODULE("_compositegenerator");

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

/*@{ Composite generator */
/**
 * Returns the native CartesianGenerator_t instance.
 * @param[in] pygenerator - the python composite generator instance
 * @returns the native cartesian instance.
 */
static CompositeGenerator_t*
PyCompositeGenerator_GetNative(PyCompositeGenerator* pygenerator)
{
  RAVE_ASSERT((pygenerator != NULL), "pygenerator == NULL");
  return RAVE_OBJECT_COPY(pygenerator->generator);
}

/**
 * Creates a python composite generator from a native composite generator or will create an
 * initial native CompositeGenerator if p is NULL.
 * @param[in] p - the native composite generator (or NULL)
 * @returns the python composite product generator.
 */
static PyCompositeGenerator*
PyCompositeGenerator_New(CompositeGenerator_t* p)
{
  PyCompositeGenerator* result = NULL;
  CompositeGenerator_t* cp = NULL;

  if (p == NULL) {
    cp = RAVE_OBJECT_NEW(&CompositeGenerator_TYPE);
    if (cp == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for composite generator.");
      raiseException_returnNULL(PyExc_MemoryError, "Failed to allocate memory for composite generator.");
    }
  } else {
    cp = RAVE_OBJECT_COPY(p);
    result = RAVE_OBJECT_GETBINDING(p); // If p already have a binding, then this should only be increfed.
    if (result != NULL) {
      Py_INCREF(result);
    }
  }

  if (result == NULL) {
    result = PyObject_NEW(PyCompositeGenerator, &PyCompositeGenerator_Type);
    if (result != NULL) {
      PYRAVE_DEBUG_OBJECT_CREATED;
      result->generator = RAVE_OBJECT_COPY(cp);
      RAVE_OBJECT_BIND(result->generator, result);
    } else {
      RAVE_CRITICAL0("Failed to create PyCompositeGenerator instance");
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to allocate memory for composite generator.");
    }
  }

done:
  RAVE_OBJECT_RELEASE(cp);
  return result;
}

/**
 * Deallocates the cartesian product
 * @param[in] obj the object to deallocate.
 */
static void _pycompositegenerator_dealloc(PyCompositeGenerator* obj)
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
 * Creates a new instance of the composite.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pycompositegenerator_new(PyObject* self, PyObject* args)
{
  PyCompositeGenerator* result = PyCompositeGenerator_New(NULL);
  return (PyObject*)result;
}

/**
 * Creates a new instance of the composite.
 * @param[in] self this instance.
 * @param[in] args arguments for creation (NOT USED).
 * @return the object on success, otherwise NULL
 */
static PyObject* _pycompositegenerator_create(PyObject* self, PyObject* args)
{
  PyObject* pymanager = NULL;
  PyObject* result = NULL;
  char* filename = NULL;
  CompositeFactoryManager_t* manager = NULL;
  CompositeGenerator_t* generator = NULL;

  if (!PyArg_ParseTuple(args, "|Oz", &pymanager, &filename)) {
    return NULL;
  }

  if (pymanager != NULL && !PyCompositeFactoryManager_Check(pymanager)) {
    raiseException_returnNULL(PyExc_ValueError, "Argument is <FactoryManager or None>, <filename or None>");
  }
  if (pymanager != NULL) {
    manager = ((PyCompositeFactoryManager*)pymanager)->manager;
  }

  if (manager != NULL && CompositeFactoryManager_size(manager) == 0) {
    raiseException_gotoTag(done, PyExc_ValueError, "The manager must at least contain one factory");
  }

  generator = CompositeGenerator_create(manager, filename);
  if (generator != NULL) {
    result = (PyObject*)PyCompositeGenerator_New(generator);
  } else {
    raiseException_gotoTag(done, PyExc_RuntimeError, "Could not create composite generator");
  }
done:
  RAVE_OBJECT_RELEASE(generator);
  return (PyObject*)result;
}

static PyObject* _pycompositegenerator_register(PyCompositeGenerator* self, PyObject* args)
{
  char* id = NULL;
  PyObject *pyfactory = NULL, *pyfilters = NULL;
  RaveObjectList_t* filters = NULL;
  if (!PyArg_ParseTuple(args, "sO|O", &id, &pyfactory, &pyfilters)) {
    return NULL;
  }

  if (!PyCompositeGeneratorFactory_Check(pyfactory)) {
    raiseException_returnNULL(PyExc_TypeError, "object must be a CompositeGeneratorFactory");
  }

  if (pyfilters != NULL) {
    if(!PyList_Check(pyfilters)) {
      raiseException_returnNULL(PyExc_TypeError, "Filters must be provided as a list");
    } else {
      Py_ssize_t nfilters = PyObject_Length(pyfilters);
      filters = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
      if (filters == NULL) {
        raiseException_gotoTag(fail, PyExc_MemoryError, "Failed to create object list");
      }
      int i = 0;
      for (i = 0; i < nfilters; i++) {
        PyObject* pyfilter = PyList_GetItem(pyfilters, i);
        if (pyfilter != NULL) {
          if (!PyCompositeFilter_Check(pyfilter)) {
            raiseException_gotoTag(fail, PyExc_AttributeError, "list of filters must only contain CompositeFilterCore instances");
          }
          if (!RaveObjectList_add(filters, (RaveCoreObject*)((PyCompositeFilter*)pyfilter)->filter)) {
            raiseException_gotoTag(fail, PyExc_AttributeError, "Could not add composite filter to filers list");
          }
        }
      }
    }
  }

  if (!CompositeGenerator_register(self->generator, id, ((PyCompositeGeneratorFactory*)pyfactory)->factory, filters)) {
    raiseException_gotoTag(fail, PyExc_AttributeError, "Could not add factory to generator");
  }

  RAVE_OBJECT_RELEASE(filters);
  Py_RETURN_NONE;
fail:
  RAVE_OBJECT_RELEASE(filters);
  return NULL;
}

static PyObject* _pycompositegenerator_getFactoryIDs(PyCompositeGenerator* self, PyObject* args)
{
  RaveList_t* ids = NULL;
  PyObject* result = NULL;

  if (!PyArg_ParseTuple(args, "")) {
    return NULL;
  }

  ids = CompositeGenerator_getFactoryIDs(self->generator);
  if (ids != NULL) {
    int i = 0, n = 0;
    n = RaveList_size(ids);
    result = PyList_New(0);
    for (i = 0; result != NULL && i < n; i++) {
      char* name = RaveList_get(ids, i);
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
  }

  RaveList_freeAndDestroy(&ids);
  return result;
fail:
  RaveList_freeAndDestroy(&ids);
  Py_XDECREF(result);
  return NULL;  
}

static PyObject* _pycompositegenerator_unregister(PyCompositeGenerator* self, PyObject* args)
{
  char* id = NULL;
  if (!PyArg_ParseTuple(args, "s", &id)) {
    return NULL;
  }
  CompositeGenerator_unregister(self->generator, id);

  Py_RETURN_NONE;
}

static PyObject* _pycompositegenerator_generate(PyCompositeGenerator* self, PyObject* args)
{
  PyObject* pyargs = NULL;
  PyObject* result = NULL;
  Cartesian_t* cartesian = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyargs)) {
    return NULL;  fprintf(stderr, "SET %p\n", PyCompositeFilter_API[PyCompositeFilter_Type_NUM]);

  }

  if (!PyCompositeArguments_Check(pyargs)) {
    raiseException_returnNULL(PyExc_AttributeError, "Expects a CompositeArgument object as argument");
  }

  cartesian = CompositeGenerator_generate(self->generator, ((PyCompositeArguments*)pyargs)->args);
  if (cartesian != NULL) {
    result =  (PyObject*)PyCartesian_New(cartesian);
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create cartesian product");
  }

  return result;
}

static PyObject* _pycompositegenerator_identify(PyCompositeGenerator* self, PyObject* args)
{
  PyObject* pyargs = NULL;
  PyObject* result = NULL;
  CompositeGeneratorFactory_t* factory = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyargs)) {
    return NULL;
  }

  if (!PyCompositeArguments_Check(pyargs)) {
    raiseException_returnNULL(PyExc_AttributeError, "Expects a CompositeArgument object as argument");
  }

  factory = CompositeGenerator_identify(self->generator, ((PyCompositeArguments*)pyargs)->args);
  if (factory != NULL) {
    result = (PyObject*)PyCompositeGeneratorFactory_New(factory);
  } else {
    PyErr_SetString(PyExc_ValueError, "Could not identify factory");
  }
  RAVE_OBJECT_RELEASE(factory);
  return result;
}

static PyObject* _pycompositegenerator_createFactory(PyCompositeGenerator* self, PyObject* args)
{
  PyObject* pyargs = NULL;
  PyObject* result = NULL;
  CompositeGeneratorFactory_t* factory = NULL;

  if (!PyArg_ParseTuple(args, "O", &pyargs)) {
    return NULL;
  }

  if (!PyCompositeArguments_Check(pyargs)) {
    raiseException_returnNULL(PyExc_AttributeError, "Expects a CompositeArgument object as argument");
  }

  factory = CompositeGenerator_createFactory(self->generator, ((PyCompositeArguments*)pyargs)->args);
  if (factory != NULL) {
    result = (PyObject*)PyCompositeGeneratorFactory_New(factory);
  } else {
    PyErr_SetString(PyExc_ValueError, "Could not create factory");
  }
  
  RAVE_OBJECT_RELEASE(factory);
  return result;
}

/**
 * All methods a cartesian product can have
 */
static struct PyMethodDef _pycompositegenerator_methods[] =
{
  {"properties", NULL, METH_VARARGS},
  {"register", (PyCFunction)_pycompositegenerator_register, 1,
    "register(id, factory)\n\n" //
    "Add a factory to the generator.\n\n"
    "id         - unique identifier for the factory\n"
    "plugin     - the factory instance\n"
  },
  {"getFactoryIDs", (PyCFunction)_pycompositegenerator_getFactoryIDs, 1,
    "getFactoryIDs() -> int\n\n" //
    "Returns the registered factory IDs.\n"
  },
  {"unregister", (PyCFunction)_pycompositegenerator_unregister, 1,
    "unregister(id)\n\n" //
    "Removes a factory from the generator.\n\n"
    "id         - unique identifier for the factory\n"
  },
  {"identify", (PyCFunction)_pycompositegenerator_identify, 1,
    "identify(arguments)\n\n" // "sddd", &
    "Identifies the factory to be used.\n\n"
    "arguments - A CompositeArgumentsCore instance"
  },
  {"createFactory", (PyCFunction)_pycompositegenerator_createFactory, 1,
    "createFactory(arguments)\n\n"
    "Identifies and creates the factory to use, will set properties before returning instance.\n\n"
    "arguments - A CompositeArgumentsCore instance"
  },
  {"generate", (PyCFunction)_pycompositegenerator_generate, 1,
    "generate()\n\n" // "sddd", &
    "Runs the composite generator.\n\n"
  },

  {NULL, NULL } /* sentinel */
};

/**
 * Returns the specified attribute in the cartesian
 * @param[in] self - the cartesian product
 */

static PyObject* _pycompositegenerator_getattro(PyCompositeGenerator* self, PyObject* name)
{
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("properties", name) == 0) {
    RaveProperties_t* properties = CompositeGenerator_getProperties(self->generator);
    if (properties != NULL) {
      PyObject* result = (PyObject*)PyRaveProperties_New(properties);
      RAVE_OBJECT_RELEASE(properties);
      return result;
    } else {
      Py_RETURN_NONE;
    }
  }
  return PyObject_GenericGetAttr((PyObject*)self, name);
}

/**
 * Returns the specified attribute in the polar volume
 */
static int _pycompositegenerator_setattro(PyCompositeGenerator* self, PyObject* name, PyObject* val)
{
  int result = -1;
  if (name == NULL) {
    goto done;
  }
  if (PY_COMPARE_STRING_WITH_ATTRO_NAME("properties", name) == 0) {
    if (val == Py_None) {
      CompositeGenerator_setProperties(self->generator, NULL);
    } else if (PyRaveProperties_Check(val)) {
      CompositeGenerator_setProperties(self->generator, ((PyRaveProperties*)val)->properties);
    } else {
      raiseException_gotoTag(done, PyExc_ValueError,"properties must be of type RavePropertiesCore or None");
    }
  }
  result = 0;
done:
  return result;
}

/*@} End of Composite product generator */

/*@{ Documentation about the type */
PyDoc_STRVAR(_pycompositegenerator_type_doc,
    "The composite type provides the possibility to create cartesian composites from a number of polar objects.\n"
    "To generate the composite, one or many polar scans or polar volumes has to be added to the generator. Then generate should be called with the expected area and an optional list of how/task quality field names.\n"
    "There are a few attributes that can be set besides the functions.\n"
    " height                       - The height in meters that should be used when generating a composite like CAPPI, PCAPPI or PMAX.\n"
    " elangle                      - The elevation angle in radians that should be used when generating a composite like PPI."
    " range                        - The range that should be used when generating the Pseudo MAX. This range is the limit in meters\n"
    "                                for when the vertical max should be used. When outside this range, the PCAPPI value is used instead.\n"
    " product                      - The product type that should be generated when generating the composite.\n"
    "                                Height/Elevation angle and range are used in combination with the products.\n"
    "                                PPI requires elevation angle\n"
    "                                CAPPI, PCAPPI and PMAX requires height above sea level\n"
    "                                PMAX also requires range in meters\n"
    " selection_method             - The selection method to use when there are more than one radar covering same point. I.e. if for example taking distance to radar or height above sea level. Currently the following methods are available\n"
    "       _pycomposite.SelectionMethod_NEAREST - Value from the nearepropertiesst radar is selected.\n"
    "       _pycomposite.SelectionMethod_HEIGHT  - Value from radar which scan is closest to the sea level at current point.\n"
    " interpolation_method         - Interpolation method is used to choose how to interpolate the surrounding values. The default behaviour is NEAREST.\n"
    "       _pycomposite.InterpolationMethod_NEAREST                  - Nearest value is used\n"
    "       _pycomposite.InterpolationMethod_LINEAR_HEIGHT            - Value calculated by performing a linear interpolation between the closest positions above and below\n"
    "       _pycomposite.InterpolationMethod_LINEAR_RANGE             - Value calculated by performing a linear interpolation between the closest positions before\n"
    "                                                                   and beyond in the range dimension of the ray\n"
    "       _pycomposite.InterpolationMethod_LINEAR_AZIMUTH           - Value calculated by performing a linear interpolation between the closest positions on each\n"
    "                                                                   side of the position, i.e., interpolation between consecutive rays\n"
    "       _pycomposite.InterpolationMethod_LINEAR_RANGE_AND_AZIMUTH - Value calculated by performing a linear interpolation in azimuth and range directions.\n"
    "       _pycomposite.InterpolationMethod_LINEAR_3D                - Value calculated by performing a linear interpolation in height, azimuth and range directions.\n"
    "       _pycomposite.InterpolationMethod_QUADRATIC_HEIGHT         - Value calculated by performing a quadratic interpolation between the closest positions before and beyond in\n"
    "                                                                   the range dimension of the ray. Quadratic interpolation means that inverse distance weights raised to the\n"
    "                                                                   power of 2 are used in value interpolation.\n"
    "       _pycomposite.InterpolationMethod_QUADRATIC_3D             - Value calculated by performing a quadratic interpolation in height, azimuth and range\n"
    "                                                                   directions. Quadratic interpolation means that inverse distance weights raised to the\n"
    "                                                                   power of 2 are used in value interpolation.\n"
    ""
    " interpolate_undetect         - If undetect should be used in interpolation or not.\n"
    "                                If undetect not should be includepropertiesd in the interpolation, the behavior will be the following:\n"
    "                                * If all values are UNDETECT, then result will be UNDETECT.\n"
    "                                * If only one value is DATA, then use that value.\n"
    "                                * If more than one value is DATA, then interpolation.\n"
    "                                * If all values are NODATA, then NODATA.\n"
    "                                * If all values are either NODATA or UNDETECT, then UNDETECT.\n"
    ""
    " date                         - The nominal date as a string in format YYYYMMDD\n"
    " time                         - The nominal time as a string in format HHmmss\n"
    " quality_indicator_field_name - If this field name is set, then the composite will be generated by first using the quality indicator field for determining\n"
    "                                radar usage. If the field name is None, then the selection method will be used instead.\n"
    "\n"
    "Usage:\n"
    " import _pycomposite\n"
    " generator = _pycomposite.new()\n"
    " generator.selection_method = _pycomposite.SelectionMethod_HEIGHT\n"
    " generator.product = \"PCAPPI\"\n"
    " generator.height = 500.0\n"
    " generator.date = \"20200201\"\n"
    " generator.date = \"100000\"\n"
    " generator.addParameter(\"DBZH\", 2.0, 3.0, -30.0)\n"
    " generator.add(_rave.open(\"se1_pvol_20200201100000.h5\").object)\n"
    " generator.add(_rave.open(\"se2_pvol_20200201100000.h5\").object)\n"
    " generator.add(_rave.open(\"se3_pvol_20200201100000.h5\").object)\n"
    " result = generator.generate(myarea, [\"se.smhi.composite.distance.radar\",\"pl.imgw.radvolqc.spike\"])\n"
    );
/*@} End of Documentation about the type */

/*@{ Type definitions */
PyTypeObject PyCompositeGenerator_Type =
{
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "CompositeGeneratorCore", /*tp_name*/
  sizeof(PyCompositeGenerator), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_pycompositegenerator_dealloc, /*tp_dealloc*/
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
  (getattrofunc)_pycompositegenerator_getattro, /*tp_getattro*/
  (setattrofunc)_pycompositegenerator_setattro, /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  _pycompositegenerator_type_doc,        /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _pycompositegenerator_methods,              /*tp_methods    RAVE_OBJECT_RELEASE(factory);
  */
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
  {"new", (PyCFunction)_pycompositegenerator_new, 1,
    "new() -> new instance of the CompositeGeneratorCore object\n\n"
    "Creates a new instance of the CompositeGeneratorCore object"
  },
  {"create", (PyCFunction)_pycompositegenerator_create, 1,
    "create() -> new instance of the CompositeGeneratorCore object with predefined factories\n\n"
    "Creates a new instance of the CompositeGeneratorCore object with predefined factories"
  },
  {NULL,NULL} /*Sentinel*/
};

MOD_INIT(_compositegenerator)
{
  PyObject *module=NULL,*dictionary=NULL;
  static void *PyCompositeGenerator_API[PyCompositeGenerator_API_pointers];
  PyObject *c_api_object = NULL;

  MOD_INIT_SETUP_TYPE(PyCompositeGenerator_Type, &PyType_Type);

  MOD_INIT_VERIFY_TYPE_READY(&PyCompositeGenerator_Type);

  MOD_INIT_DEF(module, "_compositegenerator", _pycompositegenerator_type_doc, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  PyCompositeGenerator_API[PyCompositeGenerator_Type_NUM] = (void*)&PyCompositeGenerator_Type;
  PyCompositeGenerator_API[PyCompositeGenerator_GetNative_NUM] = (void *)&PyCompositeGenerator_GetNative;
  PyCompositeGenerator_API[PyCompositeGenerator_New_NUM] = (void*)&PyCompositeGenerator_New;

  c_api_object = PyCapsule_New(PyCompositeGenerator_API, PyCompositeGenerator_CAPSULE_NAME, NULL);
  dictionary = PyModule_GetDict(module);
  PyDict_SetItemString(dictionary, "_C_API", c_api_object);
  
  ErrorObject = PyErr_NewException("_compositegenerator.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _compositegenerator.error");
    return MOD_INIT_ERROR;
  }

  import_array(); /*To make sure I get access to Numeric*/
  
  import_raveproperties();
  import_compositegeneratorfactory();
  import_compositefactorymanager();
  import_compositearguments();
  import_compositefilter();
  import_pypolarvolume();
  import_pypolarscan();
  import_pycartesian();
  import_pyarea();

  PYRAVE_DEBUG_INITIALIZE;

  return MOD_INIT_SUCCESS(module);
}
/*@} End of Module setup */
