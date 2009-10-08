/**
 * Miscellaneous helper functions for RAVE.
 * @file
 * @author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2006-
 */
#include <Python.h>
#include <arrayobject.h>
#include "rave.h"
#include "raveutil.h"
#include "rave_alloc.h"

static PyObject *ErrorObject;

#ifdef RAVE_MEMORY_DEBUG
/**
 * Flag indicating if the at exit function has been set or not.
 */
static int atExitSet = 0;
#endif

/**
 * Sets a python exception.
 */
#define Raise(type,msg) {PyErr_SetString(type,msg);}

/*
 This helper is only designed to work with 8-bit data, due to the check
 included in dbz2raw in raveutil.c
 */
static PyObject* _common_gain_offset(PyObject* self, PyObject* args)
{
  PyObject *in, *pyo;
  RaveObject inrave;
  int set = 0;
  int sets, x, y;
  double gain, offset, setgain, setoffset, nodata, undetect;
  unsigned char *indata;
  char intype;
  int instridex;
  double raw, dbz;
  char *attrname;

  if (!PyArg_ParseTuple(args, "Oidd", &in, &set, &gain, &offset))
    return NULL;

  if (!set) {
    Raise(PyExc_AttributeError,"Dataset index cannot be zero");
    return NULL;
  }
  if (!GetIntFromINFO(in, "/what/sets", &sets)) {
    Raise(PyExc_AttributeError,"Unknown number of datasets in input object");
    return NULL;
  }
  if (set > sets) {
    Raise(PyExc_AttributeError,"Not that many datasets in input object");
    return NULL;
  }

  initialize_RaveObject(&inrave); /* optional initialization */
  if (!fill_rave_object(in, &inrave, set, "image")) {
    if (!inrave.info || !inrave.data) {
      Raise(PyExc_AttributeError,"No info or data in input object");
      return NULL;
    }
  }
  indata = array_data_2d(inrave.data);
  intype = array_type_2d(inrave.data);
  instridex = array_stride_xsize_2d(inrave.data);

  pyo = PyString_FromFormat("/image%d/what/gain", set);
  attrname = PyString_AsString(pyo);
  if (!GetDoubleFromINFO(in, attrname, &setgain)) {
    Raise(PyExc_AttributeError,"No /what/gain for this dataset");
    return NULL;
  }
  pyo = PyString_FromFormat("/image%d/what/offset", set);
  attrname = PyString_AsString(pyo);
  if (!GetDoubleFromINFO(in, attrname, &setoffset)) {
    Raise(PyExc_AttributeError,"No /what/offset for this dataset");
    return NULL;
  }
  pyo = PyString_FromFormat("/image%d/what/nodata", set);
  attrname = PyString_AsString(pyo);
  if (!GetDoubleFromINFO(in, attrname, &nodata)) {
    Raise(PyExc_AttributeError,"No /what/nodata for this dataset");
    return NULL;
  }
  pyo = PyString_FromFormat("/image%d/what/undetect", set);
  attrname = PyString_AsString(pyo);
  if (!GetDoubleFromINFO(in, attrname, &undetect)) {
    Raise(PyExc_AttributeError,"No /what/undetect for this dataset");
    return NULL;
  }

  for (y = 0; y < inrave.data->dimensions[0]; y++) {
    for (x = 0; x < inrave.data->dimensions[1]; x++) {

      raw = get_array_item_2d(indata, x, y, intype, instridex);

      if ((raw != nodata) && (raw != undetect)) {

        dbz = raw2dbz(raw, setgain, setoffset);
        raw = dbz2raw(dbz, gain, offset);

        set_array_item_2d(indata, x, y, raw, intype, instridex);
      }
    }
  }

  PyErr_Clear();
  Py_INCREF(Py_None);
  return Py_None;
}

/**
 * Enables so that on exit, rave memory status is printed to stdout.
 */
static PyObject* _trigger_memory_status(PyObject* self, PyObject* args)
{
#ifdef RAVE_MEMORY_DEBUG
  if (atExitSet == 0) {
    atExitSet = 1;
    if (atexit(rave_alloc_print_statistics) != 0) {
      fprintf(stderr, "Could not set atexit function");
    }
  }
#endif

  Py_INCREF(Py_None);
  return Py_None;
}

static struct PyMethodDef _helpers_functions[] =
{
{ "CommonGainOffset", (PyCFunction) _common_gain_offset, METH_VARARGS },
{ "triggerMemoryStatus", (PyCFunction)_trigger_memory_status, METH_VARARGS },
{ NULL, NULL }
};

/**
 * Initializes the python module _helpers.
 */
PyMODINIT_FUNC init_helpers(void)
{
  PyObject* m;
  m = Py_InitModule("_helpers", _helpers_functions);
  ErrorObject = PyString_FromString("_helpers.error");
  if (ErrorObject == NULL || PyDict_SetItemString(PyModule_GetDict(m), "error",
                                                  ErrorObject) != 0)
    Py_FatalError("Can't define _helpers.error");

  import_array(); /*To make sure I get access to the Numeric PyArray functions*/
}
