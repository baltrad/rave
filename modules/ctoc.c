/**
 * Cartesian-to-cartesian transformation functionality in RAVE.
 * @file
 * @author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2006-
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION 
#include <pyravecompat.h>
#include <arrayobject.h>
#include "rave.h"

static PyObject *ErrorObject;

/**
 * Sets a python exception, then executes goto fail.
 */
#define Raise(type,msg) {PyErr_SetString(type,msg); goto fail;}

static PyObject* _ctoc_transform(PyObject* self, PyObject* args)
{
  PyObject *in = NULL, *out = NULL;
  PyObject* result = NULL;
  int method;

  RaveObject inrave, outrave;
  RaveImageStruct istruct, ostruct;
  RaveTransform2D trafo;

  unsigned char* outputdata;
  char outputtype;
  int output_stride_xsize;
  double cressxy;
  int x, y;
  int sets, i;

  trafo.inpj = NULL;
  trafo.outpj = NULL;

  if (!PyArg_ParseTuple(args, "OO", &in, &out))
    return NULL;

  if (!GetIntFromINFO(in, "/what/sets", &sets)) {
    Raise(PyExc_AttributeError,"Unknown number of datasets.");
    goto fail;
  }

  /* Loop through all images in the objects. There must be exactly the same
   number of images in input and output objects. */
  for (i = 1; i <= sets; i++) {

    initialize_RaveObject(&inrave);
    initialize_RaveObject(&outrave);

    if (!fill_rave_object(in, &inrave, i, "image")) {
      if (!inrave.info || !inrave.data) {
        goto fail;
      }
    }
    if (!fill_rave_image_info(in, &istruct, i)) {
      goto fail;
    }
    if (!fill_rave_object(out, &outrave, i, "image")) {
      if (!outrave.info || !outrave.data) {
        goto fail;
      }
    }
    if (!fill_rave_image_info(out, &ostruct, i)) {
      goto fail;
    }
    if (!GetIntFromINFO(in, "/how/i_method", &method)) {
      Raise(PyExc_AttributeError,"No i_method in input.");
      goto fail;
    }
    trafo.nodata = istruct.nodata;
    trafo.inUL.u = istruct.lowleft.u;
    trafo.inUL.v = istruct.uppright.v;
    trafo.inxscale = istruct.xscale;
    trafo.inyscale = istruct.yscale;
    trafo.inxmax = istruct.xsize;
    trafo.inymax = istruct.ysize;
    trafo.inpj = get_rave_projection(in);

    trafo.outUL.u = ostruct.lowleft.u;
    trafo.outUL.v = ostruct.uppright.v;
    trafo.outxscale = ostruct.xscale;
    trafo.outyscale = ostruct.yscale;
    trafo.outpj = get_rave_projection(out);

    trafo.data = array_data_2d(inrave.data);
    trafo.stride_xsize = array_stride_xsize_2d(inrave.data);
    trafo.type = translate_pytype_to_ravetype(array_type_2d(inrave.data));

    outputdata = array_data_2d(outrave.data);
    outputtype = translate_pytype_to_ravetype(array_type_2d(outrave.data));
    output_stride_xsize = array_stride_xsize_2d(outrave.data);

    if (method != CRESSMAN && method != INVERSE && method != UNIFORM &&
        method != CUBIC && method != BILINEAR && method != NEAREST) {
      Raise(PyExc_AttributeError,"No such interpolation method");
      goto fail;
    }
    trafo.method = method;

    /* Optional argument for more sophisticated interpolation. */
    if (!GetDoubleFromINFO(out, "/how/cressman_xy", &cressxy))
      cressxy = 0.707106781187;

    trafo.R = sqrt((ostruct.xscale * ostruct.xscale + \
        ostruct.yscale * ostruct.yscale)) * cressxy;

    for (y = 0; y < PyArray_DIMS(outrave.data)[0]; y++) {
      UV here_s;
      here_s.v = ostruct.uppright.v - ostruct.yscale * (y + 0.5);
      for (x = 0; x < PyArray_DIMS(outrave.data)[1]; x++) {
        TransformWeight* w;
        double v;
        here_s.u = (ostruct.lowleft.u + ostruct.xscale * (x + 0.5));
        w = get_weights_2d(x, y, here_s, &trafo);
        if (w && w->total_wsum) {
          v = compute_weights_2d(w);
          set_array_item_2d(outputdata, x, y, v, outputtype, output_stride_xsize);
        }
        if (w)
          free_tw(w);
      }
    }
  }

  PyErr_Clear(); // If we come here, reset the error messages
  Py_INCREF(Py_None);
  result = Py_None; // On success, set result to Py_None
fail:
  if (trafo.inpj)
    freeProjection(trafo.inpj);
  if (trafo.outpj)
    freeProjection(trafo.outpj);
  return result;
}

static PyObject* _ctoc_test(PyObject* self, PyObject* args)
{
  printf("CTOC: Test to verify that ctoc has been built cleanly.\n");

  Py_INCREF(Py_None);
  return Py_None;
}

/* Forward declaration */
#ifdef DO_DEBUG
void rave_debug(int dbg);

static PyObject* _ctoc_debug(PyObject* self, PyObject* args)
{
  int debugflag;

  if(!PyArg_ParseTuple(args,"i",&debugflag))
  return NULL;
  printf("DEBUGFLAG = %d\n",debugflag);
  rave_debug(debugflag);
  Py_INCREF(Py_None);
  return Py_None;
}
#endif

static struct PyMethodDef _ctoc_functions[] =
{
{ "transform", (PyCFunction) _ctoc_transform, METH_VARARGS },
{ "test", (PyCFunction) _ctoc_test, METH_VARARGS },
#ifdef DO_DEBUG
    { "init_debug",(PyCFunction)_ctoc_debug,METH_VARARGS},
#endif
    { NULL, NULL } };

/**
 * Initializes the python module _ctoc.
 */
MOD_INIT(_ctoc)
{
  PyObject* module = NULL;
  PyObject* dictionary = NULL;

  MOD_INIT_DEF(module, "_ctoc", NULL/*doc*/, _ctoc_functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  dictionary = PyModule_GetDict(module);
  ErrorObject = PyErr_NewException("_ctoc.error", NULL, NULL);
  if (ErrorObject == NULL || PyDict_SetItemString(dictionary, "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _ctoc.error");
    return MOD_INIT_ERROR;
  }

  import_array(); /*To make sure I get access to the Numeric PyArray functions*/
  return MOD_INIT_SUCCESS(module);
}
