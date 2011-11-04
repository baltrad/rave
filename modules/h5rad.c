/**
 * THIS IS A JUNK FILE USED TO TEST CODE.
 * @file
 * @author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2006-
 */
#include <Python.h>
#include <arrayobject.h>
#include "rave.h"

static PyObject *ErrorObject;

/**
 * Sets a python exception
 */
#define Raise(type,msg) {PyErr_SetString(type,msg);}

static int gimme(PyObject *info, RaveImageStruct *p)
{
  int ret = 1;
  PyObject *po;

  ret = GetIntFromINFO(info, "where/xsize", &p->xsize);
  if (!ret)
    Raise(PyExc_AttributeError,"No xsize in in.info");

  ret = GetIntFromINFO(info, "where/ysize", &p->ysize);
  if (!ret)
    Raise(PyExc_AttributeError,"No ysize in in.info");

  ret = GetDoubleFromINFO(info, "where/xscale", &p->xscale);
  if (!ret)
    Raise(PyExc_AttributeError,"No xscale in in.info");

  ret = GetDoubleFromINFO(info, "where/yscale", &p->yscale);
  if (!ret)
    Raise(PyExc_AttributeError,"No yscale in in.info");

  po = GetSequenceFromINFO(info, "/how/extent");
  if (PySequence_Size(po) <= 0)
    Raise(PyExc_AttributeError,
        "No extent in in.info");

  if (!GetDoubleFromSequence(po, 0, &p->lowleft.u))
    Raise(PyExc_AttributeError,"Bad read of extent");
  if (!GetDoubleFromSequence(po, 1, &p->lowleft.v))
    Raise(PyExc_AttributeError,"Bad read of extent");
  if (!GetDoubleFromSequence(po, 2, &p->uppright.u))
    Raise(PyExc_AttributeError,"Bad read of extent");
  if (!GetDoubleFromSequence(po, 3, &p->uppright.v))
    Raise(PyExc_AttributeError,"Bad read of extent");
  /*    p->lowleft.u=atof(PyString_AsString(PySequence_GetItem(po,0))); */
  /*    p->lowleft.v=atof(PyString_AsString(PySequence_GetItem(po,1))); */
  /*    p->uppright.u=atof(PyString_AsString(PySequence_GetItem(po,2))); */
  /*    p->uppright.v=atof(PyString_AsString(PySequence_GetItem(po,3))); */

  Py_DECREF(po);
  return ret;
}

static PyObject* _read_h5rad_func(PyObject* self, PyObject* args)
{
  PyObject *in, *out;
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
      printf("Failed to fill inrave.info\n");
      goto fail;
    }
    if (!fill_rave_object(out, &outrave, i, "image")) {
      if (!outrave.info || !outrave.data) {
        goto fail;
      }
    }
    if (!fill_rave_image_info(out, &ostruct, i)) {
      printf("Failed to fill outrave.info\n");
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
    trafo.inpj = NULL;

    trafo.outUL.u = ostruct.lowleft.u;
    trafo.outUL.v = ostruct.uppright.v;
    trafo.outxscale = ostruct.xscale;
    trafo.outyscale = ostruct.yscale;
    trafo.outpj = get_rave_projection(out);
    trafo.outpj = NULL;

    trafo.data = array_data_2d(inrave.data);
    trafo.stride_xsize = array_stride_xsize_2d(inrave.data);
    trafo.type = array_type_2d(inrave.data);

    outputdata = array_data_2d(outrave.data);
    outputtype = array_type_2d(outrave.data);
    output_stride_xsize = array_stride_xsize_2d(outrave.data);

    if (method != CRESSMAN && method != INVERSE && method != UNIFORM && method
        != CUBIC && method != BILINEAR && method != NEAREST) {
      Raise(PyExc_AttributeError,"No such interpolation method");
      goto fail;
    }
    trafo.method = method;

    /* Optional argument for more sophisticated interpolation. */
    if (!GetDoubleFromINFO(out, "/how/cressman_xy", &cressxy))
      cressxy = 0.707106781187;

    trafo.R = sqrt((ostruct.xscale * ostruct.xscale +\
 ostruct.yscale
        * ostruct.yscale)) * cressxy;

    for (y = 0; y < outrave.data->dimensions[0]; y++) {
      UV here_s;
      here_s.v = ostruct.uppright.v - ostruct.yscale * (y + 0.5);
      //      here_s.v=ostruct.uppright.v - ostruct.yscale*y;
      for (x = 0; x < outrave.data->dimensions[1]; x++) {
        TransformWeight* w;
        double v;
        here_s.u = (ostruct.lowleft.u + ostruct.xscale * (x + 0.5));
        //	 here_s.u=(ostruct.lowleft.u + ostruct.xscale*x);
        w = get_weights_2d(x, y, here_s, &trafo);
        if (w && w->total_wsum) {
          v = compute_weights_2d(w);
          set_array_item_2d(outputdata, x, y, v, outputtype,
                            output_stride_xsize);
        }
        if (w)
          free_tw(w);
      }
    }
  }
  if (trafo.inpj)
    pj_free(trafo.inpj);
  if (trafo.outpj)
    pj_free(trafo.outpj);
  PyErr_Clear();
  Py_INCREF(Py_None);
  return Py_None;
  fail: return NULL;
}

static PyObject* _test_h5rad_func(PyObject* self, PyObject* args)
{
  PyObject *in, *out;
  int method;

  RaveObject inrave, outrave;
  RaveImageStruct istruct, ostruct;
  RaveTransform2D trafo;

  unsigned char* outputdata;
  char outputtype;
  int output_stride_xsize;
  double cressxy;
  int x, y;
  int set = 1;

  if (!PyArg_ParseTuple(args, "OO", &in, &out))
    return NULL;

  if (!fill_rave_object(in, &inrave, set, "image")) {
    if (!inrave.info || !inrave.data) {
      goto fail;
    }
  }
  if (!fill_rave_image_info(inrave.info, &istruct, set)) {
    printf("Failed to fill inrave.info\n");
    goto fail;
  }
  if (!fill_rave_object(out, &outrave, set, "image")) {
    if (!outrave.info || !outrave.data) {
      goto fail;
    }
  }
  if (!fill_rave_image_info(outrave.info, &ostruct, set)) {
    printf("Failed to fill outrave.info\n");
    goto fail;
  }
  if (!GetIntFromINFO(inrave.info, "how/i_method", &method)) {
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
  trafo.inpj = get_rave_projection(inrave.info);

  trafo.outUL.u = ostruct.lowleft.u;
  trafo.outUL.v = ostruct.uppright.v;
  trafo.outxscale = ostruct.xscale;
  trafo.outyscale = ostruct.yscale;
  trafo.outpj = get_rave_projection(outrave.info);

  trafo.data = array_data_2d(inrave.data);
  trafo.stride_xsize = array_stride_xsize_2d(inrave.data);
  trafo.type = array_type_2d(inrave.data);

  outputdata = array_data_2d(outrave.data);
  outputtype = array_type_2d(outrave.data);
  output_stride_xsize = array_stride_xsize_2d(outrave.data);

  if (method != CRESSMAN && method != INVERSE && method != UNIFORM && method
      != CUBIC && method != BILINEAR && method != NEAREST) {
    Raise(PyExc_AttributeError,"No such interpolation method");
    goto fail;
  }
  trafo.method = method;

  /* Optional argument for more sophisticated interpolation. */
  if (!GetDoubleFromINFO(outrave.info, "how/cressman_xy", &cressxy))
    cressxy = 0.71;

  trafo.R = sqrt((ostruct.xscale * ostruct.xscale +\
 ostruct.yscale
      * ostruct.yscale)) * cressxy;

  for (y = 0; y < outrave.data->dimensions[0]; y++) {
    UV here_s;
    here_s.v = ostruct.uppright.v - ostruct.yscale * y;
    for (x = 0; x < outrave.data->dimensions[1]; x++) {
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

  if (trafo.inpj)
    pj_free(trafo.inpj);
  if (trafo.outpj)
    pj_free(trafo.outpj);
  PyErr_Clear();
  Py_INCREF(Py_None);
  return Py_None;
  fail: free_rave_object(&inrave);
  free_rave_object(&outrave);
  if (trafo.inpj)
    pj_free(trafo.inpj);
  if (trafo.outpj)
    pj_free(trafo.outpj);
  return NULL;
}

static PyObject* _test_import_func(PyObject* self, PyObject* args)
{
  PyObject *pcsmodule = NULL, *pcso = NULL;
  PyObject *pcsdef = NULL, *pcsname = NULL;
  PJ* retpj = NULL;
  int ok = 1;

  /*import pcs*/
  pcsmodule = PyImport_ImportModule("pcs");
  if (!pcsmodule) {
    printf("No pcs module\n");
    ok = 0;
  }

  //   pcsname=ok?PyMapping_GetItemString(dict,"pcs"):NULL;
  /*    printf("reading PyString\n"); */
  /*    pcsname = GetPyStringFromINFO(info, "how/pcs"); */
  /*    if(!PyString_Check(pcsname)) */
  /*       ok=0; */
  /*    printf("read PyString %s\n", PyString_AsString(pcsname)); */

  /* pcso=pcs.pcs(pcsname) */
  printf("Calling method?\n");
  pcso = PyObject_CallMethod(pcsmodule, "pcs", "s", "ps14e60n");
  printf("Called method\n");
  if (!pcso) {
    printf("NULL pcso object\n");
    ok = 0;
  }

  /* pcsdef=pcso.definition */
  pcsdef = PyObject_GetAttrString(pcso, "definition");
  if (PySequence_Size(pcsdef) <= 0)
    ok = 0;

  if (ok) {
    printf("initializing projection\n");
    retpj = initProjection(pcsdef);
    if (retpj == NULL) {
      fprintf(stderr, "Failed to initialize projection\n");
    }
  }

  Py_XDECREF(pcsmodule);
  Py_XDECREF(pcso);
  Py_XDECREF(pcsname);
  Py_XDECREF(pcsdef);

  PyErr_Clear();
  Py_INCREF(Py_None);
  return Py_None;
}

static struct PyMethodDef _h5rad_functions[] =
{
{ "read_h5rad", (PyCFunction) _read_h5rad_func, METH_VARARGS },
{ "test_h5rad", (PyCFunction) _test_h5rad_func, METH_VARARGS },
{ "test_import", (PyCFunction) _test_import_func, METH_VARARGS },
{ NULL, NULL } };

/**
 * Initializes the python module _h5rad.
 */
PyMODINIT_FUNC init_h5rad(void)
{
  PyObject* m;
  m = Py_InitModule("_h5rad", _h5rad_functions);
  ErrorObject = PyString_FromString("_h5rad.error");
  if (ErrorObject == NULL || PyDict_SetItemString(PyModule_GetDict(m), "error",
                                                  ErrorObject) != 0)
    Py_FatalError("Can't define _h5rad.error");

  import_array(); /*Access to the Numeric PyArray functions*/
}
