/* --------------------------------------------------------------------------
 $Id: mean.c,v 1.1.1.1 2006/07/14 11:31:54 dmichels Exp $
 Program:

 Description:

 Author(s):      Daniel Michelson

 Copyright:	Swedish Meteorological and Hydrological Institute, 2003

 $Log: mean.c,v $
 Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
 Project added under CVS

 -----------------------------------------------------------------------------*/
#include <Python.h>
#include <arrayobject.h>
#include "rave.h"

static PyObject *ErrorObject;

#define Raise(type,msg) {PyErr_SetString(type,msg);}

/*
 Calculates the average value within an NxN-sized kernel.
 */
static PyObject* _average_func(PyObject* self, PyObject* args)
{
  PyObject *in, *mean;

  RaveObject inrave, meanrave;
  unsigned char *indata, *meandata;
  char intype, meantype;
  int instridex, meanstridex;
  double VAL, KVAL, SUM, MEAN;
  int N, xsize, ysize;
  int x, y, k, yk, xk, nodata;

  initialize_RaveObject(&inrave);
  initialize_RaveObject(&meanrave);

  if (!PyArg_ParseTuple(args, "OOi", &in, &mean, &k))
    return NULL;

  if (!fill_rave_object(in, &inrave)) {
    if (!inrave.info || !inrave.data) {
      Raise(PyExc_AttributeError,"No info or data in input");
      goto fail;
    }
  }
  if (!getIntFromDictionary("nodata", &nodata, inrave.info)) {
    Raise(PyExc_AttributeError,"No nodata in in.info");
    goto fail;
  }

  if (!fill_rave_object(mean, &meanrave)) {
    if (!meanrave.info || !meanrave.data) {
      Raise(PyExc_AttributeError,"No info or data in mean");
      goto fail;
    }
  }

  indata = array_data_2d(inrave.data);
  intype = array_type_2d(inrave.data);
  instridex = array_stride_xsize_2d(inrave.data);

  meandata = array_data_2d(meanrave.data);
  meantype = array_type_2d(meanrave.data);
  meanstridex = array_stride_xsize_2d(meanrave.data);

  k = (int) (k / 2);
  ysize = inrave.data->dimensions[0];
  xsize = inrave.data->dimensions[1];

  /* Loop through the image */
  for (y = 0; y < ysize; y++) {
    for (x = 0; x < xsize; x++) {
      VAL = get_array_item_2d(indata, x, y, intype, instridex);

      if ((VAL != nodata) && (VAL != 0.0)) {
        SUM = 0.0;
        N = 0;

        /* Loop through the kernel */
        for (yk = -k; yk < k; yk++) {
          for (xk = -k; xk < k; xk++) {

            /* Make sure we're not out of bounds before doing anything */
            if ((((yk + k) >= 0) && ((yk + k) < ysize)) || (((xk + k) >= 0)
                && ((xk + k) < xsize))) {
              KVAL = get_array_item_2d(indata, xk + x, yk + y, intype,
                                       instridex);
              SUM += KVAL;
              N += 1;
            }
          }
        }
        MEAN = SUM / N;
        set_array_item_2d(meandata, x, y, MEAN, meantype, meanstridex);

      } else {
        set_array_item_2d(meandata, x, y, nodata, meantype, meanstridex);
      }
    }
  }
  free_rave_object(&inrave);
  free_rave_object(&meanrave);
  Py_INCREF(Py_None);
  return Py_None;
fail:
  free_rave_object(&inrave);
  free_rave_object(&meanrave);
  return NULL;
}

static struct PyMethodDef _mean_functions[] =
{
  { "average", (PyCFunction) _average_func, METH_VARARGS },
  { NULL, NULL }
};

void init_mean()
{
  PyObject* m;
  m = Py_InitModule("_mean", _mean_functions);
  ErrorObject = PyString_FromString("_mean.error");
  if (ErrorObject == NULL || PyDict_SetItemString(PyModule_GetDict(m),
                                                  "error", ErrorObject) != 0) {
    Py_FatalError("Can't define _mean.error");
  }

  import_array(); /*Access to the Numeric PyArray functions*/
}
