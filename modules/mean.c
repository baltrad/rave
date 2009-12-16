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
 * Mean
 * @file
 * @author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
 * @author Anders Henja
 * @date 2009-12-17
 */
#include <Python.h>
#include "pycartesian.h"

static PyObject *ErrorObject;

#define Raise(type,msg) {PyErr_SetString(type,msg);}

/*
 Calculates the average value within an NxN-sized kernel.
 */
static PyObject* _average_func(PyObject* self, PyObject* args)
{
  PyObject* pyobject = NULL;
  PyObject* result = NULL;
  Cartesian_t* cartesian = NULL;
  Cartesian_t* target = NULL;

  int N = 0;
  long xsize = 0, ysize = 0, x = 0, y = 0;

  if (!PyArg_ParseTuple(args, "Oi", &pyobject, &N)) {
    return NULL;
  }

  if (!PyCartesian_Check(pyobject)) {
    return NULL;
  }

  cartesian = PyCartesian_GetNative((PyCartesian*)pyobject);
  target = RAVE_OBJECT_CLONE(cartesian);

  if (target == NULL) {
    goto done;
  }
  xsize = Cartesian_getXSize(cartesian);
  ysize = Cartesian_getYSize(cartesian);

  for (x = 0; x < xsize; x++) {
    for (y = 0; y < ysize; y++) {
      double value = 0.0L;
      (void)Cartesian_getMean(cartesian, x, y, N, &value);
      Cartesian_setValue(target, x, y, value);
    }
  }

  result = (PyObject*)PyCartesian_New(target);
done:
  RAVE_OBJECT_RELEASE(cartesian);
  RAVE_OBJECT_RELEASE(target);
  return result;
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

  import_pycartesian();
}
