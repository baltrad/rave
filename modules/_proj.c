/* --------------------------------------------------------------------------
 $Id: _proj.c,v 1.2 2006/12/18 09:34:17 dmichels Exp $

 Description:

 Author(s):      Daniel Michelson, based on work contracted to Fredrik Lundh

 Copyright (c):	Swedish Meteorological and Hydrological Institute, 1997-
 All rights reserved.

 $Log: _proj.c,v $
 Revision 1.2  2006/12/18 09:34:17  dmichels
 *** empty log message ***

 Revision 1.1.1.1  2006/07/14 11:31:54  dmichels
 Project added under CVS


 * Note: if you get "no such file" errors when creating proj
 * instances, you may wish to change the (errno == 25) tests
 * in pj_init.c to (errno == ENOENT)...

 FIXME: should proj/invproj raise exceptions on HUGE_VAL?
 -----------------------------------------------------------------------------*/

#include "Python.h"
#include "projects.h"

#ifdef PJ_VERSION
#define UV projUV
#endif

typedef struct {
  PyObject_HEAD
  PJ* pj;
} ProjObject;

staticforward PyTypeObject Proj_Type;

static PyObject *PyProj_Error;

/* -------------------------------------------------------------------- */
/* Constructors                                                         */

static void _proj_error(void)
{
  PyErr_SetString(PyProj_Error, pj_strerrno(pj_errno));
}

static PyObject*
_proj_new(PyObject* self, PyObject* args)
{
  ProjObject* op;
  PyObject* in;
  int n, i;
  char** argv;

  if (!PyArg_ParseTuple(args, "O", &in))
    return NULL;

  if (!PySequence_Check(in)) {
    PyErr_SetString(PyExc_TypeError, "argument must be sequence");
    return NULL;
  }

  op = PyObject_NEW(ProjObject, &Proj_Type);
  if (op == NULL)
    return NULL;

  n = PyObject_Length(in);

  /* fetch argument array */
  argv = malloc(n * sizeof(char*));
  for (i = 0; i < n; i++) {
    PyObject* op = PySequence_GetItem(in, i);
    PyObject* str = PyObject_Str(op);
    argv[i] = PyString_AsString(str);
    Py_DECREF(str);
    Py_DECREF(op);
  }

  op->pj = pj_init(n, argv);

  free(argv);

  if (!op->pj) {
    PyObject_Free(op);
    _proj_error();
    return NULL;
  }

  return (PyObject*) op;
}

static void _proj_dealloc(ProjObject* op)
{
  if (op->pj)
    pj_free(op->pj);
  PyObject_Free(op);
}

/* -------------------------------------------------------------------- */
/* Methods                                                              */

static PyObject*
_proj_proj(ProjObject* self, PyObject* args)
{
  PyObject* in;
  PyObject* out;
  int i, n;
  UV uv;

  if (PyArg_ParseTuple(args, "(dd)", &uv.u, &uv.v)) {

    /* tuple */
    uv = pj_fwd(uv, self->pj);
    return Py_BuildValue("dd", uv.u, uv.v);

  }

  PyErr_Clear();

  /* sequence */
  if (!PyArg_ParseTuple(args, "O", &in))
    return NULL;
  if (!PySequence_Check(in)) {
    PyErr_SetString(PyExc_TypeError, "argument must be sequence");
    return NULL;
  }

  n = PyObject_Length(in);

  out = PyList_New(n);

  for (i = 0; i < n; i++) {

    /* fetch coordinate */
    PyObject* op = PySequence_GetItem(in, i);
    if (!PyArg_ParseTuple(op, "dd", &uv.u, &uv.v)) {
      Py_DECREF(op);
      Py_DECREF(out);
      return NULL;
    }
    Py_DECREF(op);

    uv = pj_fwd(uv, self->pj);

    /* store result */
    op = Py_BuildValue("dd", uv.u, uv.v);
    if (!op) {
      Py_DECREF(out);
      out = NULL;
      break;
    }
    PyList_SetItem(out, i, op);

  }

  return out;
}

static PyObject*
_proj_invproj(ProjObject* self, PyObject* args)
{
  PyObject* in;
  PyObject* out;
  int i, n;
  UV uv;

  if (PyArg_ParseTuple(args, "(dd)", &uv.u, &uv.v)) {

    /* tuple */
    uv = pj_inv(uv, self->pj);
    return Py_BuildValue("dd", uv.u, uv.v);

  }

  PyErr_Clear();

  /* sequence */
  if (!PyArg_ParseTuple(args, "O", &in))
    return NULL;
  if (!PySequence_Check(in)) {
    PyErr_SetString(PyExc_TypeError, "argument must be sequence");
    return NULL;
  }

  n = PyObject_Length(in);

  out = PyList_New(n);

  for (i = 0; i < n; i++) {

    /* fetch coordinate */
    PyObject* op = PySequence_GetItem(in, i);
    if (!PyArg_ParseTuple(op, "dd", &uv.u, &uv.v)) {
      Py_DECREF(op);
      Py_DECREF(out);
      return NULL;
    }
    Py_DECREF(op);

    uv = pj_inv(uv, self->pj);

    /* store result */
    op = Py_BuildValue("dd", uv.u, uv.v);
    if (!op) {
      Py_DECREF(out);
      out = NULL;
      break;
    }
    PyList_SetItem(out, i, op);
  }

  PyErr_Clear();

  return out;
}

static struct PyMethodDef _proj_methods[] =
{
{ "proj", (PyCFunction) _proj_proj, 1 },
{ "invproj", (PyCFunction) _proj_invproj, 1 },
{ NULL, NULL } /* sentinel */
};

static PyObject*
_proj_getattr(ProjObject* s, char *name)
{
  PyObject* res;

  res = Py_FindMethod(_proj_methods, (PyObject*) s, name);
  if (res)
    return res;

  PyErr_Clear();

  /* no attributes */

  PyErr_SetString(PyExc_AttributeError, name);
  return NULL;
}

statichere PyTypeObject Proj_Type
= {
  PyObject_HEAD_INIT(0)
  0, /*ob_size*/
  "Proj", /*tp_name*/
  sizeof(ProjObject), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_proj_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)_proj_getattr, /*tp_getattr*/
  0, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0, /*tp_hash*/
};

/* -------------------------------------------------------------------- */
/* Helpers                                                              */

static PyObject*
_proj_dmstor(PyObject* self, PyObject* args)
{
  char *s;
  if (!PyArg_ParseTuple(args, "s", &s))
    return NULL;

  return Py_BuildValue("d", dmstor(s, NULL));
}

/* -------------------------------------------------------------------- */

static PyMethodDef functions[] =
{

{ "proj", _proj_new, 1 },
{ "dmstor", _proj_dmstor, 1 },
{ NULL, NULL }

};

PyMODINIT_FUNC init_proj(void)
{
  PyObject* m;

  /* Patch object type */
  Proj_Type.ob_type = &PyType_Type;

  /* Initialize module object */
  m = Py_InitModule("_proj", functions);

  /* Create error object */
  PyProj_Error = PyString_FromString("proj.error");
  if (PyProj_Error == NULL || PyDict_SetItemString(PyModule_GetDict(m),
                                                   "error", PyProj_Error) != 0)
    Py_FatalError("can't define proj.error");
}
