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

#include "pyravecompat.h"

#ifdef USE_PROJ4_API
#include <projects.h>
#ifdef PJ_VERSION
#define UV projUV
#endif
#else
#include <proj.h>
#define UV PJ_UV
#endif

typedef struct {
  PyObject_HEAD
  PJ* pj;
#ifndef USE_PROJ4_API
  PJ_CONTEXT* context;
#endif
} ProjObject;

static PyTypeObject Proj_Type;

static PyObject *PyProj_Error;

/* -------------------------------------------------------------------- */
/* Constructors                                                         */

static void _proj_error(ProjObject* po)
{
#ifdef USE_PROJ4_API
  PyErr_SetString(PyProj_Error, pj_strerrno(pj_errno));
#else
  int e;
  if (po != NULL && po->context != NULL) {
    e = proj_context_errno(po->context);
  } else {
    e = proj_errno(0);
  }
  PyErr_SetString(PyProj_Error, proj_errno_string(e));
#endif
}

static PyObject*
_proj_new(PyObject* self, PyObject* args)
{
  ProjObject* op = NULL;
  PyObject* in = NULL;
  int n, i;
#ifdef USE_PROJ4_API
  char** argv;
#else
  char pcsdef[1025];
  memset(pcsdef, 0, sizeof(pcsdef));
#endif

  if (!PyArg_ParseTuple(args, "O", &in))
    return NULL;

  if (!PySequence_Check(in)) {
    PyErr_SetString(PyExc_TypeError, "argument must be sequence");
    return NULL;
  }
  op = PyObject_NEW(ProjObject, &Proj_Type);
  if (op == NULL)
    return NULL;
  ((ProjObject*)op)->pj = NULL;
#ifndef USE_PROJ4_API
  ((ProjObject*)op)->context = NULL;
#endif

  n = PyObject_Length(in);

  /* fetch argument array */
#ifdef USE_PROJ4_API
  argv = malloc(n * sizeof(char*));
  if (argv == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for arguments");
    Py_DECREF(op);
    return NULL;
  }
#endif
  for (i = 0; i < n; i++) {
    PyObject* pso = PySequence_GetItem(in, i);
    PyObject* str = PyObject_Str(pso);
#ifdef USE_PROJ4_API
    argv[i] = PyString_AsString(str);
#else
    strcat(pcsdef, PyString_AsString(str));
    strcat(pcsdef, " ");
#endif
    Py_DECREF(str);
    Py_DECREF(pso);
  }

#ifdef USE_PROJ4_API
  op->pj = pj_init(n, argv);
  free(argv);
#else
  op->context = proj_context_create();
  if (op->context != NULL) {
    proj_log_level(op->context, PJ_LOG_NONE);
    op->pj = proj_create(op->context, pcsdef);
  }
#endif

  if (!op->pj) {
    PyObject_Free(op);
    _proj_error(op);
    return NULL;
  }

  return (PyObject*) op;
}

static void _proj_dealloc(ProjObject* op)
{
#ifdef USE_PROJ4_API
  if (op->pj) {
    pj_free(op->pj);
  }
#else
  if (op->pj != NULL) {
    proj_destroy(op->pj);
  }
  if (op->context != NULL) {
    proj_context_destroy(op->context);
  }
#endif

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

#ifndef USE_PROJ4_API
  PJ_COORD c, outc;
#endif

  if (PyArg_ParseTuple(args, "(dd)", &uv.u, &uv.v)) {
    /* tuple */
#ifdef USE_PROJ4_API
    uv = pj_fwd(uv, self->pj);
#else
    c.uv = uv;
    outc = proj_trans(self->pj, PJ_FWD, c);
    uv = outc.uv;
#endif
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

#ifdef USE_PROJ4_API
    uv = pj_fwd(uv, self->pj);
#else
    c.uv = uv;
    outc = proj_trans(self->pj, PJ_FWD, c);
    uv = outc.uv;
#endif

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
#ifndef USE_PROJ4_API
  PJ_COORD c, outc;
#endif

  if (PyArg_ParseTuple(args, "(dd)", &uv.u, &uv.v)) {
#ifdef USE_PROJ4_API
    uv = pj_inv(uv, self->pj);
#else
    c.uv = uv;
    outc = proj_trans(self->pj, PJ_INV, c);
    uv = outc.uv;
#endif
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

#ifdef USE_PROJ4_API
    uv = pj_inv(uv, self->pj);
#else
    c.uv = uv;
    outc = proj_trans(self->pj, PJ_INV, c);
    uv = outc.uv;
#endif

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

static PyObject* _proj_getattro(ProjObject* s, PyObject* name)
{
  return PyObject_GenericGetAttr((PyObject*)s, name);
}

static PyTypeObject Proj_Type = {
  PyVarObject_HEAD_INIT(NULL, 0) /*ob_size*/
  "Proj", /*tp_name*/
  sizeof(ProjObject), /*tp_size*/
  0, /*tp_itemsize*/
  /* methods */
  (destructor)_proj_dealloc, /*tp_dealloc*/
  0, /*tp_print*/
  (getattrfunc)0 /*_proj_getattr*/, /*tp_getattr*/
  0, /*tp_setattr*/
  0, /*tp_compare*/
  0, /*tp_repr*/
  0,                            /*tp_as_number */
  0,
  0,                            /*tp_as_mapping */
  0, /*tp_hash*/
  (ternaryfunc)0,               /*tp_call*/
  (reprfunc)0,                  /*tp_str*/
  (getattrofunc)_proj_getattro, /*tp_getattro*/
  (setattrofunc)0,              /*tp_setattro*/
  0,                            /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT, /*tp_flags*/
  0,                            /*tp_doc*/
  (traverseproc)0,              /*tp_traverse*/
  (inquiry)0,                   /*tp_clear*/
  0,                            /*tp_richcompare*/
  0,                            /*tp_weaklistoffset*/
  0,                            /*tp_iter*/
  0,                            /*tp_iternext*/
  _proj_methods,                /*tp_methods*/
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

/* -------------------------------------------------------------------- */
/* Helpers                                                              */

static PyObject*
_proj_dmstor(PyObject* self, PyObject* args)
{
  char *s;
  if (!PyArg_ParseTuple(args, "s", &s))
    return NULL;

#ifdef USE_PROJ4_API
  return Py_BuildValue("d", dmstor(s, NULL));
#else
  return Py_BuildValue("d", proj_dmstor(s, NULL));
#endif
}

/* -------------------------------------------------------------------- */

static PyMethodDef functions[] =
{

{ "proj", _proj_new, 1 },
{ "dmstor", _proj_dmstor, 1 },
{ NULL, NULL }

};

MOD_INIT(_proj)
{
  PyObject* module = NULL;

  MOD_INIT_SETUP_TYPE(Proj_Type, &PyType_Type);

  /* Initialize module object */
  MOD_INIT_DEF(module, "_proj", NULL/*doc*/, functions);
  if (module == NULL) {
    return MOD_INIT_ERROR;
  }

  /* Create error object */
  PyProj_Error = PyErr_NewException("proj.error", NULL, NULL);
  if (PyProj_Error == NULL || PyDict_SetItemString(PyModule_GetDict(module), "error", PyProj_Error) != 0) {
    Py_FatalError("Can't define _proj.error");
    return MOD_INIT_ERROR;
  }

  return MOD_INIT_SUCCESS(module);
}
