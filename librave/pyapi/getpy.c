/**
 * Helper code for accessing python objects.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 1998-
 */
#include "getpy.h"
#include "pyravecompat.h"
#include "rave_alloc.h"

#ifdef USE_PROJ4_API
PJ* initProjection(PyObject* pcs)
{
  char** argv;
  int i,n;
  PJ* pj;

  n = PyObject_Length(pcs);
  argv = RAVE_MALLOC(n*sizeof(char*));

  for(i=0;i<n;i++) {
    PyObject* op = PySequence_GetItem(pcs,i);
    PyObject* str = PyObject_Str(op);
    argv[i] = (char*)PyString_AsString(str);
    Py_DECREF(str);
    Py_DECREF(op);
  }

  PyErr_Clear();

  pj = pj_init(n,argv);

  RAVE_FREE(argv);

  return pj;
}

void freeProjection(PJ* pj)
{
  if (pj != NULL)
    pj_free(pj);
}

#else
PJ* initProjection(PyObject* pcs)
{
  char pcsdef[1025];
  int i,n;
  PJ* pj;

  n = PyObject_Length(pcs);
  memset(pcsdef,0,sizeof(pcsdef));

  for(i=0;i<n;i++) {
    PyObject* op = PySequence_GetItem(pcs,i);
    PyObject* str = PyObject_Str(op);
    strcat(pcsdef, (char*)PyString_AsString(str));
    strcat(pcsdef, " ");
    Py_DECREF(str);
    Py_DECREF(op);
  }

  PyErr_Clear();
  pj = proj_create(PJ_DEFAULT_CTX, pcsdef);

  return pj;
}

void freeProjection(PJ* pj)
{
  if (pj != NULL)
    proj_destroy(pj);
}

#endif

int getDoubleFromDictionary(char* name,double* val,PyObject* dictionary)
{
  PyObject* pyo;

  pyo = PyMapping_GetItemString(dictionary,name);

  if(pyo) {
    (*val) = PyFloat_AsDouble(pyo);
    Py_DECREF(pyo);
    return 1;
  }

  PyErr_Clear();

  return 0;
}

int getIntFromDictionary(char* name, int* val, PyObject* dictionary)
{
  PyObject* pyo;

  pyo = PyMapping_GetItemString(dictionary,name);

  if(pyo) {
    (*val) = PyInt_AsLong(pyo);
    Py_DECREF(pyo);
    return 1;
  }

  PyErr_Clear();

  return 0;
}

int getDoubleFromInstance(char* name,double* val,PyObject* instance)
{
  PyObject* pyo;

  pyo = PyObject_GetAttrString(instance,name);

  if(pyo) {
    (*val) = PyFloat_AsDouble(pyo);
    Py_DECREF(pyo);
    return 1;
  }

  PyErr_Clear();
  return 0;
}

int getIntFromInstance(char* name, int* val,PyObject* instance)
{
  PyObject* pyo;

  pyo = PyObject_GetAttrString(instance,name);

  if(pyo) {
    (*val) = PyInt_AsLong(pyo);
    Py_DECREF(pyo);
    return 1;
  }

  PyErr_Clear();
  return 0;
}


int getIdxDoubleFromTuple(int idx, double* val,PyObject* tuple)
{
  PyObject* pyo;

  pyo = PySequence_GetItem(tuple,idx);

  if(pyo) {
    (*val) = PyFloat_AsDouble(pyo);
    Py_DECREF(pyo);
    return 1;
  }

  PyErr_Clear();
  return 0;
}

int getIdxIntFromTuple(int idx, int* val,PyObject* tuple)
{
  PyObject* pyo;

  pyo = PySequence_GetItem(tuple,idx);

  if(pyo) {
    (*val) = PyInt_AsLong(pyo);
    Py_DECREF(pyo);
    return 1;
  }

  PyErr_Clear();
  return 0;
}

/*
  Appends a double to a list.
*/
int AppendFloatToList(PyObject* list, double val)
{
   PyObject* tmp;
   tmp = Py_BuildValue("f", val);

   if (!PyList_Append(list, tmp)) {
      Py_XDECREF(tmp);
      PyErr_Clear();
      return 1;
   }
   Py_XDECREF(tmp);
   PyErr_Clear();
   return 0;
}


/*
  Inserts double val into a list at position i.
*/
int InsertFloatInList(PyObject* list, int i, double val)
{
   PyObject* tmp;
   tmp = Py_BuildValue("f", val);

   if (!PyList_Insert(list, i, tmp)) {
      Py_XDECREF(tmp);
      PyErr_Clear();
      return 1;
   }
   Py_XDECREF(tmp);
   PyErr_Clear();
   return 0;
}


/*
  Extracts PyFloat from a list at position i and returns it as a double.
*/
double getDoubleFromList(PyObject* list, int i)
{
   double val=0.0;
   PyObject* tmp;

   tmp = PyList_GetItem(list, i);
   val = PyFloat_AsDouble(tmp);
   PyErr_Clear();
   return val;
}
