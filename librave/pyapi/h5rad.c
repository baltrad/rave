/**
 * Accesses various contents of a RAVE INFO object.
 * @file
 * @author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2005-
 */
#include "h5rad.h"
#include "pyravecompat.h"

int GetIntFromINFO(PyObject* inobj, char* key, int* val)
{
   PyObject* pyo;

   pyo = PyObject_CallMethod(inobj, "get", "s", key);

   if(PyInt_Check(pyo)) {
      *val = (int)PyInt_AsLong(pyo);

      Py_XDECREF(pyo);
      return 1;
   }
   Py_XDECREF(pyo);
   PyErr_Clear();
   return 0;
}


int GetDoubleFromINFO(PyObject* inobj, char* key, double* val)
{
   PyObject* pyo;

   pyo = PyObject_CallMethod(inobj, "get", "s", key);

   if(PyFloat_Check(pyo)) {
      *val = PyFloat_AsDouble(pyo);

      Py_XDECREF(pyo);
      return 1;
   }
   Py_XDECREF(pyo);
   PyErr_Clear();
   return 0;
}


int GetStringFromINFO(PyObject* inobj, char* key, char** val)
{
   PyObject* pyo;

   pyo = PyObject_CallMethod(inobj, "get", "s", key);

   if(PyString_Check(pyo)) {
      *val = PyString_AsString(pyo);

      Py_XDECREF(pyo);
      return 1;
   }
   Py_XDECREF(pyo);
   PyErr_Clear();
   return 0;
}


PyObject* GetPyStringFromINFO(PyObject* inobj, char* key)
{
   PyObject* pyo;

   pyo = PyObject_CallMethod(inobj, "get", "s", key);

   if(PyString_Check(pyo)) {
      return pyo;
   }
   Py_XDECREF(pyo);
   PyErr_Clear();
   return Py_None;
}


/*
  Return sequence item from inobj using 'key'.
*/
PyObject* GetSequenceFromINFO(PyObject* inobj, char* key)
{
   PyObject *pyo, *pys;

   pyo = PyObject_CallMethod(inobj, "eval", "s", key);

   if (PySequence_Size(pyo)>0) {

      pys = Py_BuildValue("O", pyo);

      Py_XDECREF(pyo);
      //Py_XDECREF(pys);
      return pys;
   }
   PyErr_Clear();
   return Py_None;
}


/*
  Gets a C double from sequence inobj at index i.
*/
int GetDoubleFromSequence(PyObject* inobj, int i, double* val)
{
   PyObject* item=Py_None;
   char *tmps;
   int size;

   size = PySequence_Size(inobj);

   if ( (i<size) && (size>0) )
      item = PySequence_GetItem(inobj, i);

   if (PyString_Check(item)) {
      tmps = PyString_AsString(item);

      *val = atof(tmps);

      Py_XDECREF(item);
      return 1;
   }
   PyErr_Clear();
   return 0;
}


/*
  Gets a C int from sequence inobj at index i.
*/
int GetIntFromSequence(PyObject* inobj, int i, int* val)
{
   PyObject* item=Py_None;
   char *tmps;
   int size;

   size = PySequence_Size(inobj);

   if ( (i<size) && (size>0) )
      item = PySequence_GetItem(inobj, i);

   if (PyString_Check(item)) {
      tmps = PyString_AsString(item);

      *val = atoi(tmps);

      Py_XDECREF(item);
      return 1;
   }
   PyErr_Clear();
   return 0;
}


/*
  Return unicode string item from inobj using 'key'.
  DO NOT READ DIRECTLY USING THIS FUNCTION! It will leak!
*/
PyObject* getUnicodeFromINFO(PyObject* inobj, char* key)
{
   PyObject* pyo;

   pyo = PyObject_CallMethod(inobj, "findtext", "s", key);
   return pyo;
}
