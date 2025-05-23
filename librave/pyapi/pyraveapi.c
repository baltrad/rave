/* --------------------------------------------------------------------
Copyright (C) 2025 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Tools for integrating python and rave object apis.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-02-14
 */
#include "pyraveapi.h"
#include "object.h"
#include "pyerrors.h"
#include "pyravecompat.h"
#include "rave_alloc.h"
#include "rave_list.h"
#include "rave_object.h"
#include "rave_value.h"
#include "raveobject_hashtable.h"

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


RaveValue_t* PyRaveApi_RaveValueFromObject(PyObject* pyobject)
{
  RaveValue_t* ravevalue = RAVE_OBJECT_NEW(&RaveValue_TYPE);
  if (ravevalue == NULL) {
    raiseException_returnNULL(PyExc_MemoryError, "Could not create RaveValue_t");
  }
  if (!PyRaveApi_UpdateRaveValue(pyobject, ravevalue)) {
    RAVE_OBJECT_RELEASE(ravevalue);
  }
  return ravevalue;
}
 
int PyRaveApi_UpdateRaveValue(PyObject* val, RaveValue_t* ravevalue)
{
  int result = 0;

  if (val == Py_None) {
    RaveValue_setNull(ravevalue);
  } else if (PyString_Check(val)) {
    if (!RaveValue_setString(ravevalue, PyString_AsString(val))) {
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to set string");
    }
  } else if (PyBool_Check(val)) {
    RaveValue_setBoolean(ravevalue, PyInt_AsLong(val));
  } else if (PyInt_Check(val)) {
    RaveValue_setLong(ravevalue, PyInt_AsLong(val));
  } else if (PyFloat_Check(val)) {
    RaveValue_setDouble(ravevalue, PyFloat_AsDouble(val));
  } else if (PyList_Check(val) || PyTuple_Check(val)) {
    Py_ssize_t nvalues = PyObject_Length(val);
    int i = 0;
    RaveObjectList_t* rlist = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
    if (rlist == NULL) {
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to create native list");
    }
    for (i = 0; i < nvalues; i++) {
      PyObject* pyobj = NULL;
      if (PyList_Check(val)) {
        pyobj = PyList_GetItem(val, i);  /* We don't need to release this since it is internal pointer.*/
      } else {
        pyobj = PyTuple_GetItem(val, i);  /* We don't need to release this since it is internal pointer.*/
      }

      if (pyobj != NULL) {
        RaveValue_t* rvalue = PyRaveApi_RaveValueFromObject(pyobj);
        if (rvalue == NULL || !RaveObjectList_add(rlist, (RaveCoreObject*)rvalue)) {
          Py_XDECREF(pyobj);
          RAVE_OBJECT_RELEASE(rlist);
          RAVE_OBJECT_RELEASE(rvalue);
          raiseException_gotoTag(done, PyExc_AttributeError, "Could not create rave value from object or add it to list");
        }
        RAVE_OBJECT_RELEASE(rvalue);
      }
    }
    if (!RaveValue_setList(ravevalue, rlist)) {
      RAVE_OBJECT_RELEASE(rlist);
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to set list in rave value");
    }
    RAVE_OBJECT_RELEASE(rlist);    
  } else if (PyMapping_Check(val)) {
    PyObject* keys = NULL, *ko = NULL, *hashval = NULL;
    RaveObjectHashTable_t* hashtable = NULL;
    Py_ssize_t len = 0, i = 0;    

    keys = PyMapping_Keys(val);
    if (keys == NULL) {
      raiseException_gotoTag(done, PyExc_AttributeError, "Could not get keys from mapping");
    }
  
    hashtable = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
    if (hashtable == NULL) {
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to create native mapping");
    }
  
    len = PyList_Size(keys);
    for (i = 0; i < len; i++) {
      const char* key;
      ko = PyList_GetItem(keys, i); /* borrowed */
      key = PyString_AsString(ko);
      if (key != NULL) {
        hashval = PyMapping_GetItemString(val, key);
        if (hashval != NULL) {
          RaveValue_t* rvalue = PyRaveApi_RaveValueFromObject(hashval);
          if (rvalue == NULL || !RaveObjectHashTable_put(hashtable, key, (RaveCoreObject*)rvalue)) {
            Py_XDECREF(hashval);
            RAVE_OBJECT_RELEASE(hashtable);
            RAVE_OBJECT_RELEASE(rvalue);
            raiseException_gotoTag(done, PyExc_AttributeError, "Could not create rave value from object or add it to hash");
          }
          RAVE_OBJECT_RELEASE(rvalue);
        }
        Py_DECREF(hashval);
      } else {
        raiseException_gotoTag(done, PyExc_AttributeError, "Could not aquire key from hash");
      }
    }
    if (!RaveValue_setHashTable(ravevalue, hashtable)) {
      RAVE_OBJECT_RELEASE(hashtable);
      raiseException_gotoTag(done, PyExc_MemoryError, "Failed to set hash in rave value");
    }
    RAVE_OBJECT_RELEASE(hashtable);
  }
  result = 1;
done:
  return result;
}

PyObject* PyRaveApi_RaveValueToObject(RaveValue_t* value)
{
  if (RaveValue_type(value) == RaveValue_Type_Undefined || RaveValue_type(value) == RaveValue_Type_Null) {
    Py_RETURN_NONE;
  } else if (RaveValue_type(value) == RaveValue_Type_String) {
    return PyString_FromString(RaveValue_toString(value));
  } else if (RaveValue_type(value) == RaveValue_Type_Long) {
    return PyLong_FromLong(RaveValue_toLong(value));
  } else if (RaveValue_type(value) == RaveValue_Type_Double) {
    return PyFloat_FromDouble(RaveValue_toDouble(value));
  } else if (RaveValue_type(value) == RaveValue_Type_Boolean) {
    return PyBool_FromLong(RaveValue_toBoolean(value));
  } else if (RaveValue_type(value) == RaveValue_Type_List) {    
    PyObject* result = NULL;
    RaveObjectList_t* rlist = RaveValue_toList(value);

    int i = 0, rlen = RaveObjectList_size(rlist);
    result = PyList_New(0);
    if (result == NULL) {
      raiseException_returnNULL(PyExc_MemoryError, "Could not allocate memory for list");
    }
    for (i = 0; i < rlen; i++) {
      RaveValue_t* rvalue = (RaveValue_t*)RaveObjectList_get(rlist, i);
      if (rvalue != NULL) {
        PyObject* obj = PyRaveApi_RaveValueToObject(rvalue);
        if (obj == NULL || PyList_Append(result, obj) != 0) {
          Py_XDECREF(result);
          Py_XDECREF(obj);
          RAVE_OBJECT_RELEASE(rvalue);
          RAVE_OBJECT_RELEASE(rlist);
          raiseException_returnNULL(PyExc_MemoryError, "failed to create list object");
        }
      }
      RAVE_OBJECT_RELEASE(rvalue);
    }
    RAVE_OBJECT_RELEASE(rlist);
    return result;
  } else if (RaveValue_type(value) == RaveValue_Type_Hashtable) {
    RaveObjectHashTable_t* hashtable = NULL;
    RaveList_t* keys = NULL;
    PyObject* result = NULL;
    int failed = 0;
    
    result = PyDict_New();
    if (result == NULL) {
      raiseException_returnNULL(PyExc_MemoryError, "Could not create python dictionary");
    }
    hashtable = RaveValue_toHashTable(value);
    keys = RaveObjectHashTable_keys(hashtable);
    if (keys != NULL) {
      int i = 0, nlen = RaveList_size(keys);
      for (i = 0; !failed && i < nlen; i++) {
        const char* key = (const char*)RaveList_get(keys, i);
        RaveCoreObject* obj = RaveObjectHashTable_get(hashtable, key);
        if (obj != NULL && RAVE_OBJECT_CHECK_TYPE(obj, &RaveValue_TYPE)) {
          PyObject* pyobj = PyRaveApi_RaveValueToObject((RaveValue_t*)obj);
          if (pyobj != NULL) {
            if (PyDict_SetItemString(result, key, pyobj) < 0) {
              failed = 1;
              PyErr_SetString(PyExc_RuntimeError, "Failed to set item string in dictionary");
            }
          } else {
            failed = 1;
            PyErr_SetString(PyExc_RuntimeError, "Failed to create python object from rave value");
          }
          Py_XDECREF(pyobj);
        } else {
          failed = 1;
          PyErr_SetString(PyExc_RuntimeError, "Hash table item is not a rave value or is NULL");
        }
        RAVE_OBJECT_RELEASE(obj);
      }
      RaveList_freeAndDestroy(&keys);
    } else {
      PyErr_SetString(PyExc_RuntimeError, "failed to list keys in hashtable");
    }
    RAVE_OBJECT_RELEASE(hashtable);
    if (keys != NULL) {
      RaveList_freeAndDestroy(&keys);
    }

    if (failed) {
      Py_XDECREF(result);
      return NULL;
    }

    return result;
  }

  return NULL;
}
