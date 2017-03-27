/* --------------------------------------------------------------------
Copyright (C) 2017 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * utilities that are helpful for the transition to python 3
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2017-02-13
 */

#include "pyravecompat.h"


int PyRaveAPI_CompareWithASCIIString(PyObject* ptr, const char* name)
{
  int result = -1;
  if (!PyString_Check(ptr)){
#ifdef Py_USING_UNICODE
    if (PyUnicode_Check(ptr)) {
      PyObject* tmp = PyUnicode_AsEncodedString(ptr, NULL, NULL);
      if (tmp != NULL) {
        result = strcmp(PyString_AsString(tmp), name);
        Py_DecRef(tmp);
      }
    }
#endif
  } else {
    result = strcmp(PyString_AsString(ptr), name);
  }
  return result;
}

PyObject* PyRaveAPI_StringOrUnicode_FromASCII(const char *buffer)
{
#if PY_MAJOR_VERSION >= 3
  Py_ssize_t size = (Py_ssize_t)strlen(buffer);
  const unsigned char *s = (const unsigned char *)buffer;
  PyObject *unicode = NULL;
  unicode = PyUnicode_New(size, 127);
  if (!unicode) {
    return NULL;
  }
  memcpy(PyUnicode_1BYTE_DATA(unicode), s, size);
  return unicode;
#else
  return PyString_FromString(buffer);
#endif
}
