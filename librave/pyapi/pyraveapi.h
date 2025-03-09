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
 #ifndef PYRAVEAPI_H
 #define PYRAVEAPI_H

 #include "Python.h"
 #include "rave_value.h"


 RaveValue_t* PyRaveApi_RaveValueFromObject(PyObject* pyobject);

 int PyRaveApi_UpdateRaveValue(PyObject* val, RaveValue_t* ravevalue);

 PyObject* PyRaveApi_RaveValueToObject(RaveValue_t* value);

 #endif