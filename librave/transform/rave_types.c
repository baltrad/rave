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
 * Type definitions for RAVE
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-17
 */
#include "rave_types.h"

/*@{ Interface functions */
int get_ravetype_size(RaveDataType type)
{
  switch(type) {
  case RaveDataType_CHAR:
    return sizeof(char);
  case RaveDataType_UCHAR:
    return sizeof(unsigned char);
  case RaveDataType_SHORT:
    return sizeof(short);
  case RaveDataType_INT:
    return sizeof(int);
  case RaveDataType_LONG:
    return sizeof(long);
  case RaveDataType_FLOAT:
    return sizeof(float);
  case RaveDataType_DOUBLE:
    return sizeof(double);
  default:
    return -1;
  }
}
/*@} End of Interface functions */
