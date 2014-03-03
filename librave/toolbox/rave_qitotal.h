/* --------------------------------------------------------------------
Copyright (C) 2014 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Implementation of the QI-total algorithm
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2014-02-27
 */
#ifndef RAVE_QITOTAL_H
#define RAVE_QITOTAL_H
#include "rave_object.h"
#include "raveobject_list.h"
#include "rave_field.h"

/**
 * Defines QI total
 */
typedef struct _RaveQITotal_t RaveQITotal_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RaveQITotal_TYPE;

void RaveQITotal_setDatatype(RaveQITotal_t* self, RaveDataType dtype);

RaveDataType RaveQITotal_getDatatype(RaveQITotal_t* self);

int RaveQITotal_setGain(RaveQITotal_t* self, double gain);

double RaveQITotal_getGain(RaveQITotal_t* self);

void RaveQITotal_setOffset(RaveQITotal_t* self, double offset);

double RaveQITotal_getOffset(RaveQITotal_t* self);

RaveField_t* RaveQITotal_multiplicative(RaveQITotal_t* self, RaveObjectList_t* fields);

RaveField_t* RaveQITotal_additive(RaveQITotal_t* self, RaveObjectList_t* fields);

RaveField_t* RaveQITotal_minimum(RaveQITotal_t* self, RaveObjectList_t* fields);

#endif /* RAVE_QITOTAL_H */
