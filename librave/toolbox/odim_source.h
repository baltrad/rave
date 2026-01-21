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
 * Defines an odim source.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-12-10
 */
#ifndef ODIMSOURCE_H
#define ODIMSOURCE_H
#include "rave_object.h"

/**
 * Defines a Geographical Area
 */
typedef struct _OdimSource_t OdimSource_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType OdimSource_TYPE;

/**
 * Extracts an ID from the ODIM source string. For example if the source is "NOD:sekkr,WMO:01234".
 * Then the call OdimSource_getIdFromOdimSource(source, "WMO:") will return 01234
 */
char* OdimSource_getIdFromOdimSource(const char* source, const char* id);

/**
 * Extracts an ID from the ODIM source string inclusive the identifier. For example if the source is "NOD:sekkr,WMO:01234".
 * Then the call OdimSource_getIdFromOdimSource(source, "WMO:") will return WMO:01234
 */
char* OdimSource_getIdFromOdimSourceInclusive(const char* source, const char* id);

/**
 * Utility function to create a complete OdimSource_t in one line.
 * @param[in] nod - The NOD (mandatory)
 * @param[in] wmo - The WMO (mandatory)
 * @param[in] wigos - The WIGOS (mandatory)
 * @param[in] plc - The PLC (mandatory)
 * @param[in] rad - The RAD (mandatory)
 * @param[in] cccc - The CCCC (mandatory)
 * @param[in] org - The ORG (mandatory)
 * @return a created odim source
 */
OdimSource_t* OdimSource_create(const char* nod, const char* wmo, const char* wigos, const char* plc, const char* rad, const char* cccc, const char* org);

/**
 * Sets the NOD for this source.
 * @param[in] self - self
 * @param[in] nod - the nod
 * @returns 1 on success otherwise 0
 */
int OdimSource_setNod(OdimSource_t* self, const char* nod);

/**
 * Returns the NOD for this source.
 * @param[in] self - self
 * @returns the id
 */
const char* OdimSource_getNod(OdimSource_t* self);

/**
 * Sets the WMO for this source.
 * @param[in] self - self
 * @param[in] wmoid - the id
 * @returns 1 on success otherwise 0
 */
int OdimSource_setWmo(OdimSource_t* self, const char* wmoid);

/**
 * Returns the WMO for this source.
 * @param[in] self - self
 * @returns the wmo id
 */
const char* OdimSource_getWmo(OdimSource_t* self);

/**
 * Sets the WIGOS for this source.
 * @param[in] self - self
 * @param[in] wigos - the id
 * @returns 1 on success otherwise 0
 */
int OdimSource_setWigos(OdimSource_t* self, const char* wigos);

/**
 * Returns the WIGOS for this source.
 * @param[in] self - self
 * @returns the wigos id
 */
const char* OdimSource_getWigos(OdimSource_t* self);

/**
 * Sets the PLC for this source.
 * @param[in] self - self
 * @param[in] plc - the id
 * @returns 1 on success otherwise 0
 */
int OdimSource_setPlc(OdimSource_t* self, const char* plc);

/**
 * Returns the PLC for this source.
 * @param[in] self - self
 * @returns the plc id
 */
const char* OdimSource_getPlc(OdimSource_t* self);

/**
 * Sets the RAD for this source.
 * @param[in] self - self
 * @param[in] rad - the id
 * @returns 1 on success otherwise 0
 */
int OdimSource_setRad(OdimSource_t* self, const char* rad);

/**
 * Returns the RAD for this source.
 * @param[in] self - self
 * @returns the rad id
 */
const char* OdimSource_getRad(OdimSource_t* self);

/**
 * Sets the CCCC for this source.
 * @param[in] self - self
 * @param[in] cccc - the id
 * @returns 1 on success otherwise 0
 */
int OdimSource_setCccc(OdimSource_t* self, const char* cccc);

/**
 * Returns the CCCC for this source.
 * @param[in] self - self
 * @returns the cccc id
 */
const char* OdimSource_getCccc(OdimSource_t* self);

/**
 * Sets the ORG for this source.
 * @param[in] self - self
 * @param[in] org - the id
 * @returns 1 on success otherwise 0
 */
int OdimSource_setOrg(OdimSource_t* self, const char* org);

/**
 * Returns the ORG for this source.
 * @param[in] self - self
 * @returns the org id
 */
const char* OdimSource_getOrg(OdimSource_t* self);

/**
 * Returns the source string for this source.
 * Will be in the format. NOD:<nod>,WMO:<wmo>,... If WMO = 00000 then it will not be added.
 * If RAD,WIGOS,PLC is missing, they will not be added to the source string.
 * @param[in] self - self
 * @return the odim source string 
 */
const char* OdimSource_getSource(OdimSource_t* self);

#endif /* ODIMSOURCE_H */
