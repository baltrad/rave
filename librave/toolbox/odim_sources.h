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
 * Provides support for reading the odim sources from an xml-file
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-15
 */
#ifndef ODIMSOURCES_H
#define ODIMSOURCES_H
#include "rave_object.h"
#include "odim_source.h"
#include "rave_list.h"
/**
 * Defines the odim sources
 */
typedef struct _OdimSources_t OdimSources_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType OdimSources_TYPE;

/**
 * Simplified loading function, takes filename and a projection registry
 * @param[in] filename - the area file name
 * @param[in] pRegistry - the projection registry
 * @returns an area registry
 */
OdimSources_t* OdimSources_load(const char* filename);

/**
 * Adds an area to the registry. There will be no check if the
 * area already exists so it is up to the user to ensure that
 * no duplicates are added.
 *
 * @param[in] self - self
 * @param[in] area - the area to add
 * @return 1 on success otherwise 0
 */
int OdimSources_add(OdimSources_t* self, OdimSource_t* source);

/**
 * Returns the number of registered sour
 es
 * @param[in] self - self
 * @returns the number of registered sources
 */
int OdimSources_size(OdimSources_t* self);

/**
 * Returns the source with specified nod name
 * @param[in] self - self
 * @param[in] nod - the NOD identifier
 * @returns the source or NULL if no such source
 */
OdimSource_t* OdimSources_get(OdimSources_t* self, const char* nod);

/**
 * Returns the source with specified wmo identifier
 * @param[in] self - self
 * @param[in] wmo - the wmo identifier (5-digits)
 * @returns the source or NULL if no such source
 */
OdimSource_t* OdimSources_get_wmo(OdimSources_t* self, const char* wmo);

/**
 * Returns the source with specified wigos identifier
 * @param[in] self - self
 * @param[in] wmo - the wigos identifier
 * @returns the source or NULL if no such source
 */
OdimSource_t* OdimSources_get_wigos(OdimSources_t* self, const char* wigos);

/**
 * Returns the source with specified rad identifier
 * @param[in] self - self
 * @param[in] rad - the rad identifier
 * @returns the source or NULL if no such source
 */
OdimSource_t* OdimSources_get_rad(OdimSources_t* self, const char* rad);

/**
 * Returns the source with specified plc identifier
 * @param[in] self - self
 * @param[in] plc - the plc identifier
 * @returns the source or NULL if no such source
 */
OdimSource_t* OdimSources_get_plc(OdimSources_t* self, const char* plc);

/**
 * Returns the source with specified nod name. The source string will be checked in the order
 * NOD, WIGOS, WMO, RAD and finally PLC. If WMO:00000, then WMO will not be used as identifier.
 * @param[in] self - self
 * @param[in] sourcestr - the ODIM source string
 * @returns the source or NULL if no such source could be identified
 */
OdimSource_t* OdimSources_identify(OdimSources_t* self, const char* sourcestr);

/**
 * Return all NODs that has been registered in this source registry.
 * Remember to use \@ref #RaveList_freeAndDestroy that will deallocate the list for you.
 * @param[in] self - self
 * @return a list of nods (char* pointers). 
 */
RaveList_t* OdimSources_nods(OdimSources_t* self);

#endif /* AREAREGISTRY_H */
