/* --------------------------------------------------------------------
Copyright (C) 2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Provides support for reading and writing areas to and from
 * an xml-file.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-12-08
 */
#ifndef TILEREGISTRY_H
#define TILEREGISTRY_H
#include "area.h"
#include "tiledef.h"
#include "rave_object.h"
#include "raveobject_list.h"

/**
 * Represents the registry
 */
struct _TileRegistry_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveObjectList_t* tiledefs; /**< the list of tiledefs */
};


/**
 * Defines the area registry
 */
typedef struct _TileRegistry_t TileRegistry_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType TileRegistry_TYPE;

/**
 * Simplified loading function, takes filename and a projection registry
 * @param[in] filename - the area file name
 * @returns an area registry
 */
TileRegistry_t* TileRegistry_load(const char* filename);

/**
 * Adds an area to the registry. There will be no check if the
 * area already exists so it is up to the user to ensure that
 * no duplicates are added.
 *
 * @param[in] self - self
 * @param[in] area - the area to add
 * @return 1 on success otherwise 0
 */
int TileRegistry_add(TileRegistry_t* self, TileDef_t* area);

/**
 * Returns the number of registered areas
 * @param[in] self - self
 * @returns the number of registered areas
 */
int TileRegistry_size(TileRegistry_t* self);

/**
 * Returns the area at specified index
 * @param[in] self - self
 * @param[in] index - the index
 * @returns the area or NULL if no such area
 */
TileDef_t* TileRegistry_get(TileRegistry_t* self, int index);

/**
 * Returns the area with the specified id
 * @param[in] self - self
 * @param[in] id - the area id
 * @returns the found area or NULL
 */
TileDef_t* TileRegistry_getByName(TileRegistry_t* self, const char* id);


/**
 * Returns the tilles with the specified areaid
 * @param[in] self - self
 * @param[in] id - the area id
 * @returns the found area or NULL
 */
RaveObjectList_t * TileRegistry_getByArea(TileRegistry_t* self, const char* areaid);

/**
 * Removes the area at the specified index
 * @param[in] self - self
 * @param[in] index - the index of the area to remove
 */
void TileRegistry_remove(TileRegistry_t* self, int index);

/**
 * Removes the area with the specified id
 * @param[in] self - self
 * @param[in] id - the id of the area to remove
 */
void TileRegistry_removeByName(TileRegistry_t* self, const char* id);

/**
 * Writes the current registry to a xml file.
 * @param[in] self - self
 * @param[in] filename - the name of the file
 * @returns 1 on success or 0 on failure
 */
int TileRegistry_write(TileRegistry_t* self, const char* filename);

#endif /* AREAREGISTRY_H */
