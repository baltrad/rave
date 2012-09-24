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
 * Provides support for reading and writing projections to and from
 * an xml-file.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-12-09
 */
#ifndef PROJECTIONREGISTRY_H
#define PROJECTIONREGISTRY_H
#include "projection.h"
#include "rave_object.h"

/**
 * Defines the projection registry
 */
typedef struct _ProjectionRegistry_t ProjectionRegistry_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType ProjectionRegistry_TYPE;

/**
 * Loads a registry from an xml file
 * @param[in] filename - the name of the xml file
 * @returns the projection registry
 */
ProjectionRegistry_t* ProjectionRegistry_load(const char* filename);

/**
 * Adds a projection to the registry. There will be no check if the
 * projection already exists so it is up to the user to ensure that
 * no duplicates are added.
 *
 * @param[in] self - self
 * @param[in] proj - the projection to add
 * @return 1 on success otherwise 0
 */
int ProjectionRegistry_add(ProjectionRegistry_t* self, Projection_t* proj);

/**
 * Returns the number of projections in this registry
 * @param[in] self - self
 * @returns the number of projections
 */
int ProjectionRegistry_size(ProjectionRegistry_t* self);

/**
 * Returns the projection at specified position
 * @param[in] self - self
 * @param[in] index - the position of the projection
 * @returns the found projection or NULL
 */
Projection_t* ProjectionRegistry_get(ProjectionRegistry_t* self, int index);

/**
 * Returns the projection with the specified pcs id
 * @param[in] self - self
 * @param[in] pcsid - the projection id
 * @returns the found projection or NULL
 */
Projection_t* ProjectionRegistry_getByName(ProjectionRegistry_t* self, const char* pcsid);

/**
 * Removes the projection at the specified index
 * @param[in] self - self
 * @param[in] index - the index of the projection to remove
 */
void ProjectionRegistry_remove(ProjectionRegistry_t* self, int index);

/**
 * Removes the projection with the specified pcs id
 * @param[in] self - self
 * @param[in] pcsid - the projection id of the projection to remove
 */
void ProjectionRegistry_removeByName(ProjectionRegistry_t* self, const char* pcsid);

/**
 * Writes the current registry to a xml file.
 * @param[in] self - self
 * @param[in] filename - the name of the file
 * @returns 1 on success or 0 on failure
 */
int ProjectionRegistry_write(ProjectionRegistry_t* self, const char* filename);

#endif /* PROJECTIONREGISTRY_H */
