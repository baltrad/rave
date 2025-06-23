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
 * Defines an tiledef, the extent, projection, etc.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-10
 */
#ifndef TILEDEF_H
#define TILEDEF_H
#include "rave_object.h"

/**
 * Defines a Geographical TileDef
 */
typedef struct _TileDef_t TileDef_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType TileDef_TYPE;

/**
 * Sets the ID for this tiledef.
 * @param[in] tiledef - self
 * @param[in] id - the id
 * @returns 1 on success otherwise 0
 */
int TileDef_setID(TileDef_t* tiledef, const char* id);

/**
 * Returns the ID for this tiledef.
 * @param[in] tiledef - self
 * @returns the id
 */
const char* TileDef_getID(TileDef_t* tiledef);

/**
 * Sets the ID for this tiledef.
 * @param[in] tiledef - self
 * @param[in] id - the id
 * @returns 1 on success otherwise 0
 */
int TileDef_setAreaID(TileDef_t* tiledef, const char* id);

/**
 * Returns the ID for this tiledef.
 * @param[in] tiledef - self
 * @returns the id
 */
const char* TileDef_getAreaID(TileDef_t* tiledef);

/**
 * Sets the tiledef extent (lower-left, upper-right)
 * @param[in] tiledef - self
 * @param[in] llX - lower left X position
 * @param[in] llY - lower left Y position
 * @param[in] urX - upper right X position
 * @param[in] urY - upper right Y position
 */
void TileDef_setExtent(TileDef_t* tiledef, double llX, double llY, double urX, double urY);

/**
 * Returns the tiledef extent (lower-left, upper-right)
 * @param[in] tiledef - self
 * @param[out] llX - lower left X position (may be NULL)
 * @param[out] llY - lower left Y position (may be NULL)
 * @param[out] urX - upper right X position (may be NULL)
 * @param[out] urY - upper right Y position (may be NULL)
 */
void TileDef_getExtent(TileDef_t* tiledef, double* llX, double* llY, double* urX, double* urY);

#endif /* TILEDEF_H */
