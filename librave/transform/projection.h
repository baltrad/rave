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
 * Wrapper around PROJ.4
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-20
 */
#ifndef PROJECTION_H
#define PROJECTION_H
#include "projects.h"

/**
 * Defines a transformer
 */
typedef struct _Projection_t Projection_t;

/**
 * Creates a new projection instance.
 * @return a new instance or NULL on failure
 */
Projection_t* Projection_new(const char* id, const char* description, const char* definition);

/**
 * Releases the responsibility for the projection, it is not certain that
 * it will be deleted though if there still are references existing
 * to this projection.
 * @param[in] projection - the projection
 */
void Projection_release(Projection_t* projection);

/**
 * Copies the reference to this instance by increasing a
 * reference counter.
 * @param[in] projection - the projection to be copied
 * @return a pointer to the projection
 */
Projection_t* Projection_copy(Projection_t* projection);

/**
 * Returns the ID for this projection.
 * @param[in] projection - the projection
 * @return the ID for this projection
 */
const char* Projection_getID(Projection_t* projection);

/**
 * Returns the description for this projection.
 * @param[in] projection - the projection
 * @return the description for this projection
 */
const char* Projection_getDescription(Projection_t* projection);

/**
 * Returns the definition for this projection.
 * @param[in] projection - the projection
 * @return the definition for this projection
 */
const char* Projection_getDefinition(Projection_t* projection);

/**
 * Transforms the coordinates in this projection into the target projection.
 * @param[in] projection - this projection
 * @param[in] tgt - target projection
 * @param[in,out] x - coordinate
 * @param[in,out] y - coordinate
 * @param[in,out] z - coordinate (MAY BE NULL in some cases), see PROJ.4
 * @param[in] projection - the projection
 * @return 0 on failure, otherwise success
 */
int Projection_transform(Projection_t* projection, Projection_t* tgt, double* x, double* y, double* z);

/**
 *
 */
int Projection_inv(Projection_t* projection, double* x, double* y);

/**
 * Function for keeping track of some sort of metadata that should be
 * associated with this projection.
 * @param[in] ptr - a void pointer that should be stored
 */
void Projection_setVoidPtr(Projection_t* projection, void* ptr);

/**
 * Returns the void data.
 * @return the void data
 */
void* Projection_getVoidPtr(Projection_t* projection);

/**
 * Enables/disables debugging
 * @param[in] debug - 0 if debugging should be deactivated (default) and != 0 otherwise
 */
void Projection_setDebug(Projection_t* projection, int debug);

#endif
