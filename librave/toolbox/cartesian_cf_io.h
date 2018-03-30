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
 * Adaptor for cartesian CF convention files.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2017-11-24
 */
#ifndef CARTESIAN_CF_IO_H
#define CARTESIAN_CF_IO_H
#include "rave_object.h"
#include "cartesian.h"
#include "cartesianvolume.h"
#include "netcdf.h"

/**
 * Defines the cf conventions h5 adaptor for cartesian products
 */
typedef struct _CartesianCfIO_t CartesianCfIO_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType CartesianCfIO_TYPE;

/**
 * Sets the level of compression. 0 which is default means no compression.
 * 1-9 is the level of compression where 1 is lowest level and 9 is highest.
 * @param[in] self - self
 * @param[in] level - level of compression
 * @returns 1 if level was within 0 - 9 otherwise false
 */
int CartesianCfIO_setDeflateLevel(CartesianCfIO_t* self, int level);

/**
 * Returns the level of compression
 * @param[in] self - self
 * @returns level of compression where 0 means no compression and 9 is highest compression
 */
int CartesianCfIO_getDeflateLevel(CartesianCfIO_t* self);

/**
 * Reads a netcdf file store in CF convention and sets the data in the resulting
 * object.
 * @param[in] self - self
 * @param[in] const char* - the netcdf filename
 * @returns the read object on success otherwise NULL
 */
RaveCoreObject* CartesianCfIO_read(CartesianCfIO_t* self, const char* filename);

/**
 * Writes a netcdf file in CF convention format.
 * .
 * @param[in] self - self
 * @param[in] const char* - the netcdf filename
 * @param[in] obj - the object to be written
 * @returns 1 on success otherwise 0
 */
int CartesianCfIO_write(CartesianCfIO_t* self, const char* filename, RaveCoreObject* obj);

#endif /* CARTESIAN_ODIM_IO_H */
