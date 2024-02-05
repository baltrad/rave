/* --------------------------------------------------------------------
Copyright (C) 2024 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Functions for managing io cache files like static files.
 *
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-02-05
 */
#ifndef RAVE_IO_CACHE_H
#define RAVE_IO_CACHE_H

#include "rave_object.h"
#include "rave_field.h"

/**
 * Defines a Rave IO Cache instance
 */
typedef struct _RaveIOCache_t RaveIOCache_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RaveIOCache_TYPE;

/**
 * Opens a supported HDF5 file and loads it into the RaveIOCache instance.
 * @param[in] iocache - self
 * @param[in] filename - name of file to read
 * @param[in] fieldname - name of field (/fieldX) to load, if NULL, /field1 is read.
 * @returns The rave field instance on success, otherwise NULL.
 */
RaveField_t* RaveIOCache_loadField(RaveIOCache_t* iocache, const char* filename, const char* fieldname);

/**
 * Saves a rave object as specified 
 * @param[in] iocache - self
 * @param[in] obj - the field to save
 * @param[in] filename - the filename to save with.
 * @returns 1 on success, otherwise 0
 */
int RaveIOCache_saveField(RaveIOCache_t* iocache, RaveField_t* obj, const char* filename);

/**
 * Sets the compression level.
 * @param[in] iocache - self
 * @param[in] lvl - the compression level (0..9)
 */
void RaveIOCache_setCompressionLevel(RaveIOCache_t* iocache, int lvl);

/**
 * Returns the compression level
 * @param[in] iocache- self
 * @returns the compression level
 */
int RaveIOCache_getCompressionLevel(RaveIOCache_t* iocache);

/**
 * Sets the user block.
 * @param[in] iocache - self
 * @param[in] userblock - the user block
 */
void RaveIOCache_setUserBlock(RaveIOCache_t* iocache, unsigned long long userblock);

/**
 * Returns the user block.
 * @param[in] iocache - self
 * @returns the user block
 */
unsigned long long RaveIOCache_getUserBlock(RaveIOCache_t* iocache);

/**
 * Sets the sizes
 * @param[in] iocache - self
 * @param[in] sz - same as sizes.sizeof_size
 * @param[in] addr - same as sizes.sizeof_addr
 */
void RaveIOCache_setSizes(RaveIOCache_t* iocache, size_t sz, size_t addr);

/**
 * Returns the sizes
 * @param[in] iocache - self
 * @param[in] sz - same as sizes.sizeof_size
 * @param[in] addr - same as sizes.sizeof_addr
 */
void RaveIOCache_getSizes(RaveIOCache_t* iocache, size_t* sz, size_t* addr);

/**
 * Sets the symk
 * @param[in] iocache - self
 * @param[in] ik - same as sym_k.ik
 * @param[in] lk - same as sym_k.lk
 */
void RaveIOCache_setSymk(RaveIOCache_t* iocache, int ik, int lk);

/**
 * Returns the symk
 * @param[in] iocache - self
 * @param[in] ik - same as sym_k.ik
 * @param[in] lk - same as sym_k.lk
 */
void RaveIOCache_getSymk(RaveIOCache_t* iocache, int* ik, int* lk);

/**
 * Sets the istore_k value.
 * @param[in] iocache - self
 * @param[in] k - the istore_k value
 */
void RaveIOCache_setIStoreK(RaveIOCache_t* iocache, long k);

/**
 * Returns the istore_k value
 * @param[in] iocache - self
 * @returns the istore_k value
 */
long RaveIOCache_getIStoreK(RaveIOCache_t* iocache);

/**
 * Sets the meta block size
 * @param[in] iocache - self
 * @param[in] sz - the meta block size
 */
void RaveIOCache_setMetaBlockSize(RaveIOCache_t* iocache, long sz);

/**
 * Returns the meta block size
 * @param[in] iocache - self
 * @returns the meta block size
 */
long RaveIOCache_getMetaBlockSize(RaveIOCache_t* iocache);

/**
 * If an error occurs during writing, you might get an indication for why
 * by checking the error message.
 * @param[in] iocache - rave io
 * @returns the error message (will be an empty string if nothing to report).
 */
const char* RaveIOCache_getErrorMessage(RaveIOCache_t* iocache);


#endif
