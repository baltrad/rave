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
 * Functions for performing rave related IO operations, mostly ODIM-formatted HDF5 files.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-11-12
 */
#ifndef RAVE_IO_H
#define RAVE_IO_H

#include "polarvolume.h"
#include "cartesian.h"
#include "rave_object.h"

typedef enum RaveIO_ODIM_Version {
  RaveIO_ODIM_Version_UNDEFINED = -1,
  RaveIO_ODIM_Version_2_0 = 0,        /**< Currently, the only supported ODIM version (and default) */
} RaveIO_ODIM_Version;

typedef enum RaveIO_ODIM_H5rad_Version {
  RaveIO_ODIM_H5rad_Version_UNDEFINED = -1,
  RaveIO_ODIM_H5rad_Version_2_0 = 0,  /**< Currently, the only supported ODIM version (and default) */
} RaveIO_ODIM_H5rad_Version;

/**
 * Defines a Rave IO instance
 */
typedef struct _RaveIO_t RaveIO_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RaveIO_TYPE;

/**
 * Closes the HDF5 file but will keep the RaveIO instance.
 * @param[in] raveio - the rave IO instance
 */
void RaveIO_close(RaveIO_t* raveio);

/**
 * Opens a supported HDF5 file and loads it into the RaveIO instance.
 * Same as:
 * RaveIO_t* instance = RAVE_OBJECT_NEW(&RaveIO_TYPE);
 * RaveIO_setFilename(instance, filename);
 * RaveIO_load(instance);
 *
 * @param[in] filename - the HDF5 file to open
 * @returns The raveio instance on success, otherwise NULL.
 */
RaveIO_t* RaveIO_open(const char* filename);

/**
 * Loads the HDF5 file into the raveio instance.
 * @param[in] raveio - self
 * @returns the opened object
 */
int RaveIO_load(RaveIO_t* raveio);

/**
 * Saves a rave object as specified according to ODIM HDF5 format specification.
 * @param[in] raveio - self
 * @param[in] object - the object to save
 * @param[in] filename - the file name this file should have
 * @returns 1 on success, otherwise 0
 */
int RaveIO_save(RaveIO_t* raveio);

/**
 * Sets the object to be saved.
 * @param[in] raveio - self
 * @param[in] object - the object to be saved
 */
void RaveIO_setObject(RaveIO_t* raveio, RaveCoreObject* object);

/**
 * Returns the loaded object/object to be saved.
 * @param[in] raveio - self
 * @returns the object
 */
RaveCoreObject* RaveIO_getObject(RaveIO_t* raveio);

/**
 * Sets the filename that should be used when saving the object.
 * @param[in] raveio - self
 * @param[in] filename - the filename that should be used when saving.
 * @returns 1 on success, otherwise 0
 */
int RaveIO_setFilename(RaveIO_t* raveio, const char* filename);

/**
 * Returns the current filename.
 * @param[in] raveio - self
 * @returns the current filename
 */
const char* RaveIO_getFilename(RaveIO_t* raveio);

/**
 * Returns the object type for the currently opened file. Requires that
 * a RaveCoreObject has been set.
 * @param[in] raveio - the Rave IO instance
 * @returns the object type or Rave_ObjectType_UNDEFINED on error.
 */
Rave_ObjectType RaveIO_getObjectType(RaveIO_t* raveio);

/**
 * Sets the ODIM version to use when saving the file. Currently, the only
 * supported version is 2.0.
 * @param[in] raveio - self
 * @param[in] version - the version to be used
 * @returns 1 if the specified version is supported, otherwise 0.
 */
int RaveIO_setOdimVersion(RaveIO_t* raveio, RaveIO_ODIM_Version version);

/**
 * Returns the ODIM version.
 * @param[in] raveio - the Rave IO instance
 * @returns the ODIM version
 */
RaveIO_ODIM_Version RaveIO_getOdimVersion(RaveIO_t* raveio);

/**
 * Sets the ODIM h5rad version to use when saving the file. Currently, the only
 * supported version is 2.0.
 * @param[in] raveio - self
 * @param[in] version - the version to be used
 * @returns 1 if the specified version is supported, otherwise 0.
 */
int RaveIO_setH5radVersion(RaveIO_t* raveio, RaveIO_ODIM_H5rad_Version version);

/**
 * Returns the h5rad version.
 * @param[in] raveio - the Rave IO instance
 * @returns the h5rad version
 */
RaveIO_ODIM_H5rad_Version RaveIO_getH5radVersion(RaveIO_t* raveio);

#endif
