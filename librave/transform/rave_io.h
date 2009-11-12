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

/**
 * Defines a Rave IO instance
 */
typedef struct _RaveIO_t RaveIO_t;

/**
 * Creates a new RaveIO instance.
 * @return a new instance or NULL on failure
 */
RaveIO_t* RaveIO_new(void);

/**
 * Releases the responsibility for the rave IO instance, it is not certain that
 * it will be deleted though if there still are references existing
 * to this rave io.
 * @param[in] transform - the transformer
 */
void RaveIO_release(RaveIO_t* raveio);

/**
 * Copies the reference to this instance by increasing a
 * reference counter.
 * @param[in] transform - the transformer to be copied
 * @return a pointer to the scan
 */
RaveIO_t* RaveIO_copy(RaveIO_t* raveio);

/**
 * Loads a ODIM-formatted HDF5 file.
 * @param[in] filename - the file to load
 * @returns the loaded volume on success, otherwise NULL
 */
PolarVolume_t* RaveIO_loadVolume(RaveIO_t* raveio, const char* filename);

#endif
