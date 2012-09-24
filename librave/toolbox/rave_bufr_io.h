/* --------------------------------------------------------------------
Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Adaptor for polar BUFR files.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2011-11-07
 */
#ifndef RAVE_BUFR_IO_H
#define RAVE_BUFR_IO_H
#include "rave_object.h"
#include "polarscan.h"
#include "polarvolume.h"

/**
 * Defines the bufr adaptor
 */
typedef struct _RaveBufrIO_t RaveBufrIO_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RaveBufrIO_TYPE;

/**
 * Sets the table dir name where the descriptor information resides
 * @param[in] self - self
 * @param[in] dirname - the directory name
 * @return 1 on success otherwise 0
 */
int RaveBufrIO_setTableDir(RaveBufrIO_t* self, const char* dirname);

/**
 * Returns the table dir name.
 * @param[in] self - self
 * @return the directory name
 */
const char* RaveBufrIO_getTableDir(RaveBufrIO_t* self);

/**
 * Reads the bufr file.
 * @param[in] self - self
 * @param[in] filename - the filename that should be read
 * @returns the rave core object on success otherwise NULL
 */
RaveCoreObject* RaveBufrIO_read(RaveBufrIO_t* self, const char* filename);

/**
 * Tests if the specified file is a bufr file.
 * @param[in] filename - the name to check
 * @return 1 if file is a BUFR file, otherwise 0
 */
int RaveBufrIO_isBufr(const char* filename);

#endif /* RAVE_BUFR_IO_H */
