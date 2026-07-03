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
 * Adaptor for file objects when reading ODIM H5 files.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-12-18
 */
#ifndef FILE_OBJECT_ODIM_IO_H
#define FILE_OBJECT_ODIM_IO_H
#include <lazy_nodelist_reader.h>
#include "rave_object.h"
#include "hlhdf.h"
#include "file_object.h"
#include "rave_list.h"

/**
 * Defines the odim h5 adaptor for polar products
 */
typedef struct _FileObjectOdimIO_t FileObjectOdimIO_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType FileObjectOdimIO_TYPE;

/**
 * Sets the version that this io class should handle.
 * @param[in] self - self
 * @param[in] version - the odim version
 */
void FileObjectOdimIO_setVersion(FileObjectOdimIO_t* self, RaveIO_ODIM_Version version);

/**
 * Returns the version that this io class handles.
 * @param[in] self - self
 * @returns - the odim version
 */
RaveIO_ODIM_Version FileObjectOdimIO_getVersion(FileObjectOdimIO_t* self);

/**
 * If writing should be done strictly. From ODIM H5 2.4 several how-attributes are mandatory. If
 * any of these are missing and strict is set to true, then the writing will fail.
 * @param[in] self - self
 * @param[in] strict - if writing should be performed strictly or not
 */
void FileObjectOdimIO_setStrict(FileObjectOdimIO_t* self, int strict);

/**
 * If writing should be done strictly. From ODIM H5 2.4 several how-attributes are mandatory. If
 * any of these are missing and strict is set to true, then the writing will fail.
 * @param[in] self - self
 * @returns if writing should be performed strictly or not
 */
int FileObjectOdimIO_isStrict(FileObjectOdimIO_t* self);

/**
 * If an error occurs during writing, you might get an indication for why
 * by checking the error message.
 * @param[in] raveio - rave io
 * @returns the error message (will be an empty string if nothing to report).
 */
const char* FileObjectOdimIO_getErrorMessage(FileObjectOdimIO_t* self);

/**
 * Reads a scan from the nodelist and sets the data in the scan.
 * @param[in] self - self
 * @param[in] nodelist - the hdf5 node list
 * @returns the file object on success otherwise NULL
 */
FileObject_t* FileObjectOdimIO_read(FileObjectOdimIO_t* self, LazyNodeListReader_t* lazyReader);

/**
 * Fills a nodelist with information about a file object.
 * @param[in] self - self
 * @param[in] fobj - the file object
 * @param[in] nodelist - the hlhdf nodelist to fill
 * @return 1 on success otherwise 0
 */
int FileObjectOdimIO_fill(FileObjectOdimIO_t* self, FileObject_t* fobj, HL_NodeList* nodelist);

#endif /* POLAR_ODIM_IO_H */
