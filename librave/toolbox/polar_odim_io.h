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
 * Adaptor for polar ODIM H5 files.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-11-12
 */
#ifndef POLAR_ODIM_IO_H
#define POLAR_ODIM_IO_H
#include "rave_object.h"
#include "hlhdf.h"
#include "polarscan.h"
#include "polarvolume.h"

/**
 * Defines the odim h5 adaptor for polar products
 */
typedef struct _PolarOdimIO_t PolarOdimIO_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType PolarOdimIO_TYPE;

/**
 * Sets the version that this io class should handle.
 * @param[in] self - self
 * @param[in] version - the odim version
 */
void PolarOdimIO_setVersion(PolarOdimIO_t* self, RaveIO_ODIM_Version version);

/**
 * Returns the version that this io class handles.
 * @param[in] self - self
 * @returns - the odim version
 */
RaveIO_ODIM_Version PolarOdimIO_getVersion(PolarOdimIO_t* self);

/**
 * Reads a scan from the nodelist and sets the data in the scan.
 * @param[in] self - self
 * @param[in] nodelist - the hdf5 node list
 * @param[in] scan - the scan that should get the attribute and data set
 * @returns 1 on success otherwise 0
 */
int PolarOdimIO_readScan(PolarOdimIO_t* self, HL_NodeList* nodelist, PolarScan_t* scan);

/**
 * Reads a volume from the nodelist and sets the data in the volume.
 * @param[in] self - self
 * @param[in] nodelist - the hdf5 node list
 * @param[in] volume - the volume that should get the attribute and data set
 * @returns 1 on success otherwise 0
 */
int PolarOdimIO_readVolume(PolarOdimIO_t* self, HL_NodeList* nodelist, PolarVolume_t* volume);

/**
 * Fills a nodelist with information about a scan.
 * @param[in] self - self
 * @param[in] scan - the polar scan
 * @param[in] nodelist - the hlhdf nodelist to fill
 * @return 1 on success otherwise 0
 */
int PolarOdimIO_fillScan(PolarOdimIO_t* self, PolarScan_t* scan, HL_NodeList* nodelist);

/**
 * Fills a nodelist with information about a volume.
 * @param[in] self - self
 * @param[in] volume - the polar volume
 * @param[in] nodelist - the hlhdf nodelist to fill
 * @return 1 on success otherwise 0
 */
int PolarOdimIO_fillVolume(PolarOdimIO_t* self, PolarVolume_t* volume, HL_NodeList* nodelist);

#endif /* POLAR_ODIM_IO_H */
