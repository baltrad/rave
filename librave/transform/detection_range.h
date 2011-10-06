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
 * Provides functionality for creating composites.
 * @file
 * @author Harri Hohti, FMI
 * @author Daniel Michelson, SMHI (Intgration)
 * @author Anders Henja, SMHI (Adaption to rave framework)
 * @date 2011-02-16
 */
#ifndef DETECTION_RANGE_H
#define DETECTION_RANGE_H

#include "polarvolume.h"
#include "polarscan.h"
#include "rave_object.h"
#include "rave_types.h"
#include "cartesian.h"
#include "area.h"

/**
 * Defines a Detection range generator
 */
typedef struct _DetectionRange_t DetectionRange_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType DetectionRange_TYPE;

/**
 * Sets the lookup path where the cache files are stored.
 * @param[in] self - self
 * @param[in] path - the path to use for lookup tables MAY NOT BE NULL (default /tmp)
 * @return 1 on success otherwise 0
 */
int DetectionRange_setLookupPath(DetectionRange_t* self, const char* path);

/**
 * Returns the lookup path where the cache files are stored.
 * @param[in] self - self
 * @return the lookup path
 */
const char* DetectionRange_getLookupPath(DetectionRange_t* self);

/**
 * Returns the echo top.
 * @param[in] self - self
 * @param[in] pvol - the polar volume
 * @param[in] scale - the bin length
 * @param[in] threshold_dBZN - threshold for dBZN values
 * @returns a PolarScan containing the echo tops
 */
PolarScan_t* DetectionRange_top(DetectionRange_t* self, PolarVolume_t* pvol, double scale, double threshold_dBZN);

/**
 * Top field garbage should be filtered.
 * @param[in] self - self
 * @param[in] scan - the scan to filter (after top has been calculated)
 * @returns a PolarScan containing the filtered tops
 */
PolarScan_t* DetectionRange_filter(DetectionRange_t* self, PolarScan_t* scan);

#endif /* DETECTION_RANGE_H */
