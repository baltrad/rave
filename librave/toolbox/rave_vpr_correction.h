/* --------------------------------------------------------------------
Copyright (C) 2012 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Implementation of the vpr correction functionality
 * @file
 * @author Lars Norin Original code - (Swedish Meteorological and Hydrological Institute, SMHI)
 * @author Anders Henja Adapted to rave fwk (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2013-11-19
 */
#ifndef RAVE_VPR_CORRECTION_H
#define RAVE_VPR_CORRECTION_H
#include "rave_object.h"
#include "polarvolume.h"

/**
 * Defines an algorithm for vpr correction
 */
typedef struct _RaveVprCorrection_t RaveVprCorrection_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RaveVprCorrection_TYPE;

/**
 * Separates stratiform and convective rain
 * @param[in] self - self
 * @param[in] pvol - the polar volume
 * @return 1 on success otherwise 0
 */
int RaveVprCorrection_separateSC(RaveVprCorrection_t* self, PolarVolume_t* pvol);


#endif /* RAVE_VPR_CORRECTION_H */
