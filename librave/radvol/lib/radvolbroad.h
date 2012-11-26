/* --------------------------------------------------------------------
Copyright (C) 2012 Institute of Meteorology and Water Management -
National Research Institute, IMGW-PIB

This file is part of Radvol-QC package.

Radvol-QC is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Radvol-QC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Radvol-QC.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/
/**
 * Radvol-QC algorithms for assessment of distance-to-radar related effects.
 * @file radvolbroad.h
 * @author Katarzyna Osrodka (Institute of Meteorology and Water Management, IMGW-PIB)
 * @date 2012-07-12
 */
#ifndef RADVOLBROAD_H
#define RADVOLBROAD_H
#include "rave_object.h"
#include "polarvolume.h"

/**
 * Defines a RadvolBroad
 */
typedef struct _RadvolBroad_t RadvolBroad_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RadvolBroad_TYPE;

/**
 * Runs algorithm for assessment of distance-to-radar related effects with parameters from XML file
 * @param pvol - input polar volume
 * @param paramFileName - name of XML file with parameters (otherwise default values are applied)
 * @returns 1 upon success, otherwise 0
 */
int RadvolBroad_broadAssessment(PolarVolume_t* pvol, char* paramFileName);

#endif
