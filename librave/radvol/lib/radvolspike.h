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
 * Radvol-QC algorithms for spike removal.
 * @file radvolspike.h
 * @author Katarzyna Osrodka (Institute of Meteorology and Water Management, IMGW-PIB)
 * @date 2012-12-20
 */
#ifndef RADVOLSPIKE_H
#define RADVOLSPIKE_H
#include "rave_object.h"
#include "polarvolume.h"
#include "polarscan.h"
#include "radvol.h"

/**
 * Defines a RadvolSpike
 */
typedef struct _RadvolSpike_t RadvolSpike_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RadvolSpike_TYPE;

/**
 * Runs algorithm for spike removal and quality characterization with parameters from XML file
 * @param scan - input polar scan
 * @param params - struct containing algorithm-specific parameter settings
 * @param paramFileName - name of XML file with parameters (otherwise default values are applied)
 * @returns 1 upon success, otherwise 0
 */
int RadvolSpike_spikeRemoval_scan(PolarScan_t* scan, Radvol_params_t* params, char* paramFileName);

/**
 * Runs algorithm for spike removal and quality characterization with parameters from XML file
 * @param pvol - input polar volume
 * @param params - struct containing algorithm-specific parameter settings
 * @param paramFileName - name of XML file with parameters (otherwise default values are applied)
 * @returns 1 upon success, otherwise 0
 */
int RadvolSpike_spikeRemoval_pvol(PolarVolume_t* pvol, Radvol_params_t* params, char* paramFileName);

#endif
