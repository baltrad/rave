/* --------------------------------------------------------------------
Copyright (C) 2026 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Provides HOOF dealiasing of radial wind data, using torus mapping of the measurements
 * and a wind model fitted for each height layer of the whole polar volume.
 * @file
 * @author Peter Smerkol, Vito Švagelj, ARSO
 * @date 2026-11-13
 */
#ifndef HOOFDEALIAS_H
#define HOOFDEALIAS_H

#include <math.h>
#include "polarvolume.h"
#include "polarscan.h"
#include "polarscanparam.h"

/******************************************************************************/
/*Definition of default parameters.                                          */
/******************************************************************************/

#define HOOFDEALIAS_ZSECTOR    100.0     /* Height layer thickness [m] */
#define HOOFDEALIAS_ZMAX       HUGE_VAL  /* Maximum height of the height layers [m], no limit by default */
#define HOOFDEALIAS_MINPOINTS  500       /* A wind model needs more than this number of points in a height layer */
#define HOOFDEALIAS_MAXWIND    60.0      /* Maximum wind speed of a wind model [m/s] */
#define HOOFDEALIAS_TASK       "si.arso.detector.hoofdealias" /* how/task of the quality field and parameter */

/**
 * Determines whether a scan parameter is dealiased or not (how/dealiased is "True").
 * @param[in] scan - input scan
 * @param[in] quantity - the quantity to check
 * @returns int 1 if the parameter is dealiased, otherwise 0
 */
int hoofdealiased_by_quantity(PolarScan_t* scan, const char* quantity);

/**
 * Dealiases the specified quantity of all scans in a polar volume. 
 * @param[in] pvol - the polar volume, dealiased in place
 * @param[in] quantity - the quantity to dealias (e.g. "VRADH", RAVE reads ODIM 2.2 VRAD as VRADH)
 * @param[in] zsector - height layer thickness [m]
 * @param[in] zmax - maximum height above sea level of the height layers [m], HUGE_VAL for no limit
 * @param[in] minpoints - a wind model needs more than minpoints points in a height layer
 * @param[in] maxwind - maximum wind speed of a wind model [m/s]
 * @returns int 1 if at least one scan was dealiased, otherwise 0
 */
int hoofdealias_pvol_by_quantity(PolarVolume_t* pvol, const char* quantity, double zsector,
                                 double zmax, int minpoints, double maxwind);

/**
 * Dealiases the VRADH quantity of a polar volume with the default parameters.
 * @param[in] pvol - the polar volume, dealiased in place
 * @returns int 1 if at least one scan was dealiased, otherwise 0
 */
int hoofdealias_pvol(PolarVolume_t* pvol);

#endif /* HOOFDEALIAS_H */
