/**

    Copyright 2002 - 2010  Harri Hohti (minor edits Markus Peura),
    Finnish Meteorological Institute (First.Last@fmi.fi)


    This file is part of Rack.

    Rack is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Rack is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser Public License for more details.

    You should have received a copy of the GNU Lesser Public License
    along with Rack.  If not, see <http://www.gnu.org/licenses/>.

*/


#ifndef _FMI_SUNPOS_
#define _FMI_SUNPOS_

/** Computes the azimuth and elevation of the Sun at given time and location on the Earth.
 *  \par latDeg = latitude in degrees
 *  \par latDeg = longitude in degrees
 *  \par timestamp = timestamp as a string, "YYYYMMDDhhmm"
 *  \par azimuth
 */
void sunPosition(double latR, double lonR,char *timestamp,double *azimuth,double *elevation);

#endif

