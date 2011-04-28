/**

    Copyright 2001 - 2010  Markus Peura, Finnish Meteorological Institute (First.Last@fmi.fi)


    This file is part of Drain library for C++.

    Drain is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    Drain is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Drain.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifndef POLARCOORDINATEHANDLER_H_
#define POLARCOORDINATEHANDLER_H_

#include "../image/Point.h"
#include "../image/CoordinateHandler.h"

namespace drain
{

namespace radar
{

/// Handles the coordinates in an image having polar coordinate system. Assumes origin to be at left.
/** (Future versions may cope with origins at whichever direction.) 
 * 
 *
 */
class PolarCoordinateHandler : public drain::image::CoordinateHandler
{
public:
	
	PolarCoordinateHandler();
	
	virtual ~PolarCoordinateHandler();
	
	//template <class T>
	int handle(drain::image::Point2D<> &p);
	
};

}

}

#endif /*POLARCOORDINATEHANDLER_H_*/
