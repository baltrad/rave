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
#include "../image/Point.h"
#include "PolarCoordinateHandler.h"

namespace drain
{

namespace radar
{

PolarCoordinateHandler::PolarCoordinateHandler()
{
};

PolarCoordinateHandler::~PolarCoordinateHandler()
{
};

int PolarCoordinateHandler::handle(drain::image::Point2D<> &p)
{
  status = UNCHANGED;

    if (p.x < xMin)
    {
        p.x = -p.x % width;
        p.y = (p.y + height/2) % height;
        status = X_UNDERFLOW;
    }
    else if (p.x > xMax)
    {
        p.x = xMax;
        status = X_OVERFLOW | IRREVERSIBLE;
    }

    if (p.y < yMin)
    {
        p.y = (p.y + height) % height;
        status |= Y_UNDERFLOW;
    }
    else if (p.y > yMax)
    {
    	p.y = (p.y + height) % height;
        status |= Y_OVERFLOW;
    }

    return status;
};


}

}
