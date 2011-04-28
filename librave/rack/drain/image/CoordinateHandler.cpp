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
#include "CoordinateHandler.h"

namespace drain
{

namespace image
{

CoordinateHandler::CoordinateHandler(const int &width, const int &height){
	setBounds(width,height);
	status = UNCHANGED;
	name = "CoordinateHandler";
}


int CoordinateHandler::handle(int &x, int &y) const 
{

    status = UNCHANGED;

    if (x < xMin)
    {
        x = xMin;
        status = X_UNDERFLOW | IRREVERSIBLE;
    }
    else if (x > xMax)
    {
        x = xMax;
        status = X_OVERFLOW | IRREVERSIBLE;
    }

    if (y < yMin)
    {
        y = yMin;
        status |= Y_UNDERFLOW | IRREVERSIBLE;
    }
    else if (y > yMax)
    {
        y = yMax;
        status |= Y_OVERFLOW | IRREVERSIBLE;
    }

    return status;
}

Mirror::Mirror(){
	 name = "Mirror";
};

int Mirror::handle(int &x, int &y) const
{
    status = UNCHANGED;

    if (x < xMin)
    {
        status = X_UNDERFLOW;
        x -= xMin;
        x = (x%doubleWidth);

        if (x < 0)
            x += doubleWidth;
        if (x >= width)
            x = doubleWidth-x-1;
        x += xMin;
    }
    else if (x > xMax)
    {
        status = X_OVERFLOW;
        x -= xMin;
        x = (x%doubleWidth);
        if (x < 0)
            x += doubleWidth;
        if (x >= width)
            x = doubleWidth-x-1;
        x += xMin;
    }

    if (y < yMin)
    {
        status |= Y_UNDERFLOW;
        y -= yMin;
        y = (y%doubleHeight);
        if (y < 0)
            y += doubleHeight;
        if (y >= height)
            y = doubleHeight-y-1;
        y += yMin;
    }
    else if (y > yMax)
    {
        status |= Y_OVERFLOW;
        y -= yMin;
        y = (y%doubleHeight);
        if (y < 0)
            y += doubleHeight;
        if (y >= height)
            y = doubleHeight-y-1;
        y += yMin;
    }

    return status;
}


Wrapper::Wrapper(){
	name = "Wrapper";
};

int Wrapper::handle(int &x, int &y) const
{
    status = UNCHANGED;
    if (x < xMin)
    {
        status = X_UNDERFLOW;
        x -= xMin;
        x %= width;
        if (x < 0)
            x += width;
        x += xMin;
    }
    else if (x > xMax)
    {
        status = X_OVERFLOW;
        x -= xMin;
        x %= width;
        if (x < 0)
            x += width;
        x += xMin;
    }

    if (y < yMin)
    {
        status |= Y_UNDERFLOW;
        y -= yMin;
        y %= height;
        if (y < 0)
            y += height;
        y += yMin;
    }
    else if (y > yMax)
    {
        status |= Y_OVERFLOW;
        y -= yMin;
        y %= height;
        if (y < 0)
            y += height;
        y += yMin;
    }
    return status;
}


ostream & operator<<(ostream &ostr,const CoordinateHandler &handler){
	ostr << handler.name << '[' << handler.xMin << ',' << handler.yMin << ',';
	ostr << handler.xMax << ',' << handler.yMax << ']' << handler.getStatus();
	return ostr;
}

}
}




