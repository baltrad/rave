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
/*
 * Rectangle.h
 *
 *  Created on: Sep 9, 2010
 *      Author: mpeura
 */

#ifndef RECTANGLE_H_
#define RECTANGLE_H_

namespace drain {

template <class T>
class Rectangle {
public:

	Rectangle(){
		xLowerLeft = 0;
		yLowerLeft = 0;
		xUpperRight = 0;
		yUpperRight = 0;
	};

	T xLowerLeft;
	T yLowerLeft;
	T xUpperRight;
	T yUpperRight;

	bool isInside(const T &x,const T &y){
		return ((x>xLowerLeft) && (x<xUpperRight) && (y>yLowerLeft) && (y<yUpperRight));
	};
};

}

#endif /* RECTANGLE_H_ */
