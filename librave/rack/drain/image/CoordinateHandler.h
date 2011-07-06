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
#ifndef COORDINATEHANDLER_H_
#define COORDINATEHANDLER_H_


#include "Point.h"

namespace drain
{

namespace image
{

// Facility for handling coordinate under- and overflows.
/*!  
 * This base implementation forces the point (i,j) to be inside the image.
 * Any under- or overflow is tagged IRREVERSIBLE.
 * 
 */
class CoordinateHandler
{
public:


    CoordinateHandler(const int &width=0, const int &height=0);
    virtual ~CoordinateHandler(){};
    
    /// No change in coordinates has taken place.
    static const int UNCHANGED = 0;
    static const int X_OVERFLOW = 1;
    static const int X_UNDERFLOW = 2;
    static const int Y_OVERFLOW = 4;
    static const int Y_UNDERFLOW = 8;

    /** Coordinates have been changed such that an inverse move would not return to original position.
     */
    static const int IRREVERSIBLE = 128;

    //public Rectangle bounds;
    int xMin;
    /** Typically imageWidth-1.
     */
    int xMax;
    int yMin;
    /** Typically imageHeight-1.
     */
    int yMax;


	

    /** Sets the size of the image (or any rectangular area) to be accessed.
     *  TODO: rename to setGeometry ?
     * 
     * @param width
     * @param height
     */
    void setBounds(int width,int height)
    {
        this->width = width;
        this->height = height;

        doubleWidth  = 2*width;
        doubleHeight = 2*height;

        xMin = 0;
        xMax = width-1;
        yMin = 0;
        yMax = height-1;

        area = width * height;
        // cerr << "CoordinateHandler: bounds set to: " << width << "," << height << endl;
    }


    /*!
     * 
     * 
     */
    virtual int handle(int &x, int &y) const;
 	
 	/// Returns status (zero if no edges crossed).
 	inline 
 	int handle(Point2D<> & p) const { return handle(p.x,p.y);	};
 
 	string name;
 	
 	int getStatus() const{ return status; };
 	
 protected:
 	/** Width of the image(s) to be accessed.
 	 */
 	int width;
 	int doubleWidth;

 	/** Height of the image(s) to be accessed.
 	 */
 	int height;
 	int doubleHeight;

 	int area;

 	mutable int status; // = UNCHANGED;

};


/// Folds the coordinates back to the image.
class Mirror :  public CoordinateHandler
{
public:
	Mirror(); 
 	virtual int handle(int &x, int &y) const;
    //int handle(Point2D<> &p);
};

/// Repeats the image in the infinities.
class Wrapper :  public CoordinateHandler
{
public:
	Wrapper();
	virtual int handle(int &x, int &y) const;
    //int handle(Point2D<> &p);
};
    
ostream & operator<<(ostream &ostr,const CoordinateHandler &handler);
 

}
}

#endif /*COORDINATEHANDLER_H_*/
