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
#ifndef SUBCOMPOSITE_H_
#define SUBCOMPOSITE_H_


#include "Composite.h"
//#include "../util/Proj4.h"


#include "Coordinates.h"

namespace drain
{

namespace radar
{

// Defines a subregi on in a main composite.
/**
 *
 */
class SubComposite : public Composite
{
public:

	SubComposite(Composite &composite);

	virtual ~SubComposite(){};

	/** Sets the location and range of a radar.
	 * 
	 *  @param lon - location of radar site, longitude in degrees. 
	 *  @param lat - location of radar site, latitude in degrees.
	 *  @param range - range of radar, in metres.  
	 * 
	 */
	void setRadarLocation(double lon,double lat,double range = 250000){
		coordinates.setOriginDeg(lat,lon);
		this->range = range;	
	}
	
	/** Sets the location and range(=rscale*nbins) of a radar
		 *
		 *  @param lon - location of radar site, longitude in degrees.
		 *  @param lat - location of radar site, latitude in degrees.
		 *  @param rscale
		 *
		 */
		void setRadarLocation(double lon,double lat,double rscale, unsigned int nbins){
			coordinates.setOriginDeg(lat,lon);
			this->range = rscale * static_cast<double>(nbins);
		}


	/** Detects the bounding box. Computes the image dimensions within \c composite and
	 *  sets image dimensions and offset appropriately. Derives it by traversing the full 360⁰ circle.
	 *
	 *
	 *  Old: @param azimuth - the bounding box will be computed numerically by traversing the
	 *  full circle with azimuthSteps given in degrees. If zero, the circular curve will be drawn.
	 */
	void detectBoundingBox(); //unsigned int azimuthStep = 15);

	// Determines the bounding box (in image coords) inside the master composite.
	//void determineBoundingBox();

	inline virtual
	void map(const double &x,const double &y,int &i,int &j){
		composite.map(x,y,i,j);
		i -= static_cast<int>(xOffset);
		j -= static_cast<int>(yOffset);
	}

	// Adds (cumulates) image array to composite. Applies DistanceTransformFill first.
	//void flush();

	int getXOffset() const { return xOffset; };
	int getYOffset() const { return yOffset; };

    /// Traverses along the edges of the subcomposite, and derives xDiffMax and yDiffMax of neighboring beams. 
    //void detectBeamSparsity();
    float globalWeight;
    // for DisrtanceFill
    int horzDecrement;
    int vertDecrement;
    /// If true,
    //int drawMaxRange;
    double lon;
    double lat;
    double range;
    Coordinates coordinates;
protected:
    int xOffset;
    int yOffset;
    Composite & composite;
	
	
	
};

}

}

#endif /*SUBCOMPOSITE_H_*/
