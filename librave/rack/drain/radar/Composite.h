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
#ifndef COMPOSITOR_R_H_
#define COMPOSITOR_R_H_

#include "Coordinates.h" // for site coords and bin coords.

#include "../util/Rectangle.h"

#include "../util/Proj4.h"  // for geographical projection of radar data bins

#include "../image/Image.h"
#include "../image/Cumulator.h"


using namespace std;

namespace drain
{

namespace radar
{


/// An injective, cumulative radar image compositing.
/*!
  Main features:

  Injection. The main loop iterates radar (image) data points of type <Ti>,
  mapping each point to a single point in the target array of type <T>;
  hence the name injector.

  Cumulation. The compositing starts by initiating the target image,
  see open().
  Each input radar image is projected on the target image cumulatively,
  see execute(). Finally, the cumulation must be completed by
  normalizing the result with the number/weight of
  cumulated inputs, see close().
  Hence, the target which should be seen.

  The weights are stored in the first alphaChannel (index 0) of the
  target image. If the target has a second image channel, also
  the (weighted) squared values are cumulated.

  In cumulation, ...
  \f[
  x = \sqrt[p]{\frac{\sum_i {q_i}^r x_i^p}{\sum_i {q_i}^r } }
  \f]

  The default type of target image, that is, cumulation array, is
  double. If no (float valued) weighting  is applied, target image
  could be long int as well.
 */
//template <class T=double>

class Composite : public drain::image::Cumulator<double>, public drain::Rectangle<double>
{
    //public drain::image::Image<double>
public:
    /// Default constructor. The channels are DATA, COUNT, WEIGHT, WEIGHT2
    Composite(unsigned int width = 0, unsigned int height = 0);
    //, unsigned int imageChannels=1,unsigned int alphaChannels=2);
    virtual ~Composite()
    {
    };

    // Notice that someday this does NOT allocate memory. @see allocate();
    virtual
    void setGeometry(unsigned int width, unsigned int height);

	/// Bounding box in degrees in the target coordinate system. In base class, they are in lon,latitudes in degrees
    //void setBoundingBox(float latLL, float lonLL, float latUR, float lonUR);
    void setBoundingBox(double xLL, double yLL, double xUR, double yUR);

    /// In proj4 style string
    //void setProjection(const string &projection);
    /// Converts geographic coordinates to image coordinates. TODO: proj4 support
    inline virtual void map(const double & x, const double & y, int & i, int & j)
    {
        i = static_cast<int>((x - xLowerLeft) * xScale); //  xOffset
        j = static_cast<int>((y - yLowerLeft) * yScale);
    }

    /// Converts image coordinates to geographic coordinates. TODO: proj4 support
    inline virtual void mapInverse(const int & i, const int & j, double & x, double & y)
    {
        x = static_cast<double>(xLowerLeft + i / xScale); //  xOffset
        y = static_cast<double>(yLowerLeft + j / yScale);
    }

    // TODO: rename polar; northing; direction  Maybe this should be in SubComposite only?
    /// Lat,Lon in degrees.
    void addPolar(const drain::image::Image<> & src, const drain::image::Image<> & weight, float lon, float lat, float priorWeight = 1.0);

    /// Inline
    void addPolar(const drain::image::Image<> & src, float lon, float lat, float priorWeight = 1.0){
    	// This is a lazy trick. Duplicate src parameter tells that it's not weight...
    	addPolar(src,src,lon,lat,priorWeight);
    };


    /// (final?) Quality scaling (depends on users prior scale)
    double qualityScale, qualityBias;
    void updateScaling();

    inline
    const int & getDxMax() const{
        return dxMax;
    }

    inline
    const int & getDyMax() const {
        return dyMax;
    }

    /*
    inline
    int getWidth() const{
        return geometry.getWidth();
    }

    inline
    int getHeight() const{
        return geometry.getHeight();
    }
     */
	// Resizes the image array to geometry.
	void allocate();

	/// Radians
	inline const double & getXLowerLeft()  const { return xLowerLeft; }
	inline const double & getYLowerLeft()  const { return yLowerLeft; }
    inline const double & getXUpperRight() const { return xUpperRight; }
    inline const double & getYUpperRight() const { return yUpperRight; }


    Options properties;
protected:



	//drain::image::Geometry geometry;

    /// Radians
    /*
    double xLowerLeft;
    double yLowerLeft;
    double xUpperRight;
    double yUpperRight;
     */

	//double xOffset;
	//double yOffset;
	double xScale;
	double yScale;

	/// Maximum horizontal pixel distance of two beam-neighboring beams. This approximates required interpolation range. 
	/// Computed by initialize().  
	int dxMax;
	/// Maximum vertical pixel distance of two beam-neighboring beams. This approximates required interpolation range.
	/// Computed by initialize().  
	int dyMax;
  

    bool debug;
};







} // ::radar

} // ::drain

#endif /*COMPOSITOR_R_H_*/
