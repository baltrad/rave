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

#include "Composite.h"


using namespace std;


namespace drain{

namespace radar
{

Composite::Composite(unsigned int width,unsigned int height) : xScale(1), yScale(1), debug(false)
  //xLowerLeft(0), yLowerLeft(0), xUpperRight(0), yUpperRight(0),
{

		setGeometry(width,height);
};




void Composite::setGeometry(unsigned int width,unsigned int height)
{
		//this->width  = width;
		//this->height = height;
		if (width < 1)
			width = 1;
		if (height < 1)
			height = 1;
		drain::image::Cumulator<double>::setGeometry(width,height);
		//geometry.
		//setGeometry(width,height);
		//allocate(); // this should come later
		updateScaling();
};

void Composite::setBoundingBox(double xLL,double yLL,double xUR,double yUR)
{
 	static const double D2R = M_PI / 180.0;

 	xLowerLeft  = xLL*D2R;
    yLowerLeft  = yLL*D2R;
    xUpperRight = xUR*D2R;
    yUpperRight = yUR*D2R;
    updateScaling();
    properties["xLowerLeft"] = xLL;
    properties["yLowerLeft"] = yLL;
    properties["xUpperRight"] = xUR;
    properties["yUpperRight"] = yUR;

};

void Composite::updateScaling()
{
	 float l;
	 
	 l = xUpperRight - xLowerLeft;
	 if (l != 0.0)
	 	xScale = width / l;

	 l = yUpperRight - yLowerLeft;
	 if (l != 0.0)
	 	yScale = height / l;

	//cout << " bbox: " << " xLL=" << xLowerLeft  << ", yLL=" << yLowerLeft  << '\n';
    //cout << " bbox: " << " xUR=" << xUpperRight << ", yUR=" << yUpperRight << '\n';
};


void Composite::allocate(){
	drain::image::Cumulator<double>::setGeometry(width,height);
}


/*! Main function.
 */
void  Composite::addPolar(const drain::image::Image<> &src,
	const drain::image::Image<> &srcWeight,float lon, float lat, float priorWeight) {

	allocate();

	const bool WEIGHT_DATA = (&srcWeight != &src); 

    radar::Coordinates coordinates;
    coordinates.setOriginDeg(lat,lon);

	drain::image::CoordinateHandler handler = srcWeight.getCoordinateHandler();

	string source = src.properties.get("source","?,?,?");
	//source.separators=",";
	//const vector<Data> &v =  source.getVector();
	Data wd = priorWeight;
	properties["sources"] << source+"[W="+wd+"]";
	properties["method"] << method;

	if (drain::Debug > 1){
		stringstream sstr;
		sstr << "source.name()" <<  src.name << '\n';
		sstr << " site: " << " lat=" << lat << ", lon=" << lon << '\n';
		sstr << "weight: " << priorWeight << '\n';
		sstr << "scaleX: " << xScale << '*' << '\n';
		sstr << "scaleY: " << yScale << '*' << '\n';
		sstr << "p: " << p << '\n';
		sstr << "q: " << r << '\n';
		sstr << " lat=(" << lat << ") lon=(" << lon << ")\n";
		cout << sstr.str();
	}

	const unsigned int bins = src.getWidth();
	const unsigned int azimuths = src.getHeight();
	const unsigned int rscale = src.properties.get("rscale",500);

    const int WIDTH  = width; //g.getWidth();
	const int HEIGHT = height; //g.getHeight();
	
	// target x
	int x, xPrev = 0;
	// target y
	int y, yPrev = 0;

    /// Conversion coeff. from azm.index to radians (const for non-compensated)
    const float j2rad = (2.0*M_PI) / azimuths;

	// Data
	double s;
	
	// Weight
	double w = 128;
	
    double azimuth;
    float range;
    

    // In finding maximum distance between pixels of azimuthally neighboring bin
	bool computeBinSpan;
	bool binSpanDistance;


	drain::image::Point2D<> pW;

	dxMax = 1;
	dyMax = 1;

	/// Main loop: for each range, traverse azimuthally.
    for (unsigned int i=0; i<bins; i++)
        {

        range = rscale/2 + i*rscale;

		computeBinSpan = false;
		binSpanDistance = (i%10==0);

        for (unsigned int j=0; j<azimuths; j++)
        {
			azimuth = j * j2rad;

            coordinates.setBinPosition(azimuth,range);
    		map(coordinates.phiBin,coordinates.thetaBin,x,y);

            if (debug)
               if ( !(i % 50) && !(j % 90))
                {
                    cout << i << "b\t" << j << "'\t" << i*0.5 << "km \t";
                    cout << coordinates.binLatitudeDeg() << ' ' << coordinates.binLongitudeDeg() << '\t';
                    cout << coordinates.thetaBin << ' ' << coordinates.phiBin << '\t';
                    cout << x << ' ' << y << '\t';
                    cout << '\n';
                };

            // it = target.address(staticCast<int>(proj.x), staticCast<int>(proj.y));
            // If outside image, skip the rest.
            if ((x<0)||(y<0)||(x>=WIDTH)||(y>=HEIGHT)){
            	computeBinSpan = false;
            	continue;
            }
            
            if (binSpanDistance){
            	if (computeBinSpan){
					dxMax = max(dxMax,abs(x-xPrev));
					dyMax = max(dyMax,abs(y-yPrev));
				}
            	computeBinSpan = true;
				xPrev = x;
				yPrev = y;
	            //add(x, y, 25, 100);
            }
			
		    s = src.at(i,j);

            if (WEIGHT_DATA){
            	//pW.setLocation(i,j);
				//handler.handle(pW);
				//w = srcWeight.at(pW);
            	w = srcWeight.at(i,j);
            }
          
            add(x, y, s, w);

        };
    };



    cout << " dxMax=" << dxMax << " dxMax=" << dxMax << "\n";



}




} // namespace radar
}  // namespace drain
