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
#ifndef POLARTOGEOGRAPHICAL_H_
#define POLARTOGEOGRAPHICAL_H_

#include <math.h>
#include "../image/Image.h"
#include "../image/ImageOp.h"
#include "../image/Cumulator.h"

#include "../util/ProjectionFrame.h"

using namespace std;
using namespace drain::image;



namespace drain
{

namespace radar
{
	
	
 /*! Converts polar image to cartesian. Radar (origin) is expected to be at left, and first line is the
  *  pointing to North.  
  */
template <class T = unsigned char,class T2 = unsigned char>

class PolarToGeographical : public image::ImageOp<T,T2>
{
public:
	PolarToGeographical(){};
	virtual ~PolarToGeographical(){};

	Image<T2> &filter(const Image<T> &src, Image<T2> &dst) const;
	
	Coordinates coordinates;
	
	//string projStr;
	mutable ProjectionFrame proj;
};

template <class T,class T2>
Image<T2> &PolarToGeographical<T,T2>::filter(const Image<T> &src, Image<T2> &dst) const
{
	
	//Proj4 proj;
	//proj.setProjection(projStr);
	cerr << "kukkuu" << endl;
	const image::Geometry &g = src.getGeometry();
	const int bins = g.getWidth();
	const int azimuths = g.getHeight();
	const bool weighted = (g.getChannelCount() > 1);  // alpha?
	
	const int width  = this->parameters.get("width",bins);
	const int height = this->parameters.get("height",width);
	proj.setGeometry(width,height);
	
	const string channelString = this->parameters.get("channels","D");
	vector<double> origin(2); 
	this->parameters.get("coords",drain::Data("25,65")).splitTo(origin);
	//drain::Data coordData = this->parameters.get("coords","25,65");
	//coordData.splitTo(origin);
	
	drain::radar::Coordinates coordinates;
	coordinates.setOriginDeg(origin[0],origin[1]);
	
	const int channelCount = channelString.length();
	
	const int halfWidth = width/2;
	float r;
	double a;
	double s;
	double c;
	
	cerr << "AZ:  "      << azimuths << '\n';
	cerr << "BINS: "     << bins << '\n';
	cerr << "CHANNELS: " << channelCount << '\n';
		
	Cumulator<double> cumulator;
	cerr << "Cumu: \n";
	cumulator.setGeometry(width,width);
	cerr << "Cumu2: \n";
	
	 
	
	const float az2rad = (2.0f*M_PI)/azimuths; 
	
	double x;
	double y;
	
	for (int j = 0; j < azimuths; ++j) {
		a = az2rad*j;
		s = sin(a);
		c = cos(a);
		//if (weighted){
		for (int i = 0; i < bins; ++i) {
			r = (i*halfWidth)/bins;
			coordinates.setBinPosition(a,r);
			proj.project(coordinates.thetaBin,coordinates.phiBin);
			cumulator.add(static_cast<int>(coordinates.phiBin),static_cast<int>(coordinates.thetaBin),src.at(i,j),255);
	  	}
		/*
		}
		else {
			for (int i = 0; i < bins; ++i) {
				r = (i*halfWidth)/bins;
				cumulator.add(halfWidth + static_cast<int>(s*r),halfWidth + static_cast<int>(c*r),src.at(i,j),255);
				//cumulator.add(halfWidth + static_cast<int>(s*r),halfWidth + static_cast<int>(c*r)) = src.at(i,j);
	  	  }
		}
		*/
	 }
	
	 
	 	
	cerr << "cart temp: \n";
	//image::BufferedImage<double> temp;

	cumulator.extractTo(dst,channelString);
	
	
	//for (int i = 0; i < width; ++i) 	dst.at(i,i) = 255;
	
	
	return dst;	
}

}

}

#endif /*POLARTOGEOGRAPHICAL_H_*/
