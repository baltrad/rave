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
#ifndef POLARTOCARTESIAN_H_
#define POLARTOCARTESIAN_H_

#include <math.h>

#include "../image/Image.h"
#include "../image/ImageOp.h"
#include "../image/Cumulator.h"

using namespace drain::image;


namespace drain
{

namespace radar
{

using namespace std;
	
 /*! Converts polar image to cartesian. Radar (origin) is expected to be at left, and first line is the
  *  pointing to North.  
  */
template <class T = unsigned char,class T2 = unsigned char>
class PolarToCartesian : public image::ImageOp<T,T2>
{
public:

	PolarToCartesian(const string & p = "500,255") : image::ImageOp<T,T2>("PolarToCartesian",
			"Conversion from polar to simple Cartesian coordinates.","width,nodata",p){
	};


	//PolarToCartesian();
	virtual ~PolarToCartesian(){};
	
	virtual void filter(const Image<T> &src, Image<T2> &dst) const;
	virtual void filterInjective(const Image<T> &src, Image<T2> &dst) const;
	
	
};

template <class T,class T2>
void PolarToCartesian<T,T2>::filter(const Image<T> &src, Image<T2> &dst) const
{

	if (drain::Debug > 2)
		cout << "Staring PolarToCartesian\n";

	const int bins   = src.getWidth();  // todo param
	const int sweeps = src.getHeight();
	
	
	//const int imageChannels = 1;
	
	const int channel = 0; //this->parameters.get("channel",0);
	const long width = this->parameters.get("width",bins); // long for ODIM
	const int halfWidth = width/2;
	
	const double nodata
		= this->parameters.get("nodata",src.properties.get("@where/nodata",500.0)); // double for ODIM

	// ODIM
	Options &p = dst.properties;
	p = src.properties;
	p["@what/nodata"] = nodata;

	static double scale = (src.properties.get("/where/rscale",500.0) * 0.5 * width)/bins;
	p["/where/xscale"] = scale;
	p["/where/yscale"] = scale;

	p["/where/xsize"] = width;
	p["/where/ysize"] = width;

	double lat = src.properties.get("/where/lat",0.0);
	double lon = src.properties.get("/where/lon",0.0);

	stringstream sstr;
	sstr << "+proj=aeqd +lat_0="<< lat << " +lon_0="<< lon <<" +no_defs";
	//		"// +R=" << EARTH_RADIUS << " +no_defs";
	p["/dataset1/where/projdef"] = sstr.str();
	//string prefix="/dataset1";
	// xscale is the m/pixel resolution at the centerpoint of the image (hence cos).

	drain::radar::Coordinates coordinates;
	coordinates.setOriginDeg(lat,lon);
	double latMin,lonMin,latMax,lonMax;
	const float range = src.properties.get("rscale",500.0) * width; //src.properties.get("nbins",500.0);
	coordinates.getBoundingBox(range,latMin,lonMin,latMax,lonMax);
	p["/where/LL_lon"] = lonMin * RAD_TO_DEG;
	p["/where/LL_lat"] = latMin * RAD_TO_DEG;
	p["/where/UR_lon"] = lonMax * RAD_TO_DEG;
	p["/where/UR_lat"] = latMax * RAD_TO_DEG;


	const double azimuth2sweep = sweeps/(2.0*M_PI);
	//dst.setGeometry(width,width,imageChannelCount(),src.getAlphaChannelCount());
	
	const bool alphaChannel = (src.getAlphaChannelCount() > 0);
	
	ImageView<T2> srcView(src,channel,1);
	ImageView<T2> srcAlpha;
	ImageView<T2> dstAlpha;
	if (alphaChannel){
	  srcAlpha.viewImage(src.getAlphaChannel());
	  dst.setGeometry(width,width,1,1);
	  dstAlpha.viewChannel(dst,1);
	}
	else
	  dst.setGeometry(width,width,1,0);
	
	int sweep;
	int bin;
	
	
	int di,di2,dj,dj2;
	
	//CoordinateHandler h;
	//h.setBounds(src.getWidth(),sweeps);
	//for (int k = 0; k < channels; k++)
	  for (int j = 0; j < width; j++) {
		dj  = j-halfWidth;
		dj2 = dj*dj;
		for (int i = 0; i < width; i++) {
			di  = i-halfWidth;
			di2 = di*di;
			bin = (static_cast<int>(bins*sqrt(di2 + dj2)))/halfWidth;
			sweep = static_cast<int>(azimuth2sweep * (atan2(di,-dj)+2.0*M_PI)) % sweeps;
			//if (h.handle(bin,sweep)==0)
			if (bin < bins){
			   dst.at(i,j) = srcView.at(bin,sweep);
			   if (alphaChannel)
			   	 dstAlpha.at(i,j) = srcAlpha.at(bin,sweep);
			}
			else
				dst.at(i,j) = static_cast<T>(nodata);
		}
	  }
}

// needs postprocessing
template <class T,class T2>
void PolarToCartesian<T,T2>::filterInjective(const Image<T> &src, Image<T2> &dst) const
{
	const image::Geometry &g = src.getGeometry();
	const int bins = g.getWidth();
	const int azimuths = g.getHeight();
	//const bool weighted = (g.getChannelCount() > 1);  // alpha?
	
	const int width = this->parameters.get("width",bins);

	const string channelString = this->parameters.get("channels","D");
	
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
	
	for (int j = 0; j < azimuths; ++j) {
		a = az2rad*j;
		s = sin(a);
		c = cos(a);
		//if (weighted){
		for (int i = 0; i < bins; ++i) {
			r = (i*halfWidth)/bins;
			//if (i==j) 
			//	cerr << (float) src.at(i,j) << " => ";
			cumulator.add(halfWidth + static_cast<int>(s*r),halfWidth + static_cast<int>(c*r),src.at(i,j),255);
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

	cumulator.extractTo(dst,channelString);
	
	

}

}

}

#endif /*POLARTOCARTESIAN_H_*/
