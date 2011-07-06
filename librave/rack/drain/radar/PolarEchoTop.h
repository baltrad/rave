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
#ifndef POLARECHOTOP_H_
#define POLARECHOTOP_H_

#include <algorithm>
#include "Geometry.h"

/*
#include "../image/Image.h"
#include "../image/ImageView.h"
#include "../image/ImageOp.h"

*/

#include "PolarProduct.h"

namespace drain
{

namespace radar
{

//using namespace std;

	
/*! The main idea is to compute one equidistant arc at a time.
 */
template <class T = unsigned char,class T2 = unsigned char>
class PolarEchoTop : public PolarProduct<T,T2>
{
public:

	PolarEchoTop(const string p="500,max,64,0.01") : PolarProduct<T,T2>("PolarEchoHeight",
			"Computes minimum or maximum altitude of echo.","width,mode,threshold,scale,errorCoeff",p)  {
		/*		this->setInfo("Computes minimum or maximum altitude of echo.",
		"width,mode,threshold,scale,errorCoeff",p);
		*/
	};
	//virtual ~PolarEchoTop(){};
	void filter(const image::Image<T> &src,image::Image<T2> &dst) const ;
};



template <class T,class T2>
void PolarEchoTop<T,T2>::filter(const image::Image<T> &src,image::Image<T2> &dst) const {
   const unsigned int srcWidth  = src.getWidth();
   const unsigned int srcHeight = src.getHeight();
   
   const bool topMode = (this->parameters.get("mode","max") == "max");
   const float threshold = this->parameters.get("threshold",(int)64);  
   const float scale = this->parameters.get("scale",0.01);
   const float errorScale   = this->parameters.get("errorCoeff",1.0);    
   
   const unsigned int width =  this->parameters.get("width",src.getWidth());
   const unsigned int height = srcHeight; 
   dst.setGeometry(width,height,1,1);
   dst.properties["/where/nbins"] = width;
   
   // Altitude difference
   image::Image<T2> &dstAlpha = dst.getAlphaChannel();
   
   radar::Geometry radarGeometry;
   float binDepth = src.properties.get("@where/rscale",500.0);
   
   Data elevations = src.properties.get("@where/elangle","0.123");
   elevations.trim();
   vector<float> e;
   elevations.splitTo(e," ");
   map<float,unsigned int> elevationMap;
   for (unsigned int i=0; i<e.size(); i++)
      elevationMap[M_PI/180.0*e[i]] = i; 
   //elevations.splitTo(radarGeometry.elevationAngles," ");
   
   const unsigned int sweeps = radarGeometry.elevationAngles.size();
   
   Data bins = src.properties.get("@where/nbins");
   bins.trim();
   vector<unsigned int> binVector;
   bins.splitTo(binVector," ");
   if (binVector.size() != sweeps){
   		binVector.resize(sweeps,srcWidth);
   } 
   
   for (unsigned int i = 0; i < sweeps; ++i) {
		radarGeometry.elevationAngles[i] *= M_PI/180.0;
   }
    
   
    /// bin (index) of source beam (equidistant bin retrieval).
    unsigned int bin;

    /// bin (index) of the resulting "beam" on Earth surface.
    unsigned int groundBin;

   	/// Elevation angle (radians).
    float eta;

	/// On-earth angle.
    float beta;

	
    if (drain::Debug > 2){
		cout << "Starting EchoTop\n";
		cout << "\nParams:\n";	
		cout << this->parameters << '\n';
	
		cout << "\nGeometry:\n";	
		cout << src.getGeometry() << '\n';
		cout << "\nProperties:\n";	
		cout << src.properties << '\n';
		cout << "\nElevs: " << elevations << "\n";
	
		for (unsigned int i=0; i< radarGeometry.elevationAngles.size(); i++){
			cout << "'" <<  radarGeometry.elevationAngles[i] << "', ";	
		}
		cout << "\n";
    }
	
	double groundDistance = 0.0;
	T altitude = 0;
	//T altitudeDifference = 0;
	//bool weighted = src
	
	
	// Main loop
	//for (unsigned int k=0; k<sweeps; k++){
	// This forces sweeps to be addressed in order
	
	if (!topMode)
		dst.fill(image::Intensity::max<T2>());
	
	for (typename map<float,unsigned int>::iterator it = elevationMap.begin(); it != elevationMap.end(); it++){
		
		const unsigned int k = it->second;
		eta = it->first;
		
		cerr << "Angle " << eta << endl;

	  const drain::image::Image<T> & srcSweep = src.getChannel(k);
   	  const unsigned int srcWidth  = srcSweep.getWidth();
  	
	  //eta  = radarGeometry.elevationAngles[k];
	  
      for (unsigned int i=0; i<width; i++){

        // Scale in width
        groundBin = (i * srcWidth) / width;
      
		groundDistance = groundBin*binDepth + binDepth/2.0;
		beta = Geometry::betaFromGround(groundDistance);
		
		// aloft
	  	bin = static_cast<unsigned int>(Geometry::beamFromEtaBeta(eta,beta) / (float)binDepth);
		//cerr << "eta = " << eta << "\t beta = " << beta << "\t bin = " << bin << endl;
	  	
	  	if (bin < binVector[k]) {
	  	
	  		altitude = static_cast<T2>(scale * Geometry::heightFromEtaBeta(eta,beta));
			  
			 if (topMode)
 			  for (unsigned int j=0; j<height; j++){
			
				if (srcSweep.at(bin,j) > threshold){
			  	   //if (altitude > dst.at(i,j)){
			 		  dst.at(i,j) = altitude;
			 		  dstAlpha.at(i,j) = image::Intensity::max<T2>();  // "reset"
			  	  // }	
			  	}
			  	else {
		  			if (altitude < dstAlpha.at(i,j)){
		  				dstAlpha.at(i,j) = altitude;
			  		}
			  	}
		  	}
		  	// Bottom mode
		  	else {
		  	  for (unsigned int j=0; j<height; j++){
		  	  	if (altitude < dst.at(i,j)){  // = no echo yet
		  	  		if (srcSweep.at(bin,j) > threshold){
		  	  			dst.at(i,j) = altitude;
		  	  		}
		  	  		else
		  	  			dstAlpha.at(i,j) = altitude;
		  	  	}		
				
		  	}		
        }
       
      }
      //cerr << "Altitude at end ("<< groundDistance<<"m): " << altitude/scale << endl;
    }

	}
	if (errorScale > 0.0)
      for (unsigned int i=0; i<width; i++)
      	for (unsigned int j=0; j<height; j++){
      		const T2 &d = dst.at(i,j);
      		T2 &a = dstAlpha.at(i,j);
      		a = static_cast<T2>( errorScale * ((d>a) ? d-a : a-d));
  	 	}
   
   	
};


} // ::image

} // ::drain

#endif /*POLARECHOTOP_H_*/
