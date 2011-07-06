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
#ifndef POLARMAXECHO_H_
#define POLARMAXECHO_H_

#include "Geometry.h"

#include "../image/Image.h"
#include "../image/ImageView.h"
#include "../image/ImageOp.h"

//#include "geometry.h"

namespace drain
{

namespace radar
{

//using namespace std;

	
/*! The main idea is to compute one equidistant arc at a time.
 */
template <class T = unsigned char,class T2 = unsigned char>
class PolarMaxEcho : public image::ImageOp<T,T2>
{
public:
	PolarMaxEcho(const string p=""){
		this->setInfo("Computes maximum altitude of echo. POLAR COORDINATES.","width",p);
	};
	//virtual ~PolarMax(){};
	void filter(const image::Image<T> &src,image::Image<T2> &dst) const ;
};

template <class T,class T2>
void PolarMaxEcho<T,T2>::filter(const image::Image<T> &src,image::Image<T2> &dst) const
{
   const unsigned int srcWidth  = src.getWidth();
   const unsigned int srcHeight = src.getHeight();
   
   const unsigned int width =  this->parameters.get("width",src.getWidth());
   const unsigned int height = srcHeight; 
   dst.setGeometry(width,height,1,1);
   
   radar::Geometry radarGeometry;
   float binDepth = src.properties.get("BINDEPTH",500.0);
   
   Data elevations = src.properties.get("ELEVATIONS","0.123");
   elevations.trim();
   elevations.splitTo(radarGeometry.elevationAngles," ");
   
   const unsigned int sweeps = radarGeometry.elevationAngles.size();
   
   Data bins = src.properties.get("BINS");
   bins.trim();
   bins.splitTo(radarGeometry.bins," ");
   if (radarGeometry.bins.size() != sweeps){
   		radarGeometry.bins.resize(sweeps,srcWidth);
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

	
    if (false){
		cout << "Starting Max\n";
		cout << "\nParams:\n";	
		//cout << this->parameters << '\n';
		cout << " width" << width << '\n';
	
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
	
	
	// Main loop
	for (unsigned int k=0; k<sweeps; k++){

	  const drain::image::Image<T> & srcSweep = src.getChannel(k);
   	  const unsigned int srcWidth  = srcSweep.getWidth();
  	
	  eta  = radarGeometry.elevationAngles[k];
	  
      for (unsigned int i=0; i<width; i++){

        // Scale in width
        groundBin = (i * srcWidth) / width;
      
		groundDistance = groundBin*binDepth + binDepth/2.0;
		beta = Geometry::betaFromGround(groundDistance);
		
		bin =  static_cast<unsigned int>(Geometry::beamFromEtaBeta(eta,beta) / binDepth);
		
	  	if (bin < radarGeometry.bins[k]) {
	  	
	  		for (unsigned int j=0; j<height; j++){
		  		T2  s = static_cast<T2>(srcSweep.at(bin,j));
		  		T2 &d = dst.at(bin,j);
				if (s > d) 
				  d = s;	
			}
		
        }
       
      }
      //cerr << "Altitude at end ("<< groundDistance<<"m): " << altitude/scale << endl;
    }
   
};


} // ::image

} // ::drain

#endif /*POLARMAX_H_*/
