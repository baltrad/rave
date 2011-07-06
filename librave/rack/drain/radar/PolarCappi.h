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
#ifndef POLARCAPPI_H_
#define POLARCAPPI_H_

#include "PolarProduct.h"

namespace drain
{

namespace radar
{

using namespace std;

/*! Constant-altitude planar position indicator.
 *  The main idea is to compute one equidistant arc at a time.
 *  This means that the nearest beams (up and down) can be computed
 *  in advance, as well as the respective distances (hence weights).
 */
template <class T = unsigned char,class T2 = unsigned char>
class PolarCappi : public PolarProduct<T,T2>
{
public:

	PolarCappi(const string & p = "500,500,0.8") : PolarProduct<T,T2>("PolarCappi",
			"Constant-altitude planar position indicator.","width,altitude,beamwidth",p){
	};

	// ODIM  TODO
	//  COPY defaults, so src should be in.
	//void setStandardProperties(const string &product, double lat,double lat,double rscale,double elangle);

	//virtual ~PolarCappi(){};

	virtual void filter(const image::Image<T> &src,image::Image<T2> &dst) const;
};

/*
void setStandardProperties(const string &product, double lat,double lon,double rscale,double elangle = 0.0){
	this->setParamater("product",product);
	this->setParamater("lat",lat);
	this->setParamater("lon",lon);
	this->setParamater("rscale",rscale);
	this->setParamater("elangle",elangle);
}
 */

template <class T,class T2>
void PolarCappi<T,T2>::filter(const image::Image<T> &src,image::Image<T2> &dst) const
{
	if (drain::Debug > 0)
		cerr << "Computing "<< this->name << '\n' << this->parameters << '\n';

	PolarProductParameters p;
	p.initialize("PCAPPI",this->parameters,src,dst);

	if (drain::Debug > 2){
		cerr << this->parameters;
		cerr << "---------------\nSRC" << src.properties << "\n";
		cerr << "---------------\nDST" << dst.properties << "\n";
	}

	if (drain::Debug > 1)
		cerr << " Starting CAPPI";
	const int altitude = this->parameters.get("altitude",0); // CAPPI intersection altitude

	dst.setGeometry(p.dstWidth,p.dstHeight,1,1);
	//cerr << dst.properties;

	if (drain::Debug > 1){
		cerr << " cappi source: " << src.getGeometry() << '\n';
		cerr << "   elevations [" << p.radarGeometry.elevationAngles.size() << "] \n";// << elevations << '\n';
		cerr << "   elevations (rad): ";
		for (unsigned int i=0; i< p.radarGeometry.elevationAngles.size(); i++){
			cerr << "'" <<  p.radarGeometry.elevationAngles[i] << "', ";
		}
		cerr << '\n';
		cerr << " dst: " << dst.getGeometry() << '\n';
	}


	drain::image::ImageView<T2> dstAlpha(dst.getAlphaChannel());

	drain::image::ImageView<T> srcUpper;
	drain::image::ImageView<T> srcLower;

	//bool SOURCE_WEIGHT = false;
	drain::image::ImageView<T> srcUpperAlpha;
	drain::image::ImageView<T> srcLowerAlpha;

	const unsigned int srcDataChannels = p.elangle.size();
	const unsigned int srcWeightChannels = src.getChannelCount() - srcDataChannels;

	// bin (index) of source beam (equidistant bin retrieval).
	int bin;

	// bin (index) of the resulting pseudo cappi beam (on earth surface).
	unsigned int groundBinSrc;

	// On-earth distance.
	float groundDistance;

	// On-earth angle.
	float beta;

	// Elevation index of the nearest beam BELOW the point,
	// or (-1) if all the beams are above.
	int kLower;

	// Elevation index of the nearest beam ABOVE the point.
	// or (-1) if all the beams are below.
	int kUpper;

	// Percentage of full power of the lower beam.
	float wLower;

	// Percentage of full power of the upper beam.
	float wUpper;

	// Sum of the weights
	float wNorm;

	// Larger of the weights
	float wMax;

	// The elevation angle (in radians) of the
	// beam that would hit the point.
	float elevationR;
	float eUpper;
	float eLower;


	// Convert (other) string_data to numbers (to maximize speed)

	const float beamWidthDeg = p.radarGeometry.beamWidth * 180.0/M_PI;
	//beamWidthDeg = src.properties.get("beamwidth",beamWidthDeg);  // TODO
	//beamWidthDeg = this->parameters.get("beamwidth",beamWidthDeg);
	p.radarGeometry.beamWidth = this->parameters.get("beamwidth",beamWidthDeg) * M_PI/180.0;

	if (drain::Debug > 2){
		cout << "Starting CAPPI\n";
		cout << "\nParams:\n";
		cout << this->parameters << '\n';

		cout << "\nGeometry:\n";
		cout << src.getGeometry() << '\n';
		cout << "\nProperties:\n";
		cout << src.properties << '\n';
		//cout << "\nElevs: " << elevations << "\n";
		cout << " srcs   " <<  srcDataChannels << "\n";
		cout << " alphas " <<  srcWeightChannels << "\n";
	}


	//cout << "\n";

	// Main loop
	for (unsigned int i=0; i<p.dstWidth; i++){

		// Scale (in width)
		groundBinSrc = (i * p.srcWidth) / p.dstWidth;
		//cappiBin = (i * width) / srcWidth;

		// IMPOSSIBLE to violate
		if (groundBinSrc < p.dstWidth) {

			groundDistance = static_cast<double>(groundBinSrc * p.rscale[0] + p.rscale[0]/2);
			beta = groundDistance / Geometry::EARTH_RADIUS_43;

			bin = static_cast<int>(Geometry::beamFromBetaH(beta, altitude) /  p.rscale[0]);
			elevationR = p.radarGeometry.etaFromBetaH(beta, altitude);



			/*! Find out the closest curves above and below.
			 */
			p.radarGeometry.findClosestElevations(elevationR, kLower, eLower, kUpper, eUpper);

			if (drain::Debug > 5){
				cerr << " bin=" << bin << '\t';
				cerr << " kLower,kUpper=" << kLower << ',' << kUpper << '\n';
			}

			/*! Compute weights of the upper and lower beams
			 *  from their gaussian power curve:
			 *  The greater the distance to the beam, the lower the weight.
			 *
			 *  Applies source data channel(s) if available. Accepts one channel (reporting quality for every channel)
			 *  and several channels (each alpha channel corresponding to data channels)
			 */
			if (kUpper != -1)
			{
				wUpper = p.radarGeometry.normalizedBeamPower(eUpper - elevationR);
				srcUpper.viewChannel(src,kUpper);
				if (srcWeightChannels){
					unsigned int a = srcDataChannels + min((unsigned int)kUpper,srcWeightChannels-1);
					srcUpperAlpha.viewChannel(src,a);
					if (drain::Debug > 6)
						cerr << " CAPPI(upper): viewing weight channel: " << a << '\n';
				}
			}
			else
				wUpper = 0.0;

			if (kLower != -1)
			{
				wLower = p.radarGeometry.normalizedBeamPower(eLower - elevationR);
				srcLower.viewChannel(src,kLower);
				if (srcWeightChannels){
					unsigned int a =  srcDataChannels + min((unsigned int)kLower,srcWeightChannels-1);
					srcLowerAlpha.viewChannel(src,a);
					if (drain::Debug > 6)
						cerr << " CAPPI(lower): viewing weight channel: " << a << '\n';
				}
			}
			else
				wLower = 0.0;

			wNorm = wLower + wUpper;
			wMax =  max(wLower,wUpper); // * 255.0;

			/*
				cout << " wL=" << wLower;
				cout << " wU=" << wUpper;
				cout << " w="  << wNorm << '\n';
			 */
			const float WLOWER = wLower;
			const float WUPPER = wUpper;

			if (kLower != -1) {

				//const drain::image::image<T> &lower = source.channel(k_lower);

				if (kUpper != -1){
					// between two beams
					for (unsigned int j=0; j<p.dstHeight; j++){
						if (srcWeightChannels){
							wLower = WLOWER / 255.0 * srcLowerAlpha.at(bin,j);
							wUpper = WUPPER / 255.0 * srcUpperAlpha.at(bin,j);
							wNorm = wLower + wUpper;
							wMax =  max(wLower,wUpper);
						}
						dst.at(i,j) = //static_cast<T>(0.5*srcLower.at(bin,j) + 0.5*srcUpper.at(bin,j));
								static_cast<T>( (wLower*srcLower.at(bin,j) + wUpper*srcUpper.at(bin,j))/wNorm);
						dstAlpha.at(i,j) = static_cast<T2>(wMax * 255.0);
					}
				}
				else {
					/// over the highest beam
					for (unsigned int j=0; j<p.dstHeight; j++){
						dst.at(i,j) = srcLower.at(bin,j);
						if (srcWeightChannels){
							wLower = WLOWER  / 255.0 * srcLowerAlpha.at(bin,j);  // TODO scale
						}
						dstAlpha.at(i,j) = static_cast<T2>(wLower * 255.0);
					}
				}
			}
			else
				/// under the lowest beam
				if (kUpper != -1){
					for (unsigned int j=0; j<p.dstHeight; j++){
						dst.at(i,j) = srcUpper.at(bin,j);
						if (srcWeightChannels){
							wUpper = WUPPER  / 255.0 * srcUpperAlpha.at(bin,j); // TODO scale
						}
						dstAlpha.at(i,j) = static_cast<T2>(wUpper * 255.0);
					}
				}
			/// no beams anywhere... something's wrong
				else {
					std::cerr << " pseudo_cappi<T>:: ERROR no beams ";
					std::cerr << std::endl;
				}



		}


	}


};


} // ::image

} // ::drain

#endif /*POLARCAPPI_H_*/
