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
#ifndef POLARPRODUCT_H_
#define POLARPRODUCT_H_

#include "Geometry.h"

#include "../image/Image.h"
#include "../image/ImageView.h"
#include "../image/ImageOp.h"

//#include "geometry.h"

namespace drain
{

namespace radar
{

using namespace std;
//using namespace drain::image;


class PolarProductParameters {

public:

	/** Copies and sets standard variables from the source image and operator parameters to destination.
	 *
	 */
	template <class T,class T2>
	void initialize(const string &productName, const Options & productParameters,
			const image::Image<T> &src,image::Image<T2> &dst){

		srcWidth = src.getWidth();
		srcHeight = src.getHeight();


		dstWidth =  productParameters.get("width",srcWidth);
		dstHeight = srcHeight;

		const Options & s = src.properties;


		// Scalars
		lat = s.get("/where/lat",60.0);
		lon = s.get("/where/lon",20.0);

		// Vectors
		if (drain::Debug > 3)
			cerr << "ProductParams split start" << endl;

		s.get("@where/rscale").splitTo(rscale);
		s.get("@where/rstart").splitTo(rstart);
		s.get("@where/elangle").splitTo(elangle);
		s.get("@where/nbins").splitTo(nbins);
		// s["@what/parameter"].splitTo(pars);
		//cerr << "ProductParams split" << endl;
		if (drain::Debug > 3)
			cerr << " elangle size=" << elangle.size() << endl;

		radarGeometry.elevationAngles.resize(elangle.size());

		for (unsigned int i = 0; i < elangle.size(); ++i) {
			radarGeometry.elevationAngles[i] = elangle[i]*M_PI/180.0;
		}

		if (drain::Debug > 3)
			cerr << "ProductParams split ok" << endl;

		Options & d = dst.properties;

		d = s;
		d["/how/rack/prodpars"] = productParameters.getParameterValues();
		//d["/what/rscale"] = ;
		d["/what/rstart"] = 0;
		d["/what/object"] = "SCAN";
		d["/what/product"] = productName;
		//d["/what/prodpar"] = static_cast<double>(altitude);
		//dst.properties["product"] << altitude;
		unsigned int rscaleDst = d.get("/where/rscale");
		d["/where/rscale"] = (rscaleDst * srcWidth) / dstWidth;
		d["/where/nbins"] = dstWidth;
		//d["/where/altitude"] = altitude;

	};

	radar::Geometry radarGeometry;

	double lat;
	double lon;
	vector<double> rscale;
	vector<double> rstart;
	vector<double> elangle;
	vector<unsigned int> nbins;

	vector<string> dataParam;

	unsigned int srcWidth;
	unsigned int srcHeight;

	unsigned int dstWidth;
	unsigned int dstHeight;

};


/*! A base class for polarimetric products.
 *  Typically, derived products will have following parameters:
 *  -# width of a
 */
template <class T = unsigned char,class T2 = unsigned char>
class PolarProduct : public image::ImageOp<T,T2>
{
public:

	PolarProduct(const string &name = "PolarProductOp",const string &description="Polar product generator.",
			   const string & parameterNames="", const string & defaultValues = "") :
				   image::ImageOp<T,T2>(name,description,parameterNames,defaultValues){
	};



};




} // ::image

} // ::drain

#endif /*POLARCAPPI_H_*/
