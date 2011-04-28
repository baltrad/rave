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
#include "SubComposite.h"

#include "../image/DistanceTransformFillOp.h"
#include "../image/RecursiveRepairerOp.h"
#include "../image/ThresholdOp.h"

namespace drain
{

namespace radar
{

/**! author mpeura
 *
 */
SubComposite::SubComposite(Composite &composite) :  composite(composite)
{
	lat = 0.0;
	lon = 0.0;
	range = 0.0;
	
	xOffset = 0;
	xOffset = 0;
	p = composite.p;
	r = composite.r;
	method = composite.method;

	globalWeight = 1.0;
	horzDecrement = -1;
	vertDecrement = 0;
	
}


void SubComposite::detectBoundingBox(){

	const unsigned int azimuthStep = 12;
	const float D2R = M_PI/180.0;
	const int w =  composite.getWidth();
	const int h =  composite.getHeight();

	int iMin = w-1; // numeric_limits<int>::max();
	int jMin = h-1; // numeric_limits<int>::max();
	int iMax = 0;   // numeric_limits<int>::min();
	int jMax = 0;   // numeric_limits<int>::min();

	//const bool drawCircle = (azimuthStep == 0);
	//if (drawCircle) 	  azimuthStep = 1;

	// Jumps around 360 degrees, finding out coordinate bounds
	// TODO: why not via Coordinates.getBoundingBox ?
	// Answer: composite may map coords though a projection?
	float azimuth;
	int i;
	int j;
	
	for (int a = 0; a < 360; a += azimuthStep) {
    	azimuth = a * D2R;
    	coordinates.setBinPosition(azimuth,range);
		//cout << a <<  " : " << coordinates.binLatitudeDeg() << ',' << coordinates.binLongitudeDeg() << '\t';
		composite.map(coordinates.phiBin,coordinates.thetaBin,i,j);
		/*
		if ((i>=0)&&(i<w)&&(j>=0)&&(j<h)){
			if (drawCircle)
		   	  composite.add(i,j,255,255);
		}
		*/
		iMin = min(i,iMin);
		jMin = min(j,jMin);
		iMax = max(i,iMax);
		jMax = max(j,jMax);
	}

	//cout << " min: " << iMin << ',' << jMin << '\n';
	//cout << " max: " << iMax << ',' << jMax << '\n';
	// Finally, crop by image dimensions.
	iMin = max(0,iMin);
	jMin = max(0,jMin);
	iMax = min(w-1,iMax);
	jMax = min(h-1,jMax);
	//cout << " min: " << iMin << ',' << jMin << '\n';
	//cout << " max: " << iMax << ',' << jMax << '\n';

	/// TODO warning if suspiciously large
	setGeometry(static_cast<unsigned int>(iMax-iMin)+1,static_cast<unsigned int>(jMax-jMin)+1);
	xOffset = iMin;
	yOffset = jMin;
	//cout << " size: " << width << ',' << height << '\n';

	// TODO: auto rectDecrement
  }

	/*
	void SubComposite::flush(){
		composite.allocate();

		drain::image::Image<double> result;
		drain::image::Image<double> resultWeight;

		extractTo(result,"d");
		extractTo(resultWeight,"w");

    	drain::image::DistanceTransformFillExponentialOp<double,double> op;
    	//TODO drain::image::RecursiveRepairerOp<double,double> op2;
    	   	
    	// Auto mode
    	if (horzDecrement == 0){
    		/// TODO  LOGIIKKA MUUTETTU 02/2010, TOIMIIKOHAN?
    		op.parameters["horz"] = static_cast<float>(dxMax)/10.0; // 128/(dxMax+1);
    		op.parameters["vert"] = static_cast<float>(dyMax)/10.0; // 128/(dyMax+1);  // WAS dxMax ??
    	}
    	// User defined
    	else {
    		op.parameters["horz"] = horzDecrement;
    		op.parameters["vert"] = vertDecrement;
    	}
    	
    	cerr << op.parameters << '\n';

    	op.filter(result,resultWeight,result,resultWeight);

    	drain::image::ThresholdOp<double,double>("8").filter(resultWeight,resultWeight);

    	composite.addImage(result,resultWeight,globalWeight,xOffset,yOffset);


	}
	*/

}

}
