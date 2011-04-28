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
#ifndef FUZZYPEAKOP_H_
#define FUZZYPEAKOP_H_

#include <math.h>


#include "SequentialImageOp.h"

using namespace std;

namespace drain
{

namespace image
{

/// FuzzyPeak correction: intensitys is mapped as f' = f^(gamma)
/*! FuzzyPeak correction
 * 
 *
 */
template <class T = unsigned char,class T2 = unsigned char>
class FuzzyPeakOp : public SequentialImageOp<T,T2>
{
public:

	//FuzzyPeakOp(const Data &option);
	FuzzyPeakOp(const string & p = "0,128,255") :
		SequentialImageOp<T,T2>("FuzzyPeakOp","Gaussian-like peak.","location,width,scaleDst",p){
		//this->setInfo("Fuzzy Gaussian-like peak.","location,width,scaleDst",p);
	}
	
	FuzzyPeakOp(double location,double width, double scaleDst=255.0) :
		SequentialImageOp<T,T2>("FuzzyPeakOp","Gaussian-like peak.","location,width,scaleDst",""){
		this->setParameter("location",location);
		this->setParameter("width",width);
		this->setParameter("scaleDst",scaleDst);
		//this->setInfo("Fuzzy Gaussian-like peak.","location,width,scaleDst",p);
	}

	/// - scale of source image -
	//FuzzyPeakOp(double width,double location){
	void initialize() const {
		double w = this->parameters.get("width",Intensity::max<T2>()/2.0);
		width2 = w*w;
		location = this->parameters.get("location",0);
		scaleDst = this->parameters.get("scaleDst",static_cast<double>(Intensity::max<T2>()));
		if (drain::Debug > 3)
			cerr << "FuzzyPeak: " << this->location << ',' << this->width2 << ',' << this->scaleDst << '\n';
	};
	
	inline 
	void filterValue(const T &src, T2 &dst) const {
		static double x;
		x = static_cast<double>(src) - this->location ;
		
		dst = static_cast<T2>(this->scaleDst*width2 / (width2 + x*x));
		  
	}; 
	
protected:

	mutable double width2;
	mutable double location;
	mutable double scaleDst;
	
};

}
}

#endif /*GAMMAOP_H_*/
