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
#ifndef FUZZYTHRESHOLDOP_H_
#define FUZZYTHRESHOLDOP_H_

#include <math.h>


//#include "SequentialImageOp.h"
#include "ThresholdOp.h"

using namespace std;

namespace drain
{

namespace image
{

/// FuzzyThreshold correction: intensity is mapped as f' = f^(gamma)
/*! FuzzyThreshold correction
 * 
 *   @see
 */
template <class T = unsigned char,class T2 = unsigned char>
class FuzzyThresholdOp : public SequentialImageOp<T,T2>
	// public ThresholdOp<T,T2>
{
public:

	/// Future versions may use dstMin, dstMax, srcMin, srcMax
	FuzzyThresholdOp(const string & p = "0,1,0,255") :
		SequentialImageOp<T,T2>("FuzzyThresholdOp","Smooth transition.","location,width,dstMin,dstMax",p){
	}

	/// - scale of source image -
	void initialize() const {
		//cout << this->parameters << '\n';
		location = this->parameters.get("location",0);
		width = this->parameters.get("width",(double)Intensity::max<T2>()/2.0);
		float min = this->parameters.get("dstMin",(double)Intensity::min<T2>());
		float max = this->parameters.get("dstMax",(double)Intensity::max<T2>());
		scaleDstHalf  = (min+max)/2.0;
		scaleDstCoeff = (max-min)/2.0;
		if (drain::Debug > 3){
			cerr << "FuzzyTheshold: " << this->location << ',' << this->width << ',';
			cerr << " " << scaleDstHalf << ' ' << scaleDstCoeff << '\n';
		}
	};

	inline 
	void filterValue(const T &src, T2 &dst) const {
		double x = static_cast<double>(src) - location;
		if (x>0){
			dst = static_cast<T2>(scaleDstHalf + (scaleDstCoeff * x)/(width + x));	
		}
		else {
			dst = static_cast<T2>(scaleDstHalf + (scaleDstCoeff * x)/(width - x));	
		}
	}; 

protected:

	mutable double location;
	mutable double width;
	mutable double scaleDstHalf;
	mutable double scaleDstCoeff;

};

}
}

#endif /*GAMMAOP_H_*/
