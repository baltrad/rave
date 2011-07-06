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
#ifndef HIGHPASSOP_H_
#define HIGHPASSOP_H_

//#include "SlidingWindowAverage.h"
#include "ImageOp.h"

#include "FastAverageOp.h"
#include "MathOpPack.h"


namespace drain
{

namespace image
{

/// Difference of image and its low-pass filtered copy. 
/**
 *   Finds details of one-pixel scale. 
 *   Implements
 *   The simplest square high-pass filter
 *   -1  -1  -1
 *   -1   8  -1
 *   -1  -1  -1 
 * 
 *   The result can be scaled with coeff and transposed with bias.
 *   Internally, applies the fast SlidingStripeAverage and SubtractOp .
 */
template <class T=unsigned char,class T2=unsigned char>
class HighPassOp : public ImageOp<T,T2>
{
public:

	HighPassOp(const string & p = "3,3,1.0,0"){
    	this->setInfo("High-pass filter for recognizing details.",
    		"width,height,scale,coeff",p);
	};
    
	virtual void filter(const Image<T> &src,Image<T2> &dst) const {
		makeCompatible(src,dst);
		
		const int  width = this->parameters.get("width",3);
		const int height = this->parameters.get("height",width);
		FastAverageOp<T,T2>(width,height).filter(src,dst);
		
		const float scale = this->parameters.get("scale",-1.0/(float)width*(float)height); // ????
		const float bias  = this->parameters.get("bias",0);
		//ScaledSubtractionOp<T,T2>(scale).filter(src,dst,dst);
		SubtractionOp<T,T2>(scale,bias).filter(src,dst,dst);
		
	};


};


}

}

#endif /*HIGHPASSOP_H_*/
