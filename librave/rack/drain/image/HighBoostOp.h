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
#ifndef HIGHBOOSTOP_H_
#define HIGHBOOSTOP_H_

//#include "SlidingWindowAverage.h"
#include "ImageOp.h"

#include "FastAverageOp.h"
#include "MathOpPack.h"


namespace drain
{

namespace image
{


// Returns src - scale*src2
template <class T=unsigned char,class T2=unsigned char>
class ScaledSubtractionOp: public MathOp<T,T2>
{
public:

   ScaledSubtractionOp(const string & p = "1.0,0") : MathOp<T,T2>(p) {
	   this->setInfo("An arithmetic operator.","scale,bias",p);
   };

   void filterValue(const T &src, const T2 &src2, T2 &dst) const {
   		dst = MathOp<T,T2>::limit(static_cast<double>(src) - this->scale*src2 + this->bias);
   };
};

/// Enhances details by mixing original image and result of HighPassOp op.
/**
 */
template <class T=unsigned char,class T2=unsigned char>
class HighBoostOp : public ImageOp<T,T2>
{
public:

	HighBoostOp(const string & p = "3,3,0.5"){
    	this->setInfo("Mixture of original and high-pass filtered image.",
    		"width,height,mixCoeff",p);
	};
    
	virtual void filter(const Image<T> &src,Image<T2> &dst) const {
		makeCompatible(src,dst);
		
		const int  width = this->parameters.get("width",3);
		const int height = this->parameters.get("height",width);
		FastAverageOp<T,T2>(width,height).filter(src,dst);
		
		const string &coeff = this->parameters.get("mixCoeff","0.5"); // ????
		//SubtractionOp<T,T2>(1,0).filter(src,dst,dst);
		ScaledSubtractionOp<T,T2>(coeff).filter(src,dst,dst);
	};


};


}

}

#endif /*HIGHPASSOP_H_*/
