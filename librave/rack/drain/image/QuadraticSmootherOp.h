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
#ifndef QUADRATIC_SMOOTHEROP
#define QUADRATIC_SMOOTHEROP "QuadraticSmootherOp Markus.Peura@iki.fi"

#include "WindowOp.h"

#include "FastAverageOp.h"
#include "MathOpPack.h"
//#include "SlidingStripeAverageOp.h"

namespace drain
{
namespace image
{


//  see double smoother... TODO delicate smoother
// -------------------------------------------------
/*! This operator produces
 *  <CODE>
 *  F2 = cF + (1-c)M{ cF + (1-c)M{F} }
 *     = cF + (1-c)cM{F} + (1-c)^2M^2{F}
 *  </CODE>
 *  where 
 *  <CODE>F</CODE> is an image,
 *  <CODE>M{}</CODE> is a <CODE>W x W</CODE> WindowAverage operator, and
 *  <CODE>c</CODE> is a coefficient.
 */
template <class T=unsigned char,class T2=unsigned char>
class QuadraticSmootherOp: public WindowOp<T,T2>
{

public:

	QuadraticSmootherOp(const string & p = "5,5,0.5") :
		WindowOp<T,T2>("QuadraticSmootherOp",
				"Smoothes image twice, mixing original image with coeff*100%.",
				"width,height,coeff", p){
	};

	virtual void initialize() const {
	}

	virtual void filter(const Image<T> &src,Image<T2> &dst) const {

		const int width  = this->getParameter("width",3);
		const int height = this->getParameter("height",width);
		const double coeff  = this->getParameter("coeff",0.5);
		//double coeffInside  = this->getParameter("coeff",0.5);

		Image<T> tmp;
		FastAverageOp<T,T>(width,height).filter(src,tmp);
		MixerOp<T,T>(coeff).filter(src,tmp,tmp);
		FastAverageOp<T,T2>(width,height).filter(tmp,dst);
		MixerOp<T,T2>(coeff).filter(src,dst,dst);

	}


};




} // namespace drain

} //namespace image


#endif
