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
#ifndef Double_SMOOTHEROP
#define Double_SMOOTHEROP "DoubleSmootherOp Markus.Peura@iki.fi"

#include "WindowOp.h"

#include "FastAverageOp.h"
#include "MathOpPack.h"
//#include "SlidingStripeAverageOp.h"

namespace drain
{
namespace image
{


//  TODO delicateSmoother (quadraticsmoother)
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
class DoubleSmootherOp: public WindowOp<T,T2>
{

public:

	DoubleSmootherOp(const string & p = "5,5,0.5") :
		WindowOp<T,T2>("DoubleSmootherOp",
				"Smoothes image twice, mixing first result with coeff*100%.",
				"width,height,coeff", p){
	};

	virtual void initialize() const {
	}

	virtual void filter(const Image<T> &src,Image<T2> &dst) const {

		const int width  = this->getParameter("width",3);
		const int height = this->getParameter("height",width);
		const double coeff  = this->getParameter("coeff",0.5);

		Image<T2> tmp;
		FastAverageOp<T,T2>(width,height).filter(src,tmp);
		MixerOp<T,T2>(coeff).filter(src,tmp,dst);
		//FastAverageOp<T,T2>(width,height).filter(tmp,dst);

	}


};




} // namespace drain

} //namespace image


#endif
