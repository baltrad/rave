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
#ifndef GradientOP_H_
#define GradientOP_H_


#include <math.h>

#include "ImageOp.h"

namespace drain
{

namespace image
{


/**! Computes spatial horizontal derivative, dx.
 *    src(i,j)-src(i-1,j). 
 * 
 */
template <class T=unsigned char,class T2=unsigned char>
class GradientOp : public ImageOp<T,T2> 
{
public:

	//GradientOp() : bias(0), scale(1), offsetLo(1), offsetHi(1) {};

	
	void filter(const Image<T> &src,Image<T2> &dst,int diLow,int djLow,int diHigh,int djHigh) const {

			makeCompatible(src,dst);

			const int width  = src.getWidth();
			const int height = src.getHeight();

			const float bias  = this->getParameter("bias",0.0);

			const int iSpan = diLow+diHigh;
			const int jSpan = djLow+djHigh;
			const float span = sqrt(iSpan*iSpan + jSpan*jSpan);

			if (span == 0.0)
				throw (runtime_error("GradientOp: zero span"));

			const float scale = this->getParameter("scale",1.0f) / span;

			if (drain::Debug > 0){
				cerr << this->name;
				cerr << " bias=" << bias;
				cerr << " scale=" << scale;
				cerr << '\n';
			}

			Point2D<> pLo;
			Point2D<> pHi;
			const CoordinateHandler & h = src.getCoordinateHandler();

			for (int j = 0; j < height; j++) {
				for (int i = 0; i < width; i++) {
					pLo.setLocation(i-diLow,j-djLow);
					h.handle(pLo);
					pHi.setLocation(i+diHigh,j+djHigh);
					h.handle(pHi);
					dst.at(i,j) = static_cast<T2>(bias + scale*(src.at(pHi) - src.at(pLo)));
				}
			}
		};


/*
		
  protected:
	mutable float bias;  // TODO full scale! Intensity::
	mutable float scale;
	mutable int offsetLo;
	mutable int offsetHi;
	*/
	//virtual void filter(const Image<T> &src,Image<T2> &dst) const = 0;
};


/**! Computes spatial horizontal derivative, dx.
 *    src(i,j)-src(i-1,j). 
 * 
 */
template <class T=unsigned char,class T2=unsigned char>
class GradientHorizontalOp : public GradientOp<T,T2>
{
public:
	
	GradientHorizontalOp(){
		this->setInfo("Horizontal intensity Gradient","bias,scale,span","0,1,2");
	};

	void filter(const Image<T> &src,Image<T2> &dst) const {
		const int span = this->getParameter("span",2);
		if (span>0)
			GradientOp<T,T2>::filter(src, dst, span/2, 0, (span+1)/2, 0);
		else
			GradientOp<T,T2>::filter(src, dst, span/2, 0, (span-1)/2, 0);
	}
		
};

/**! Computes spatial vertical derivative, dy.
 *    src(i,j)-src(i,j-1). 
 * 
 */
template <class T=unsigned char,class T2=unsigned char>
class GradientVerticalOp : public GradientOp<T,T2>
{
public:
	
	GradientVerticalOp(){
		this->setInfo("Vertical intensity Gradient","bias,scale,span","0,1,2");
	};

	void filter(const Image<T> &src,Image<T2> &dst) const {
		const int span = this->getParameter("span",2);
		if (span>0)
			GradientOp<T,T2>::filter(src, dst, 0,span/2, 0, (span+1)/2);
		else
			GradientOp<T,T2>::filter(src, dst, 0,span/2, 0, (span-1)/2);
	};
	
};

}
}


#endif /*GradientOP_H_*/
