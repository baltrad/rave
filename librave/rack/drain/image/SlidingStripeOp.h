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
#ifndef SLIDINGSTRIPEOP_H_
#define SLIDINGSTRIPEOP_H_


#include "SlidingWindowOp.h"
//#include "SlidingStripe.h"

namespace drain
{

namespace image
{

/// A class for Nx1 or 1xN windows.
/**
 * 
 *
 * 
 */
template <class T=unsigned char,class T2=unsigned char>
class SlidingStripe : public SlidingWindow<T,T2>
{
public:
	 SlidingStripe(int width=1,int height=1) {
	   this->setSize(width,height);
	 };
	 	  
	 virtual void setSize(unsigned int width,unsigned int height = 1){
	 	
	 	SlidingWindow<T,T2>::setSize(width,height);
	 	
	 	if (height == 1){
	 		horizontal = true;	
	 	}
	    else if (width == 1){
	    	horizontal = false;
	    }
	    else 
	      cerr << "Warning: SlidingStripe::set not 1xN or Nx1 window. " << *this;
	    
	    
	    
	 }
	
	protected:
	 	bool horizontal;
		 
};

/// Speed-optimized operator for computations that can be carried out with subsequent 1xH and Wx1 operations.
/** The user should not set the size of the stripe, but "width" and "height" of Op parameters. 
 * 
 *  TODO: this is actually stripeD op, because 1xn  => 1xn * 1x1
 */
template <class T=unsigned char,class T2=unsigned char>
class SlidingStripeOp : public WindowOp<T,T2>
{
public:
	SlidingStripeOp(SlidingStripe<T,T2> & stripe1, SlidingStripe<T2,T2> & stripe2, const string & p = "") :
		WindowOp<T,T2>(p), stripe1(stripe1), stripe2(stripe2) {
	};
	
	// Ongelma se että stripeä ei voi käyttää viestintuojana
	virtual void initialize() const {
		virtualWidth  = this->parameters.get("width",3);		
		virtualHeight = this->parameters.get("height",virtualWidth);
	}
	
	
	/// This should be final.
	/// If you want to redefine something, try changing initialize() first.
	virtual void filter(const Image<T> &src,Image<T2> &dst) const {
			
		initialize(); 
		
		Image<T2> dstTemp;
		
		//cerr << "Sliding horizontal \n";
		
		SlidingWindowOp<T,T2> op(stripe1); //,"width,height","0,0");
		//cerr << "Sliding horizontal \n";
		op.setParameter("width",virtualWidth);
		op.setParameter("height",1);
		op.filter(src,dstTemp);
	    
		SlidingWindowOp<T2,T2> op2(stripe2); //,"width,height","0,0");
		//cerr << "Sliding vertical \n";
	    op2.setParameter("width",1);
		op2.setParameter("height",virtualHeight);
		//cout << "ssPARS: " << op.parameters;
		op2.filter(dstTemp,dst);
		
	};
	
	/// This should be final.
	/// If you want to redefine something, try changing initialize() first.
	virtual void filter(const Image<T> &src,const Image<T> &srcWeight,
		Image<T2> &dst, Image<T2> &dstWeight){
			
		initialize(); 
		
		Image<T2> dstTemp;
		Image<T2> dstTempWeight;
		
		SlidingWindowOp<T,T2> op(stripe1); //,"width,height","0,0");
		
		//cerr << "Sliding Hhorizontal \n";
		op.setParameter("width",virtualWidth);
		op.setParameter("height",1);
		op.filter(src,srcWeight,dstTemp,dstTempWeight);
	    
	    SlidingWindowOp<T2,T2> op2(stripe2); //,"width,height","0,0");
		//cerr << "Sliding vertical \n";
	    op2.setParameter("width",1);	
		op2.setParameter("height",virtualHeight);
		op2.filter(dstTemp,dstTempWeight,dst,dstWeight);
		
	};
	
protected:
	SlidingStripe<T,T2>  & stripe1;
	SlidingStripe<T2,T2> & stripe2;

	
private:
	mutable unsigned int virtualWidth;
	mutable unsigned int virtualHeight;
	
};

}

}

#endif /*SLIDINGSTRIPEOP_H_*/
