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
#ifndef SLIDINGWINDOWMEDIANOP_H_
#define SLIDINGWINDOWMEDIANOP_H_


#include "SlidingWindowHistogram.h"

namespace drain
{

namespace image
{


/** 
 *
 */
template <class T=unsigned char,class T2=unsigned char>
class SlidingWindowMedian : public SlidingWindowHistogram<T,T2> {

  public:
	
	SlidingWindowMedian(){ //const string & p = "5,5,0.5"){
		//this->setInfo("A pipeline implementation of window median.","width,height,percentage",p);
	};
	
	//Value 0.5 corresponds to standard median
	void setLimit(float percentage){ limitPercentage = percentage;}
	// int width = 0, int height = 0, float limitPercentage=0.5) : SlidingWindowHistogram<T,T2>(width,height) {
	//	setLimit(limitPercentage);
	virtual void initialize(){
		cerr << "SlidingWindowMedian::initialize()\n";
		//setLimit(this->parameters.get("percentage",0.5));
		SlidingWindowHistogram<T,T2>::initialize();
	}
	
	virtual void write(){
		this->dst.at(this->location) = this->histogram.getMedian(this->limitPercentage);
	}
	
   protected: 
	float limitPercentage;  
	
  };	
  
template <class T=unsigned char,class T2=unsigned char>
class SlidingWindowMedianOp : public SlidingWindowOp<T,T2>
{
public:
	SlidingWindowMedianOp(const string & p = "5,5,0.5")
		: SlidingWindowOp<T,T2>(medianWindow,"SlidingWindowMedian","A pipeline implementation of window median.","width,height,percentage",p){
	};
	
	virtual void initialize() const {
		SlidingWindowOp<T,T2>::initialize();
		this->medianWindow.setLimit(this->getParameter("percentage",0.5));
	}
	
	
protected:
	mutable SlidingWindowMedian<T,T2> medianWindow;
};

}
}

#endif // MEDIAN
