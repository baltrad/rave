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
#ifndef SLIDINGWINDOWHISTOGRAM_H_
#define SLIDINGWINDOWHISTOGRAM_H_



#include "SlidingWindow.h"
#include "../util/Histogram.h"

namespace drain
{

namespace image
{


/** 
 *
 */
template <class T=unsigned char,class T2=unsigned char>
class SlidingWindowHistogram : public SlidingWindow<T,T2> {

  public:
	SlidingWindowHistogram() {};
		
	/*
	SlidingWindowHistogram(const string & p) {
		this->setInfo("An optimized pipeline algorithm for computing window histograms.","",p);
		//this->parameters.setAllowedKeys("width,height,percentage");
		//this->parameters.set(p);
		this->setSize(1,1);
	};
	*/ 
	/*
		// int width = 0, int height = 0){ 
		// float limitPercentage=0.5){
		this->parameters.setAllowedKeys("width,height");
		this->parameters.set(p);	
		//this->setSize(width,height);
		//setLimit(limitPercentage);
	}
	*/
	
	
	
	
	/** 
	 *   TODO: weight = > weightSUm
	 *   TODO: multiple channels (banks)?
	 */
	void initialize(){
		cerr << "SlidingWindowHistogram::initialize()\n";
		
		//unsigned int w = this->parameters.get("width",1);
		//unsigned int h = this->parameters.get("height",w);
		//this->setSize(w,h);
		
		
		//cerr << this->parameters << "\n";
		
		//area = this->getArea();
		histogram.clear();
		histogram.setSize(256);
		histogram.setSampleCount(this->getArea());
		//histogram.resize(256,0);
		
		this->location.setLocation(0,0);
		
		//cerr << "area" << area << "\n";
		//cerr << this->dst.getCoordinateHandler();
		
		for (int i = this->iMin; i <= this->iMax; i++) {
			for (int j = this->jMin; j <= this->jMax; j++) {
				this->p.x = i;
				this->p.y = j;
				this->dst.getCoordinateHandler().handle(this->p);
				++histogram[this->src.at(this->p)];
			}
		}
		
	}
	
		
	virtual void updateHorz(int dx){
		
		const int xOld = this->getXOld(dx); // location.x + (dx>0 ? iMin-1 : iMax+1);
		const int xNew = this->getXNew(dx); //location.x + (dx>0 ? iMax     : iMin);

		for (int j = this->jMin; j <= this->jMax; j++) {
			this->p.y = this->location.y+j;	
			
			this->p.x = xOld;
			this->dst.getCoordinateHandler().handle(this->p);
			--histogram[this->src.at(this->p)];
			
			this->p.x = xNew;
			this->dst.getCoordinateHandler().handle(this->p);
			++histogram[this->src.at(this->p)];
		}
	}
	

	virtual void updateVert(int dy){
		
		const int yOld = this->getYOld(dy); // location.y + (dy>0 ? jMin-1 : jMax+1);
		const int yNew = this->getYNew(dy); // location.y + (dy>0 ? jMax     : jMin);
		
		for (int i=this->iMin; i<=this->iMax; i++) {
			this->p.x = this->location.x+i;
			
			this->p.y = yOld;
			this->dst.getCoordinateHandler().handle(this->p);
			--histogram[this->src.at(this->p)];
			
			this->p.y = yNew;
			this->dst.getCoordinateHandler().handle(this->p);
			++histogram[this->src.at(this->p)];
			
		}
		
	}
	
	virtual void write() = 0;
	
	
	protected:
	
	drain::Histogram<unsigned int> histogram;
	
	
	
  };	

}
}

#endif
