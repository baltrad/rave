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
#ifndef FLOODFILL2_H
#define FLOODFILL2_H

#include "ImageOp.h"
#include "CoordinateHandler.h"


namespace drain
{
namespace image
{
	
//template <class T = unsigned char, class T2 = unsigned char>
//class FloodFillRecursion;  // see bottom of this file

// The actual operator is defined at the bottom of this file.


/// Helper class (not an operator itself) applied by FloodFillOp and SegmentAreaOp
/**
 *   \author Markus.Peura@fmi.fi
 */
template <class T = unsigned char, class T2 = unsigned char>
//template <class T, class T2>
class FloodFillRecursion {
  public: 
	 FloodFillRecursion(const Image<T> &s,Image<T2> &d) : src(s), dst(d), h(s.getCoordinateHandler()) {};
	 
	 /// Fills the segment having constant intensity, that is, src.at(i0,j0).
	 unsigned long int fill(unsigned int i, unsigned int j, T2 fillValue){
	 	dst.setGeometry(src.getWidth(),src.getHeight(),
	 		max(1u,dst.getImageChannelCount()),dst.getAlphaChannelCount()); 
	 	return fill(i,j,fillValue,src.at(i,j),src.at(i,j));
	  };
	  
	  /// Fills the segment having intensity between min and max.
	  unsigned long int fill(unsigned int i, unsigned int j, T2 fillValue, T min, T max = Intensity::max<T>()){
	  	value = fillValue;
	    anchorMin = min;
	 	anchorMax = max;
	    size = 0;
	   // cout << "range:" << (int)anchorMin  << '-' << (int)anchorMax  << '\n';
	    fill8(i,j);
	    //cout << "size:" << size  << '\n';
	    return size;
	  };
	  
  protected:
	 const Image<T> &src;
	 Image<T2> &dst;
	 CoordinateHandler &h;
	mutable T  anchorMin;
	mutable T  anchorMax;
	mutable T2 value;
	mutable Point2D<> p;
	mutable unsigned long int size;
	
    void fill8(unsigned int i,unsigned int j){
    
     	
     	// Is outside segment?
     	if (src.at(i,j) < anchorMin)
     		return;
     	if (src.at(i,j) > anchorMax)
     		return;
     	
     	// Is visited?
     	if (dst.at(i,j) == value)
     		return;
     	
     	
     	// MAIN ACTION:
     	dst.at(i,j) = value;
    	size++;
     	
     	p.setLocation(i-1,j);
    	if ((h.handle(p) & CoordinateHandler::IRREVERSIBLE) == 0)
    		fill8(p.x,p.y);
     	p.setLocation(i,j-1);	
     	if ((h.handle(p) & CoordinateHandler::IRREVERSIBLE) == 0)
     	    fill8(p.x,p.y);
     	p.setLocation(i+1,j);	
     	if ((h.handle(p) & CoordinateHandler::IRREVERSIBLE) == 0)
     	    fill8(p.x,p.y);
     	p.setLocation(i,j+1);	
     	if ((h.handle(p) & CoordinateHandler::IRREVERSIBLE) == 0)
     	    fill8(p.x,p.y);
    }

};


/**
 *   \author Markus.Peura@fmi.fi
 */
template <class T = unsigned char, class T2 = unsigned char>
class FloodFillOp : public ImageOp<T,T2> {
	
public:
	FloodFillOp(const string & p = "1,1,128,255,255"){
    	this->setInfo("Fills an area starting at (i,j) having intensity in [min,max], with a value.",
    		"i,j,min,max,value",p);
	};
    
    virtual void makeCompatible(const Image<T> &src,Image<T2> &dst) const  {
    		dst.setGeometry(src.getWidth(),src.getHeight(),
	 			max(1u,dst.getImageChannelCount()),dst.getAlphaChannelCount()); 
	}
    	
    virtual void filter(const Image<T> &src,Image<T2> &dst) const {
   	  		makeCompatible(src,dst);
   	  		const int i = this->parameters.get("i",0);
   	   		const int j = this->parameters.get("j",0);
   	   		const T min = this->parameters.get("min",128);
   	   		const T max = this->parameters.get("max",(int)Intensity::max<T>());
   	   		const T2 value = this->parameters.get("value",(int)Intensity::max<T2>());
   	   		FloodFillRecursion<> f(src,dst);
   	   		f.fill(i,j,value,min,max);
   	};
	
};


}
}
#endif /* FLOODFILL_H_ */

