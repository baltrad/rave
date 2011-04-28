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
#ifndef RUNLENGTHOP_H_
#define RUNLENGTHOP_H_

#include <stdexcept>


#include "ImageOp.h"


namespace drain
{

namespace image
{

/// Computes lengths of horizontal or vertical segments.
/** Fast distance transform using 8-directional distance function.
 *  Class D is used for computing distances?
 */
template <class T=unsigned char,class T2=unsigned char> //,class D=int>
class RunLengthOp : public ImageOp<T,T2>
{
public:
    
	/**
	 *   \param p - mode ("horz" or "vert") and threshold.
	 *
	 *   This operator does not check if dst==src.
	 */
	RunLengthOp(const string &p = "horz,128") :
		ImageOp<T,T2>("RunLength","Computes lengths of horizontal or vertical segments of intensity above threshold.",
				"mode,threshold",p) {
	};

	void filter(const Image<T> &src, Image<T2> &dst) const {
		const string & mode = this->parameters.get("mode","horz");
		if (mode == "horz"){
			filterHorz(src,dst);
		}
		else if (mode == "vert"){
			filterVert(src,dst);
		}
		else {
			// throw;
			throw runtime_error(string("RunLengthOp: illegal mode '") + mode + "'\n" + this->help());	
		}
		//return dst; 	
	}

	void filterHorz(const Image<T> &src, Image<T2> &dst) const ;
	void filterVert(const Image<T> &src, Image<T2> &dst) const ;
    
};
  

template <class T,class T2>
void RunLengthOp<T,T2>::filterHorz(const Image<T> &src, Image<T2> &dst) const {
	
	makeCompatible(src,dst);
	
	const T threshold = static_cast<T>(this->parameters.get("threshold",128.0));
	const int width  = src.getWidth();
	const int height = src.getHeight();
	
	Point2D<int> p;
	CoordinateHandler &handler = src.getCoordinateHandler();
	int hResult = 0;
	
	int pos;
	int length;
	T lengthT;
	const T lengthMax = Intensity::max<T>();
	
	for (int j=0; j<height; j++){
		
	  pos = 0;
	  length = 0;
				
	  while (pos<width){
	
		
		/// SCAN OVER EMPTY AREA
		pos = pos+length; // jump over last segment
		while (pos < width){
			p.setLocation(pos,j);
			hResult = handler.handle(p);
			if (hResult && CoordinateHandler::IRREVERSIBLE)
				break;
			if (src.at(p) >= threshold)
				break;
			dst.at(p) = 0;
			pos++;
		}
		
		// calculate  (note max _span_ = w)
		for (length=0; length<width; length++){
			p.setLocation(pos+length,j);
			hResult = handler.handle(p);
			if (hResult && CoordinateHandler::IRREVERSIBLE)
				break;
			if (src.at(p) < threshold)
				break;
		}	
		// mark
		lengthT = (length < lengthMax) ? length : lengthMax;
		for (int i=0; i<length; i++){
			p.setLocation(pos+i,j);
			handler.handle(p);  // handled ok above
			dst.at(p) = lengthT; 
		}
		//pos = pos+length;
		
	  }
	}
	
	
		
}


// kesken?
template <class T,class T2>
void RunLengthOp<T,T2>::filterVert(const Image<T> &src, Image<T2> &dst) const {
	
	makeCompatible(src,dst);
	
	const T threshold = static_cast<T>(this->parameters.get("threshold",128.0));
	const int width  = src.getWidth();
	const int height = src.getHeight();
	
	
	Point2D<int> p;
	CoordinateHandler &handler = src.getCoordinateHandler();
	int hResult = 0;
	
	int pos;
	int length;
	T lengthT;
	const T lengthMax = Intensity::max<T>();
	
	for (int i=0; i<width; i++){
		
	  pos = 0;
	  length = 0;
				
	  while (pos<height){
		
		// skip	
		pos = pos+length;
		while (pos < height){
			p.setLocation(i,pos);
			hResult = handler.handle(p);
			if (hResult && CoordinateHandler::IRREVERSIBLE)
				break;
			if (src.at(p) >= threshold)
				break;
			dst.at(p) = 0;
			pos++;
		}
		
		// calculate  (note max _span_ = w)
		for (length=0; length<height; length++){
			p.setLocation(i,pos+length);
			hResult = handler.handle(p);
			if (hResult && CoordinateHandler::IRREVERSIBLE)
				break;
			if (src.at(p) < threshold)
				break;
		}	
		// mark
		lengthT = (length < lengthMax) ? length : lengthMax;
		for (int j=0; j<length; j++){
			p.setLocation(i,pos+j);
			handler.handle(p);  // handled ok above
			dst.at(p) = lengthT; 
		}
		pos = pos+length;  // WARNING - UNNEEDED?
		
	  }
	}
	
	
		
}



}
}

#endif /* RUNLENGTHOP_H_*/
