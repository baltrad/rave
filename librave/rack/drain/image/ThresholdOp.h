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
#ifndef THRESHOLDOP_H_
#define THRESHOLDOP_H_

#include <cstdarg>

#include "SequentialImageOp.h"


namespace drain
{
namespace image
{

/// Thresholding performed in several ways.
/**! 
 *   The given threshold may be absolute or relative (0.0...1.0),
 *   in which case the intensity range is assumed to be that returned
 *   by ImageOp::getScale(). For example, in the case of unsigned char,
 *   a relative threshold of 0.5 corresponds to absolute threshold 128.  
 *   
 * 
 */
template <class T=unsigned char,class T2=unsigned char>
class ThresholdOp: public SequentialImageOp<T,T2>
{
public:

	/// Default constructor. Accepts threshold and replace values.
 	ThresholdOp(const string & p = "128,0"):
		SequentialImageOp<T,T2>("ThesholdOp","Strict thresholding.","threshold,replace",p){
 		/*
 		Data data = Intensity::max<T2>();
 		if (!d.empty())
 			data = d;
 		this->setInfo("Thresholding operator.","threshold,replace",data);
 		*/
 	};
    
    /*
     * ThresholdOp(T t,T2 replace=0){
		setThreshold(t);
		this->replace = replace;
	};
	*/

	void initialize() const {
		threshold = static_cast<T2>(this->getParameter("threshold",1.0));
 		replace   = static_cast<T2>(this->getParameter("replace",0.0));
 	}

	void setThreshold(T t){
		this->setParameter("threshold",1.0);
		//this->threshold = t;
	};

	/*
	void jurpo(...){
		cerr << "keijo";
	}
	*/

	/*
	void setRelativeThreshold(float t){   // TODO
		threshold = 255;
	};
	*/

   
	inline 
	void filterValue(const T &src, T2 &dst) const {
		const T2 s = static_cast<T2>(src);
		dst = (s >= threshold) ? s : replace; 
	};

	
   

protected:
	mutable T threshold;
	mutable T2 replace;
}; 





}
}

#endif /*THRESHOLD_H_*/
