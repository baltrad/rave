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
#ifndef SEQUENTIALIMAGEOP_H_
#define SEQUENTIALIMAGEOP_H_


#include "ImageOp.h"

namespace drain
{

namespace image
{

/**! Base class for designing image processing based simple pixel iterations. 
 * 
 */
template <class T=unsigned char,class T2=unsigned char>
class SequentialImageOp : public ImageOp<T,T2> 
{
public:

	SequentialImageOp(const string &name = "SequentialImageOp",
			const string &description="Image processing operator.",
			const string & parameterNames="",
			const string & defaultValues = "") : ImageOp<T,T2>(name,description,parameterNames,defaultValues) {};

	virtual ~SequentialImageOp(){};
	
	/// Initializes (converts) the parameters of the operator.
	/** In this method, it is recommended to at least implement conversion of
	 *  parameter map values to mutable member variables.
	 */
	virtual void initialize() const = 0;

	// Final
	virtual void filter(const Image<T> &src,Image<T2> &dst) const {
		
		initialize();
		makeCompatible(src,dst);
		
		typename std::vector<T>::const_iterator s = src.begin();
        typename std::vector<T2>::iterator d;
        typename std::vector<T2>::const_iterator dEnd = dst.end();
		
        for (d=dst.begin(); d!=dEnd; d++,s++)
        	filterValue(*s,*d);

		//return dst;

	};
	
	
	//inline 
	virtual void filterValue(const T &src, T2 &dst) const = 0;
			//{ dst = static_cast<T2>(src);};
	
};


}

}

#endif /*SEQUENTIALIMAGEOP_H_*/
