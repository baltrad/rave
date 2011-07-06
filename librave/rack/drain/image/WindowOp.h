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
#ifndef WINDOWOP_H_
#define WINDOWOP_H_

#include "ImageOp.h"

namespace drain
{

namespace image
{

/**
 *
 *  TODO: const string name;
 */
template <class T=unsigned char,class T2=unsigned char>
class WindowOp : public ImageOp<T,T2>
{
  public:
	
	WindowOp(const string &name = "SequentialImageOp",
			const string &description="Image processing operator.",
			const string & parameterNames="width,height",
			const string & defaultValues = "0,0") : ImageOp<T,T2>(name,description,parameterNames,defaultValues) {};

	virtual ~WindowOp(){};
	
	//virtual
	void setSize(unsigned int width,unsigned int height = 0){
		this->parameters["width"] = width;
		if (height == 0)
			height = width;
		this->parameters["height"] = height;
	};
};



}

}

#endif /*WINDOWOP_H_*/
