/**

    Copyright 2001 - 2011

    This file is part of Rack.

    Rack is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Rack is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser Public License for more details.

    You should have received a copy of the GNU Lesser Public License
    along with Rack.  If not, see <http://www.gnu.org/licenses/>.

 */

#ifndef DATASCALEOP_H_
#define DATASCALEOP_H_

#include <drain/image/MathOpPack.h>

namespace drain
{

namespace radar
{

using namespace std;
//using namespace H5;

/// Scales data, without changing noData. TODO: undetect
template <class T=unsigned char,class T2=unsigned char>
class DataScaleOp: public drain::image::MathOp<T,T2> {

public:

	DataScaleOp(float scale = 1.0,float bias = 0.0,T noDataSrc=255,T noDataDst=255) :
			image::MathOp<T,T2>("DataScaleOp","Scales intensities, respecting nodata code.",scale,bias),
		noDataIn(noDataSrc), noDataOut(noDataDst) {
	};

	void setNoDataCode(T noDataSrc,T noDataDst=255){
		this->noDataIn  = noDataSrc;
		this->noDataOut = noDataDst;
	};

	void filterValue(const T &src, const T2 &src2, T2 &dst) const {
		if (src == 0)
			dst = 0;
		else if (src == noDataIn)
			dst = noDataOut;
		else
			dst = image::MathOp<T,T2>::limit(this->scale*static_cast<double>(src) + this->bias);
	};

protected:
	T  noDataIn;
	T2 noDataOut;
};
} // ::image
} // ::drain

#endif /* DATASCALEOP_H_ */
