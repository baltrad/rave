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
#ifndef GAMMAOP_H_
#define GAMMAOP_H_

#include <math.h>

//#include <limits>

//#include "BufferedImage.h"
#include "SequentialImageOp.h"

using namespace std;

namespace drain
{

namespace image
{

/// Gamma correction: intensitys is mapped as f' = f^(gamma)
/*! Gamma correction
 * 
 *  NOTE. Design for paramters may vary, since multichannel could be handled by giving
 *  a value for each: 1.2,1.4,0.7 for example. 
 */
template <class T = unsigned char,class T2 = unsigned char>
class GammaOp : public SequentialImageOp<T,T2>
{
public:

	GammaOp(const string & p = "1.0") : SequentialImageOp<T,T2>("GammaOp",
			"Gamma correction for brightness.","gamma,scaleSrc,scaleDst","1,255,255"){
		this->setParameters(p);
	};

	virtual void initialize() const {
		gammaInv = 1.0 / this->parameters.get("gamma",1.0);
		scaleSrc = this->parameters.get("scaleSrc",Intensity::max<T>());
		scaleDst = this->parameters.get("scaleDst",Intensity::max<T2>());
	};

	
	inline 
	void filterValue(const T &src, T2 &dst) const {
		dst = static_cast<T2>(scaleDst * pow(static_cast<double>(src)/scaleSrc, gammaInv));
	}; 
	
protected:
	mutable double gammaInv;
	mutable double scaleSrc;
	mutable double scaleDst;
	
};

}
}

#endif /*GAMMAOP_H_*/
