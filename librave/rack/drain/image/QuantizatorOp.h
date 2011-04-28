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
#ifndef QUANTIZATOROP_H_
#define QUANTIZATOROP_H_

#include "SequentialImageOp.h"


namespace drain
{
namespace image
{

/// TODO: CopyOP should be quite similar.
/**! 
 *   
 * 
 */
template <class T=unsigned char,class T2=unsigned char>
class QuantizatorOp: public SequentialImageOp<T,T2>
{
public:

	QuantizatorOp(const string & p = ""){
		this->setInfo("Quantize to n bits. Makes sense for integer data only.","bits",p);
		//setBits(8*sizeof(T)); 
		//setBits(8*sizeof(T));
	};

	void initialize() const {//setBits(unsigned int bits){
		
		const unsigned int bits = this->parameters.get("bits",8*sizeof(T));
		
		mask = 0;
		
		// Create 111111...
      	for (unsigned int i = 0; i < bits; i++)
			mask = (mask << 1) | 1;
      
      	// Complete 1111110000
      	mask = mask << (sizeof(T)*8 - bits);
      
        bitShift = (sizeof(T2)-sizeof(T))*8;
	};
   
    // TODO: recognize float and int images?
	inline 
	void filterValue(const T &src, T2 &dst) const {
		dst = (src&mask) << bitShift; 
	};

	
protected:
	mutable long int mask;
	mutable int bitShift;
}; 





}
}

#endif /*QUANTIZE_H_*/
