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
#ifndef QUALITYOVERRIDEOP_H_
#define QUALITYOVERRIDEOP_H_


#include "ImageOp.h"

namespace drain
{

namespace image
{

/** Overwrites pixels of destination image in locations where destination weight is lower.
 * 
 */
template <class T=unsigned char,class T2=unsigned char>
class QualityOverrideOp : public ImageOp<T,T2>
{
public:
	QualityOverrideOp(){};
	
	inline 
	virtual void filter(const Image<T> &src,Image<T2> &dst) const {
		makeCompatible(src,dst);
		//const unsigned int k = src.getChannelCount();
		//const BufferedView<T> s(src,0,k-1);
		//BufferedView<T2> d(dst,0,k-1);
		// Todo: alpha % i
		filter(src.getImageChannels(),src.getAlphaChannel(),
			dst.getImageChannels(),dst.getAlphaChannel());
	};
	
	virtual void filter(const Image<T> &src,const Image<T> &srcWeight,
		Image<T2> &dst,Image<T2> &dstWeight) const ;
	
};


template <class T,class T2>
void QualityOverrideOp<T,T2>::filter(const Image<T> &src,const Image<T> &srcWeight,
		Image<T2> &dst,Image<T2> &dstWeight) const {
			
		makeCompatible(src,dst);
		makeCompatible(srcWeight,dstWeight);
		
		for (unsigned int k=0; k<src.getChannelCount(); k++){
		
			typename std::vector<T>::const_iterator s    = src.getChannel(k).begin();
			typename std::vector<T>::const_iterator sEnd = src.getChannel(k).end();
        	typename std::vector<T>::const_iterator sw = srcWeight.begin();
			//T2 sw2;
			
        	typename std::vector<T2>::iterator d = dst.getChannel(k).begin();
        	typename std::vector<T2>::iterator dw = dstWeight.begin();
        
        	while (s != sEnd){
        		//sw2 = static_cast<T2>(*sw);
        		//cerr << (float)*sw << ' ' << (float)*dw;
        		if (*sw >= *dw){  // important >= instead of >
        			*dw = *sw; 
        			*d  = *s; //static_cast<T2>(*s);
        			//cerr << ' '<< (float)*dw ;
        		}
        		//cerr << '\n';
        		s++;
        		sw++;
        		d++;
        		dw++;
        	}
		}
};


}

}

#endif /*QUALITYOVERRIDE_H_*/
