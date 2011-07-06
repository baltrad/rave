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
#ifndef RECURSIVEREPAIREROP_H_
#define RECURSIVEREPAIREROP_H_

#include "CopyOp.h"
#include "FastAverageOp.h"
#include "QualityOverrideOp.h"


// debugging
#include "File.h"

namespace drain
{

namespace image
{

/** Image restoration utility applying recursive interpolation from neighboring pixels. 
 * 	
 */
template <class T=unsigned char,class T2=unsigned char>
class RecursiveRepairerOp : public ImageOp<T,T2> //: public WindowOp<T,T2>
{
public:
	
	RecursiveRepairerOp(const string &p = "5,5,3,0.9"): ImageOp<T,T2>("RecursiveRepairerOp",
			"Applies weighted averaging window repeatedly, preserving the pixels of higher weight.",
					"width,height,loops,decay",p) {
	};
	
	inline 
	virtual void filter(const Image<T> &src,Image<T2> &dst) const {
		makeCompatible(src,dst);
		filter(src.getImageChannels(),src.getAlphaChannel(),
			dst.getImageChannels(),dst.getAlphaChannel());
		/*
		const unsigned int k = src.getChannelCount();
		cerr << "Rec k=" << k; 
		const BufferedView<T> s(src,0,k-1);
		BufferedView<T2> d(dst,0,k-1); // täsä oli pugi
		filter(s,src.getChannel(k-1),d,dst.getChannel(k-1));
		*/
	}
	
	
	virtual void filter(const Image<T> &src,const Image<T> &srcWeight,
		Image<T2> &dst,Image<T2> &dstWeight) const ;
		
};


template <class T,class T2>
void RecursiveRepairerOp<T,T2>::filter(const Image<T> &src,const Image<T> &srcWeight,
	Image<T2> &dst,Image<T2> &dstWeight) const {
	
		makeCompatible(src,dst);
		makeCompatible(srcWeight,dstWeight);
				
		//const unsigned int channels = src.getChannelCount();
		const Data & width = this->parameters.get("width","5");
		const Data & height = this->parameters.get("height",width); 
		const unsigned int loops = this->parameters.get("loops",3);
		
		if (loops == 0) {
			CopyOp<T,T2>().filter(src,dst);
			CopyOp<T,T2>().filter(srcWeight,dstWeight);
			return;
		}
		else if (loops == 1) {
			FastAverageOp<T,T2>(width,height).filter(src,srcWeight,dst,dstWeight);
			QualityOverrideOp<T,T2> q;
			q.filter(src,srcWeight,dst,dstWeight);
			return;
		}

		// loops > 1


		Image<T2> dst2;
		Image<T2> dstWeight2;

		makeCompatible(src,dst2);
		makeCompatible(srcWeight,dstWeight2);

		/*
		Image<T2> *s  = NULL;
		Image<T2> *sW = NULL;
		Image<T2> *d  = NULL;
		Image<T2> *dW = NULL;
		 */

		/// First loop
		/// Odd number of loops
		if ((loops & 1)==1){
			FastAverageOp<T,T2>(width,height).filter(src,srcWeight,dst,dstWeight);
			QualityOverrideOp<T,T2>().filter(src,srcWeight,dst,dstWeight);
		}
		/// Even number of loops
		else {
			FastAverageOp<T,T2>(width,height).filter(src,srcWeight,dst2,dstWeight2);
			QualityOverrideOp<T,T2>().filter(src,srcWeight,dst2,dstWeight2);
		}

		/// Remaining loops
		FastAverageOp<T2,T2> avg2(width,height);
		QualityOverrideOp<T2,T2> q2;
		for (int i = loops-1; i >= 0; i--){

			const bool b = (i&1)==0;

			Image<T2> &s  = b ? dst : dst2;
			Image<T2> &sW = b ? dstWeight : dstWeight2;
			Image<T2> &d  = b ? dst2 : dst;
			Image<T2> &dW = b ? dstWeight2 : dstWeight;

			stringstream sstr;
			sstr.width(2);
			sstr.fill('0');

			sstr << "rec-" << i;

			avg2.filter(s,sW,d,dW);
			//File::write(dst,sstr.str() + "aD.png");
			//File::write(dstWeight,sstr.str() + "aW.png");
			q2.filter(s,sW,d,dW);
			//File::write(dst2,sstr.str() + "bD.png");
			//File::write(dstWeight2,sstr.str() + "bW.png");

		}







};

}

}

#endif /*RECURSIVEREPAIRER_H_*/
