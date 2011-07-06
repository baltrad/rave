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
#ifndef COPYOP_H_
#define COPYOP_H_

#include "SequentialImageOp.h"

namespace drain
{

namespace image
{

/** Copies intensities. Does not scale them. Uses
 *  Internally, uses SequentialImageOp.
 *  @see ScaleOp
 */	
template <class T=unsigned char,class T2=unsigned char>
class CopyOp : public ImageOp<T,T2>
{
public:
	CopyOp(const string & p = "f,f"){
		this->setInfo("Copies channels. f=full image, i=image channels, a=alpha channel(s), 0=1st, 1=2nd,...",
		"srcView,dstView",p);
	};

	virtual void filter(const Image<T> &src,Image<T2> &dst) const {
		
		ImageView<T>  srcView;
		//const char s = this->parameters.get("srcView",'f');
		const Data &s = this->getParameter("srcView","f");
		switch (s[0]) {
    		case 'f': // full image
				srcView.viewImage(src);
				break;
			case 'i':  // image channels (ie. excluding alpha)
				srcView.viewChannels(src,src.getImageChannelCount());
				break;
			case 'a': // alpha channel(s)
				//srcView.viewImage(src.getAlphaChannel());
				srcView.viewChannels(src,src.getAlphaChannelCount(),src.getImageChannelCount());
				break;
			default:
				unsigned int k = s;
				//if ((k==0)&&(s[0]!=)
				//if (!s.isType<int>()){
				//cerr << "Unsupported channel option:" << this->parameters.get("srcView");
				//throw runtime_error("CopyOp: illegal source channel code.");
				//};
				srcView.viewChannel(src,k);
		}
		
		//cerr << src.getGeometry() << '\n';
		//cerr << srcView.getGeometry() << '\n';
		
		
		makeCompatible(srcView,dst);
		//cerr << dst.getGeometry() << " own="<< dst.hasOwnBuffer() <<'\n';
		
		
		ImageView<T2> dstView;
		const Data &d = this->getParameter("dstView","f");
		
		switch (d[0]) {
    		case 'f': // full image
    			dstView.viewImage(dst);
				break;
			case 'i':  // image channels (ie. excluding alpha)//makeCompatible(srcView,dst);
				dstView.viewChannels(dst,dst.getImageChannelCount());
				break;
			case 'a': // alpha
				//cerr << "alpha\n";
				if (dst.getAlphaChannelCount() == 0)
					dst.setAlphaChannelCount(srcView.getChannelCount());
				//cerr << "alpha\n";
				dstView.viewChannels(dst,dst.getAlphaChannelCount(),dst.getImageChannelCount());
				break;
			default:
				//int k = s;
				/*
				if (!d.isType<int>()){
					cerr << "Unsupported channel option:" << d;
					throw runtime_error("CopyOp: illegal source channel code.");	
				};
				*/
				unsigned int kd = d;
				if (dst.getChannelCount() <= kd)
					dst.setChannelCount(kd+1);
				dstView.viewChannel(dst,kd);
				break;
		}

		if (Debug > 0){
			cerr << srcView.getGeometry() << '\n';
			cerr << dstView.getGeometry() << '\n';
		}

		typename std::vector<T>::const_iterator si = srcView.begin();
        typename std::vector<T2>::iterator di;
        typename std::vector<T2>::const_iterator dEnd = dstView.end();
		for (di=dstView.begin(); di!=dEnd; di++,si++)
			*di = static_cast<T2>(*si);



	};

	//inline 	virtual void filterValue(const T &src, T2 &dst){ dst = static_cast<T2>(src);};
};

}

}

#endif /*COPYOP_H_*/
