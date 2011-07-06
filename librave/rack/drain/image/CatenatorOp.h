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
#ifndef CATENATOROP_H_
#define CATENATOROP_H_

#include "ImageOp.h"

namespace drain
{

namespace image
{

template <class T=unsigned char,class T2=unsigned char>
class CatenatorOp : public ImageOp<T,T2>
{
public:
	

	CatenatorOp(const string & p="vert") : ImageOp<T,T2>("CatenatorOp","Catenates images, mode=vert|depth (vertically or in-depth, by adding channels). Horz not yet implemented.",
		"mode",p)
	{};

	void filter(const Image<T> &src,Image<T2> &dst) const {
		const string mode = this->parameters.get("mode","depth");
		if (mode == "depth")
			catenateInDepth(src,dst);
		else if (mode == "vert")
			catenateVertically(src,dst);
		else
			throw runtime_error(mode + ": invalid option for Catenator.");
	};
	
	/// The width of the image is the maximum of dst width, if src initialized, else src width.
	void catenateInDepth(const Image<T> &src,Image<T2> &dst) const;
	void catenateVertically(const Image<T> &src,Image<T2> &dst) const;
};


template <class T,class T2>
void CatenatorOp<T,T2>::catenateInDepth(const Image<T> &src,Image<T2> &dst) const {
	
	const int imageChannelsSrc = src.getImageChannelCount();
	
	const int width  = dst.getWidth()>0 ? dst.getWidth() : src.getWidth();
	const int height = dst.getHeight()>0 ? dst.getHeight() : src.getHeight();
	const int imageChannelsDst = dst.getImageChannelCount();
	const int alphaChannels    = max(src.getAlphaChannelCount(),dst.getAlphaChannelCount());
	
	dst.setGeometry(width,height,imageChannelsSrc + imageChannelsDst,alphaChannels);
	
	Point2D<int> p;
	for (int k = 0; k < imageChannelsSrc; ++k) {
		ImageView<T>  channelSrc(src,k);
		ImageView<T2> channelDst(dst,imageChannelsDst + k);
		const int w = min(width,(int)src.getWidth());
		for (p.y = 0; p.y < height; ++p.y) {
			for (p.x = 0; p.x < w; ++p.x) {
				channelDst.at(p) = channelSrc.at(p);
			}
		}
	}
	//cout << "GEOM NOW " << dst.getGeometry() << '\n';
	
};

template <class T,class T2>
void CatenatorOp<T,T2>::catenateVertically(const Image<T> &src,Image<T2> &dst) const {
	
	
	const int width  = dst.getWidth()>0 ? dst.getWidth() : src.getWidth();
	const int heightDst = dst.getHeight();
	const int channelsDst = dst.getChannelCount();
	
	const int heightSrc = src.getHeight();
	const int channelsSrc = src.getChannelCount();
	//const int alphaChannels    = max(src.getAlphaChannelCount(),dst.getAlphaChannelCount());
	
	const int start = width*heightDst*channelsDst;
	
	dst.setGeometry(width,heightDst*channelsDst+heightSrc*channelsSrc,1,0);
	
	
	//Point2D<int> p;
	for (int k = 0; k < channelsSrc; ++k) {
		//BufferedView<T>  channelSrc(src,k);
		//BufferedView<T2> channelDst(dst,channelsDst + k);
		const int w = min(width,(int)src.getWidth());
		//for (p.y = 0; p.y < height; ++p.y) {
		//	for (p.x = 0; p.x < w; ++p.x) {
		for (int j=0; j<heightSrc; j++) {
			for (int i=0; i<w; i++) {
				//dst.at(start+src.address(i,j,k)) = src.at(i,j,k);
				dst.at(start+src.address(i,j,k)) = static_cast<T2>(src.at(i,j));
				//dst.at(i,j,k) = src.at(i,j,k);
			}
		}
	}
	//cout << "GEOM NOW " << dst.getGeometry() << '\n';
	
};

}

}

#endif /*CATENATOR_H_*/
