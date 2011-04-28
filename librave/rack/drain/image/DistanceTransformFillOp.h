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
#ifndef DISTANCETRANSFORMFILLOP_H_
#define DISTANCETRANSFORMFILLOP_H_
	
	//#include "BufferedImage<T>.h"
	//#include "Operator.h"
	//#include "Point.h"
	
#include "DistanceTransformOp.h"
//#include "File.h"
	
	
namespace drain
{
	
namespace image
{
	
	/**
	 *  Semantics. In filtering, if there are as many alpha channels as image channels, the channels
	 *  are treated separately. Else, the image channels are treated as colors, and the only
	 *  alpha channel (or the last of many) will serve as propagation criterion.
	 */
	template <class T=unsigned char,class T2=unsigned char>
	class DistanceTransformFillOp : public DistanceTransformOp<T,T2> 
	{
	public:
		
			
	    DistanceTransformFillOp(const string & p = "5,-1,-1") : 
	    	DistanceTransformOp<T,T2> (this->distanceLinear) {
	    	this->setInfo("Spreads intensities to ranges defined by alpha intensities.",
	    	"horz,vert,diag",p);
		};
		
		/// For derived classes.
		DistanceTransformFillOp(DistanceModel<T2> & distanceModel) : 
	    	DistanceTransformOp<T,T2>(distanceModel) {
	    }; 
	    
		~DistanceTransformFillOp(){};
	
		
		void filter(const Image<T> &src, Image<T2> &dest) const;
		void filter(const Image<T> &src, const Image<T> &srcAlpha, 
			Image<T2> &dst,	Image<T2> &dest) const;
	
		void filterDownRight(const Image<T> &src,  const Image<T> &srcAlpha,
			Image<T2> &dst, Image<T2> &destAlpha) const;
		
		void filterUpLeft(const Image<T> &src,  const Image<T> &srcAlpha,
			Image<T2> &dst, Image<T2> &destAlpha) const ;
	
	};
		
template <class T,class T2>
void DistanceTransformFillOp<T,T2>::filter(
	const Image<T> &src, Image<T2> &dst) const {
		if (drain::Debug > 2){
			cerr << "DistanceTransformFillOp<T,T2>::filter\n" << this->getParameters() << '\n';
		}
		makeCompatible(src,dst);
		const unsigned int iCh = src.getImageChannelCount();

		const unsigned int aCh = src.getAlphaChannelCount();
		if (aCh == iCh){
			for (unsigned int i = 0; i < iCh; ++i)
				filter(src.getChannel(i), src.getChannel(iCh+i), dst.getChannel(i), dst.getChannel(iCh+i));
		}
		else {
			const ImageView<T> srcView(src,0,iCh);
			ImageView<T2> dstView(dst,0,iCh);
			filter((const Image<T>&)srcView, src.getAlphaChannel(), (Image<T2> &)dstView, dst.getAlphaChannel());
		}
};
		
template <class T,class T2>
void DistanceTransformFillOp<T,T2>::filter(
		const Image<T> &src, const Image<T> &srcAlpha, 
			Image<T2> &dst, Image<T2> &dstAlpha) const {

			// TODO
			//if (filterWithTmp(src,srcAlpha,dst,dstAlpha)) return;
			
			dst.name = "DTF target";
			
			makeCompatible(src,dst);
			makeCompatible(srcAlpha,dstAlpha);
			
			this->initialize();
			
			filterDownRight(src,srcAlpha,dst,dstAlpha);
			//drain::image::File::write(dst,"dtd-i.jpg");
			//drain::image::File::write(dstAlpha,"dtd-i.png");
			
			filterUpLeft(dst,dstAlpha,dst,dstAlpha);
			
			//drain::image::File::write(dst,"dtu-i.jpg");
			//drain::image::File::write(dstAlpha,"dtu-i.png");
			
		}
		
		
template <class T,class T2>
void DistanceTransformFillOp<T,T2>::filterDownRight(const Image<T> &src,const Image<T> &srcAlpha, 
	Image<T2> &dst, Image<T2> &dstAlpha) const {
	
		
			const int width  = src.getWidth();
			const int height = src.getHeight();
			const CoordinateHandler & coordinateHandler = src.getCoordinateHandler();
			
			unsigned int iChannels = src.getImageChannelCount();
	
			// proximity (inverted distance)
			float d;
		    float dPrev;
				
			// Intensity (graylevel)
			//T g;
		//cerr << "iChannels: " << iChannels << endl;
		vector<T2> pixel(iChannels);
	
		//cerr << "koe" << endl;
			
			Point2D<> p; 
			
			Point2D<> t;
			int &tx = t.x;
       	 	int &ty = t.y;
        
			//coordinateOverflowHandler.setBounds(srcDist.getBounds());
			
			for (ty=0; ty<height; ty++){
				for (tx=0; tx<width; tx++){
					
					// Take source value as default
					d = srcAlpha.at(t);
					src.getPixel(t,pixel); 
					
					// Compare to previous value
					dPrev = dstAlpha.at(t);
					if (dPrev > d){
						d = dPrev;
						dst.getPixel(t,pixel); 					
					}
					
					// Compare to upper left neighbour 	 
					p.setLocation(tx-1,ty-1);
					coordinateHandler.handle(p);
					dPrev = this->distanceModel.decreaseDiag(dstAlpha.at(p));
					if (dPrev > d){
						d = dPrev;
						dst.getPixel(p,pixel);				
					} 
	
					// Compare to upper neighbour 
					p.setLocation(tx,ty-1);
					coordinateHandler.handle(p);
					dPrev = this->distanceModel.decreaseVert(dstAlpha.at(p));
					if (dPrev > d){
						d = dPrev;
						dst.getPixel(p,pixel);			
					} 
	
					// Compare to upper right neighbour 
					p.setLocation(tx+1,ty-1);
					coordinateHandler.handle(p);
					dPrev = this->distanceModel.decreaseDiag(dstAlpha.at(p));
					if (dPrev > d){
						d = dPrev;
						dst.getPixel(p,pixel);					
					} 
	
					// Compare to left neighbour 
					p.setLocation(tx-1,ty);
					coordinateHandler.handle(p);
					dPrev = this->distanceModel.decreaseHorz(dstAlpha.at(p));
					if (dPrev > d){
						d = dPrev;
						dst.getPixel(p,pixel);					
					} 
	
					if (d>0){
						dstAlpha.at(t) = static_cast<T2>(d);
						dst.putPixel(t, pixel);
					}
	
				}
			}
			// return dst;
		};
		
template <class T,class T2>
void DistanceTransformFillOp<T,T2>::filterUpLeft(
	const Image<T> &src, const Image<T> &srcAlpha,
			Image<T2> &dst, Image<T2> &dstAlpha) const {
	
			const int width  = src.getWidth();
			const int height = src.getHeight();
			const CoordinateHandler & coordinateHandler = src.getCoordinateHandler();
			
			// proximity (inverted distance)
			float d;
			float dPrev;
			
			vector<T> pixel(src.getImageChannelCount());
				
			Point2D<> p;
			
			Point2D<> t;
			int &tx = t.x;
       	 	int &ty = t.y;
        
       
			//coordinateHandler.setBounds(width,height);
			for (ty=height-1; ty>=0; ty--){
				for (tx=width-1; tx>=0; tx--){
					
					// Source
					d = srcAlpha.at(t); 
					src.getPixel(t,pixel);
					
					// Compare to previous value
					dPrev = dstAlpha.at(t);
					if (dPrev > d){
						d = dPrev;
						dst.getPixel(t,pixel);
					}
					
					// Compare to lower left neighbour 	 
					p.setLocation(tx-1,ty+1);
					coordinateHandler.handle(p);
					dPrev = this->distanceModel.decreaseDiag(dstAlpha.at(p));
					if (dPrev > d){
						d = dPrev;
						dst.getPixel(p,pixel);					
					} 
	
					// Compare to lower neighbour 
					p.setLocation(tx,ty+1);
					coordinateHandler.handle(p);
					dPrev = this->distanceModel.decreaseVert(dstAlpha.at(p));
					if (dPrev > d){
						d = dPrev;
						dst.getPixel(p, pixel);					
					} 
	
					// Compare to lower right neighbour 
					p.setLocation(tx+1,ty+1);
					coordinateHandler.handle(p);
					dPrev = this->distanceModel.decreaseDiag(dstAlpha.at(p));
					if (dPrev > d){
						d = dPrev;
						dst.getPixel(p,pixel);					
					} 
	
					// Compare to right neighbour 
					p.setLocation(tx+1,ty);
					coordinateHandler.handle(p);
					dPrev = this->distanceModel.decreaseHorz(dstAlpha.at(p));
					if (dPrev > d){
						d = dPrev;
						dst.getPixel(p,pixel);					
					} 
	
					if (d>0){
						dstAlpha.at(t) = static_cast<T2>(d);
						dst.putPixel(t, pixel);
					}
	
	
			}
		}
	};

	template <class T=unsigned char,class T2=unsigned char>
	class DistanceTransformFillExponentialOp : public DistanceTransformFillOp<T,T2> 
	{
	public:
		 
	    DistanceTransformFillExponentialOp(const string & p = "5,-1,-1") : 
	    	DistanceTransformFillOp<T,T2> (this->distanceExponential) {
	    	this->setInfo("Spreads intensities to ranges defined by alpha intensities.",
	    	"horz,vert,diag",p);
		};
	};

	
}
}
	
	
#endif /*DISTANCETRANSFORMFILL_H_*/
