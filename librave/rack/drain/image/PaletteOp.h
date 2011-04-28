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
#ifndef PALETTEOP_H_
#define PALETTEOP_H_

#include "ImageOp.h"
#include "CopyOp.h"
#include "File.h" // ?

namespace drain
{

namespace image
{

/// Colorizes an image of 1 channel to an image of N channels by using a palette image as a lookup table. 
/*! Treats an RGB truecolor image of N pixels as as a palette of N colors.
 *  - 
 *  - 
 */
template <class T=unsigned char,class T2=unsigned char>
class PaletteOp : public ImageOp<T,T2>
{
public:
  
	PaletteOp(const Image<T2> & palette) : ImageOp<T,T2>("PaletteOp",
			"Applies colour palette to an image","file","") {
		setPalette(palette);
	   	 //this->setInfo("Sets palette","file",filename);
	  	 //this->parameters.setAllowedKeys("file");
	   	 //this->parameters.set(p);
	};

	PaletteOp(const string & filename = "") : ImageOp<T,T2>("PaletteOp",
			"Applies colour palette to an image","file",filename) {
		if (!filename.empty()){
			Image<> palette;
			File::read(palette,this->getParameter("file",""));
			setPalette(palette);
		}
   	};
  
   virtual ~PaletteOp(){};
   
   void initialize() const {
	   /*
   		if (this->parameters.hasKey("file")){
   			Image<> palette;
   			File::read(palette,this->getParameter("file",""));
   			setPalette(palette);
   		}
	    */
   };
   
   void makeCompatible(const Image<T> &src,Image<T2> &dst) const;
   
   void filter(const Image<T> &src,Image<T2> &dst) const ;

   /// Creates an internal palette by taking the colors on the diagonal from lower left to upper right corner.
   /*! Maximally 256 colors.
    * 
    */
   void setPalette(const Image<T> &palette) const; //,unsigned int maxColors = 256);

   /// Creates a gray palette ie. "identity mapping" from gray (x) to rgb (x,x,x).
   // TODO T 256, T2 32768
   void setGrayPalette(unsigned int iChannels=3,unsigned int aChannels=0,float brightness=0.0,float contrast=1.0) const {

	   const unsigned int colors = 256;
	   const unsigned int channels = iChannels+aChannels;

	   paletteImage.setGeometry(colors,1,iChannels,aChannels);

	   T g;
	   const float origin = (Intensity::min<T>()+Intensity::max<T>())/2.0;

	   for (unsigned int i = 0; i < colors; ++i) {

		   g = Intensity::limit<T>( contrast*(static_cast<float>(i)-origin) + brightness+origin );
		   for (unsigned int k = 0; k < iChannels; ++k)
			   paletteImage.at(i,0,k) = g;

		   for (unsigned int k = iChannels; k < channels; ++k)
		   			   paletteImage.at(i,0,k) = 255;  // TODO 16 bit?
	   }
   }

protected:

   mutable Image<T2> paletteImage;
    
};


template <class T,class T2>
void PaletteOp<T,T2>::setPalette(const Image<T> &palette) const {
	//const Geometry & gPal = palette.getGeometry();
	
	const unsigned int width  = palette.getWidth();
	const unsigned int height = palette.getHeight();
	const unsigned int channels = palette.getChannelCount();

	const unsigned int colors = 256;
	paletteImage.setGeometry(colors,1,palette.getImageChannelCount(),palette.getAlphaChannelCount());
	for (unsigned int i = 0; i < colors; ++i) {
		for (unsigned int k = 0; k < channels; ++k) {
			paletteImage.at(i,0,k) = palette.at((i*width)/colors,(i*height)/colors,k);	
		}
	}
}

template <class T,class T2>
void PaletteOp<T,T2>::makeCompatible(const Image<T> &src,Image<T2> &dst) const {
	//const Geometry &gSrc = src.getGeometry();
	//const Geometry &gPal = paletteImage.getGeometry();
	//dst.setGeometry(gSrc.getWidth(),gSrc.getHeight(),gPal.getImageChannelCount(),gPal.getAlphaChannelCount());
	const unsigned int alphaChannels = max(paletteImage.getAlphaChannelCount(),src.getAlphaChannelCount()); 
	dst.setGeometry(src.getWidth(),src.getHeight(),paletteImage.getImageChannelCount(),alphaChannels);
}


template <class T,class T2>
void PaletteOp<T,T2>::filter(const Image<T> &src,Image<T2> &dst) const {
	
	initialize();
	
	//if (&src == &dst){
	if (src.hasOverlap(dst)){
		throw string("PaletteOp: cannot handle src==dst (extra memory not implemented)");
	}
	
	makeCompatible(src,dst);
	
	
	//const Geometry & g = dst.getGeometry();
	const unsigned int width  = dst.getWidth();
	const unsigned int height = dst.getHeight();
	const unsigned int channels = min(paletteImage.getChannelCount(),dst.getChannelCount());
	
	for (unsigned int i = 0; i < width; ++i) {
		//cerr << "Palette: " << i << '\t' << '\n';
		for (unsigned int j = 0; j < height; ++j) {
			for (unsigned int k = 0; k < channels; ++k) {
				dst.at(i,j,k) = paletteImage.at(src.at(i,j)&255,0,k);  // % was not ok! for char?
			}
		}
	}
	
	if (!paletteImage.hasAlphaChannel()){
		if (src.hasAlphaChannel() && dst.hasAlphaChannel()){
			CopyOp<T,T2> op;
			op.filter(src.getAlphaChannel(),dst.getAlphaChannel());		
		}
	}
	
	//return dst;
}

}

}

#endif /*PALETTEOP_H_*/
