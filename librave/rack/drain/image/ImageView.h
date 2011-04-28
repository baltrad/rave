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
#ifndef IMAGEVIEW_H_
#define IMAGEVIEW_H_

#include <sstream> 
#include <stdexcept>
#include "Image.h"

namespace drain
{
namespace image
{


template <class T>
class Image;

/// A reference to one or several subsequent channels of a BufferedImage.
/** It does not have memory itself. 
 *  @see BufferedImage
 *  Note: the name is not logical, as this view does NOT have a buffer...l
 */
template <class T=unsigned char>
class ImageView : public Image<T>
{
public:
	
	ImageView(){ this->buffer=NULL; };
	ImageView(const Image<T> &image, int channelStart=0, int channelCount=1);

	/// Views all the channels of an image.
	void viewImage(const Image<T> & src){
		viewChannels(src,src.getChannelCount());
		setGeometry(src.getGeometry());
	}
	
	/// Views a single channel of an image.
	void viewChannel(const Image<T> &image,unsigned int channel){
		viewChannels(image,1,channel);
	}
	
	/// Views \code channelCount channels of an image, starting from \code channelCount.
	void viewChannels(const Image<T> &image,unsigned int channelCount, unsigned int channelStart = 0);
	
	
	/// Concatenates all the channels to a single view.
	void viewImageFlat(const Image<T> &src){
		viewChannels(src,src.getChannelCount());
		this->geometry.setGeometry(src.getWidth(),src.getHeight()*src.getChannelCount(),1);
		cerr << "viewImageFlat" << this->geometry << endl;
	};
	
	
	//void setView(const Image<T> &image, unsigned int channelCount = 0, unsigned int channelStart = 0);

	void setView(std::vector<T> & v,unsigned int width,unsigned int height, 
		unsigned int imageChannelCount, unsigned int alphaChannelCount, unsigned int channelStart=0); 
	
	/// Overrides (Image). Does nothing.
    //virtual void setGeometry(int width,int height,int imageChannelCount=1,int alphaChannelCount=0);

	/// Overrides that of BufferedImage. Does nothing but checks if request would be illegal ie. would change geometry.
    void setGeometry(const Geometry &geometry){
		// todo: warning if tries to change?
		if (this->getGeometry().getVolume() != geometry.getVolume()){
			stringstream sstr;
			sstr << "ImageView<T>::setGeometry: tried to change ";
			sstr << " '" << this->getGeometry() << "' to '" << geometry << "'\n";
			//throw (sstr.str());
			throw runtime_error(sstr.str());
		}
		// 
	};
 	
    
    
 	/*
 	virtual const BufferedImage<T> &getChannel(int index) const;
    virtual       BufferedImage<T> &getChannel(int index) ;
	*/

 	virtual inline int address(const int &i) const {
    	return (this->offset + i);  };
 	
 	virtual inline int address(const int &i,const int &j) const {
    	return (this->offset + i + this->geometry.getWidth()*j);  };

    virtual inline int address(const int &i,const int &j,const int &k) const {
		return (this->offset + i + this->geometry.getWidth()*j + this->geometry.getArea()*k);  };

	
	bool isSet(){ return !this->hasOwnBuffer(); }; 

protected:

	/// For ImageView.
     //	int channelStart;
	
	
	/// For ImageView.
    //int offset;
    
	
    /// Does nothing.
	virtual void initialize(){
		//this->channel.clear();  // is this needed?
		cerr <<  "Bview init" << endl;
	};
	
};

template <class T>
ImageView<T>::ImageView(const Image<T> &image,int channelStart, int channelCount){
	//cerr << "ImageView() --> "  << image.name << endl;
	viewChannels(image,channelCount,channelStart);
}


	
/// Views \code channelCount channels of an image, starting from \code channelCount.
template <class T>
void ImageView<T>::viewChannels(const Image<T> &image,unsigned int channelCount, unsigned int channelStart)
{
	//void  ImageView<T>::setView(const BufferedImage<T> &image, unsigned int channelCount, unsigned int channelStart){
	
	const Geometry g = image.getGeometry();
	
	if (channelCount == 0)  // ???????????
		channelCount = g.getChannelCount(); 
	
	const int i = g.getImageChannelCount();
	const int a = g.getAlphaChannelCount();

	int imageChannelCount = i - channelStart;
	if (imageChannelCount < 0)
		imageChannelCount = 0;
	
	int alphaChannelCount = channelCount - imageChannelCount;
	if (alphaChannelCount > a)
		alphaChannelCount = a;


	/*
	unsigned int imageChannelCount; 
	unsigned int alphaChannelCount;
	
	if ((channelStart + channelCount) > i){
		imageChannelCount = i - channelStart;
    	alphaChannelCount = channelCount - imageChannelCount;
	}	
    else {
    	imageChannelCount = channelCount;
     	alphaChannelCount = 0;
    }
    */
	
	
	stringstream sstr;
	sstr << image.name << "{" << channelStart;
	if (channelCount > 1){
		sstr <<  "..." << (channelStart+channelCount-1);
	}
	sstr <<  "}";
	this->name = sstr.str();
	
	//cerr << "creating view " << this->name << endl;
	
	int currentChannelStart;
	if (g.getArea() != 0)
		 currentChannelStart = image.getOffset()/g.getArea();
	else
		currentChannelStart = 0; // correct
	
	setView((vector<T> &)image.getBuffer(), g.getWidth(), g.getHeight(),
		imageChannelCount,alphaChannelCount,channelStart + currentChannelStart);
	
	setCoordinateHandler(image.getCoordinateHandler()); 
}


template <class T>
void ImageView<T>::setView(std::vector<T> & v,unsigned int width,unsigned int height,
	unsigned int imageChannelCount, unsigned int alphaChannelCount,unsigned int channelStart) 
{
		Geometry &g = this->geometry;
		
		this->setMax();
	
		if (&(this->ownBuffer) != &v)
			this->ownBuffer.resize(0);
		
    	this->buffer = &v;

    	g.setGeometry(width,height,imageChannelCount,alphaChannelCount);
    	setCoordinateHandler(this->defaultCoordinateHandler);

		this->offset = g.getArea() * channelStart;
	
		if ((this->offset + g.getVolume()) > this->buffer->size()){
			stringstream sstr;
			sstr << "ImageView<T>::setView: requested allocation exceeds available buffer.\n";
			sstr << " buffer size :" << this->buffer->size() << endl;
			sstr << " offset :" << this->offset << endl;
			sstr << " volume :" << g.getVolume() << endl;
			throw runtime_error(sstr.str());
		}
	
}
 


}
}

#endif /*BUFFEREDVIEW_H_*/
