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
#ifndef IMAGE_H_
#define IMAGE_H_ "Image 1.3,  2010.05.05 Markus.Peura@fmi.fi"


// TODO: rename to WritableImage ?

#include <stdlib.h>  // for exit(-1) TODO: obsolete, use exceptions.
#include <stdexcept>
#include <vector>
#include <ostream>

#include "../util/Debug.h"
#include "../util/Options.h"
#include "Geometry.h"

#include "Intensity.h"
#include "CoordinateHandler.h"
#include "ImageView.h"

#include "Point.h"


namespace drain
{

/// General debugging // TODO: move to more general place
//static unsigned int Debug;

/// Namespace for images and image processing tools.
namespace image
{

template <class T>
class ImageView;

/// A class for images having using an own memory resource.
/*!
 *  Essentially, BufferedImage is a std::vector, hence a 1D structure in which
 *  further dimensions are implemented by Geometry.
 *  
 *  @see BufferedView
 * 
 */
template <class T=unsigned char>
class Image  //: public Image
{
public:


    Image();

	// Creates a new image having the geometry of \code image. 
	Image(const Image<T> &image);

    Image(int width,int height,int channelCount=1,int alphaChannelCount=0);

 	virtual ~Image(){buffer = NULL;};
 
 	inline
    virtual const Geometry & getGeometry() const {
		return geometry;
	}

 	/// Collapses the image to zero size, releasing memory. @see clear().
 	void reset(){
 		setGeometry(0,0,0,0);
 	}

	/// Defines the dimensions of the image: width x height x (imageChannelCoount + alphaChannelCount).
	/// Convenience. Calls setGeometry(geom).
	/// Final!
	void setGeometry(unsigned int width,unsigned int height,unsigned int imageChannelCount = 1,
		unsigned int alphaChannelCount = 0){
		setGeometry( Geometry(width,height,imageChannelCount,alphaChannelCount) ); 
	};
	
	//inline   
	virtual void setGeometry(const Geometry &g)
	{
		geometry.setGeometry(g);	
		initialize();
	}

		
	/// Sets the number of image channels and alpha channels.
	inline 
	void setChannelCount(unsigned int k,unsigned int ak = 0){
		//const Geometry &g = getGeometry();
		setGeometry(getWidth(),getHeight(),k,ak);	
	};
	
	inline 
	void setAlphaChannelCount(unsigned int k){
		setGeometry(getWidth(),getHeight(),getImageChannelCount(), k);	
	};
	
	// Frequently needed convenience functions.
	inline const unsigned int & getWidth() const { return geometry.getWidth();};
	inline const unsigned int & getHeight() const { return geometry.getHeight();};
	inline const unsigned int & getChannelCount() const { return geometry.getChannelCount();};
	inline const unsigned int & getImageChannelCount() const { return geometry.getImageChannelCount();};
	inline const unsigned int & getAlphaChannelCount() const { return geometry.getAlphaChannelCount();};
	inline const unsigned int & getVolume() const { return geometry.getVolume();};
	
	inline
	const vector< ImageView<T> > & getChannelVector() const {
		updateChannelVector();
		return channel;
	};


	virtual const Image<T> &getChannel(unsigned int index) const;
    virtual       Image<T> &getChannel(unsigned int index);
    virtual const Image<T> &getAlphaChannel(unsigned int index = 0) const;
    virtual       Image<T> &getAlphaChannel(unsigned int index = 0);
	
	/// Returns a view to image channels of this image.
	virtual const Image<T> &getImageChannels() const {
		views.resize(2);
		views[0].viewChannels(*this,geometry.getImageChannelCount());
		return views[0];
	}
	
	/// Returns a view to image channels of this image.
    virtual       Image<T> &getImageChannels(){
    	views.resize(2);
    	views[0].viewChannels(*this,geometry.getImageChannelCount());
    	return views[0];	
    }
    
    /// Returns a view to alpha channels of this image.
    virtual const Image<T> &getAlphaChannels() const {
    	views.resize(2);
		views[1].viewChannels(*this,geometry.getAlphaChannelCount(),geometry.getImageChannelCount());
		return views[1]; //alphaChannels;
	}
	
	/// Returns a view to alpha channels of this image.
    virtual       Image<T> &getAlphaChannels(){
    	views.resize(2);
    	views[1].viewChannels(*this,geometry.getAlphaChannelCount(),geometry.getImageChannelCount());
    	return views[1]; //alphaChannels;	
    }
   
    // Extensions (new methods):
    inline
    void setCoordinateHandler(CoordinateHandler &handler);

    inline
    CoordinateHandler &getCoordinateHandler() const;


	inline
    const T & at(int i) const {
    	return (*buffer)[ address(i) ];  // slow?
    }
    
    inline 
    T & at(int i) {
    	return (*buffer)[ address(i) ];  // slow?
    }

	inline
	const T & at(const int &i, const int &j) const {
    	return (*buffer)[ address(i,j) ];
	};

	inline
	T & at(const int &i, const int &j){
    	return (*buffer)[ address(i,j) ];
	};

	inline
	const T & at(const int &i, const int &j, const int &k) const {
    	return (*buffer)[ address(i,j,k) ];
	};

	inline
	T & at(const int &i, const int &j, const int &k) {
    	return (*buffer)[ address(i,j,k) ];
	};

	inline
	const T & at(const Point2D<int> &p) const {
    	return (*buffer)[ address(p.x,p.y) ];
	};

	inline
	T & at(const Point2D<int> &p) {
    	return (*buffer)[ address(p.x,p.y) ];
	};
	
	void getPixel(const Point2D<> &p, std::vector<T> & pixel) const;
  	void putPixel(const Point2D<> &p, const vector<T> & pixel);

    
    
	/// For sequential (in-place) computations
    virtual typename vector<T>::const_iterator begin() const {
		typename vector<T>::const_iterator it = buffer->begin();
		advance(it, offset);
    	return it;
	}
	
    virtual typename vector<T>::iterator begin() {
    	typename vector<T>::iterator it = buffer->begin();
		advance(it, offset);
    	return it;
	}
	
    virtual typename vector<T>::const_iterator end() const {
    	typename std::vector<T>::const_iterator it = buffer->begin();
		advance(it, offset + geometry.getVolume());
    	return it;
	}
	
    virtual typename vector<T>::iterator end() {
    	typename std::vector<T>::iterator it = buffer->begin();
		advance(it, offset + geometry.getVolume());
    	return it;
	}

	/// The 1D address of a pixel in the vector 
    virtual inline int address(const int &i) const {
        return i;  
    };

	/// The 1D address of a pixel in the vector 
    virtual inline int address(const int &i,const int &j) const {
        return (i + geometry.getWidth()*j);  
    };

    /// The 1D address of a pixel in the vector 
    virtual inline int address(const int &i, const int &j, const int &k) const {
        return (i + geometry.getWidth()*j + geometry.getArea()*k);
    };


	const std::vector<T> & getBuffer() const;

	// This is dangerous. Consider resize().
	std::vector<T> & getBuffer();
  
  	
    ///  Fills all the channels with %value.
	inline
    void fill(T value){
    	for (typename vector<T>::iterator it=begin(); it!=end(); it++)
    		  *it = value;
    }

    /// Sets intensities to zero. Does not change geometry. @see reset();
    inline
    void clear(){fill(0);};

	// Tells if this image has an own buffer or views the buffer of another image.
	bool hasOwnBuffer() const;

	// Tells if image have share memory image.

	bool hasOverlap(const Image<T> & img) const {
		typename vector<T>::const_iterator b = begin();
		typename vector<T>::const_iterator e = end();
		typename vector<T>::const_iterator bi = img.begin();
		typename vector<T>::const_iterator ei = img.end();
		return ((bi>=b) && (bi<e)) || ((ei>=b) && (ei<e));
	}

	// Test with T != T2 will always fail.
	template <class T2>
	bool hasOverlap(const Image<T2> & img) const {
		return false;
	}

	// Tells if this image has a alpha channel. Single-channel image never has.
	bool hasAlphaChannel() const { return (getAlphaChannelCount() > 0);};

	inline
	const unsigned int & getOffset() const {return offset;};

	/// Prints image debugging data to the standard error buffer.
	void debug(ostream &ostr = cout) const;
	
	void setName(const string &n){ name = n; };

	const string & getName() const { return name; };

	/// Probably kicked to properties, see below.
	string name;

    Options properties;
    
    /*! Expected maximum intensity. This value will be used when scaling intensities. User may change the value. 
     *  By default, for \em char and \em short int this value will be 255 and 65535, respectively, else 1.0. 
     */     
    inline
	T getMax() const {return max;};
	
	inline
	void setMax(T max = Intensity::max<T>()){this->max = max;};
	
protected:

	///
	virtual void updateChannelVector() const;

	// needed?
	T max;

	Geometry geometry;

	// Initializes the channel vector, if not yet initialized.
	virtual void initialize();

    CoordinateHandler * coordinateHandler;
    CoordinateHandler defaultCoordinateHandler;

    std::vector<T> *buffer;
    std::vector<T> ownBuffer;

	unsigned int offset;
	
private:

	/// Container for storing referneces to each channel. Lazy update. 
	/// Changed only when a channel is requested with getChannel().
    mutable vector< ImageView<T> > channel;
    
    mutable vector< ImageView<T> > views;
    
    // experimental
    //mutable BufferedView<T> imageChannels;
    //mutable BufferedView<T> alphaChannels;
};



//unsigned int ImageDebug = 0;





template <class T>
Image<T>::Image() //:  imageArray(&defaultVector)
{
	setMax();
	buffer = & ownBuffer;
    setGeometry(0,0,0);
    setCoordinateHandler(this->defaultCoordinateHandler);
}

template <class T>
Image<T>::Image(const Image<T> &image) //:  imageArray(&defaultVector)
{
	max = image.max;
	buffer = & ownBuffer;
	setGeometry(image.getGeometry());
	setCoordinateHandler(this->defaultCoordinateHandler);
}


template <class T>
Image<T>::Image(int width,int height, int imageChannelCount, int alphaChannelCount) 
	// : imageArray(&defaultVector)
{
	setMax();
	buffer =& ownBuffer;
    setGeometry(width,height,imageChannelCount,alphaChannelCount);
	setCoordinateHandler(this->defaultCoordinateHandler);
}


template <class T>
void Image<T>::setCoordinateHandler(CoordinateHandler &handler)
{
    coordinateHandler = &handler;
    coordinateHandler->setBounds(getWidth(), getHeight());
}


template <class T>
CoordinateHandler &Image<T>::getCoordinateHandler() const
{
    return *coordinateHandler;
}




template <class T>
void Image<T>::getPixel(const Point2D<int> &p,std::vector<T> &pixel) const
{
	static unsigned int i;
    for (i=0; i<pixel.size(); i++)
    	pixel[i] = at(p.x,p.y,i); 

};

template <class T>
void Image<T>::putPixel(const Point2D<int> &p,const std::vector<T> &pixel)
{
	static unsigned int i;
    for (i=0; i<pixel.size(); i++)
    	at(p.x,p.y,i) = pixel[i]; 
};

template <class T>
void Image<T>::updateChannelVector() const {

	const unsigned int channelCount = getChannelCount();

	if (channel.size() != channelCount){
		channel.resize(channelCount);
		for (unsigned int i = 0; i < channelCount; ++i) {
			channel[i].viewChannel(*this,i);
		}
	}

}

template <class T>
Image<T> &Image<T>::getChannel(unsigned int index)
{
	
	const unsigned int channelCount = getChannelCount();
	
	if (index >= channelCount){
		// TODO Exception
		cerr << " getChannel() illegal index: " << index << endl;
		exit(-1);
	}
	
	if (channelCount == 1){
		return *this;
	}
	
	// ERROR? Risk of missing channels when re-allocating?
	/*
	if (channel.size() != channelCount){
		channel.resize(channelCount);
	}
	channel[index].viewChannel(*this,index);
    */
	updateChannelVector();

    return channel[index];
};

template <class T>
const Image<T> &Image<T>::getChannel(unsigned int index) const
{
	const unsigned int channelCount = getGeometry().getChannelCount();
	
	if (index >= channelCount){
		// TODO Exception
		//cerr << " getChannel() illegal index: " << index << endl;
		exit(-1);
	}
	
	if (channelCount == 1){
		return *this;
	}

	/* vanha
	if (channel.size() != channelCount){
   		channel.resize(channelCount);
	}
    channel[index].viewChannel(*this,index);
    */
	
    // uusi
    updateChannelVector();
    
    return channel[index];
    
};

template <class T>
Image<T> &Image<T>::getAlphaChannel(unsigned int index)
{
	unsigned int alphaChannelCount = getGeometry().getAlphaChannelCount();

	if (index < alphaChannelCount){
		//cerr << "getAlphaChannel returns " << getGeometry().getChannelCount()-1 - index << endl;
		return getChannel(getGeometry().getChannelCount()-1 - index);
	}
	else {
		cerr << " getAlphaChannel() illegal alpha index: " << index << endl;
		exit(-1);
	}
  		
};

template <class T>
const Image<T> &Image<T>::getAlphaChannel(unsigned int index) const
{
	unsigned int alphaChannelCount = getGeometry().getAlphaChannelCount();

	if (index < alphaChannelCount){
		return getChannel(getGeometry().getChannelCount()-1 - index);
	}
	else {
		cerr << " getAlphaChannel() illegal index: " << index << endl;
		exit(-1);
	}
};

/**
 *  Notice, this returns the whole vector, even when called
 *
 */
template <class T>
const vector<T> & Image<T>::getBuffer() const {
	return *(this->buffer); 
}

template <class T>
vector<T> & Image<T>::getBuffer() {
	return *(this->buffer); 
}


template <class T>
void Image<T>::initialize(){
	
	offset = 0;
	channel.clear();
	
	//cerr << "BufferedImage<T>::initialize BEGIN " << this->geometry.toString(gStr) << endl;
	//cerr << " name='" << this->name << "'" <<endl;
	/*
	if (!hasOwnBuffer()){
		cerr << "WARNING: tried initializing (resizing) an extrenal buffer" << endl;
		return;	
	}
	*/

	if ((*buffer).size() != getVolume()){

		if (!hasOwnBuffer())
			throw runtime_error("WARNING: tried initializing (resizing) an extrenal buffer");

		(*buffer).resize(getVolume());
	}
	
	
}

template <class T>
bool Image<T>::hasOwnBuffer() const
{
	return (buffer == &ownBuffer);
}


// DEBUGGING
// TODO: change channels to getChVect
template <class T>
void Image<T>::debug(ostream &ostr) const
{
	int channelCount = getGeometry().getChannelCount();
	vector<const Image<T> *> channels(channelCount+1);
	
	//cerr << "...debug: init THIS" << endl;
	channels[0] = this;
	for (int i = 0; i < channelCount; ++i) {
		//cerr << "...debug: init channel #" << i << endl;
		channels[i+1] = &getChannel(i);
	}
	
	ostr << "\nImage debug info for "<< this->name << ", type=";
	//cout << " Type: ";
	if (hasOwnBuffer())
		ostr << "BUFFER\n";
	else
	    ostr<< "REFERENCE\n";
	
	if (this->coordinateHandler == NULL)
		ostr << "WARNING: CoordHandler undefined!";
	  //ostream << " Coord Handler defined: " << (this->coordinateHandler != NULL) << "\n";
	
	for (int i = 0; i < channelCount+1; ++i) {
		const Image<T> *img = channels[i];
		string info;
		long unsigned int a  = 0;
		//long unsigned int a2 = 0;
		long unsigned int m = 0;
		if (img != NULL){
			img->getGeometry().toString(info);
			a  = img->address(0,0);
			//a2 = img->address(1,1);
			m = (long unsigned int)&(img->at(0,0));
		}
		else 
			info = "Null";
			
		if (i == 0)
			ostr << "image:";
		else	
			ostr << "ch #" << (i-1);
			
		//ostr << ":\t" << info << "(" << a << ','<< a2 << ")" << "# " << m << endl;
		ostr << ":\t" << info << " [" << a << "]" << " #" << m << endl;
	}
	ostr << " Default array: size= "<< ownBuffer.size( )
		 << " address= " << (long unsigned int)&ownBuffer[0] << endl;
		 
		
		
	ostr << " Current array: size= " << buffer->size( )
		 << " address= " << (long unsigned int)&((*buffer)[0]) << endl
		 << "  = " << (long unsigned int)&(*begin()) << " -- " << (long unsigned int)&(*end()) << endl;
	
}

/*
template <class T>
void BufferedImage<T>::setName(const string &n){
	name = n;	
}
*/


}

}

#endif /*BUFFEREDIMAGE_H_*/
