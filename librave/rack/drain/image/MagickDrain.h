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

#ifndef MAGICKDRAIN_H_
#define MAGICKDRAIN_H_





#ifdef DRAIN_MAGICK_yes
#include <Magick++.h>
#endif

/// In compilation, use "Magick*-config" to get libs and includes right.


#include "Image.h"

//#include "Point.h"

namespace drain
{

namespace image
{

class MagickDrain
{
public:
	//MagickDrain();
	//virtual ~MagickDrain();
	
#ifdef DRAIN_MAGICK_yes
	/// Converts drain::image to MagickImage
	/// Does not support other types than <unsigned char>
	template<class T>
	static void convert(const Image<T> &drainImage, Magick::Image &magickImage);
	
	/// Converts MagickImage to drain::image
	/// Does not support other types than <unsigned char>
	template<class T>
	static void convert(Magick::Image &magickImage, Image<T> &drainImage);
	
#endif
	
};



#ifdef DRAIN_MAGICK_yes

// Looks like Magick inverts alpha channel
template <class T>
void MagickDrain::convert(const Image<T> &drainImage, Magick::Image &magickImage) {

	Magick::ImageType type = Magick::TrueColorMatteType;

	int imageChannels = drainImage.getImageChannelCount();
	int alphaChannels = drainImage.getAlphaChannelCount();

	if (alphaChannels > 1){
		std::cerr << "Warning: multiple alpha channel image, using 1st only \n";
		std::cerr << "  Image geometry:" << drainImage.getGeometry() << '\n';
		alphaChannels = 1;
	}

	const Image<T> *red = NULL, *green = NULL, *blue = NULL, *alpha = NULL;

	//string info;
	//drainImage.getGeometry().toString(info);
	//cout << "MagickDrain: " << drainImage.getGeometry() << endl;
	/*
	cerr << "img.chs:" << imageChannels  << endl;
	cerr << "alpha.chs:" << alphaChannels  << endl;
	cerr << "Extra debug:"  << endl;
	drainImage.debug();
	 */

	switch (imageChannels){
	case 0:
		if (alphaChannels == 0){
			std::cerr << "Error: zero channel image!\n";
		}
		else {
			type = Magick::GrayscaleType;
			red   =& drainImage.getChannel(0);
			green =& drainImage.getChannel(0);
			blue  =& drainImage.getChannel(0);
		}
		break;
	case 1:
		type = Magick::GrayscaleType;
		red   =& drainImage.getChannel(0);
		green =& drainImage.getChannel(0);
		blue  =& drainImage.getChannel(0);
		if (alphaChannels > 0) {
			type = Magick::GrayscaleMatteType;
			alpha =& drainImage.getAlphaChannel();
		}
		break;
	case 2:
		if (!alphaChannels) {
			type  =  Magick::GrayscaleMatteType;
			std::cerr << "Notice: 2 channel image, storing 2nd as alpha channel \n";
			red   =& drainImage.getChannel(0);
			green =& drainImage.getChannel(0);
			blue  =& drainImage.getChannel(0);
			alpha =& drainImage.getChannel(1);
		}
		else {
			std::cerr << "Notice: (2+alpha ) channel image, creating 'RGAA' image instead of 'RGBA'.\n";
			type = Magick::TrueColorMatteType;
			red   =& drainImage.getChannel(0);
			green =& drainImage.getChannel(1);
			blue  =& drainImage.getAlphaChannel();
			alpha =& drainImage.getAlphaChannel();
		}
		break;
	case 3:
		type  = Magick::TrueColorType;
		red   =& drainImage.getChannel(0);
		green =& drainImage.getChannel(1);
		blue  =& drainImage.getChannel(2);
		if (alphaChannels){
			type  =  Magick::TrueColorMatteType;
			alpha =& drainImage.getAlphaChannel();
		}
		break;
	default:
		type  =  Magick::TrueColorMatteType;
		red   =& drainImage.getChannel(0);
		green =& drainImage.getChannel(1);
		blue  =& drainImage.getChannel(2);

		if (alphaChannels){
			cerr << "Warning: (" << imageChannels << "+alpha) channel image, using (3+alpha). \n";
			alpha =& drainImage.getAlphaChannel();
		}
		else if (imageChannels == 4){
			cerr << "Notice: 4 channel image, storing 4th as alpha channel \n";
			alpha =& drainImage.getChannel(3);
		};
		imageChannels = 3;	// WHY?
	}


	const int width = drainImage.getGeometry().getWidth();
	const int height = drainImage.getGeometry().getHeight();


	//Magick::Image magickImage(Magick::Geometry(width,height),Magick::Color("black"));


	try {
		magickImage.classType(Magick::DirectClass);  // here?
		magickImage.size(Magick::Geometry(width,height));
		//magickImage.read("cow.png");

		magickImage.modifyImage(); // actually: _prepare_to_ modify
		magickImage.type(type);   //  The order copied from Magick examples


		Magick::PixelPacket *pixel_cache = magickImage.getPixels(0,0,width,height);


		Point2D<> p;
		int &i = p.x;
		int &j = p.y;

		// TODO: const unsigned drainBits = numeric_limits<unsigned char>::is_integer ?  numeric_limits<unsigned char>::max() :  //
		const unsigned int drainMax = numeric_limits<unsigned char>::max();
		const int shiftBits = magickImage.depth() - 8;  // ????


		int rowAddress = 0;
		int address = 0;

		for (j=0; j<height; j++){
			rowAddress = j*width;

			for (i=0; i<width; i++){
				address = i + rowAddress;
				pixel_cache[address].red =   (red->at(i,j)   << shiftBits);
				pixel_cache[address].green = (green->at(i,j) << shiftBits);
				pixel_cache[address].blue =  (blue->at(i,j)  << shiftBits);
				if (alpha != NULL)
					//pixel_cache[address].opacity = ((alpha->at(i,j))  << shiftBits);  //WARNING!!
					// Looks like Magick does NOT invert alpha channel IN THIS DIRECTION
					pixel_cache[address].opacity = ((drainMax-alpha->at(i,j))  << shiftBits);  //WARNING!!
			}
		}
		//cerr << "synching" << endl;
		//pixel_cache[rowAddress + 10 + width] = Magick::Color("green");

		magickImage.syncPixels();
	}
	catch (Magick::Error& e) {
		// because 'Error' is derived from the standard C++ exception, it has a 'what()' method
		cerr << "a Magick++ error occurred: " << e.what() << endl;
	}
	catch ( ... ) {
		cerr << "MagickDrain: an unhandled error has occurred; exiting application." << endl;
		exit(1);
	}


	// cerr << "magickImage.type = " << magickImage.type() << endl;

	// Store comments as KEY=VALUE pairs
	stringstream sstr;
	map<string,Data>::const_iterator it;
	for (it = drainImage.properties.begin(); it != drainImage.properties.end(); it++){
		sstr << it->first << '=' << it->second << '\n';
	}
	magickImage.comment(sstr.str());


}
#endif



#ifdef DRAIN_MAGICK_yes

// Looks like Magick inverts alpha channel
template <class T>
void  MagickDrain::convert(Magick::Image &magickImage, Image<T> &drainImage) {

	const int w = magickImage.columns();
	const int h = magickImage.rows();

	//drainImage.setGeometry(w,h)

	// TODO: redChannel = &drainImage.at(0,0,0);

	switch (magickImage.type()){
	case Magick::GrayscaleType:
		drainImage.setGeometry(w,h,1);
		magickImage.write(0,0,w,h,"I",Magick::CharPixel,&drainImage.at(0,0));
		break;
	case Magick::GrayscaleMatteType:
		drainImage.setGeometry(w,h,1,1);
		magickImage.write(0,0,w,h,"I",Magick::CharPixel,&drainImage.at(0,0,0));
		magickImage.write(0,0,w,h,"A",Magick::CharPixel,&drainImage.at(0,0,1));
		break;
		//    case Magick::RGBColorspace:
	case Magick::PaletteType: // just test status
	case Magick::TrueColorType:
		drainImage.setGeometry(w,h,3);
		magickImage.write(0,0,w,h,"R",Magick::CharPixel,&drainImage.at(0,0,0));
		magickImage.write(0,0,w,h,"G",Magick::CharPixel,&drainImage.at(0,0,1));
		magickImage.write(0,0,w,h,"B",Magick::CharPixel,&drainImage.at(0,0,2));
		break;
	case Magick::PaletteMatteType:
	case Magick::TrueColorMatteType:
		drainImage.setGeometry(w,h,3,1);
		magickImage.write(0,0,w,h,"R",Magick::CharPixel,&drainImage.at(0,0,0));
		magickImage.write(0,0,w,h,"G",Magick::CharPixel,&drainImage.at(0,0,1));
		magickImage.write(0,0,w,h,"B",Magick::CharPixel,&drainImage.at(0,0,2));
		magickImage.write(0,0,w,h,"A",Magick::CharPixel,&drainImage.at(0,0,3));
		//      magickImage.write(0,0,w,h,"A",Magick::CharPixel,&drainImage.alphaChannel(0)[0]);
		break;
		//    default:
	default:
		stringstream sstr;
		sstr << "operator<<(image,magickImage) : Magick type " << magickImage.type() << " not handled.";
		throw runtime_error(sstr.str());
	}

	// TODO contradictory!
	if (drainImage.getAlphaChannelCount()>0){
		//Image<> & alpha = drainImage.getAlphaChannel();
		//NegateOp<>().filter(alpha,alpha);  // Looks like Magick inverts alpha channel IN THIS DIRECTION.
		//ScaleOp<>(-1.0,255).filter(alpha,alpha);  // Looks like Magick inverts alpha channel
	}

	std::stringstream sstr(magickImage.comment());  // dont touch sstr!!
	drainImage.properties.reader.read(sstr);

	if (drain::Debug > 5){ // TODO static (does not work)
			cerr << "read magickImage.type  = " << magickImage.type() << '\n';
			cerr << "comment='" << magickImage.comment() << "'\n";
			cerr << "::::::::::::::::::::\n";
			cerr << drainImage.properties;
	}


}

#endif




}  // image

}  // drain

#endif /*MAGICKDRAIN_H_*/
//#endif // ImageMagick
