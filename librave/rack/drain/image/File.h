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
#ifndef DRAIN_FILE_H_
#define DRAIN_FILE_H_

#include "MagickDrain.h"
#include "FilePng.h"

#include <string>

#include "Image.h"


namespace drain
{
namespace image
{

using namespace std;

class File
{
public:
	//((File(const string &filename);

	//virtual ~File();

	template <class T>
	static void read(Image<T> &img,const string &path){
#ifdef DRAIN_MAGICK_yes
		Magick::Image magickImage;
		magickImage.read(path);
		MagickDrain::convert(magickImage,img);
#else
		// Todo PNM support
		FilePng::read(img,path);
#endif
	};

	//static void read(Image<unsigned char> &image,const string &path);

	template <class T>
	static void write(const Image<T> &img,const string &path){
#ifdef DRAIN_MAGICK_yes

		Magick::Image magickImage;
		magickImage.size("1x1");
		magickImage.magick("RGBA");

		if (Debug > 1)
			cerr << "Converting image to Magick image." << endl;

		MagickDrain::convert(img,magickImage);

		if (Debug > 0)
			cerr << "Writing image" << img.getName() << " to "<< path << endl;

		magickImage.write(path);
#else
		FilePng::write(img,path);
#endif
	};

	//static void write(const Image<unsigned char> &image,const string &path);

	
};


}  // image

}  // drain

#endif /*FILE_H_*/
