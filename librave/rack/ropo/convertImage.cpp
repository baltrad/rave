/**

    Copyright 2011  Martin Raspaud <martin.raspaud@smhi.se>
    Copyright 2010  Markus Peura,
    Finnish Meteorological Institute (First.Last@fmi.fi)


    This file is part of Rack.

    Rack is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Rack is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser Public License for more details.

    You should have received a copy of the GNU Lesser Public License
    along with Rack.  If not, see <http://www.gnu.org/licenses/>.

 */

#include <string.h>
#include <vector>
#include <iterator>
#include "convertImage.h"
#include <drain/util/Debug.h>

void convertImage(const FmiImage &src,drain::image::Image<Byte> &dst){

	image_info((FmiImage *)&src);

	dst.setGeometry(src.width,src.height,src.channels);
	//const unsigned int size = dst.getVolume();

	const std::vector<Byte>::const_iterator & end = dst.end();
	Byte *b = src.array;
	for (std::vector<Byte>::iterator it = dst.begin(); it != end; it++,b++){
		*it = *b+10;
	};

	//for (int i = 0; i < src.volume; ++i)
	//	dst.at(i) = src.array[i];

	/*
	for (int k = 0; k < src.channels; ++k) {
		for (int j = 0; j < src.height; ++j) {
			for (int i = 0; i < src.width; ++i) {
				dst.at(i,j,k) = get_pixel((FmiImage *)&src,i,j,k);
			}
		}
	}
	 */



	//dst.properties["COMMENT"] = src.comment_string;

}

void convertImage(const drain::image::Image<Byte> &src,FmiImage &dst){

	dst.type = TRUE_IMAGE;
	dst.width  = src.getWidth();
	dst.height = src.getHeight();
	dst.channels = src.getChannelCount();
        dst.heights = new int[dst.channels + 1];
	initialize_image(&dst);
	image_info(&dst);

	const std::vector<Byte>::const_iterator & end = src.end();
	Byte *b = dst.array;
	for (std::vector<Byte>::const_iterator it = src.begin();
			it != end; it++,b++){
		*b = *it;
	};

	/*
	for (int i = 0; i < dst.volume; ++i)
		dst.array[i] = src.at(i);
	 */

	/*
	for (int k = 0; k < dst.channels; ++k) {
		for (int j = 0; j < dst.height; ++j) {
			for (int i = 0; i < dst.width; ++i) {
				  put_pixel(&dst,i,j,k,src.at(i,j,k));
			}
		}
	}*/

	//dst.properties["COMMENT"] = src.comment_string;
	dst.comment_string[0]='\0';

}


///
void viewImage(const drain::image::Image<Byte> &src,FmiImage &dst){

	// see initialize_image

	// Geometry
	dst.width = src.getWidth();
	dst.height = src.getHeight();
	dst.channels = src.getChannelCount();
        dst.heights = new int[dst.channels + 1];
	dst.area = dst.width*dst.height;
	dst.volume = dst.area * dst.channels;
	dst.max_value = 255; // unneeded?

	// Coordinate handlers
	dst.coord_overflow_handler_x = BORDER;
	dst.coord_overflow_handler_y = BORDER;

	// Image memory
	//dst.type = TRUE_IMAGE;
	dst.type = LINK_IMAGE;

	//dst.array = (Byte *)(& src.at(0));
	dst.array = (Byte *)&(*src.begin());
	if (drain::Debug > 0) {
		cerr << "   Image: " << (long int) & src.at(0) << endl;
		cerr << "   Image: " << (long int) &*src.begin() << endl;
		cerr << "FmiImage: " << (long int) dst.array << endl;
	}

	// Comments
	//strncpy(dst.comment_string,src.properties.get("COMMENT").c_str(),MAX_COMMENT_LENGTH);
	//strncpy(dst.comment_string,"");
	dst.comment_string[0]='\0';

	/*
	std::vector<Byte>::const_iterator it2;
	 Byte *b = src.array;
	for (std::vector<Byte>::iterator it = dst.begin();
			it != dst.end(); it++){
		*it = *b;
		b++;
	};
	 */


}

/*
struct fmi_image{
  int width,height,channels;
  //  int *channel_mapping;
  int area,volume;
  int max_value;
  // depth ?
  // unsigned char **array;
  Byte *array;
  CoordOverflowHandler coord_overflow_handler_x, coord_overflow_handler_y;
  //  unsigned char *stream;
  char comment_string[MAX_COMMENT_LENGTH];
  FmiImageFormat format;
  FmiImageType type;
};
*/
