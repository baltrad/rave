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

    You should have received a copy  of the GNU General Public License
    along with Drain.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifndef FILEPNG_H_
#define FILEPNG_H_


#include <string>
#include <iostream>
#include <fstream>
#include <exception>

#include <png.h>

#include "Image.h"
#include "../util/Time.h"


namespace drain
{
namespace image
{

using namespace std;

/// For reading and writing images in PNG format
/** A lighter alternative for Magick++ which supports tens of image formats.
 *  Portable Network Graphics (PNG) format is a widely adapted, flexible image format
 *  for gray level and color images. It supports lossless compression and alpha channels.
 */
class FilePng
{
public:

	template <class T>
	static void read(Image<T> &image,const string &path,
			int png_transforms = (PNG_TRANSFORM_PACKING || PNG_TRANSFORM_EXPAND));

		template <class T>
	static void write(const Image<T> &image,const string &path);
	/// TODO: Write options (PNG_*)
	
};

/** Reads a png file to drain::Image.
 *  Converts indexed (palette) images to RGB or RGBA.
 *  Scales data to 8 or 16 bits, according to template class.
 *  Floating point images will be scaled as 16 bit integral (unsigned short int).
 *
 */
template <class T>
void FilePng::read(Image<T> &image,const string &path, int png_transforms ) {

	// Try to open the file
	FILE *fp = fopen(path.c_str(), "rb");
	if (fp == NULL)
		throw runtime_error(string("FilePng: could not open file: ") + path);

	// For checking magic code (signature)
	const unsigned int PNG_BYTES_TO_CHECK=4;
	png_byte buf[PNG_BYTES_TO_CHECK];

	/* Read in some of the signature bytes */
	if (fread(buf, 1, PNG_BYTES_TO_CHECK, fp) != PNG_BYTES_TO_CHECK)
		throw runtime_error(string("FilePng: suspicious size of file: ") + path);

	/* Compare the first PNG_BYTES_TO_CHECK bytes of the signature.
	   Return nonzero (true) if they match */
	if (png_sig_cmp(buf, (png_size_t)0, PNG_BYTES_TO_CHECK) != 0)
		throw runtime_error(string("FilePng: not a png file: ")+path);

	png_structp  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,NULL,NULL,NULL);
	if (!png_ptr){
		throw runtime_error(string("FilePng: problem in allocating image memory for: ")+path);
	}

	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr){
	    png_destroy_read_struct(&png_ptr,(png_infopp)NULL, (png_infopp)NULL);
		throw runtime_error(string("FilePng: problem in allocating info memory for: ")+path);
	}

	/*
	png_infop end_info = png_create_info_struct(png_ptr);
	if (!end_info){
	    png_destroy_read_struct(&png_ptr, &info_ptr,(png_infopp)NULL);
	    throw runtime_error(string("FilePng: problem in allocating end_info memory for: ")+path);
	}
	*/

	// This may be unstable. According to the documentation, if one uses the high-level interface png_read_png()
	// one can only configure it with png_transforms flags (PNG_TRANSFORM_*)
	png_set_palette_to_rgb(png_ptr);

	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, PNG_BYTES_TO_CHECK);

	/// Main action
	if (drain::Debug > 2)
		cerr << "png read starting\n";
	png_read_png(png_ptr, info_ptr, png_transforms, NULL);


	/// Read comments
	if (drain::Debug > 2)
		cerr << "png image comments:\n";

	int num_text;
	png_textp text_ptr;
	png_get_text(png_ptr, info_ptr,&text_ptr, &num_text);
	for (int i = 0; i < num_text; ++i) {
		if (drain::Debug > 2)
			cerr << text_ptr[i].key << '=' << text_ptr[i].text << '\n';
		image.properties[text_ptr[i].key] = text_ptr[i].text;
	}


	/// Copy to drain::Image
	const int width  = png_get_image_width(png_ptr, info_ptr);
	const int height = png_get_image_height(png_ptr, info_ptr);
	const int channels = png_get_channels(png_ptr, info_ptr);
	const int bit_depth = png_get_bit_depth(png_ptr, info_ptr);

	switch (channels){
	case 4:
		image.setGeometry(width,height,3,1);
		break;
	case 3:
		image.setGeometry(width,height,3);
		break;
	case 2:
		image.setGeometry(width,height,1,1);
		break;
	case 1:
		image.setGeometry(width,height,1);
		break;
	default:
		throw runtime_error(string("FilePng: invalid channel count in : ")+path);
	}

	if (drain::Debug > 2){
		cerr << "png geometry ok\n";
		cerr << "png channels =" << channels << "\n";
		cerr << "png bit_depth=" << bit_depth << "\n";
		cerr << image.getGeometry() << "\n";
	}

	if ((bit_depth!=8) && (bit_depth != 16)){
		fclose(fp);
		png_destroy_read_struct(&png_ptr,&info_ptr, (png_infopp)NULL);
		//png_free_data(png_ptr,info_ptr,PNG_FREE_ALL,-1);  // ???
		throw runtime_error(string("FilePng: unsupported bit depth in : ")+path);
	}

	png_bytep *row_pointers = png_get_rows(png_ptr, info_ptr);
	png_bytep p;
	int i0;
	for (int j = 0; j < height; ++j) {
		p = row_pointers[j];
		for (int i = 0; i < width; ++i) {
			for (int k = 0; k < channels; ++k) {
				i0 = channels*i + k;
				if (bit_depth == 8) {
					image.at(i,j,k) = p[i0];
				}
				else {
					image.at(i,j,k) = p[i0*2] + (p[i0*2+1]<<8);
				}
			}
		}
	}

	fclose(fp);
	png_destroy_read_struct(&png_ptr,&info_ptr, (png_infopp)NULL);
	//png_free_data(png_ptr,info_ptr,PNG_FREE_ALL,-1);  // ???

	//png_destroy_read_struct(&png_ptr,(png_infopp)NULL, (png_infopp)NULL);
	//png_destroy_info_struct(png_ptr,&info_ptr);


}

/** Writes drain::Image to a png image file applying G,GA, RGB or RGBA color model.
 *  Writes in 8 or 16 bits, according to template class.
 *  Floating point images will be scaled as 16 bit integral (unsigned short int).
*/
template <class T>
void FilePng::write(const Image<T> &image,const string &path){


	FILE *fp = fopen(path.c_str(), "wb");
	if (!fp){
		throw runtime_error(string("FilePng: could not open file : ")+path);
	}

	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png_ptr){
		throw runtime_error(string("FilePng: could not allocate memory for: ")+path);
	}

	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr){
	   png_destroy_write_struct(&png_ptr,(png_infopp)NULL);
	   throw runtime_error(string("FilePng: could not allocate info memory for: ")+path);
	}

	// Required.
	png_init_io(png_ptr, fp);

	// Optional.
	png_set_filter(png_ptr, 0,PNG_FILTER_HEURISTIC_DEFAULT);

	// Optional
	png_set_compression_level(png_ptr, Z_DEFAULT_COMPRESSION);

	const int width  = image.getWidth();
	const int height = image.getHeight();
	const int channels = image.getChannelCount();

	int color_type;
	switch (channels) {
		case 4:
			color_type = PNG_COLOR_TYPE_RGBA;
			break;
		case 3:
			color_type = PNG_COLOR_TYPE_RGB;
			break;
		case 2:
			color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
			break;
		case 1:
			color_type = PNG_COLOR_TYPE_GRAY;
			break;
		default:
			throw runtime_error(string("FilePng: unsupported channel count: "));
	}


	const int byte_depth = sizeof(T);
	const int bit_depth  = byte_depth <= 2 ? byte_depth*8 : 16;
	if (drain::Debug > 1){
		if (byte_depth > 2)
			cerr << "FilePng: source image " << byte_depth << " byte data, converting to 2 bytes (16bit uInt).\n";
	}
	// TODO: check integral vs float types, instead

	if (drain::Debug > 2){
		cerr << image.getGeometry() << "\n";
		cerr << "png bit_depth=" << bit_depth << "\n";
	}

	// Set header information
	png_set_IHDR(png_ptr, info_ptr, width, height,bit_depth, color_type,
			PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    // interlace_type, compression_type, filter_method

	// Optional
	// png_set_tIME(png_ptr, info_ptr, mod_time);
	// png_set_text(png_ptr, info_ptr, text_ptr, num_text);

	// Optional: text content
	const int text_size = image.properties.size() + 2;
	png_text text_ptr[text_size];
	drain::Time time;
	text_ptr[0].key = (char *)"Creation time";
	text_ptr[0].text = (char *)time.str().c_str();
	text_ptr[0].text_length = strlen(text_ptr[0].text); // ???
	text_ptr[0].compression = PNG_TEXT_COMPRESSION_NONE;
	//text_ptr[1].key = (char *)"Comment";
	//text_ptr[1].text = (char *)"Test comment";
	text_ptr[1].key = (char *)"Software";
	text_ptr[1].text = (char *)"drain/image/FilePng Markus.Peura[c]fmi.fi";
	text_ptr[1].text_length = strlen(text_ptr[1].text);
	text_ptr[1].compression = PNG_TEXT_COMPRESSION_NONE;

	int i=2;
	for (map<string,Data>::const_iterator it = image.properties.begin(); it != image.properties.end(); it++){
		text_ptr[i].key  = (char *)it->first.c_str();
		text_ptr[i].text = (char *)(it->second.substr(0,79)).c_str();
		text_ptr[i].text_length = it->second.length();
		text_ptr[i].compression = PNG_TEXT_COMPRESSION_NONE;
		i++;
	}
	png_set_text(png_ptr, info_ptr, text_ptr, 2);

	// Create temporary image array.
	png_byte data[height][width*channels*byte_depth];
	for (int k = 0; k < channels; ++k) {
		// 8 bit
		if (byte_depth == 1){
			for (int i = 0; i < width; ++i) {
				//i0 = i*channels + k;
				for (int j = 0; j < height; ++j) {
					data[j][i*channels + k] = static_cast<png_byte>(image.at(i,j,k));
				}
			}
		}
		// 16 bit
		else {
			int i0;
			for (int i = 0; i < width; ++i) {
				i0 = (i*channels + k)*2;
				for (int j = 0; j < height; ++j) {
					data[j][i0+1] = static_cast<png_byte>(image.at(i,j,k)&255);
					data[j][i0  ] = static_cast<png_byte>((image.at(i,j,k)>>8)&255);
				}
			}
		}
	}
	png_bytep row_pointers[height];
	for (int j = 0; j < height; ++j) {
		row_pointers[j] = data[j];
	}
	//cerr << "Preparing to write..." << endl;
	png_set_rows(png_ptr, info_ptr, row_pointers);

	// Main operation
	int png_transforms = PNG_TRANSFORM_IDENTITY  || PNG_TRANSFORM_SHIFT;
	png_write_png(png_ptr, info_ptr, png_transforms, NULL);

	fclose(fp);
	png_destroy_write_struct(&png_ptr,&info_ptr);
	//png_free_data(png_ptr,info_ptr,PNG_FREE_ALL,-1);
}

}

}

#endif /*FILEPng_H_*/
