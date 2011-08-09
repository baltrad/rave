/**
    Copyright 2010 Markus Peura,
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
    along with Rack.  If not, see <http://www.gnu.org/licenses/>. */

/*
 * convertImage.h
 *
 *  Created on: Nov 15, 2010
 *      Author: mpeura
 */
#ifndef __CONVERT_IMAGE__
#define __CONVERT_IMAGE__

extern "C" {
	#include "fmi_image.h"
}

#include <drain/image/Image.h>

/*/ Converts a drain image to an old image struct. */
void convertImage(const drain::image::Image<Byte> &src,FmiImage &dst);

void viewImage(const drain::image::Image<Byte> &src,FmiImage &dst);

/*/ Converts an old image struct to a drain Image. */
void convertImage(const FmiImage &src,drain::image::Image<Byte> &dst);


#endif
