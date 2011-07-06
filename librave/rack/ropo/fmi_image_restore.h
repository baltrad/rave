/**

    Copyright 2001 - 2010  Markus Peura,
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


#ifndef __FMI_IMAGE_RESTORE__
#define __FMI_IMAGE_RESTORE__

#include "fmi_image.h"


void mark_image(FmiImage *target,FmiImage *prob,Byte threshold,Byte marker);

void restore_image(FmiImage *source,FmiImage *target,FmiImage *prob,Byte threshold);

#endif
