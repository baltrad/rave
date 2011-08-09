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
    along with Rack.  If not, see <http://www.gnu.org/licenses/>. */

#include "fmi_util.h"
#include "fmi_image.h"
#include "fmi_image_filter.h"
#include "fmi_image_filter_texture.h"

void detect_texture(FmiImage *source,FmiImage *trace){
  canonize_image(source,trace);
  fill_image(trace,0);
  detect_horz_maxima(source,trace);
  fmi_debug(1,"koe");
  detect_vert_maxima(source,trace);
  if (FMI_DEBUG(1)) 
    write_image("texture",trace,PGM_RAW);
  /*  detect_vert_maxima(source,trace); */
}

