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
#include "fmi_util.h"
#include "fmi_image.h"
#include "fmi_image_filter.h"
#include "fmi_image_filter_line.h"

/* source = VALUE TO BE PROPAGATED; IF NULL VALUE=1*/
/* mask = control image */


void detect_horz_line_segments(FmiImage *target,FmiImage *trace,unsigned char min_length,unsigned char min_elevation){

  detect_vert_maxima(target,trace); /* fmi_image_filter.h */
  if (FMI_DEBUG(1)) write_image("maxima",trace,PGM_RAW);

  /* fmi_image_filter.h */
  mask_image(trace,trace,min_elevation,0); 
  if (FMI_DEBUG(1)) write_image("maxima2",trace,PGM_RAW);

  /* go right, measure segment length... */
  propagate_right(NULL,trace,trace,1,put_pixel);   
  if (FMI_DEBUG(1)) write_image("prop",trace,PGM_RAW);

  /* ...and spread it back left */
  propagate_left(trace,trace,trace,0,put_pixel);  
  if (FMI_DEBUG(1)) write_image("prop2",trace,PGM_RAW);

  mask_image(trace,trace,min_length,0); 
}
  
void detect_horz_edge_segments(FmiImage *target,FmiImage *trace,unsigned char min_length,unsigned char min_elevation){
  
  detect_horz_edges(target,trace); /* fmi_image_filter.h */
  if (FMI_DEBUG(1)) write_image("maxima",trace,PGM_RAW);

  /* fmi_image_filter.h */
  mask_image(trace,trace,min_elevation,0); 
  if (FMI_DEBUG(1)) write_image("maxima2",trace,PGM_RAW);

  /* go right, measure segment length... */
  propagate_right(NULL,trace,trace,1,put_pixel);   
  if (FMI_DEBUG(1)) write_image("prop",trace,PGM_RAW);

  /* ...and spread it back left */
  propagate_left(trace,trace,trace,0,put_pixel);  
  if (FMI_DEBUG(1)) write_image("prop2",trace,PGM_RAW);

  mask_image(trace,trace,min_length,0); 
}
  
void detect_vert_line_segments(FmiImage *target,FmiImage *trace,unsigned char min_length,unsigned char min_elevation){
  //int i,j,k,length;
  //  const int CRITICAL_LENGTH=4;
  //int g1,g2,gd,gmax;
  // FmiImage trace;
  // canonize_image(target,&trace);
  //  info(target);
  //info(trace);


  detect_horz_maxima(target,trace); /* fmi_image_filter.h */
  if (FMI_DEBUG(1)) write_image("maxima",trace,PGM_RAW);

  /* fmi_image_filter.h */
  mask_image(trace,trace,min_elevation,0); 
  if (FMI_DEBUG(1)) write_image("maxima2",trace,PGM_RAW);

  /* go right, measure segment length... */
  propagate_up(NULL,trace,trace,1,put_pixel);   
  if (FMI_DEBUG(1)) write_image("prop",trace,PGM_RAW);

  /* ...and spread it back left */
  propagate_down(trace,trace,trace,0,put_pixel);  
  if (FMI_DEBUG(1)) write_image("prop2",trace,PGM_RAW);

  mask_image(trace,trace,min_length,0); 
  //release_image(trace);
}

