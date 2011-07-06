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
#include "fmi_image.h"

void image_average_horz(FmiImage *source,FmiImage *vert){
  register i,j,k;
  int sum;

  // risky?
  if ((vert->width!=1)||(vert->height!=source->height))
    initialize_vert_stripe(vert,source->height);

  //  for (k=0;k<source->;k++){
  for (j=0;j<source->height;j++){
    sum=0;
    for (i=0;i<source->width;i++)
      sum+=get_pixel(source,i,j,0);
    put_pixel_direct(vert,j,sum/source->width);
    //    put_pixel(vert,0,j,0,sum/source->width);
    /*
    put_pixel(source,sum/255,j,0,255);
    put_pixel(source,sum/255+1,j,0,0);
    put_pixel(source,sum/255-1,j,0,0);
    */
  }
  
}

void image_average_vert(FmiImage *source,FmiImage *vert){
  register i,j,k;
  int sum;

  // risky?
  if ((vert->width!=1)||(vert->height!=source->height))
    initialize_vert_stripe(vert,source->height);

  //  for (k=0;k<source->;k++){
  for (j=0;j<source->height;j++){
    sum=0;
    for (i=0;i<source->width;i++)
      sum+=get_pixel(source,i,j,0);
    put_pixel_direct(vert,j,sum/source->width);
  }
}

