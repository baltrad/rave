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
#include "fmi_image_histogram.h"

void morph_closing(FmiImage *source,FmiImage *target,int w,int h){
  FmiImage temp;
  canonize_image(source,&temp);
  canonize_image(source,target);
  pipeline_process(source,&temp,w,h,histogram_max);
  pipeline_process(&temp,target,w,h,histogram_min);
  if (FMI_DEBUG(4)) write_image("debug_morph_closing",target,PGM_RAW);
  /* reset_image(&temp); */
}

void morph_opening(FmiImage *source,FmiImage *target,int w,int h){
  FmiImage temp;
  canonize_image(source,&temp);
  canonize_image(source,target);
  pipeline_process(source,&temp,w,h,histogram_min);
  pipeline_process(&temp,target,w,h,histogram_max);
  /* reset_image(&temp); */
}

/*void morph_opening(FmiImage *source,FmiImage *target){ */
/*} */


void distance_transform(FmiImage *source,FmiImage *target){
	register int i,j,k,s,t;
  /*  canonize_image(source,target); */
  if (source!=target)
    copy_image(source,target);

  for (j=0;j<target->height;j++)
    for (i=0;i<target->width;i++)
      for (k=0;k<target->channels;k++){
	t=get_pixel(target,i,j,k);
	s=get_pixel(target,i-1,j,k);
	if (s>0) s--;
	if (s>t) {
	  put_pixel(target,i,j,k,s);
	  t=s;}
	s=get_pixel(target,i,j-1,k);
	if (s>0) s--;
	if (s>t) 
	  put_pixel(target,i,j,k,s);
      }

  for (j=target->height-1;j>=0;j--)
    for (i=target->width-1;i>0;i--)
      for (k=0;k<target->channels;k++){
	t=get_pixel(target,i,j,k);
	s=get_pixel(target,i+1,j,k);
	if (s>0) s--;
	if (s>t){
	  put_pixel(target,i,j,k,s);
	  t=s;}
	s=get_pixel(target,i,j+1,k);
	if (s>0) s--;
	if (s>t) 
	  put_pixel(target,i,j,k,s);
      }
}
