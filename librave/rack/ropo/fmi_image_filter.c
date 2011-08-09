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

#include <math.h> 
#include "fmi_util.h"
#include "fmi_image.h"
#include "fmi_image_filter.h"

void detect_vert_gradient(FmiImage *source,FmiImage *trace){
  int i,j,k;
  int j_upper,j_lower;
  int g;
  canonize_image(source,trace);

  for (k=0;k<source->channels;k++){
    for (j=0;j<source->height;j++){
      j_upper=j-1;
      j_lower=j+1;
      for (i=0;i<source->width;i++){
	g = (get_pixel(source,i,j_upper,k)-get_pixel(source,i,j_lower,k))/2+128;
	if (g>254) g=254;
	if (g<2)   g=2;
	put_pixel(trace,i,j,k,g);}
      }
    }
}

void detect_horz_gradient(FmiImage *source,FmiImage *trace){
  int i,j,k;
  int i_left,i_right;
  int g;
  canonize_image(source,trace);

  for (k=0;k<source->channels;k++){
    for (i=0;i<source->width;i++){
      i_left=i-1;
      i_right=i+1;
      for (j=0;j<source->height;j++){
	g = (get_pixel(source,i_right,j,k)-get_pixel(source,i_left,j,k))/2+128;
	if (g>254) g=254;
	if (g<2)   g=2;
	put_pixel(trace,i,j,k,g);}
      }
    }
}


void detect_vert_maxima(FmiImage *source,FmiImage *trace){
	int i,j,k;
	Byte g,g_upper,g_lower,gmax;
	canonize_image(source,trace);
	/*check_image_properties(source,trace); */

	for (k=0;k<source->channels;k++){
		for (j=1;j<source->height-1;j++){
			for (i=0;i<source->width;i++){
				g = get_pixel(source,i,j,k);
				g_upper = get_pixel(source,i,j-1,k);
				g_lower = get_pixel(source,i,j+1,k);
				gmax = MAX(g_upper,g_lower);
				/*put_pixel(trace,i,j,k,gmax); */
				if (g>gmax)
					put_pixel(trace,i,j,k,(Byte)(g-gmax));
				else
					put_pixel(trace,i,j,k,0);
			}
		}
	}
}

/* calculate RELATIVE "shoulder", 242 => 2   393 => 3 , divide by should avg */
void detect_vert_maxima2(FmiImage *source,FmiImage *trace){
  int i,j,k;
  Byte g,g_sum; /*g_upper,g_lower; */
  int gt;
  canonize_image(source,trace);
  /*check_image_properties(source,trace); */

  for (k=0;k<source->channels;k++)
    for (j=1;j<source->height-1;j++)
      for (i=0;i<source->width;i++){
	g=get_pixel(source,i,j,k);
	/*	g_upper=get_pixel(source,i,j-1,k);
		g_lower=get_pixel(source,i,j+1,k);
		gt=(2*g-g_upper-g_lower)/(1+g_upper+g_lower);
	*/
	g_sum=(get_pixel(source,i,j-1,k)+get_pixel(source,i,j+1,k));
	gt=(2*g-g_sum)/(1+g_sum);
	gt=MAX(0,gt);
	/*	gt=pseudo_sigmoid(128,gt); */
	put_pixel(trace,i,j,k,gt);
      }
}

void detect_horz_edges(FmiImage *source,FmiImage *trace){
  int i,j,k;
  int g,g2;
  canonize_image(source,trace);
  /*check_image_properties(source,trace); */

  for (k=0;k<source->channels;k++){
    for (j=1;j<source->height-1;j++){
      for (i=0;i<source->width;i++){
	 g=get_pixel(source,i,j,k)-get_pixel(source,i,j-1,k);
	 g2=get_pixel(source,i,j,k)-get_pixel(source,i,j+1,k);
	 g=MAX(g,g2);
	 g=MAX(0,g);
	 put_pixel(trace,i,j,k,g);}
    }
  }
  if (FMI_DEBUG(3)) write_image("edges",trace,PGM_RAW);
}

void detect_horz_maxima(FmiImage *source,FmiImage *trace){
  int i,j,k;
  unsigned char g,g_left,g_right,gmax,gt;
  canonize_image(source,trace);

  for (k=0;k<source->channels;k++){
    for (i=1;i<source->width-1;i++){
      for (j=0;j<source->height;j++){
	g=get_pixel(source,i,j,k);
	g_left=get_pixel(source,i-1,j,k);
	g_right=get_pixel(source,i+1,j,k);
	gmax=MAX(g_left,g_right); 
	if (g>gmax){
	  gt=get_pixel(trace,i,j,k);
	  put_pixel(trace,i,j,k,MAX(gt,g-gmax));}
	/*	else */
	/*  put_pixel(trace,i,j,k,0); */
      }
    }
  }
}

void detect_vert_edges(FmiImage *source,FmiImage *trace){
  int i,j,k;
  int g,g2;
  canonize_image(source,trace);
  /*check_image_properties(source,trace); */

  for (k=0;k<source->channels;k++){
    for (i=1;i<source->width-1;i++){
     for (j=0;j<source->height;j++){
       g =get_pixel(source,i,j,k)-get_pixel(source,i-1,j,k);
       g2=get_pixel(source,i,j,k)-get_pixel(source,i+1,j,k);
       g=MAX(g,g2);
       g=MAX(0,g);
       put_pixel(trace,i,j,k,g);}
    }
  }
  if (FMI_DEBUG(5)) write_image("debug_edges",trace,PGM_RAW);
}

void iir_right(FmiImage *source,FmiImage *trace,int promille){
  int i,j,k;
  int g,g_old;
  canonize_image(source,trace);
  for (k=0;k<source->channels;k++)
    for (j=0;j<source->height;j++){
      g_old=0;
      for (i=0;i<source->width;i++){
	g=get_pixel(source,i,j,k);
	g=MAX(g,g_old);
	put_pixel(trace,i,j,k,g);
	g_old=g*promille/1000;}
    }
  if (FMI_DEBUG(5)) write_image("debug_iir_right",trace,PGM_RAW);
}

void iir_left(FmiImage *source,FmiImage *trace,int promille){
  int i,j,k;
  int g,g_old;
  canonize_image(source,trace);
  for (k=0;k<source->channels;k++)
    for (j=0;j<source->height;j++){
      g_old=0;
      for (i=source->width-1;i>=0;i--){
	g=get_pixel(source,i,j,k);
	g=MAX(g,g_old);
	put_pixel(trace,i,j,k,g);
	g_old=g*promille/1000;}
    }
  if (FMI_DEBUG(5)) write_image("debug_iir_left",trace,PGM_RAW);
}

void iir_up(FmiImage *source,FmiImage *trace,int promille){
  int i,j,k;
  int g,g_old;
  canonize_image(source,trace);
  for (k=0;k<source->channels;k++)
    for (i=0;i<source->width;i++){
      g_old=0;
      for (j=0;j<source->height;j++){
	g=get_pixel(source,i,j,k);
	g=MAX(g,g_old);
	put_pixel(trace,i,j,k,g);
	g_old=g*promille/1000;}
    }
  if (FMI_DEBUG(5)) write_image("debug_iir_up",trace,PGM_RAW);
}

void iir_down(FmiImage *source,FmiImage *trace,int promille){
  int i,j,k;
  int g,g_old;
  canonize_image(source,trace);
  for (k=0;k<source->channels;k++)
    for (i=0;i<source->width;i++){
      g_old=0;
      for (j=source->height-1;j>=0;j--){
	g=get_pixel(source,i,j,k);
	g=MAX(g,g_old);
	put_pixel(trace,i,j,k,g);
	g_old=g*promille/1000;}
    }
  if (FMI_DEBUG(5)) write_image("debug_iir_down",trace,PGM_RAW);
}




void mask_image(FmiImage *source,FmiImage *mask,Byte threshold,Byte c){
  register int i;
  check_image_properties(source,mask);
  if (FMI_DEBUG(3)) image_info(source);

  for (i=0;i<source->volume;i++)
    if (mask->array[i]<threshold)
      source->array[i]=c;

}

void threshold_image(FmiImage *source,FmiImage *target,Byte threshold){
  register int i;
  Byte c;
  for (i=0;i<source->volume;i++){
    c=source->array[i];
    target->array[i]=(c>threshold)?c:0;
  }
}

void binarize_image(FmiImage *source,FmiImage *target,Byte c){
  register int i;
  for (i=0;i<source->volume;i++)
    target->array[i]=(source->array[i]>c)?255:0;
}

#define MAXVAL 250
void propagate_right(FmiImage *source,FmiImage *domain,FmiImage *target, signed char slope,void (* put_func)(FmiImage *,int,int,int,Byte)){
  register int i,j,k;
  int c;


  check_image_properties(domain,target);
  if (source!=NULL) check_image_properties(domain,source);
  for (k=0;k<domain->channels;k++)
    for (j=0;j<domain->height;j++){
      c=0;
      for (i=0;i<domain->width;i++){
	/*for (i=1;i<domain->width-1;i++){ */
	/*	printf("\nDomain (%d,%d)=%d ",i,j,get_pixel(domain,i,j,k)) ; */
	if (((int)get_pixel(domain,i,j,k))>0){
	  if (c==0){
	    /*  printf("N "); */
	    if (source!=NULL){ 
	      c=(int)get_pixel(source,i,j,k);
	      if (c==0) c=1;}
	    else 
	      c=1;}
	  else {
	    c=c+slope;
	    if (c>MAXVAL) c=MAXVAL-1;
	    if (c<1)   c=1;}}
	else 
	  c=0;
	/*	c2=get_pixel(target,i,j,k); c=MAX(c,c2); */
	put_func(target,i,j,k,(Byte)c);
      }
    }
}

void propagate_left(FmiImage *source,FmiImage *domain,FmiImage *target, signed char slope,void (* put_func)(FmiImage *,int,int,int,Byte)){
  register int i,j,k;
  int c;

  for (k=0;k<domain->channels;k++)
    for (j=0;j<domain->height;j++){
      c=0;
      for (i=domain->width-1;i>=0;i--){
      /*for (i=domain->width-2;i>0;i--){ */
	if (((int)get_pixel(domain,i,j,k))>0){
	  if (c==0){
	    /* START MARKER MODE ... */
	    if (source!=NULL){ 
	      c=(int)get_pixel(source,i,j,k); /* ... with local start value */
	      if (c==0) c=1;}
	    else 
	      c=1;}                /* ... with value 1 */
	  else {
	    /* CONTINUE MARKER MODE  */
	    c=c+slope;
	    if (c>MAXVAL) c=MAXVAL-2;
	    if (c<1)   c=1;}}
	else /* domain==0 */
	  c=0; /* RETURN TO SEARCH MODE */
	/*c2=get_pixel(target,i,j,k);	c=MAX(c,c2); */
	put_func(target,i,j,k,(Byte)c);}
    }
}

void propagate_up(FmiImage *source,FmiImage *domain,FmiImage *target, signed char slope,void (* put_func)(FmiImage *,int,int,int,Byte)){
  register int i,j,k;
  int c;

  for (k=0;k<domain->channels;k++)
    for (i=0;i<domain->width;i++){
      c=0;
      for (j=0;j<domain->height;j++){
	if (get_pixel(domain,i,j,k)>0){
	  if (c==0){
	    if (source!=NULL){ 
	      c=get_pixel(source,i,j,k);
	      if (c==0) c=1;}
	    else 
	      c=1;}
	  else {
	    c=c+slope;
	    if (c>MAXVAL) c=MAXVAL-3;
	    if (c<1)   c=1;}}
	else
	  c=0;
	/* c2=get_pixel(target,i,j,k); c=MAX(c,c2); */
	put_func(target,i,j,k,(unsigned char)c);}
    }
}
     

void propagate_down(FmiImage *source,FmiImage *domain,FmiImage *target, signed char slope,void (* put_func)(FmiImage *,int,int,int,Byte)){
  register int i,j,k;
  int c;

  for (k=0;k<domain->channels;k++)
    for (i=0;i<domain->width;i++){
      c=0;
      for (j=domain->height-1;j>=0;j--){
	if (get_pixel(domain,i,j,k)>0){
	  if (c==0){
	    if (source!=NULL){
	      c=get_pixel(source,i,j,k);
	      if (c==0) c=1;}
	    else 
	      c=1;}
	  else {
	    c=c+slope;
	    if (c>MAXVAL) c=MAXVAL-4;
	    if (c<1)   c=1;}}
	else
	  c=0;
	/*c2=get_pixel(target,i,j,k); 	c=MAX(c,c2); */
	put_func(target,i,j,k,(unsigned char)c);}
    }
}
   

void horz_seg_lengths(FmiImage *source,FmiImage *target){
  propagate_right(NULL,source,target,1,put_pixel);  
  if (FMI_DEBUG(5)) write_image("debug_horz_seg_lengths_1",target,PGM_RAW);
  propagate_left(target,source,target,0,put_pixel);  
  if (FMI_DEBUG(4)) write_image("debug_horz_seg_lengths",target,PGM_RAW);
}

void vert_seg_lengths(FmiImage *source,FmiImage *target){
    propagate_up(NULL,  source,target,1,put_pixel);  
    if (FMI_DEBUG(5)) write_image("debug_vert_seg_lengths_1",target,PGM_RAW);
    propagate_down(target,source,target,0,put_pixel);  
    if (FMI_DEBUG(4)) write_image("debug_vert_seg_lengths",target,PGM_RAW);
}

void row_statistics(FmiImage *source,Byte *nonzero,Byte *avg,Byte *pow){
  register int i,j;
  int nz,s;
  long int s2;
  Byte c;
  for (j=0;j<source->height;j++){
    nz=0;
    s=0;
    s2=0;
    for (i=0;i<source->width;i++){
      c=get_pixel(source,i,j,0);
      /*nz+=(c>0); */
      if (c>0) ++nz;
      s+=c;
      s2+=c*c;}
    /*    printf("%d nz=%d s=%d s2=%d (w=%d)\n",j,nz,s,s2,source->width); */
    if (nonzero!=NULL) nonzero[j]=(255*nz)/source->width;
    if (avg!=NULL)   avg[j]=s/source->width;
    if (pow!=NULL) pow[j]=sqrt(s2/source->width);
  }
}

void col_statistics(FmiImage *source,Byte *nonzero,Byte *avg,Byte *pow){
  register int i,j;
  int nz,s;
  long int s2;
  Byte c;
  for (i=0;i<source->width;i++){
    nz=0;
    s=0;
    s2=0;
    for (j=0;j<source->height;j++){
      c=get_pixel(source,i,j,0);
      nz+=(c>0);
      s+=c;
      s2+=c*c;}
    if (nonzero!=NULL) nonzero[j]=255*nz/source->width;
    if (avg!=NULL)   avg[j]=s/source->width;
    if (pow!=NULL) pow[j]=sqrt(s2/source->width);
  }
}

