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

#include <stdio.h>
/*#include <limits.h> */
#include "fmi_util.h"
#include "fmi_image.h"
#include "fmi_image_filter.h"
#include "fmi_image_histogram.h"
#include "fmi_image_filter_speck.h"

/* THIS IS THE GOOD OLD BINARY PROBE */

#define UNVISITED 0
#define DETECTED  128
#define DONE      255

/* shared with the new fmi_image_speck  */
FmiImage *PROBE_DOMAIN;
FmiImage *PROBE_SOURCE;
FmiImage *PROBE_TARGET;
FmiImage PROBE_BOOK[1]; 

Histogram PROBE_SPECK_HISTOGRAM;
int  (* PROBE_SPECK_HISTOGRAM_INFO)(Histogram);

/*=========================================================================*/

/* unit steps in four directions */
int ROTX(int dir){ 
  static int rotx[4]={ 1, 0, 0,-1};
  return (rotx[dir&3]); 
} 

int ROTY(int dir){ 
  static int roty[4]={ 0,-1, 1, 0};   
  return (roty[dir&3]); 
} 

/* stack-minimisation trick */
int ROT_CODE(int i,int j){ 
  return ( (((i+j)&64)+2*((i-j)&64))/64 ); 
} 


/* trace = book keeping image */

/* subroutine: process single speck */
/*void probe_speck(FmiImage *target,FmiImage *trace,int i,int j,unsigned char min_value,Histogram PROBE_SPECK_HISTOGRAM){ */
/*void probe_speck(FmiImage *domain,FmiImage *trace,int i,int j,unsigned char min_value){ */
void probe_speck(int i,int j,unsigned char min_value){
  /* ,int *area,int histogram[256],int *perimeter){ */
  int dir;
  static unsigned char g;

  if (!legal_coords(PROBE_DOMAIN,i,j)){          /* OUTSIDE IMAGE  */
    PROBE_SPECK_HISTOGRAM[HIST_SIZE]++;
    PROBE_SPECK_HISTOGRAM[HIST_PERIMx3]+=3;
    PROBE_SPECK_HISTOGRAM[HIST_SUM_I]+=i;
    PROBE_SPECK_HISTOGRAM[HIST_SUM_J]+=j;
    PROBE_SPECK_HISTOGRAM[HIST_SUM_II]+=i*i;
    PROBE_SPECK_HISTOGRAM[HIST_SUM_JJ]+=j*j;
    PROBE_SPECK_HISTOGRAM[HIST_SUM_IJ]+=i*j;
    return;}

  if ((g=get_pixel(PROBE_DOMAIN,i,j,0))<min_value){    /* OUTSIDE SPECK  */
    PROBE_SPECK_HISTOGRAM[HIST_SIZE]++;
    PROBE_SPECK_HISTOGRAM[HIST_PERIMx3]+=3;
    PROBE_SPECK_HISTOGRAM[HIST_SUM_I]+=i;
    PROBE_SPECK_HISTOGRAM[HIST_SUM_J]+=j;
    PROBE_SPECK_HISTOGRAM[HIST_SUM_II]+=i*i;
    PROBE_SPECK_HISTOGRAM[HIST_SUM_JJ]+=j*j;
    PROBE_SPECK_HISTOGRAM[HIST_SUM_IJ]+=i*j;
    return;}

  if (get_pixel(PROBE_TARGET,i,j,0)!=UNVISITED)   /* ALREADY MARKED */
    return; 

  put_pixel(PROBE_TARGET,i,j,0,VISITED);
  /*  ++(*area);   */
  PROBE_SPECK_HISTOGRAM[HIST_AREA]++;
  g=get_pixel(PROBE_SOURCE,i,j,0);
  PROBE_SPECK_HISTOGRAM[g]++;
  if (g<PROBE_SPECK_HISTOGRAM[HIST_MIN])
    PROBE_SPECK_HISTOGRAM[HIST_MIN]=g;
  if (g>PROBE_SPECK_HISTOGRAM[HIST_MAX])
    PROBE_SPECK_HISTOGRAM[HIST_MAX]=g;

  /*
    histogram[HIST_SUM_I]+=i;
    histogram[HIST_SUM_J]+=j;
    histogram[HIST_SUM_II]+=i*i;
    histogram[HIST_SUM_JJ]+=j*j;
    histogram[HIST_SUM_IY]+=i*j;
  */
  /*  if (histogram[HIST_SUM_II]>(INT_MAX/16)) */
  /*  printf("SUM_II=%d\n",histogram[HIST_SUM_II]); */

  dir=ROT_CODE(i,j);
  probe_speck(i+ROTX(dir  ),j+ROTY(dir  ),min_value);
  probe_speck(i+ROTX(dir+1),j+ROTY(dir+1),min_value);
  probe_speck(i+ROTX(dir+2),j+ROTY(dir+2),min_value);
  probe_speck(i+ROTX(dir+3),j+ROTY(dir+3),min_value);
  return;
}

/* subroutine: process single speck */
/*void propagate_attribute(FmiImage *domain,FmiImage *trace,int i,int j,unsigned char min_value,unsigned char attribute){ */
void propagate_attribute(int i,int j,unsigned char min_value,unsigned char attribute){
  int dir;
  if (!legal_coords(PROBE_DOMAIN,i,j))    return; /* OUTSIDE IMAGE  */
  if (get_pixel(PROBE_DOMAIN,i,j,0)<min_value)  return; /* OUTSIDE SPECK  */
  if (get_pixel(PROBE_BOOK, i,j,0)==DONE) return;
  put_pixel(PROBE_TARGET,i,j,0,attribute);
  put_pixel(PROBE_BOOK,i,j,0,DONE);
  if (FMI_DEBUG(1)){
    /*    fprintf(stderr," i=%d j=%d area=%d\n",i,j,area);fflush(stderr); */
  };
  dir=ROT_CODE(i,j);
  propagate_attribute(i+ROTX(dir  ),j+ROTY(dir  ),min_value,attribute);
  propagate_attribute(i+ROTX(dir+1),j+ROTY(dir+1),min_value,attribute);
  propagate_attribute(i+ROTX(dir+2),j+ROTY(dir+2),min_value,attribute);
  propagate_attribute(i+ROTX(dir+3),j+ROTY(dir+3),min_value,attribute);
}

/* MAIN PROCESS (recursive) */
/*void traverse_image(FmiImage *target,FmiImage *trace,int i,int j,int min_value){ */
void traverse_image(int i,int j,int min_value){
  /*  static int area,perimeter; */
  /*  static int histogram[256]; */
  /*  static Histogram histogram; */
  /*  Histogram histogram; */
  int attribute;
  int dir;
  if (!legal_coords(PROBE_DOMAIN,i,j)) return;
  /*  if (get_pixel(trace,i,j,0)!=UNVISITED) return; */
  if (get_pixel(PROBE_BOOK,i,j,0)!=UNVISITED) return;
  
  /*  fprintf(stderr,"traverse, i=%d, j=%d \n",i,j); */
  if (get_pixel(PROBE_DOMAIN,i,j,0)<min_value){
    /*    put_pixel(trace,i,j,0,0); */
    put_pixel(PROBE_BOOK,i,j,0,DETECTED);
    /* SEARCH MODE (continue searching segments in image) */
        dir=ROT_CODE(i,j);
	/*dir=0; */
    traverse_image(i+ROTX(dir  ),j+ROTY(dir  ),min_value);
    traverse_image(i+ROTX(dir+1),j+ROTY(dir+1),min_value);
    traverse_image(i+ROTX(dir+2),j+ROTY(dir+2),min_value);
    traverse_image(i+ROTX(dir+3),j+ROTY(dir+3),min_value);}
  else{
    /* SPECK PROBE MODE */
    /* DETECT SPECK, COMPUTE ITS AREA, HISTOGRAM, PERIMETER ON THE RUN */
    /*    histogram[HIST_AREA]=0; */
    /*    histogram[HIST_PERIMx3]=0; */
    /*    clear_histogram(histogram); */

    clear_histogram_full(PROBE_SPECK_HISTOGRAM);
    /*    PROBE_SPECK_HISTOGRAM[HIST_MIN]=255; */
    PROBE_SPECK_HISTOGRAM[HIST_MIN]=255;
    PROBE_SPECK_HISTOGRAM[HIST_SUM_I]=0;
    PROBE_SPECK_HISTOGRAM[HIST_SUM_J]=0;
    PROBE_SPECK_HISTOGRAM[HIST_SUM_II]=0;
    PROBE_SPECK_HISTOGRAM[HIST_SUM_JJ]=0;
    PROBE_SPECK_HISTOGRAM[HIST_SUM_IJ]=0;


    /*    probe_speck(target,trace,i,j,min_value,PROBE_SPECK_HISTOGRAM); */
    probe_speck(i,j,min_value);
    put_pixel(PROBE_BOOK,i,j,0,DETECTED);

    /*    attribute=histogram_function(PROBE_SPECK_HISTOGRAM); */
    attribute=PROBE_SPECK_HISTOGRAM_INFO(PROBE_SPECK_HISTOGRAM);
    if (histogram_scaling_function!=NULL)
      attribute=histogram_scaling_function(histogram_scaling_parameter,attribute);

    if (FMI_DEBUG(5)){
      fprintf(stderr," found speck, i=%d, j=%d  \n",i,j);
      fprintf(stderr,"        size: %d \n",(int)PROBE_SPECK_HISTOGRAM[HIST_AREA]);
      fprintf(stderr,"   function value: %d \n",attribute);
      fflush(stderr);}
	
    if (attribute<1)  attribute=1;
    if (attribute>250)  attribute=250;
    propagate_attribute(i,j,min_value,attribute);
    fmi_debug(5,"mark_speck_size finished");
  }
}

/* CLIENT (STARTER) */
void Binaryprobe(FmiImage *domain,FmiImage *source,FmiImage *trace,int (* histogram_function)(Histogram),unsigned char min_value){ 
  register int i,j;
  /*  fprintf(stderr,"\tHIST=%d\n",histogram_function(h)); */
  /* fprintf(stderr,"\tHIST=%d\n",histogram_area(h)); */
  fmi_debug(3,"filter_specks");
  if (source->channels!=1) 
    fmi_error("filter_specks: other than single-channel source");
  PROBE_DOMAIN=domain;
  PROBE_SOURCE=source;
  PROBE_TARGET=trace;
  canonize_image(source,PROBE_BOOK);
  canonize_image(source,PROBE_TARGET);
  fill_image(PROBE_BOOK,UNVISITED);
  fill_image(trace,0);

  PROBE_SPECK_HISTOGRAM_INFO=histogram_function;


  fmi_debug(4,"filter_specks...");

  for (i=0;i<source->width;i++)
    for (j=0;j<source->height;j++)
      /*traverse_image(source,trace,i,j,min_value); */
      traverse_image(i,j,min_value);

  fmi_debug(4,"filter_specks, DONE.");

  if (FMI_DEBUG(5))
    write_image("probe",PROBE_BOOK,PGM_RAW);
}

void detect_specks(FmiImage *source,FmiImage *trace,unsigned char min_value,int (* histogram_function)(Histogram)){ 
  Binaryprobe(source,source,trace,histogram_function,min_value);
}

/* DEBUGGING... */
void test_rotation(FmiImage *target,FmiImage *trace,int i,int j,int rec_depth){
  int dir=1;
  /*int k,l; */
  if (!legal_coords(target,i,j)) return;
  if (get_pixel(trace,i,j,0)!=UNVISITED) return;
  /*  fprintf(stderr,"\ti=%d \tj=%d \tdepth=%d\n",i,j,rec_depth); fflush(stderr); */
  put_pixel(trace,i,j,0,10+dir);
  /*  fprintf(stderr,"ok\n"); fflush(stderr); */
  /*  for (i=0;i<target->width;i++) */
  /*  for (j=0;j<target->height;j++){ */

  if ((rec_depth&255)==0)
    fprintf(stderr,"\ti=%d \tj=%d \tdepth=%d\n",i,j,rec_depth); fflush(stderr);
  dir=ROT_CODE(i,j);
  put_pixel(trace,i,j,0,10+dir);
  put_pixel(target,i,j,0,(rec_depth>>8)&255);
  test_rotation(target,trace,i+ROTX(dir  ),j+ROTY(dir  ),rec_depth+1);
  test_rotation(target,trace,i+ROTX(dir+1),j+ROTY(dir+1),rec_depth+1);
  test_rotation(target,trace,i+ROTX(dir+2),j+ROTY(dir+2),rec_depth+1);
  test_rotation(target,trace,i+ROTX(dir+3),j+ROTY(dir+3),rec_depth+1);
  /*  fmi_debug(1,"finito"); */
}


