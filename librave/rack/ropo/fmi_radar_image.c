/**

    Copyright 2011 Martin Raspaud, SMHI <martin.raspaud@smhi.se>
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
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include "fmi_util.h"
#include "fmi_image.h"
#include "fmi_image_arith.h"
#include "fmi_image_filter.h"
#include "fmi_image_filter_morpho.h"
#include "fmi_image_filter_line.h"
#include "fmi_image_histogram.h"
#include "fmi_image_filter_speck.h"
#include "fmi_meteosat.h"
#include "fmi_radar_image.h"


float fmi_radar_sweep_angles[FMI_RADAR_SWEEP_COUNT]={0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 11.0, 20.0, 45.0};
float fmi_radar_bin_depth = 0.0;

#define RGBCOUNT 14
int dbz_rgb[RGBCOUNT][4]={
  {-30,    0,  0,  0},
  {-20,    0,  0,128},
  {-10,    0,  0,192},
  { -5,    0, 96,255},
  {  0,    0,160, 92},
  { 10,   64,255,  0},
  { 20,  192,192,  0},
  { 30,  224,160,  0},
  { 40,  255,128,  0},
  { 50,  255, 64,  0},
  { 60,  255, 32, 64},
  { 70,  255,  0,128},
  { 80,  255,  0,192},
  { 90,  255,  0,255}
};

void
setup_context(FmiImage * source)
{
  fmi_radar_bin_depth = source->bin_depth;
}

int abs_dbz_to_byte(Dbz dbz){
  int g;
  g=dbz*2+64;
  if (g<0) g=0;
  if (g>255) g=255;
  return  (g);
}

int rel_dbz_to_byte(Dbz dbz){ 
  int g;
  g=dbz*2;
  if (g<0) g=0;
  if (g>255) g=255;
  return g;
}

int abs_dbz_to_int(Dbz dbz){
  return(dbz*2+64);
}

int rel_dbz_to_int(Dbz dbz){ 
  return(dbz*2);
}



int byte_to_abs_dbz(Byte byte){ 
  return ((byte-64)/2);
}

int byte_to_rel_dbz(Byte byte){ 
  return (byte/2);
}

float radians_to_degree(float radians){ return (radians*360.0/(2.0*PI));}
float degrees_to_radian(float degrees){ return (degrees*2.0*PI/360.0);}

int bin_to_metre(int bin){
  return ((1+2*bin) * fmi_radar_bin_depth/2);
  /*    = (0.5+bin) * fmi_radar_bin_depth */
}

int metre_to_bin(int metre){
  return(metre/fmi_radar_bin_depth);
}

int bin_to_altitude(int sweep_bin,float sweep_angle){
  /*  const float b=(0.5+(float)sweep_bin)*fmi_radar_bin_depth; */
  const float b=bin_to_metre(sweep_bin);
  const float a=EARTH_RADIUS43;
  /* by cosine rule */
  return (int)sqrt(b*b+a*a-2.0*a*b*cos(PI/2.0+degrees_to_radian(sweep_angle)))-EARTH_RADIUS43;
}

int altitude_to_bin(int altitude,float sweep_angle){
  const float c=EARTH_RADIUS43+altitude;
  const float a=EARTH_RADIUS43;
  const float gamma=degrees_to_radian(sweep_angle)+PI/2.0;
  const float beta=PI-gamma-asin(a*sin(gamma)/c);
  /* by sine rule */
  /* sin(gamma)/c = sin(beta)/b  => b=sin(beta)*c/sin(gamma) */
  return (sin(beta)*c/sin(gamma)/fmi_radar_bin_depth);
}

int ground_to_bin(int g_metre,float sweep_angle){
  float beta; /* angle<RADAR,GROUND_POINT> */
  float gamma; /* angle<BIN_RADAR,BIN_GROUND_POINT> */
  beta=(float)g_metre/(float)EARTH_RADIUS43;
  gamma=PI-degrees_to_radian(sweep_angle)-PI/2.0-beta;
  /*  printf("...%f\n",sweep_angle); */
  /* SINE RULE */
  /* sin(beta)/BIN = sin(gamma) / EARTH_RADIUS */
  return (int)(sin(beta)/sin(gamma)*(float)EARTH_RADIUS43/fmi_radar_bin_depth);
  /*  return (int)(2.0*sweep_angle); */
}

int bin_to_ground(int bin,float sweep_angle){
  float x,y;
  const float m=bin_to_metre(bin);
  const float chi=degrees_to_radian(sweep_angle);
  x=m*cos(chi);
  y=m*sin(chi);
  return (EARTH_RADIUS43*atan(x/(y+(float)EARTH_RADIUS43)));
  /* COSINE RULE?    */
}

int bin_to_bin(int sweep_bin,float sweep_angle,float target_sweep_angle){
  return  ground_to_bin( bin_to_ground(sweep_bin,sweep_angle), target_sweep_angle);
}

void xy_to_polar(int i,int j,int *theta,int *radius){
  /*  const int radius=250; */
  i=(i-250)*2;
  j=(j-250)*2;
  *radius=(int)sqrt((double)i*i+j*j);
  *theta=180*atan2(i,-j)/3.14;
  if (*theta<0) 
    *theta+=360;
}

void volume_to_cappi(FmiImage *volume,int height,FmiImage *cappi){
  int i,j,k;
  /* VERTICAL INTERPOLATION SCHEME */
  int channels = volume->channels;
  int *bin;
  Byte b,b_upper,b_lower;
  /*  const Byte PSEUDOCAPPI_MARKER=31; */
  const Byte PSEUDOCAPPI_MARKER=1;
  /* sweep indices */
  int upper,lower;
  int h,h_upper,h_lower;
  int mix;

  bin = (int *)malloc(sizeof(int) * volume->sweep_count);

  /* assumption: max area = area of lowest ppi */
  /*
    Can't work with variable ray counts.
    Actually, that shouldn't be channels, should it ?

    martin.raspaud@NOSPAM.smhi.se, Fri Aug  5 11:04:42 2011.
  if (volume->channels==1){
    fmi_debug(0,"warning: computing CAPPI of only one PPI?");
    channels=volume->channels;
    split_to_channels(volume,volume->height/FMI_RADAR_RAY_COUNT);
  }
  */
  setup_context(volume);

  canonize_image(&volume[1],cappi);
  for (i=0;i<cappi->width;i++){
    upper=-1;
    lower=-1;
    h_upper=INT_MAX;
    h_lower=0;
    for (k=0;k<volume->channels;k++){
      bin[k]=ground_to_bin(i*fmi_radar_bin_depth,volume[k+1].elevation_angle);
      if (bin[k]>=volume->width) 
        bin[k]=volume->width-1;
      h=bin_to_altitude(bin[k],volume[k+1].elevation_angle);
      if ((h<=height)&&(h>=h_lower)){
        lower=k;
        h_lower=h;
      }
      if ((h>=height)&&(h<=h_upper)){
        upper=k;
        h_upper=h;
      }
    }

    if (FMI_DEBUG(5))
      fprintf(stderr,"lower(%d)=%dm [%d]\t upper(%d)=%dm [%d]\n",lower,h_lower,bin[lower],upper,h_upper,bin[upper]);


    if (upper==-1)
      /* NO UPPER SWEEP AVAILABLE */
      for (j=0;j<cappi->height;j++){
	b=get_pixel(volume,bin[lower],j,lower);
	b=b&~PSEUDOCAPPI_MARKER;
	put_pixel(cappi,i,j,0,b);
      }
    else if (lower==-1)
      /* NO LOWER SWEEP AVAILABLE */
      for (j=0;j<cappi->height;j++){
	b=get_pixel(volume,bin[upper],j,upper);
	b=b&~PSEUDOCAPPI_MARKER;
	/*	b=(i*255)/cappi->width; */
	put_pixel(cappi,i,j,0,b);
     }
    else {
      /* OK - BETWEEN TWO SWEEPS  */
      mix=(255*(height-h_lower))/(h_upper-h_lower);
      if (mix>255) mix=255;
      if (mix<0)   mix=0;
      for (j=0;j<cappi->height;j++){
	b_lower=get_pixel(volume,bin[lower],j,lower);
	b_upper=get_pixel(volume,bin[upper],j,upper);
	b=(mix*b_upper+(255-mix)*b_lower)/255;
	b=b | PSEUDOCAPPI_MARKER;
	/*	b=128; */
	put_pixel(cappi,i,j,0,b);
      }
    }


  }
  split_to_channels(volume,channels);
  free(bin);
}

void dump_sweep_info(){
  int i,j;
  float beta;
  int b,alt;
  for (j=0;j<FMI_RADAR_SWEEP_COUNT;j++){
	  printf("#[%d]",j);
	  printf("\t%3.1f\t",fmi_radar_sweep_angles[j]);
  }
  printf("\n");
  for (i=0;i<250000;i+=100){
    printf("%d",i);
    for (j=0;j<FMI_RADAR_SWEEP_COUNT;j++){
      beta=fmi_radar_sweep_angles[j];
      /*	  printf("\t[%.1f]%d",fmi_radar_sweep_angles[j],ground_to_bin(i,fmi_radar_sweep_angles[j])); */
      b=ground_to_bin(i,beta);
      b=MIN(b,FMI_RADAR_BIN_COUNT);
      printf("\t%3d",b);
      alt=bin_to_altitude(b,beta);
      printf("%7d",alt);
    }
    /*    printf("\t ppi0:'%d' ppi45 %d",i/500,bin_to_bin(i/500,fmi_radar_sweep_angles[0],fmi_radar_sweep_angles[FMI_RADAR_SWEEP_COUNT-1])); */
    /*    printf("\t alt=%d",bin_altitude(ground_to_bin(i,45),45)); */
    printf("\n");
  }
  /*fprintf(stderr,"plot ",i); */
  fprintf(stderr,"plot ");
  for (j=0;j<FMI_RADAR_SWEEP_COUNT;j++)
    fprintf(stderr,"\\\n'sweep.dat' using 1:%d title \"%.1f�\",",2*j+3,fmi_radar_sweep_angles[j]);
  fprintf(stderr,"\b \n");
  
}


/* enhances invidual responses on lines with large total response */
void enhance_horz(FmiImage *trace){
 register int i,j,k;
 int m,c;
 int *n;
 n=(int*)malloc(trace->height*sizeof(int));
 /* int m[trace->height]; */
 for (k=0;k<trace->channels;k++){
   for (j=0;j<trace->height;j++){
      n[j]=0;
      for (i=0;i<trace->width;i++){
	if (get_pixel(trace,i,j,k)>0)
	  n[j]++;
      }
      /*n[j]=MIN(n[j],trace->width); */
      /*      n[j]=MAX(n[j],0); */
   }
   for (j=0;j<trace->height;j++){
     m=(n[(j-1)%trace->height]+2*n[j]+n[(j+1)%trace->height])/4;
     m=MAX(m,n[j]);
     for (i=0;i<trace->width;i++){
       /* put_pixel(trace,i,j,k,255); */
       c=get_pixel(trace,i,j,k)*m/trace->width;
       c=MIN(c,255);
       put_pixel(trace,i,j,k,c);}
   }
 } 
 free(n);
}

void enhance_horz255(FmiImage *trace,Byte row_statistic[]){
  register int i,j;
  register int c,s;
  
  for (j=0;j<trace->height;j++){
    s=row_statistic[j];
    for (i=0;i<trace->width;i++){
      c=get_pixel(trace,i,j,0)*s/255;
      put_pixel(trace,i,j,0,c);
    }
  }
} 

/* REMOVE DISTINCT HORIZONTAL LINES ??? */
/* detect pixels brighter than neighborhood */
/* compute lengths of detected segments, store as segment intesities  */
/* replace segments longer than max_length pixels with neighborhood mean */


/* FUZZY PRODUCT OF vert AND horz lengths; VERT CUT FIRST! */
void detect_horz_segments(FmiImage *source,FmiImage *trace,int max_width,int min_length){
  /*  register int i,j; */
  /*  int min_length=60; */
  FmiImage horz;
  FmiImage vert;

  canonize_image(source,trace);
  canonize_image(source,&horz);
  canonize_image(source,&vert);

  /*  threshold_image(source,trace,min_intensity); */

  /* COMPUTE VERTICAL RUN LENGTHS; SCALE */
  vert_seg_lengths(source,&vert);
  semisigmoid_image_inv(&vert,max_width);
  translate_intensity(&vert,255,0);
  threshold_image(&vert,trace,32);
  threshold_image(&vert,&vert,96);

  if (FMI_DEBUG(5)) write_image("debug_horz_segments_vert",&vert,PGM_RAW);

  /* COMPUTE VERTICAL RUN LENGTHS; SCALE */
  horz_seg_lengths(&vert,&horz);
  /*  semisigmoid_image(&horz,min_length);  */
  sigmoid_image(&horz,min_length,1); 

  /*  multiply_image255_sigmoid(&vert,&horz,trace); */
  multiply_image255_sigmoid(trace,&horz,trace);

  if (FMI_DEBUG(5)) write_image("debug_horz_segments",trace,PGM_RAW);
  /*reset(&horz); */
  /*reset(&vert); */
}


/* UNITY-WIDTH SEGMENTS */
/* CLIENTS: detect_emitters2 */
void detect_emitters(FmiImage *source,FmiImage *trace,int min_elevation,int min_length){

	canonize_image(source,trace);

	/* FIND VERT MAXIMA... */
	/* DEBUG threshold_image(source,trace,min_elevation); */
	/*detect_vert_maxima(source,trace); */


	detect_vert_maxima(source,trace);
	/*put_pixel(trace,20,20,0,200); */
	threshold_image(trace,trace,min_elevation);
	if (FMI_DEBUG(5)) write_image("debug_emitter_1_vert_max",trace,PGM_RAW);

	/* ... AND COMPUTE HORZ LENGTHS */
	horz_seg_lengths(trace,trace);
	threshold_image(trace,trace,min_length-1);
	semisigmoid_image(trace,min_length);

	if (FMI_DEBUG(4)) write_image("debug_emitter",trace,PGM_RAW);

}

void detect_emitters2(FmiImage *source,FmiImage *trace,int min_intensity,int min_length,int max_width){
  /*register int i,j; */

  /*  FmiImage unity_width_segments; */
  FmiImage candidate;
  FmiImage mask,mask2;
  /*  FILE *fp; // debug */
  FmiImage temp;

  canonize_image(source,&temp);
  canonize_image(source,trace);
  canonize_image(source,&candidate);
  initialize_vert_stripe(&mask,source->height);
  initialize_vert_stripe(&mask2,source->height);

  /*  canonize_image(source,&vert); */

  /* PHASE 1: marginal (ray) sums of unity-width segments */
  /*
    detect_vert_maxima(source,&temp); // unity-width segments
    threshold_image(&temp,&temp,2); // ROBUST ANYWAY
    if (FMI_DEBUG(4)) write_image("debug_emitter_1_seg1",&temp,PGM_RAW);
    horz_seg_lengths(&temp,&temp);
    threshold_image(&temp,&temp,min_length-1); 
    semisigmoid_image(&temp,min_length); 
    if (FMI_DEBUG(5)) write_image("debug_emitter_2_seg2",&temp,PGM_RAW);
  */

  /* COMPUTE UNITY-WIDTH SEGMENTS... */
  detect_emitters(source,&temp,min_intensity/2,min_length/2);
  /* ...TO BE USED AS TENTATIVE EVIDENCE. */
  image_average_horz(&temp,&mask);
  pipeline_process(&mask ,&mask2,0,1,histogram_max);
  pipeline_process(&mask2,&mask ,0,1,histogram_mean);
  /*semisigmoid_image(&mask,32); */
  sigmoid_image(&mask,16,2);
  if (FMI_DEBUG(5)) write_image("debug_emitter2_3_stripe",&mask,PGM_RAW);

  /* PHASE 2: calculate candidates by rigid vert x horz probing */
  /*  copy_image(source,trace); */
  /*
    threshold_image(source,trace,min_intensity);
    
    vert_seg_lengths(trace,&temp);
    translate_intensity(&temp,0,255);
    sigmoid_image(&temp,max_width,-1);
    threshold_image(&temp,&temp,64); 
    if (FMI_DEBUG(4)) write_image("debug_emitter_4_vert_run",&temp,PGM_RAW);
    
    horz_seg_lengths(&temp,&candidate);
    sigmoid_image(&candidate,min_length,2);
    threshold_image(&candidate,&candidate,64); 
    if (FMI_DEBUG(4)) write_image("debug_emitter_5_horzrun",&candidate,PGM_RAW);
  */

  /* THEN, DETECT HORIZONTAL SEGMENTS... */
  threshold_image(source,&temp,min_intensity); /* seems: was needed */
  detect_horz_segments(&temp,&candidate,max_width,min_length);
  threshold_image(&candidate,trace,16);
  /* pipeline_process(&candidate,&temp,4,1,histogram_max); */
  /* pipeline_process(&temp,trace,3,1,histogram_mean); */
  /* pipeline_process(&temp,trace,4,1,histogram_max); */
  if (FMI_DEBUG(5)) write_image("debug_emitter2_6_raw",trace,PGM_RAW); 

  /* PHASE 3: weight PHASE 2 image by rowsum obtained in PHASE 1 */
  /*enhance_horz255(trace,row_nonzero); */
  multiply_image255_flex(trace,&mask,trace);
  semisigmoid_image(trace,32);  /* 16 */

  if (FMI_DEBUG(4)) write_image("debug_emitter2",trace,PGM_RAW);
  /*reset_image(&temp); */
  /*reset_image(&candidate); */
}


/*----------------- */

void smooth_signal(Byte signal[],int length){
  register int i;
  Byte *signal2;
  signal2=(Byte*)malloc(sizeof(Byte)*length);
  for (i=0;i<length;i++)
    signal2[i]=(signal[(i-1)%length] + 2*signal[i] +signal[(i+1)%length])/4;
  for (i=0;i<length;i++)
    signal[i]=MAX(signal[i],signal2[i]);
  free(signal2);
}

void detect_emitters2old(FmiImage *source,FmiImage *trace,int min_intensity,int min_length,int max_width){
  register int i,j;
  Byte *row_nonzero;
  Byte *row_avg;
  Byte *row_pow;
  /*  FmiImage unity_width_segments; */
  FmiImage candidate;
  FILE *fp; /* debug */
  FmiImage temp;

  row_nonzero=(Byte*)malloc(sizeof(Byte)*source->height);
  row_avg=(Byte*)malloc(sizeof(Byte)*source->height);
  row_pow=(Byte*)malloc(sizeof(Byte)*source->height);

  canonize_image(source,&temp);
  canonize_image(source,trace);
  canonize_image(source,&candidate);
  /*  canonize_image(source,&vert); */

  /* PHASE 1: marginal (ray) sums of unity-width segments */
  detect_vert_maxima2(source,&temp); /* unity-width segments */
  threshold_image(&temp,&temp,2); /* ROBUST ANYWAY */
  if (FMI_DEBUG(4)) write_image("debug_emitter_seg1",&temp,PGM_RAW);
  /* no intensity-min_length compensation -> box logic: */
  horz_seg_lengths(&temp,&temp);
  sigmoid_image(&temp,min_length,1); 
  threshold_image(&temp,&temp,126); 
  if (FMI_DEBUG(5)) write_image("debug_emitter_seg2",&temp,PGM_RAW);
  row_statistics(&temp,row_nonzero,row_avg,row_pow); 
  for (i=0;i<trace->height;i++){
    /*    if (row_nonzero[i]<16) row_nonzero[i]=0; */
    j=128+pseudo_sigmoid(1,((int)row_nonzero[i])-16);      /* =16/255 *100% */
    j=MAX(j,0);
    j=MIN(j,255); 
    /*    if (row_nonzero[i]<16) j=0; */
    row_nonzero[i]=j;}
  /*  for (i=0;i<32;i+=1) printf("%d\t%d\n",i,128+pseudo_sigmoid(1,i-16)/2);      */
  smooth_signal(row_nonzero,trace->height);
  if (FMI_DEBUG(5)) {
    fp=fopen("row_stats.txt","w");
    for (j=0;j<temp.height;j++){
      fprintf(fp,"%d\t%d\t%d\n",row_nonzero[j],row_avg[j],row_pow[j]);
      put_pixel(&temp,row_nonzero[j],j,0,255);
      put_pixel(&temp,row_avg[j],j,0,254);
      put_pixel(&temp,sqrt(row_pow[j]),j,0,253);
    }
    fclose(fp);
  }
  if (FMI_DEBUG(4)) write_image("debug_emitter_horz_spread",&temp,PGM_RAW);

  /* PHASE 2: calculate candidates by rigid vert x horz probing */
  copy_image(source,trace);
  threshold_image(trace,trace,min_intensity);

  vert_seg_lengths(trace,&temp);
  translate_intensity(&temp,0,255);
  sigmoid_image(&temp,max_width,-1);
  threshold_image(&temp,&temp,64); 
  if (FMI_DEBUG(4)) write_image("debug_emitter_vert_run",&temp,PGM_RAW);

  horz_seg_lengths(&temp,&candidate);
  sigmoid_image(&candidate,min_length,2);
  threshold_image(&candidate,&candidate,64); 
  if (FMI_DEBUG(4)) write_image("debug_emitter_horz_run",&candidate,PGM_RAW);

  pipeline_process(&candidate,&temp,5,1,histogram_max);
  pipeline_process(&temp,trace,3,1,histogram_mean);
  if (FMI_DEBUG(4)) write_image("debug_emitter_trace",trace,PGM_RAW); 

  /* PHASE 3: weight PHASE 2 image by rowsum obtained in PHASE 1 */
  enhance_horz255(trace,row_nonzero);
  semisigmoid_image(trace,32);  /* 16 */

  if (FMI_DEBUG(4)) write_image("debug_emitter_trace_enh",trace,PGM_RAW);
  /*reset_image(&temp); */
  /*reset_image(&candidate); */
  
  free(row_pow);
  free(row_avg);
  free(row_nonzero);
}

/*----------------- */



void detect_sun(FmiImage *source,FmiImage *trace,int min_intensity,int max_width,int min_total_length){
  /*  FmiImage mask; */
  const int closing_dist=4;
  const int min_seg_length=4;

  canonize_image(source,trace);
  /*  initialize_vert_stripe(&mask,source->height); */

  /*   */
  threshold_image(source,trace,min_intensity);
  morph_closing(trace,trace,closing_dist,0); 
  detect_horz_segments(trace,trace,max_width,min_seg_length);

  horz_seg_lengths(trace,trace);
  /* if (FMI_DEBUG(4)) write_image("debug_sun_length",trace,PGM_RAW);  */
  sigmoid_image(trace,min_total_length,4);
  /*  if (FMI_DEBUG(4)) write_image("debug_sun_length2",trace,PGM_RAW);  */

  /* IF YOU USE THIS, CLEAR-SKY NORMALIZE THIS!
     image_average_horz(trace,&mask);
     multiply_image255_flex(trace,&mask,trace);
  */
  /*  threshold_image(trace,trace,64); */
  if (FMI_DEBUG(4)) write_image("debug_sun",trace,PGM_RAW); 

}


void detect_sun2(FmiImage *source,FmiImage *trace,int min_intensity,int max_width,int min_total_length,int azimuth,int elevation){
  register int j,mj=source->height/2;
  FmiImage mask;
  const int closing_dist=4;
  /*const int min_seg_length=4; */

  fmi_debug(2,"sun2");

  canonize_image(source,trace);

  if ((elevation<-2)||(elevation>20)){
    fill_image(trace,0);
    return;}

  /*  detect_sun(source,trace,min_intensity,min_length,max_width);  */
  threshold_image(source,trace,min_intensity);
  morph_closing(trace,trace,closing_dist,0); 
  if (FMI_DEBUG(5)) write_image("debug_sun1",trace,PGM_RAW);
  detect_horz_segments(trace,trace,max_width,min_total_length);
  if (FMI_DEBUG(5)) write_image("debug_sun2",trace,PGM_RAW);
  /*morph_closing(trace,trace,closing_dist,0); */
  /*  fmi_debug(5,"debug_sun_4"); */

  initialize_vert_stripe(&mask,source->height);
  for (j=0;j<source->height;j++){
    put_pixel_direct(&mask,(j+mj+azimuth)%source->height,pseudo_gauss_int(max_width,j-mj));
    /*    put_pixel(trace,0,(j+mj+azimuth)%source->height,0,pseudo_gauss_int(width,j-mj)); */
  }
  if (FMI_DEBUG(5)) write_image("debug_sun3",&mask,PGM_RAW);
  /*  fmi_debug(5,"debug_sun_4"); */
  multiply_image255_flex(trace,&mask,trace);
 
 if (FMI_DEBUG(4)) write_image("debug_sun",trace,PGM_RAW); 
 /* fmi_debug(2,"sun2"); */
}






/* 
   BASIC IDEA
   - Detect specks, of which size < "max_area" and intensity > "min_intensity"
   - SPECK must be pronounced, not just thresholded
*/
void detect_ships(FmiImage *source,FmiImage *prob,int min_intensity,int max_area){
  /* register int i; */
  int max_radius;
  /*  FmiImage trace;   */
  /*  FmiImage trace; */
  /*FmiImage trace4; */

  FmiImage temp1;
  FmiImage temp2;
  /*  FmiImage temp3; */
  FmiImage specks;
  FmiImage virtual;
  FmiImage lines;

  max_radius=sqrt(max_area)/2;

  canonize_image(source,&temp1);
  fmi_debug(2,"--------------");
  if (FMI_DEBUG(2)) image_info(&temp1);
  fmi_debug(2,"==============");

  canonize_image(source,&temp2);
  /*  image_info(&temp2); */
  /* canonize_image(source,&temp3); */
  canonize_image(source,&specks);
  canonize_image(source,&virtual);
  canonize_image(source,&lines);

  /*  fmi_debug(2,"--------------"); */
  /*
    image_info(&temp1);
    image_info(&temp2);
    image_info(&specks);
    image_info(&virtual);
    image_info(&lines);
  */
  fmi_debug(2,"remove ships2");

  /* high-boost filter for detecting specks */
  pipeline_process(source,&temp1,max_radius+2,max_radius+2,histogram_mean);
  subtract_image(source,&temp1,&specks); 
  threshold_image(&specks,&specks,min_intensity);
  if (FMI_DEBUG(5)) write_image("debug_ship_speck1",&specks,PGM_RAW);

  /* cut off short segments */
  propagate_right(&specks,&specks,&specks,-16,put_pixel);
  propagate_left(&specks,&specks,&specks,0,put_pixel_min);
  threshold_image(&specks,&specks,min_intensity/2);
  if (FMI_DEBUG(4)) write_image("debug_ship_speck2",&specks,PGM_RAW);

  /* create virtual echoes (= locations of possible sidelobe echoes) */
  iir_up(&specks,&temp1,975);
  iir_down(&specks,&temp2,975);
  max_image(&temp1,&temp2,&virtual);
  if (FMI_DEBUG(4)) write_image("debug_ship_virtual",&virtual,PGM_RAW);


  /* detect possible sidelobe echoes in data */
  /*  edges */
  detect_vert_edges(source,&temp1);
  threshold_image(&temp1,&temp1,8);
  /*  emphasize long segments, prune short segments */
  propagate_up(NULL,&temp1,&temp2,1*8,put_pixel);
  propagate_down(&temp2,&temp2,&lines,0,put_pixel);
  threshold_image(&lines,&lines,2*8);
  if (FMI_DEBUG(3)) write_image("debug_ship_edges",&lines,PGM_RAW);

  /*  detect suspicious segments = connect HORZ segments */
  pipeline_process(source,&temp2,1,0,histogram_max);
  pipeline_process(&temp2,&temp1,1,0,histogram_min);
  if (FMI_DEBUG(4)) write_image("debug_ship_edges2",&temp1,PGM_RAW);
  propagate_right(NULL,&temp1,&temp2,1,put_pixel);
  propagate_left(&temp2,&temp2,&temp1,0,put_pixel);
  if (FMI_DEBUG(4)) write_image("debug_ship_edges3",&temp1,PGM_RAW);

  /* here we get the "real" sidelobes */
  subtract_image(&lines,&temp1,&lines);
  pipeline_process(&lines,&temp1,0,1,histogram_max);
  pipeline_process(&temp1,&lines,0,2,histogram_mean);
  if (FMI_DEBUG(4)) write_image("debug_ship_edges4",&lines,PGM_RAW);

  /*  multiply_image(&lines,&virtual,&lines); */
  min_image(&lines,&virtual,&lines);
  if (FMI_DEBUG(3)) write_image("debug_ship_edges5",&lines,PGM_RAW);

  /* combine ships and their sidelobes */
  max_image(&specks,&lines,prob);

  pipeline_process(prob,&temp1,1,2,histogram_mean); /* smooth a bit */
  average_images(prob,&temp1,prob);
  multiply_image_scalar255(prob,768);
  /*  if (FMI_DEBUG(3)) write_image("ships",prob,PGM_RAW); */
  
  /*  reset_image(&trace);  */
  fmi_debug(2,"remove ships2: reset temp1");
  fmi_debug(2,"--------------");
  if (FMI_DEBUG(2)) image_info(&temp1);
  fmi_debug(2,"==============");
  reset_image(&temp1);
  fmi_debug(2,"remove ships2: reset temp2");
  reset_image(&temp2);
  fmi_debug(2,"remove ships2: reset specks1");
  reset_image(&specks);
  reset_image(&virtual);
  reset_image(&lines);
}


void detect_doppler_anomaly(FmiImage *source,FmiImage *target,int width, int height,int threshold){
  /* peura hack 10.12.2002:�gaussian  - ADDS PASSIVE ARE NEAR ZERO */
  /*  histogram_threshold=threshold; */
  histogram_threshold=128;
  pipeline_process(source,target,width,height,histogram_variance_rot);
  sigmoid_image(target,threshold,threshold/16);
  /*invert_image(target); */
  if (FMI_DEBUG(4)) write_image("debug_doppler",target,PGM_RAW);
  /*pipeline_process(source,target,1,1,histogram_mean); */
}


/* NEW! */
void detect_ground_echo_minnetgrad(FmiImage *source,int ppi_count,FmiImage *prob,int gradT,int altT){
  register int i,j;
  /*register int h,l; */
  register int l;
  int grad; /*,alt_delta,alt_m; */
  /*const int altMAX=altT*3; */
  const int gradD=ABS(gradT);
  const int image_height=source[0].height;
  const int image_width=source[0].width;
  /*  int g_max,g_max_alt,g_span_max; */
  int g;         /* current gray level */
  int altD;   /* altitude difference */
  int altM;   /* altitude mean */
  /*  int p,p_min; */
  int p;  /* prob of this */
  int p_max;  /* max prob obtained */
  int pGRAD;  /* prob according to current grad */
  int pALT;   /* prob according to current (grad-halfway) altitude */
  int g_min;     /* minimum intensity obtained (before this) */
  int pGRADmax = 0; /* for debug images */

  FmiImage debug_grad_raw,debug_grad;
  FmiImage median0;
  float *altitude;
  int *bin;
  
  altitude = (float *)malloc(source->width * sizeof(float));
  bin = (int *)malloc(source->width * sizeof(int));

  

  setup_context(source);

  fmi_debug(2,"detect_ground_echo");

  canonize_image(&source[0],prob);
  fill_image(prob,0);

  canonize_image(&source[0],&median0);
  /*  histogram_sample_count=(5*3)*3/4; */
  histogram_sample_count=3;
  pipeline_process(&source[0],&median0,1,1,histogram_median2);

  if (FMI_DEBUG(5)){
    canonize_image(&source[0],&debug_grad_raw);
    fill_image(&debug_grad_raw,0);
    canonize_image(&source[0],&debug_grad);
    fill_image(&debug_grad,0);
  }

  

  if (FMI_DEBUG(3))
    fprintf(stderr," intensity_grad %.2d per 1000m, half_altitude %d \n",gradT,altT);

  /* traverse image, one cirle at the time */
  for (i=0;i<image_width;i++){
    /* save BIN INDICES and ALTITUDES to array */
    for (l=0;l<ppi_count;l++){
      bin[l]=bin_to_bin(i,source[1].elevation_angle,source[l+1].elevation_angle);
      altitude[l]=bin_to_altitude(i,source[l+1].elevation_angle);
    }
    /* traverse one circular sweep */
    for (j=0;j<image_height;j++){
      /*g_max=-1000; */
      /*      g_span_max=-1000; */
      /*g_max_alt=-1000; */
      /* */
      p_max=0;
      /*      g_min=get_pixel(&source[0],i,j,0); */
      g_min=get_pixel(&median0,i,j,0);
      if (FMI_DEBUG(5))
    	  pGRADmax=0;
      for (l=1;l<ppi_count;l++){
	if (g_min==0)
	  break;
	/* RAW INTENSITY */
	g=get_pixel(&source[l],i,j,0);
	if (g==NO_DATA)
	  break;
	altD=altitude[l]-altitude[l-1]+100; /* STABILITY TRICK */
	altM=(altitude[l]+altitude[l-1])/2;
	/*	if (altD==0)	  altD==100;  // WARNING? */

	grad = (1000*(g-g_min))/altD;


	pGRAD=128+pseudo_sigmoid(gradD,-(grad-gradT))/2;

	if (FMI_DEBUG(5))
	  if (pGRAD>pGRADmax)
	    pGRADmax=pGRAD;

	/* fuzzify by altitude (255 at 0m, 128 at half-m) */
	pALT = 128-pseudo_sigmoid(altD,altM-altT)/2;
	
	p=pGRAD*pALT/256;
	if (p<24) p=0;

	if (p>p_max) p_max=p;
	if (g<g_min) g_min=g;
      }
      if (FMI_DEBUG(5)){
	put_pixel_max(&debug_grad_raw,i,j,0,pGRADmax);
	put_pixel_max(&debug_grad,i,j,0,p_max);
	/*if (p>get_pixel(prob,i,j,0))  */
	/*  put_pixel(&debug,i,j,0,16*h+l); */
      }
      
      put_pixel_max(prob,i,j,0,p_max);
    }
  }
  if (FMI_DEBUG(5)){
    write_image("debug_ground_1grad_raw",&debug_grad_raw,PGM_RAW);
    write_image("debug_ground_2grad",&debug_grad,PGM_RAW);
    write_image("debug_ground_3grad",prob,PGM_RAW);
  }
  free(bin);
  free(altitude);
}

void detect_ground_echo_mingrad(FmiImage *source,int ppi_count,FmiImage *prob,int intensity_grad,int half_altitude){
  register int i,j;
  register int l;  /* h */
  int grad,alt_delta,alt_m;
  const int max_altitude=half_altitude*3;
  const int half_width=ABS(intensity_grad);
  const int image_height=source[0].height;
  const int image_width=source[0].width;

  int p; /*,p_min; */
  FmiImage debug_grad_raw,debug_grad;
  float *altitude;
  int *bin;
  
  altitude = (float *)malloc(source->width * sizeof(float));
  bin = (int *)malloc(source->width * sizeof(int));

  setup_context(source);

  fmi_debug(2,"detect_ground_echo");

  canonize_image(&source[0],prob);
  fill_image(prob,0);

  if (FMI_DEBUG(5)){
    canonize_image(&source[0],&debug_grad_raw);
    fill_image(&debug_grad_raw,255);
    canonize_image(&source[0],&debug_grad);
    fill_image(&debug_grad,0);
  }

  if (FMI_DEBUG(3))
    fprintf(stderr," intensity_grad %.2d per 1000m, half_altitude %d \n",intensity_grad,half_altitude);

  for (i=0;i<image_width;i++){
    for (l=0;l<ppi_count;l++){
      bin[l]=bin_to_bin(i,source[1].elevation_angle,source[l+1].elevation_angle);
      altitude[l]=bin_to_altitude(i,source[l+1].elevation_angle);
    }
    for (j=0;j<image_height;j++){
      /*
      grad_min=0;
      g_max=0;
      g_span_max=0;
      g_max_alt=0;
      */
      for (l=0;l<ppi_count-1;l++){
	
	/* RAW INTENSITY */
	alt_m=(altitude[l+1]+altitude[l])/2;

	if (alt_m>max_altitude) 
	  break;

	alt_delta=(altitude[l+1]-altitude[l]);
	grad=1000*(get_pixel(&source[l+1],i,j,0)-get_pixel(&source[l],i,j,0))/alt_delta;

	/*
	if (grad>grad_max)
	  grad_max=grad;
	if (grad+grad_max<grad_net_min)
	  grad_net_min=grad+grad_max;
	*/

	/* RAW INTENSITY GRADIENT */
	
	/* fuzzify by gradient (e.g. -10dbz / 1000m  => 128b) */

	p=128+pseudo_sigmoid(half_width,-(grad-intensity_grad));
	if (p<0)   p=0;
	if (p>255) p=255;

	if (FMI_DEBUG(5)){
	    put_pixel_min(&debug_grad_raw,i,j,0,128+grad);
	    put_pixel_max(&debug_grad,i,j,0,p);
	    /*if (p>get_pixel(prob,i,j,0))  */
	    /*  put_pixel(&debug,i,j,0,16*h+l); */
	}

	/* fuzzify by altitude (255 at 0m, 128 at half-m) */
	p=(pseudo_gauss(half_altitude,alt_m))*p/256;
	/*	  put_pixel_max(prob,i,j,0,(p+p_min)/2); */
	put_pixel_max(prob,i,j,0,p);

      }
    }
  }
  if (FMI_DEBUG(5)){
    write_image("debug_ground_1grad_raw",&debug_grad_raw,PGM_RAW);
    write_image("debug_ground_2grad",&debug_grad,PGM_RAW);
    write_image("debug_ground_3grad",prob,PGM_RAW);
  }
  free(bin);
  free(altitude);
}

void detect_too_warm(FmiImage *source,FmiImage *prob,FmiImage *meteosat,Celsius c50,Celsius c75,int min_intensity,int min_size){
  FmiImage specks;
  int b50=celsius_to_meteosatbyte(c50);
  int b75=celsius_to_meteosatbyte(c75);
  
  canonize_image(source,&specks);
  detect_specks(source,&specks,min_intensity,histogram_area);
  semisigmoid_image(&specks,min_size);
  if (FMI_DEBUG(5)) write_image("debug_warm_specks",&specks,PGM_RAW);

  copy_image(meteosat,prob);  
  sigmoid_image(prob,b50,b75-b50);
  /*  sigmoid_image(prob,b50,b50-b75); */
  if (FMI_DEBUG(5)) write_image("debug_warm_temp",prob,PGM_RAW);

  multiply_image255(prob,&specks,prob);
  if (FMI_DEBUG(4)) write_image("debug_warm",prob,PGM_RAW);
}

void remove_thin_horz_lines(FmiImage *target,int min_elevation,int weight){
  int i,j,k;
  /*int gd,gmax; */
  unsigned char g1,g2;
  int n;
  int min_length;
  float w;
  FmiImage trace;

  min_length=10;

  weight=10;
  fmi_debug(1,"remove_sun/emitter_lines");
  printf("weight=%d \n",weight);

  canonize_image(target,&trace);


  /*  detect_horz_segments(target,&trace,min_length,min_elevation); */
  /* fmi_image_filter_line.h : */
  detect_horz_line_segments(target,&trace,min_length,min_elevation); 

  /*printf("n=%d w=%f weight=%d \n",n,w,weight); */
  printf("weight=%d \n",weight);

  /*  fmi_error("remove_emitter_lines?"); */

  /*  fill_image(trace,0); */
  for (k=0;k<target->channels;k++){
    for (j=0;j<target->height;j++){

      
      n=0;
      /* cumulate horz segments */
      for (i=0;i<trace.width;i++){
	if (get_pixel(&trace,i,j,k)>0) 
	  n++;}
      
      /*      w=weight*((float)n)/(float)trace.width;*/
      w=weight*n/trace.width;

      /*      if (w>1.0) w=1.0; */
      /*      printf("n=%d w=%f weight=%f \n",n,w,weight); */
      
      /* IR filtering (strengthen lines) */
      for (i=1;i<trace.width;i++){
	g1=get_pixel(&trace,i,j,k);
	g2=w*(float)get_pixel(&trace,i-1,j,k);
	put_pixel(&trace,i,j,k,MAX(g1,g2));}
      for (i=trace.width-2;i>0;i--){
	g1=get_pixel(&trace,i,j,k);
	g2=w*(float)get_pixel(&trace,i+1,j,k);
	put_pixel(&trace,i,j,k,MAX(g1,g2));}
	
      /* DELETE / COVER */
      for (i=0;i<trace.width;i++){
	if (get_pixel(&trace,i,j,k)>2){    /* > CRITICAL_LENGTH */
	  g1=get_pixel(target,i,j-1,k);
	  g2=get_pixel(target,i,j+1,k);
	  /*	  put_pixel(target,i,j,k,(g1+g2)/2);} */
	  put_pixel(target,i,j,k,SUN);}
	  /*	  put_pixel(target,i,j,k,MIN(g1,g2));} */
	/*put_pixel(target,i,j,k,255);} */
      }

    }
  }
  write_image("lines",&trace,PGM_RAW);
}


void enhance_horz_lines2(FmiImage *trace,int weight){
  int i,j,k;
  int hrad;
  int vrad;
  int n,count;
  /*  int n1[trace->height],n2[trace->height]; */
  Histogram hist; /*,weights; */
  FmiImage trace2;
  fmi_debug(3,"enhance_horz_lines2");

  /*  for (i=0;i<256;i++) histogram_weighted_mean2_weights[i]=i+1; */
  for (i=0;i<256;i++) histogram_weights[i]=i+1;

  canonize_image(trace,&trace2);
  /*  copy_image(trace,&trace2); */
  pipeline_process(trace,&trace2,4,1,histogram_max);
  pipeline_process(&trace2,trace,4,0,histogram_mean);
  /*  threshold_image(trace,trace,8,0,histogram_max); */

  /*  histogram_median2_count=(2*hrad+1)*(2*vrad+1)*7/8; */

  if (FMI_DEBUG(4)){
    fprintf(stderr," **** weight=%d\n",weight);
    /*fprintf(stderr," count=%d\n",count); */
  }

  vrad=1;
  hrad=4;
  count=(2*hrad+1)*(2*vrad+1) * weight;

  for (k=0;k<trace->channels;k++){
    for (j=0;j<trace->height;j++){
      n=0;
      for (i=0;i<trace->width;i++)  n+=(get_pixel(trace,i,j,k)>0?1:0);
      if (n>5){
	histogram_sample_count=count* n/trace->width;
	/* RISK 2010  +null */
	initialize_histogram(trace,hist,hrad,vrad,0,j,NULL);
	for(i=0;i<trace->width;right(trace,hist,hrad,vrad,&i,&j))
	  put_pixel(&trace2,i,j,k,histogram_median2(hist));}
    }
  }
  copy_image(&trace2,trace);
}

/* enhance by */
/* height: hrad */
void enhance_vert_lines(FmiImage *trace,int weight){
  int i,j,k;
  int hrad;
  int vrad;
  int n,count;
  /* int n1[trace->height],n2[trace->height]; */
  Histogram hist; /*,weights; */
  FmiImage trace2;
  fmi_debug(3,"enhance_horz_lines2");

  /*  for (i=0;i<256;i++) histogram_weighted_mean2_weights[i]=i+1; */
  for (i=0;i<256;i++) histogram_weights[i]=i+1;

  canonize_image(trace,&trace2);
  /*  copy_image(trace,&trace2); */
  pipeline_process(trace,&trace2,4,1,histogram_max);
  pipeline_process(&trace2,trace,4,0,histogram_mean);
  /*  threshold_image(trace,trace,8,0,histogram_max); */
  /*  histogram_median2_count=(2*hrad+1)*(2*vrad+1)*7/8; */

  for (k=0;k<trace->channels;k++){
    for (i=0;i<trace->width;i++){ 
      n=0;
      for (j=0;j<trace->height;j++) n+=(get_pixel(trace,i,j,k)>0?1:0);
      /*vrad=   weight * n/trace->width; */
      /*hrad= 4*weight * n/trace->width; */
      /*    count=weight*weight*(2*hrad+1)*(2*vrad+1)*n/trace->width; */
      vrad=4;
      hrad=1;
      count=(2*hrad+1)*(2*vrad+1) * weight * n/trace->width;
      histogram_sample_count=count;
      /* RISK 2010 */
      initialize_histogram(trace,hist,hrad,vrad,0,j,NULL);
      for(j=0;j<trace->height;up(trace,hist,hrad,vrad,&i,&j))
	/*	put_pixel(&trace2,i,j,k,histogram_median(hist,count)); */
      	/*put_pixel(&trace2,i,j,k,histogram_weighted_mean2(hist)); */
	/*	put_pixel(&trace2,i,j,k,histogram_cumul_bottom(hist,count)); */
	put_pixel(&trace2,i,j,k,histogram_median2(hist));
    }
  }
  /*  invert_image(&trace2); */
  /*  pipeline_process(&trace2,trace,0,1,histogram_max); */
  copy_image(&trace2,trace);
}

/* */
void put_nonzero(FmiImage *img,int x,int y,int channel,Byte c){
  if (c>0)  img->array[channel*img->area + y*img->width + x]=c;
}

void hide_segments(FmiImage *target,FmiImage *trace){
  /*register int i,j,k; */
  /*int n,count; */
  FmiImage trace2; /*,glue; */
  /*  canonize_image(target,&glue); */

  /*
  copy_image(trace,&trace2);
  pipeline_process(&trace2,trace,0,1,histogram_max);
  if (FMI_DEBUG(2)) write_image("hide",trace,PGM_RAW);
  */

  /*subtract_image(trace,&trace2,trace); */

  /* GLUE SEPARATE DOTS with trace2 */
  pipeline_process(trace,&trace2,0,1,histogram_min);
  max_image(target,&trace2,&trace2);  
  if (FMI_DEBUG(2)) write_image("glue",&trace2,PGM_RAW);
  /*  copy_image(target,&trace2); */

  /*  return;  */

  /*  propagate_up(&trace2,trace,target,8,put_pixel_max); */
  /*  propagate_down(&trace2,trace,target,8,put_pixel_max); */
  propagate_up(&trace2,trace,target,0,put_nonzero);
  /*  propagate_up(&trace2,trace,target,0,put_pixel_min); */
  propagate_down(&trace2,trace,target,0,put_pixel_max);
  reset_image(&trace2);
}

/* first full, then slope/256 % at max distance  */
void distance_compensation_die(FmiImage *image,int slope){
  register int i,j,k,l;
  for (k=0;k<image->channels;k++)
    for (i=0;i<image->width;i++){
      l=((image->width-i*slope/256)/image->width);
      for (j=0;j<image->height;j++)
	put_pixel(image,i,j,k,get_pixel(image,i,j,k)*l);
    }
}

/* first mul by 1, then by coeff at max distance  */
void distance_compensation_mul(FmiImage *image,int coeff){
  register int i,j,k,l,m;
  for (k=0;k<image->channels;k++)
    for (i=0;i<image->width;i++){
      l=image->width+(coeff-1)*i;
      for (j=0;j<image->height;j++){
	m=get_pixel(image,i,j,k)*l/image->width;
	if (m>DATA_MAX) m=DATA_MAX;
	put_pixel(image,i,j,k,m);
      }
    }
}

/* div by [1...slope] (at origin... at max distance) */
/* NOTICE! ROUNDS 0.001 => 1 */
void distance_compensation_div(FmiImage *image,int slope){
  register int i,j,k;
  int l,m;
  for (k=0;k<image->channels;k++)
    for (i=0;i<image->width;i++){
      /*      l=(image->width-i)/image->width; */
      l=1024*slope*i/image->width+1024;
      for (j=0;j<image->height;j++){
	m=get_pixel(image,i,j,k);
	put_pixel(image,i,j,k,m*1024/l+(m>0));
      }
    }
}

/*
void tyrannize(FmiImage *image,int threshold_intensity,int max_size,int max_intensity){
  
}
*/

void remove_horz_lines(FmiImage *target,int min_length,int min_elevation,int weight){
	/*
  int i,j,k;
  int gd,gmax;
  unsigned char g1,g2;
  int n;
  float w;
	 */
  FmiImage trace,trace2,trace3;
  /*  const int   up_edges[1][3]={0,1,-1}; */
  /* const int down_edges[1][3]={-1,1,0}; */
  canonize_image(target,&trace);
  canonize_image(target,&trace2);
  canonize_image(target,&trace3);

  fmi_debug(1,"remove_emitters2");
  if (FMI_DEBUG(2)) printf(" min_elevation=%d\n",min_elevation);
  detect_horz_edges(target,&trace);
  distance_compensation_die(&trace,50);
  threshold_image(&trace,&trace,min_elevation);

  /* COMPUTE "HORIZONTAL" LENGTHS OF SEGMENTS */
  fill_image(&trace2,0);
  propagate_right(NULL,&trace,&trace2,+1,put_pixel);
  limit_image_intensities(&trace2,0,32);
  fill_image(&trace,0);
  propagate_left(&trace2,&trace2,&trace,0,put_pixel);
  if (FMI_DEBUG(2)) printf(" min_length=%d\n",min_length);
  threshold_image(&trace,&trace,min_length);
  if (FMI_DEBUG(2)) write_image("debug_edges",&trace,PGM_RAW);
  /* AT THIS POINT, WE HAVE SEGMENTS LONGER THAN n, MARKED WITH n+ */

  /* COMPUTE VERTICAL WIDTH BY A WAVE FORTH AND BACK, IN BOTH DIRS */
  fill_image(&trace2,0);
  propagate_up(&trace,target,&trace2,-8,put_pixel);
  propagate_down(&trace2,target,&trace2,0,put_pixel_min);

  fill_image(&trace3,0);
  propagate_down(&trace,target,&trace3,-8,put_pixel);
  propagate_up(&trace3,target,&trace3,0,put_pixel_min);

  max_image(&trace2,&trace3,&trace3);
  if (FMI_DEBUG(2)) write_image("debug_edges2",&trace3,PGM_RAW);
  threshold_image(&trace3,&trace3,min_length/2);
  /* AT THIS POINT, TOO THICK SEGMENTS HAVE BEEN PRUNED */

  /*  propagate_left(&trace,&trace,&trace,0,put_pixel); */
  enhance_horz_lines2(&trace3,weight);
  if (FMI_DEBUG(2)) write_image("debug_lines",&trace3,PGM_RAW);

  invert_image(&trace3);
  mask_image(target,&trace3,255,EMITTER);
  /* hide_segments(target,&trace3); */
  /*  if (FMI_DEBUG(2)) write_image("edges2",&trace,PGM_RAW); */

  /*  detect_horz_edge_segments(target,&trace,min_length,min_elevation); */
  /*  convolve(target,&trace,1,3,up_edges,2); */
  reset_image(&trace);
  reset_image(&trace2);
  reset_image(&trace3);
}

/* start_intensity = threshold at the origin */
/* radius          = range of 50% effect */
/* weight          = steepness, 1=steep, 100=soft */
/*void detect_insect_band(FmiImage *source,FmiImage *prob,int start_intensity,int radius,int slope){ */
void detect_insect_band(FmiImage *source,FmiImage *prob,int start_intensity,int radius,int slope){
  register int i,j,k;
  int threshold,temp;
  canonize_image(source,prob);
  for (k=0;k<source->channels;k++)
    for (i=0;i<source->width;i++){
      temp=i-radius;
      threshold=start_intensity*(256-pseudo_sigmoid(slope,temp))/512+1;
      /*      printf("%d\t",threshold); */
      for (j=0;j<source->height;j++){
	temp=threshold-get_pixel(source,i,j,k);
	if (temp<=-threshold) temp=-threshold;
	/*	temp=MAX(temp,-threshold); */
	if (get_pixel(source,i,j,k)>DATA_MIN)
	  put_pixel(prob,i,j,k,128+127*temp/threshold);
	else
	  put_pixel(prob,i,j,k,0);
      }
      /*put_pixel(prob,i,j,k,threshold); */
    }
}

void detect_biomet(FmiImage *source,FmiImage *prob,int intensity_max,int intensity_delta,int altitude_max,int altitude_delta){
  register int i,j,k;
  int f,h;
  
  setup_context(source);
  /*int threshold,temp; */
  canonize_image(source,prob);
  for (k=0;k<source->channels;k++)
    for (i=0;i<source->width;i++){

      h=bin_to_altitude(i,source[k+1].elevation_angle);
      /*printf("bin %d[%d]: %d metres",i,k,h); */
      h=(pseudo_sigmoid(altitude_delta,altitude_max-h)+255)/2;
      /*      printf(", prob=%d\n",h); */
      /*printf("bin %d[%d]: %d\n",i,k,h); */
      for (j=0;j<source->height;j++){
        /*      for (j=0;j<2;j++){ */
        f = get_pixel(source,i,j,k);
        if ((f==0)||(f==NO_DATA)){
          f=240;  /* ? */
          put_pixel(prob,i,j,k,0);
        }
        else {
          f=(pseudo_sigmoid(intensity_delta,intensity_max-f)+255)/2;
          put_pixel(prob,i,j,k,f*h/255);
        }
        /*	printf("bin %d[%d]: prob=%d \n",i,k,h); */
        /*	put_pixel(prob,i,j,k,f); */
        /*	put_pixel(prob,i,j,k,129); */
      }
      /*put_pixel(prob,i,j,k,threshold); */
    }
}

void pgm_to_ppm_radar(FmiImage *source,FmiImage *target){
  register int i;
  /*  Byte i,r,g,b; */
  ColorMap256 map;

  for (i=0;i<256;i++){
    /*    colorcode(i,&r,&g,&b); */
    map[i][0]=i;
    map[i][1]=i;
    map[i][2]=i;}

  #include "fmi_radar_codes.inc"

  copy_image_properties(source,target);
  target->channels=3;
  initialize_image(target);
  map_channel_to_256_colors(source,0,target,map);
}

/*  */
void pgm_to_pgm_print(FmiImage *source,FmiImage *target){
  register int i;
  int g;
  canonize_image(source,target);
  for (i=0;i<source->volume;i++){
    g = source->array[i];
    if (g>0)
      g = ((g-64)/16)*32+96;
      /*  g = ((g-64)/16)*32+128; */
    if (g>255) g=255;
    target->array[i] =255-g;
  }
}

/* INVERTED */
void pgm_to_pgm_print2(FmiImage *source,FmiImage *target){
  register int i;
  int g;
  canonize_image(source,target);
  for (i=0;i<source->volume;i++){
    g = source->array[i];
    if (g>0){
      g = ((g-64)/16)*32+64;
      /* g = ((g-64)/8)*32+64; */
      if (g>=255) g=254;
      g=255-g;
    }
    target->array[i] =255-g;
  }
}

void pgm_to_ppm_radar_iris(FmiImage *source,FmiImage *target){
  register int i;
  int temp;
  ColorMap256 map;

  for (i=0;i<DATA_MIN;i++){
    map[i][0]=0;
    map[i][1]=0;
    map[i][2]=0;}

  /*for (i=0;i<16;i++) */
  /*fprintf(stderr,"map[%3d]= %3d,%3d,%3d\n",i,map[i][0],map[i][1],map[i][2]); */


  for (i=DATA_MIN;i<256;i++){
    temp=128+pseudo_sigmoid(16,i-112)/2;
    map[i][0]=MIN(temp,255);
    temp=pseudo_gauss(32,i-92)+64+pseudo_sigmoid(16,i-144)/4;
    map[i][1]=MIN(temp,255);
    temp=pseudo_gauss(24,i-64)*2/3+128+pseudo_sigmoid(32,i-176)/2;
    map[i][2]=MIN(temp,255);
  }

  map[255][0]=255;
  map[255][1]=255;
  map[255][2]=255;

  copy_image_properties(source,target);
  target->channels=3;
  initialize_image(target);
  map_channel_to_256_colors(source,0,target,map);
  /*for (i=0;i<16;i++) */
  /*fprintf(stderr,"map[%3d]= %3d,%3d,%3d\n",i,map[i][0],map[i][1],map[i][2]); */

}

void pgm_to_redgreen(FmiImage *source,FmiImage *target){
  register int i;
  ColorMap256 map;

  for (i=0;i<255;i++){
    map[0][0]= (i>128) ? pseudo_sigmoid(64,128) : 0;
    map[0][1]= (i<128) ? 255-pseudo_sigmoid(64,128) : 0;
    map[0][2]=0;}

  copy_image_properties(source,target);
  target->channels=3;
  initialize_image(target);
  map_channel_to_256_colors(source,0,target,map);
}



/* SAMPLING */

/* sample images */
void virtual_rhi(FmiImage *volume){
  int i,j=0,k,j_start=0,j_end;
  FmiImage target;
  setup_context(volume);
  copy_image_properties(volume,&target);
  target.channels=1;
  initialize_image(&target);
  fill_image(&target,0);
  for (i=0;i<volume->width;i++){
    j_start=0;
    for (k=0;j<volume->channels;k++){
      j_end=bin_to_bin(i,0,volume[k+1].elevation_angle);
      for (j=j_start;j<j_end;j++)
	put_pixel(&target,i,j,0,5);
      j_start=j_end;}
  }
}

void gradient_rgb(FmiImage *volume){
  FmiImage target[1+3],cart;
  const int sweeps=volume->height/360;

  fprintf(stderr,"gradient_rgb: %d sweeps\n",sweeps);

  if (sweeps<4)
    fmi_error("gradient_rgb: needs at least 4 sweeps");

  copy_image_properties(&volume[1],target);
  target->channels=3;
  initialize_image(target);
  /* channels_to_link_array(target,&target[1]); */
  split_to_link_array(target,3,&target[1]);

  subtract_image128(&volume[2],&volume[1],&target[1]);
  /*  fill_image(&target[2],128); */
  subtract_image128(&volume[3],&volume[1],&target[2]);
  subtract_image128(&volume[4],&volume[1],&target[3]);

  /*  split_to_channels(target,3); */
  canonize_image(target,&cart);
  to_cart(target,&cart,NO_DATA);

  write_image("Grad",target,PPM_RAW);
  write_image("Gradc",&cart,PPM_RAW);
}


/* pointwise images */

int histogram_dominate_anaprop(Histogram h){
  register int i,i_max;
  i_max=1;
  for (i=1;i<DATA_MIN;i++)
    if (h[i]>h[i_max])
      i_max=i;
  if (h[i_max]==0)
    return 0;
  else
    return i_max;
}

