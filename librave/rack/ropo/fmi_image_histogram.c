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
#include <math.h> /* sqrt() */
#include "fmi_util.h"
#include "fmi_image.h"
#include "fmi_image_filter.h"
#include "fmi_image_histogram.h"

/*FmiImage *histogram_weight_image=NULL; */
/*histogram_weight_image=NULL; */
int histogram_sample_count=0;
int (* histogram_scaling_function)(int param, int value)=NULL;
int histogram_scaling_parameter=128;

int histogram_semisigmoid(int a,int x){ /* typical histogram_scaling_function */
  return  (255*(x)/(a+x));
}

int histogram_semisigmoid_inv(int a,int x){ /* typical histogram_scaling_function */
  return  (255-255*(x)/(a+x));
}



void clear_histogram_full(Histogram hist){
  register int i;
  for (i=0;i<HISTOGRAM_SIZE;i++)  hist[i]=0;
  hist[HIST_MIN]=255;
}



/*void set_histogram_sample_count(int width,int height); */

/* WINDOW (or template or mask)  - BASED FILTERING */

void convolve(FmiImage *source,FmiImage *target,int mask_width,int mask_height,int **mask,int divisor){
  register int  i,j; /*,k,s,t; */

  fmi_debug(2,"convolve");
  /*  printf("\t hey ,%d) =",divisor); */
  /*fflush(stdout); */
  /*  printf("\t(%d,%d) =",mask_width,mask_height); */
  for (j=0;j<mask_height;j++){
    for (i=0;i<mask_width;i++){
      printf("\t(%d,%d) =",i,j);
      fflush(stdout);
      printf(" %d",mask[j][i]);
      fflush(stdout);
      printf("\n");}
  }
  /*fmi_debug(2,"convolve"); */
  if (FMI_DEBUG(2)) write_image("conv",target,PGM_RAW);
}


/* --------------------------------------- */

int histogram_sum(Histogram h){
  register int  i;
  int sum;
  sum=0;
  for (i=0;i<256;i++)
    sum+=h[i];
  return sum;
}


int histogram_median_biased(Histogram h,int count){
  register int  i;
  int sum;
  sum=0;
  /*  sum2=histogram_sum(h)/2; */
  for (i=0;i<256;i++){
    sum+=h[i];
    if (sum>=count) 
      return i;}
  return 255;
}

int histogram_median_biased_top(Histogram h,int count){
  register int  i;
  int sum;
  sum=0;
  for (i=255;i>=0;i--){
    sum+=h[i];
    if (sum>=count) return i;}
  return 0;
}

/* --------------------------------------- */

int histogram_size(Histogram h){
  return h[HIST_SIZE];
}

int histogram_area(Histogram h){
  return ABS(h[HIST_AREA]);
}

int histogram_area_inv255(Histogram h){
  return 255-ABS(h[HIST_AREA]);
}

int histogram_area2(Histogram h){
  return pseudo_sigmoid(255,ABS(h[HIST_AREA]));
}

int histogram_area2_inv255(Histogram h){
  return 255-pseudo_sigmoid(1,ABS(h[HIST_AREA]));
}

int histogram_perimeter(Histogram h){
  return h[HIST_PERIMx3];
}

int histogram_perimeter2(Histogram h){
  return  pseudo_sigmoid(255,h[HIST_PERIMx3]);
}

/*
int histogram_perimeter_normalized(Histogram h){
return h[HIST_PERIMx3]/3;
}
*/

int histogram_compactness(Histogram h){
  /* (A=�r�, P=2�r maxA=�(P/2�)�=P/4� */
  /* 255*4� = 3204  "theoretical coeff" */
  /* * circle aliasing coeff sqrt2 = 4500 */
  return (4500*ABS(h[HIST_AREA])/(h[HIST_PERIMx3]*h[HIST_PERIMx3]+1));
}

int histogram_min(Histogram h){
  return histogram_median_biased(h,1);
}

int histogram_max(Histogram h){
  return histogram_median_biased_top(h,1);
}

int histogram_range(Histogram h){
  return (histogram_median_biased_top(h,1)-histogram_median_biased(h,1));
}


int histogram_median(Histogram h){ /* computationally heavy */
  register int  i;
  int sum,count;
  sum=0;
  count=histogram_sum(h)/2;
  for (i=0;i<256;i++){
    sum+=h[i];
    if (sum>=count) 
      return i;}
  return 255;
}

/*histogram_median2_count=0; */
void histogram_median2_reset(Histogram h){  /*stupid? */
  histogram_sample_count=histogram_sum(h);
}

/* use this instead, precalculate COUNT */
int histogram_median2(Histogram h){ 
  register int  i;
  int sum;
  sum=0;
  for (i=0;i<256;i++){
    sum+=h[i];
    if (sum>=histogram_sample_count) 
      return i;}
  return 255;
}

/* use this instead, precalculate sample COUNT */
int histogram_median2_top(Histogram h){ 
  register int  i;
  int sum;
  sum=0;
  for (i=255;i>=0;i--){
    sum+=h[i];
    if (sum>=histogram_sample_count) 
      return i;}
  return 0;
}

int histogram_mean(Histogram h){
  register int  i;
  int sum,s;
  sum=0;
  s=0;
  for (i=0;i<256;i++){
    s+=h[i];
    sum+=h[i]*i;}
  return (sum/s);
}

int histogram_mean_nonzero(Histogram h){
  register int  i;
  int sum,s;
  sum=0;
  s=0;
  /* i=1,2,... */
  for (i=1;i<256;i++){
    s+=h[i];
    sum+=h[i]*i;}
  if (s>0)
    return (sum/s);
  else
    return 0;
}

int histogram_mean2(Histogram h){
  register int  i;
  int sum;
  sum=0;
  /*  s=0; */
  for (i=0;i<256;i++){
    /*    s+=h[i]; */
    sum+=h[i]*i;}
  return (sum/histogram_sample_count);
}

int histogram_mean_weighted(Histogram h){
  register int  i;
  int sum,s;
  sum=0;
  s=h[HIST_SIZE];
  if (s<1) s=1;
  for (i=0;i<256;i++){
    sum+=h[i]*i;}
    /*    s+=histogram_weights[i]*h[i]; */
    /*sum+=histogram_weights[i]*h[i]*i;} */
  return (sum/s);
}

int (* histogram_mean_weighted_pyramid)(Histogram h) = histogram_mean_weighted;

int histogram_variance_rot(Histogram h){
  register int  i,n;
  int x,y,sum_x,sum2_x,sum_y,sum2_y;
  int N;
  N=1;
  sum_x  = 0;
  sum2_x = 0;
  sum_y  = 0;
  sum2_y = 0;

  /* NOTE k=1,... */
  for (i=1;i<256;i++){
    n=h[i];
    N += n;
    /*w=n; */
    /*w=(float)n; */
    /*    x=cos(((float)i)*2.0*PI/255.0); */
    /* y=sin(((float)i)*2.0*PI/255.0); */
    x=histogram_cosine[i]-128;
    y=histogram_sine[i]-128;
    sum_x  += n*x;
    sum2_x += n*x*x;
    sum_y  += n*y;
    sum2_y += n*y*y;
  }

  return pseudo_sigmoid (histogram_threshold,(sum2_x-sum_x*sum_x/N + sum2_y-sum_y*sum_y/N)/N/128);
  /*return pseudo_sigmoid(128.0,255*(sum2_x+sum2_y)); */
}

/*
int histogram_weighted_mean(Histogram h,Histogram weights){
  register int  i;
  int sum,s;
  sum=0;
  s=0;
  for (i=0;i<256;i++){
    s+=weights[i]*h[i];
    sum+=weights[i]*h[i]*i;}
  return (sum/s);
}
*/

/*
int histogram_weighted_mean2(Histogram h){
  register int  i;
  int sum,s;
  sum=0;
  s=0;
  for (i=0;i<256;i++){
    s+=histogram_weights[i]*h[i];
    sum+=histogram_weights[i]*h[i]*i;}
  return (sum/s);
}
*/

int histogram_dom(Histogram h){
  register int  i,i_max;
  i_max=0;
  for (i=0;i<256;i++)
    if (h[i]>h[i_max])
      i_max=i;
  return i_max;
}

int histogram_dom_nonzero(Histogram h){
  register int  i,i_max;
  i_max=1;
  for (i=1;i<256;i++)
    if (h[i]>h[i_max])
      i_max=i;
  if (h[i_max]==0)
    return 0;
  else
    return i_max;
}

int histogram_meanX(Histogram h){
  if (h[HIST_SIZE]>0) 
    return h[HIST_SUM_I]/h[HIST_SIZE];
  else
    return -1;
}

int histogram_meanY(Histogram h){
  if (h[HIST_SIZE]>0) 
    return h[HIST_SUM_I]/h[HIST_SIZE];
  else
    return -1;
}

int histogram_principal_component_ratio(Histogram h){
  /* LONG INT was not enough! */
  double x,y,xx,xy,yy,n;
  double Cxx,Cxy,Cyy,SQRT,ans;
  x= h[HIST_SUM_I];
  y= h[HIST_SUM_J];
  xx=h[HIST_SUM_II];
  xy=h[HIST_SUM_IJ];
  yy=h[HIST_SUM_JJ];
  /*  A=h[HIST_AREA]; */
  /*  n=h[HIST_PERIMx3]; */
  n=h[HIST_SIZE];
  
  /*  histogram_dump_stats(h); return 127; */

  if (n==0) return 255;
  Cxx=(xx-x*x/n)/n;
  Cyy=(yy-y*y/n)/n;
  Cxy=(xy-x*y/n)/n;
  /*
    Cxy=(h[HIST_SUM_IJ]-h[HIST_SUM_I]*h[SUM_Y]/A)/A;
    Cyy=(h[SUM_YY]-h[SUM_Y]*h[SUM_Y]/A)/A;
  */
  /*  SQR=((Cxx+Cyy)*(Cxx+Cyy)-4*(Cxx*Cyy-Cxy*Cxy)); */
  SQRT=sqrt((Cxx-Cyy)*(Cxx-Cyy)+4*Cxy*Cxy);
  /*  SQRT=sqrt((double)((Cxx+Cyy)*(Cxx+Cyy)-4*(Cxx*Cyy-Cxy*Cxy))) */
  /*  SQRT=sqrt((double)SQR); */
  ans=255.0*(Cxx+Cyy-SQRT)/(Cxx+Cyy+SQRT);
  /*ans=sqrt(Cxx/(Cyy)); */
  if ((Cxx+Cyy)==0) 
    return 255;
  else
    return /*255*sqrt((Cxx+Cyy-SQRT)/(Cxx+Cyy+SQRT)); */
      ans;
  /*return 16*d/(a+1); */
  /*return 255*D; */
}

int histogram_smoothness(Histogram h){
  int temp,temp2;
  temp=histogram_compactness(h);
  temp2=histogram_principal_component_ratio(h);
  if (temp>255) temp=255;
  if (temp2<1) temp2=1;
  return sqrt(255*temp*temp/temp2);
}

/* HISTOGRAM WINDOW - BASED FILTERING */
/* ==================================================================== */


void histogram_dump_nonzero(Histogram h){
  register int  i;
  /*  int sum,s; */
  /*sum=0; */
  /*s=0; */
  /*  for (i=0;i<HISTOGRAM_SIZE;i++) */
  for (i=0;i<HISTOGRAM_SIZE;i++)
    if (h[i]!=0)
      fprintf(stderr,"histogram[%d]=%d\n",i,(int)h[i]);
}


void histogram_dump(Histogram h){
  register int  i;
  int sum,s;
  sum=0;
  s=0;
 for (i=0;i<256;i++){
   /*   fprintf(stderr," %d \t %d\t",i,h[i]); */
    s+=h[i];
    sum+=h[i]*i;}
  fprintf(stderr,"\n sum=%d \t hits=%d \t mean=%d \n",sum,s,sum/s);
}

void histogram_dump_stats(Histogram h){
  char *format="%d\t%d %s\n";
  printf(format,h[HIST_SIZE],HIST_SIZE,"HIST_SIZE");
  printf(format,h[HIST_SUM],HIST_SUM,"HIST_SUM");
  printf(format,h[HIST_MIN],HIST_MIN,"HIST_MIN");
  printf(format,h[HIST_AREA],HIST_AREA,"HIST_AREA");
  printf(format,h[HIST_MAX],HIST_MAX,"HIST_MAX");
  printf(format,h[HIST_PERIMx3],HIST_PERIMx3,"HIST_PERIMx3");
  printf(format,h[HIST_SUM_I],HIST_SUM_I,"HIST_SUM_I");
  printf(format,h[HIST_SUM_J],HIST_SUM_J,"HIST_SUM_J");
  printf(format,h[HIST_SUM_II],HIST_SUM_II,"HIST_SUM_II");
  printf(format,h[HIST_SUM_JJ],HIST_SUM_JJ,"HIST_SUM_JJ");
  printf(format,h[HIST_SUM_IJ],HIST_SUM_IJ,"HIST_SUM_IJ");
  printf(format,h[HIST_MIN_I],HIST_MIN_I,"HIST_MIN_I");
  printf(format,h[HIST_MIN_J],HIST_MIN_J,"HIST_MIN_J");
  printf(format,h[HIST_MAX_I],HIST_MAX_I,"HIST_MAX_I");
  printf(format,h[HIST_MAX_J],HIST_MAX_J,"HIST_MAX_J");
}


void initialize_histogram(FmiImage *source,Histogram histogram,int hrad,int vrad,int i,int j,int (* hist_func)(Histogram)){
  int k,m,n,w;
  float alpha;

  clear_histogram(histogram);   /*full? */

  /* EXCLUSIVE INITS */
  if (hist_func==histogram_mean_weighted){
    fmi_debug(2,"initialize_histogram: histogram_mean_weighted, source:");
    image_info(source);
    fmi_debug(2,"initialize_histogram: histogram_mean_weighted, weight:");
    /* canonize_images(source,histogram_weight_image); */
    image_info(histogram_weight_image);
    histogram[HIST_SIZE]=0;
    for (k=0;k<source->channels;k++)
      for (m=-hrad;m<=hrad;m++) 
	for (n=-vrad;n<=vrad;n++){
	  w=get_pixel(histogram_weight_image,i+m,j+n,k);
	  histogram[get_pixel(source,i+m,j+n,k)]+=w;
	  histogram[HIST_SIZE]+=w;
	}
  }
  else {
    for (k=0;k<source->channels;k++)
      for (m=-hrad;m<=hrad;m++) 
	for (n=-vrad;n<=vrad;n++) 
	  ++histogram[get_pixel(source,i+m,j+n,k)];
  } 

  /* quick add - check if ok? */
  if (histogram_sample_count==0)
    histogram_sample_count=(2*hrad+1)*(2*vrad+1)/2;

  /* ADDED INITS */
  if (hist_func==histogram_variance_rot)
    for (i=0;i<256;i++){
      alpha=((float)i)/255*2.0*PI;
      histogram_cosine[i]=128+127*cos(alpha);
      histogram_sine[i]  =128+127*sin(alpha);
    }
  fmi_debug(2,"initialize_histogram");
} 


/*
void initialize_histogram_trigon(){
  int i;
  float alpha;
  for (i=0;i<256;i++){
    alpha=((float)i)/255*2.0*PI;
    histogram_cosine[i]=128+127*cos(alpha);
    histogram_sine[i]  =128+127*sin(alpha);
  }
}
*/

void (* histogram_window_up)(FmiImage    *,Histogram,int,int,int *,int *);
void (* histogram_window_down)(FmiImage  *,Histogram,int,int,int *,int *);
void (* histogram_window_right)(FmiImage *,Histogram,int,int,int *,int *);
void (* histogram_window_left)(FmiImage  *,Histogram,int,int,int *,int *);
/*left(FmiImage *source,Histogram histogram,int hrad,int vrad,int *i,int *j) */


void up(FmiImage *source,Histogram histogram,int hrad,int vrad,int *i,int *j){
  register int  m,k;
  k=0;
    for (m=-hrad;m<=hrad;m++){
      --histogram[get_pixel(source,(*i)+m,(*j)-vrad  ,k)];
      ++histogram[get_pixel(source,(*i)+m,(*j)+vrad+1,k)];}
  (*j)++;
}
void down(FmiImage *source,Histogram histogram,int hrad,int vrad,int *i,int *j){
  register int  m,k;
  k=0;
    for (m=-hrad;m<=hrad;m++){
      --histogram[get_pixel(source,(*i)+m,(*j)+vrad  ,k)];
      ++histogram[get_pixel(source,(*i)+m,(*j)-vrad-1,k)];}
  (*j)--;
}

void right(FmiImage *source,Histogram histogram,int hrad,int vrad,int *i,int *j){
  register int  n,k;
  k=0;
    for (n=-vrad;n<=vrad;n++){
      --histogram[get_pixel(source,*i-hrad  ,*j+n,k)];
      ++histogram[get_pixel(source,*i+hrad+1,*j+n,k)];}
  (*i)++;
}

void left(FmiImage *source,Histogram histogram,int hrad,int vrad,int *i,int *j){
  register int  n,k;
  /*  const int height=vrad*2+1; */
  k=0;
  /*   for (k=0;k<source->channels;k++) */
    for (n=-vrad;n<=vrad;n++){
      --histogram[get_pixel(source,*i+hrad  ,*j+n,k)];
      ++histogram[get_pixel(source,*i-hrad-1,*j+n,k)];}
  (*i)--;
}



void up_w(FmiImage *source,Histogram histogram,int hrad,int vrad,int *i,int *j){
  register int  m,w;
  register int  ii;
  register int  jo=*j-vrad;
  register int  jn=*j+vrad+1;
  for (m=-hrad;m<=hrad;m++){
    ii=*i+m;
    w=get_pixel(histogram_weight_image,ii,jo,0);
    histogram[get_pixel(source,ii,jo,0)] -= w;
    histogram[HIST_SIZE] -= w;
    w=get_pixel(histogram_weight_image,ii,jn,0);
    histogram[get_pixel(source,ii,jn,0)] += w;
    histogram[HIST_SIZE] += w;
  }
  (*j)++;
}
void down_w(FmiImage *source,Histogram histogram,int hrad,int vrad,int *i,int *j){
  register int  m,w;
  register int  ii;
  register int  jo=*j+vrad;
  register int  jn=*j-vrad-1;
  /*  k=0; */
  for (m=-hrad;m<=hrad;m++){
    ii=*i+m;
    w=get_pixel(histogram_weight_image,ii,jo,0);
    histogram[get_pixel(source,ii,jo,0)] -= w;
    histogram[HIST_SIZE] -= w;
    w=get_pixel(histogram_weight_image,ii,jn,0);
    histogram[get_pixel(source,ii,jn,0)] += w;
    histogram[HIST_SIZE] += w;
      }
  (*j)--;
}

void right_w(FmiImage *source,Histogram histogram,int hrad,int vrad,int *i,int *j){
  register int  n,w;
  register int  jj;
  register int  io=*i-hrad;
  register int  in=*i+hrad+1;
  for (n=-vrad;n<=vrad;n++){
    jj=*j+n;
    w=get_pixel(histogram_weight_image,io,jj,0);
    histogram[get_pixel(source,io,jj,0)] -= w;
    histogram[HIST_SIZE] -= w;
    w=get_pixel(histogram_weight_image,in,jj,0);
    histogram[get_pixel(source,in,jj,0)] += w;
    histogram[HIST_SIZE] += w;
  }
  (*i)++;
}

void left_w(FmiImage *source,Histogram histogram,int hrad,int vrad,int *i,int *j){
  register int  n,w;
  register int  jj;
  register int  io=*i+hrad;
  register int  in=*i-hrad-1;
  for (n=-vrad;n<=vrad;n++){
    jj=*j+n;
    w = get_pixel(histogram_weight_image,io,jj,0);
    histogram[w * get_pixel(source,io,jj,0)] -= w;
    histogram[HIST_SIZE] -= w;
    w = get_pixel(histogram_weight_image,in,jj,0);
    histogram[get_pixel(source,in,jj,0)] += w;
    histogram[HIST_SIZE] += w;
  }
  (*i)--;
}

void pipeline_process_col_major(FmiImage *source,FmiImage *target,int hrad,int vrad,int (* histogram_function)(Histogram),Histogram histogram){
  int i,j,k;
  i=j=k=0;

  fmi_debug(4,"pipeline_process_col_major");
  /* MAIN LOOP */
  while (1){

    /* UP */
    while (j<source->height-1){
      /*      if (0){	fprintf(stderr," - sum=i%d\t j=%d\n",i,j);} */
      histogram_window_up(source,histogram,hrad,vrad,&i,&j);
      put_pixel(target,i,j,k,histogram_function(histogram));
      if (0){	fprintf(stderr," - sum=i%d\t j=%d\n",i,j);}
      /*dump_histogram(histogram);  */
    }

    /* ONE STEP RIGHT */
    if (i<source->width-1){
      histogram_window_right(source,histogram,hrad,vrad,&i,&j);
      put_pixel(target,i,j,k,histogram_function(histogram));}
    else
      return;

    /* DOWN */
    while (j>0){
      histogram_window_down(source,histogram,hrad,vrad,&i,&j);
      put_pixel(target,i,j,k,histogram_function(histogram));
    }

    /* ONE STEP RIGHT */
    if (i<source->width-1){
      histogram_window_right(source,histogram,hrad,vrad,&i,&j);
      put_pixel(target,i,j,k,histogram_function(histogram));
    }
    else
      return;
  }
}

void pipeline_process_row_major(FmiImage *source,FmiImage *target,int hrad,int vrad,int (* histogram_function)(Histogram),Histogram histogram){
  int i,j,k;
  i=j=k=0;
  fmi_debug(4,"pipeline_process_row_major");

  /* MAIN LOOP */
  while (1){

    /* RIGHT */
    while (i<source->width-1){
      histogram_window_right(source,histogram,hrad,vrad,&i,&j);
      put_pixel(target,i,j,k,histogram_function(histogram));
    }

    /* ONE STEP UP */
    if (j<source->height-1){
      histogram_window_up(source,histogram,hrad,vrad,&i,&j);
      put_pixel(target,i,j,k,histogram_function(histogram));}
    else
      return;

    /* LEFT */
    while (i>0){
      histogram_window_left(source,histogram,hrad,vrad,&i,&j);
      put_pixel(target,i,j,k,histogram_function(histogram));
    }

    /* ONE STEP UP */
    if (j<source->height-1){
      histogram_window_up(source,histogram,hrad,vrad,&i,&j);
      put_pixel(target,i,j,k,histogram_function(histogram));}
    else
      return;
  }
}


void pipeline_process(FmiImage *source,FmiImage *target,int hrad,int vrad,int (* histogram_function)(Histogram)){
  /*int i,j; */
  /*  register int  k,m,n; */
  /*  FmiImage *target_ptr; */
  Histogram histogram;
  int width, height; /*,count; */

  /* INITIALIZE */
  width=hrad*2+1;
  height=vrad*2+1;
  /*  histogram_median2_count=(width*height+1)/2; */

  /*
  if (histogram_function==histogram_mean_weighted_pyramid){ 
  fmi_debug(1,"pipeline_process: pyramid");
  for (i=0;i<128;i++)  histogram_weights[i]=i+1;
  for (i=0;i<128;i++)  histogram_weights[128+i]=128-i;
  }
  */

  histogram_window_up    = up;
  histogram_window_down  = down;
  histogram_window_right = right;
  histogram_window_left  = left;

  if (histogram_function==histogram_mean_weighted){ 
    fmi_debug(2,"pipeline_process: histogram_mean_weighted");
    if (histogram_weight_image==NULL)
      fmi_error("pipeline_process: histogram_weight_image==NULL");
    histogram_window_up    = up_w;
    histogram_window_down  = down_w;
    histogram_window_right = right_w;
    histogram_window_left  = left_w;
  }

  fmi_debug(4,"pipeline_process");
  if (FMI_DEBUG(4))
    printf(" width=%d\t height=%d\n",width,height);

  /* INITIALIZE */
  initialize_histogram(source,histogram,hrad,vrad,0,0,histogram_function);
  /*  dump_histogram(histogram); */
      
  /*  i=j=k=0; */
  put_pixel(target,0,0,0,histogram_function(histogram));

  if (histogram_function==histogram_mean_weighted){ 
    fmi_debug(2,"pipeline_process: histogram_mean_weighted");
  }

  if (width>height)
    pipeline_process_row_major(source,target,hrad,vrad,histogram_function,histogram);
  else
  /*  printf(" width=%d\t height=%d\n",width,height); */
    pipeline_process_col_major(source,target,hrad,vrad,histogram_function,histogram);
}
