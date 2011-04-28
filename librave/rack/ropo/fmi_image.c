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
//#include <stdio.h>
#include <math.h>
#include "fmi_image.h"
#include "fmi_util.h"


/*img->array=NULL;*/

char FmiImageFormatExtension[7][4]={
  "xxx",
  "pbm",
  "pgm",
  "ppm",
  "pbm",
  "pgm",
  "ppm"
};

int FMI_IMAGE_COMMENT=YES;

/*
  void int_to_iris_rgb(int dbz,int *r,int *g,int *b){
  *r=128.0+127.0*pseudo_sigmoi(0.05,dbz-96.0);
  *g=255.0*pseudo_gauss(80.0,dbz-80.0);
  *b=255.0*pseudo_gauss(24.0,dbz-48.0); 
  *r=pseudo_sigmoid(0.1,*r)*(*r);
  *g=pseudo_sigmoid(0.1,*g)*(*g);
  *b=pseudo_sigmoid(0.1,*b)*(*b);
  }
*/

int initialize_image(FmiImage *img){
  img->type=TRUE_IMAGE;
  //  img->format=UNDEFINED;
  img->area=img->width*img->height;
  img->volume=img->area*img->channels;
  
  fmi_debug(2,"initialize_image");
  if (FMI_DEBUG(3)) image_info(img);
  img->array=(Byte *) malloc(img->volume);
  //  img->channel_mapping=(int *) malloc(img->channels);
  //  img->coord_overflow_handler=WRAP;
  img->coord_overflow_handler_x=BORDER;
  img->coord_overflow_handler_y=BORDER;
  img->max_value=255;
  img->comment_string[0]='\0';
  return 1;
}

int initialize_horz_stripe(FmiImage *img,int width){
  img->width=width;
  img->height=1;
  img->channels=1;
  fmi_debug(2,"initialize_horz_image");
  initialize_image(img);
  img->coord_overflow_handler_y=TILE;
  return 1;
}

int initialize_vert_stripe(FmiImage *img,int height){
  img->width=1;
  img->height=height;
  img->channels=1;
  fmi_debug(2,"initialize_horz_image");
  initialize_image(img);
  img->coord_overflow_handler_x=TILE;
  return 1;
}

int link_image_segment(FmiImage *source,int start_row,int rows,FmiImage *linked){
  //  int a=rows*source->width;
  
  fmi_debug(3,"link_image_segment");
  
  if (start_row+rows > source->height*source->channels)
    fmi_error("link_image_segment: segment overflow");
  
  if (linked==source)
    fmi_error("link_image_segment: dont-like self-linking");
  
  if (linked->type==TRUE_IMAGE)
    fmi_debug(3,"WARNING: true image to link image without reset (free)");

  copy_image_properties(source,linked);
  linked->channels=1;
  linked->height=rows;
  linked->area=linked->width*linked->height; // safety
  linked->volume=linked->area*linked->channels;
  linked->type=LINK_IMAGE;
  linked->array=&(source->array[start_row*source->width]);
  if (FMI_DEBUG(2)) 
    image_info(linked);
  return 1;
}

void link_image_channel(FmiImage *source,int channel,FmiImage *linked){
  link_image_segment(source,channel*source->height,1*source->height,linked);
}

void split_to_link_array(FmiImage *source,int segments,FmiImage *target){
  int hs,ht,k;
  if (segments==0)
    fmi_error("split_to_link_array: zero channels?");
  hs=(source->height*source->channels);
  if (hs%segments!=0)
    fmi_error("split_to_link_array: height not divisible by # segments ");
  ht=hs/segments;
  for (k=0;k<segments;k++)
    link_image_segment(source,k*ht,ht,&target[k]);
}

void split_to_channels(FmiImage *target,int channels){
  int hs;
  if (channels<=0)
    fmi_error("split_to_channels: zero channels?");
  hs=(target->height*target->channels);
  if (hs%channels!=0)
    fmi_error("split_to_channels: height not divisible by #channels ");
  else{
    target->channels=channels;
    target->height=hs/channels;
    target->area=target->height*target->width;
  }
  fprintf(stderr,"split_to_channels: %d channels\n",channels);
}



// channel-division independent
void concatenate_images_vert(FmiImage *source,int count,FmiImage *target){
  register int i,j,k;
  // START FROM FIRST
  fmi_debug(3,"concat base");
  copy_image_properties(source,target);
  if (FMI_DEBUG(3)) image_info(target);
  // ADD THE REST
  for (k=1;k<count;k++){
    if (FMI_DEBUG(3)) fprintf(stderr,"concatenating #%d\n",k);
    if (FMI_DEBUG(3)) image_info(&source[k]);
    if (source[k].width!=target->width)
      fmi_error("concatenate_images: not implemented for variable width");
    if (source[k].channels!=target->channels)
      fmi_error("concatenate_images: not implemented for variable channel counts");
    target->height+=source[k].height;
  }
  fmi_debug(3,"to be concat");
  if (FMI_DEBUG(3)) image_info(target);
  initialize_image(target);
  if (FMI_DEBUG(3)) image_info(target);
  // CONCATENATE
  j=0;
  for (k=0;k<count;k++){
    for (i=0;i<source[k].volume;i++)
      target->array[j+i]=source[k].array[i];
    j+=source[k].volume;
  }
}

// NO IMAGE ARRAY SIZE CONTROL
// SEGMENTATION FAULT RISK
/*
  int reclaim_channels(FmiImage *img,int channels){
  img->channels=channels;
  img->volume=img->area*img->channels; 
  return 1;
  }
*/

int copy_image_properties(FmiImage *sample,FmiImage *target){
  target->width    =sample->width;
  target->height   =sample->height;
  target->channels =sample->channels;
  target->max_value=sample->max_value;
  //  target->format=sample->format; olis ehk� OK
  target->coord_overflow_handler_x=sample->coord_overflow_handler_x;
  target->coord_overflow_handler_y=sample->coord_overflow_handler_y;

  target->area=target->width*target->height;
  target->volume=target->channels*target->area;


  return 1;
}

int check_image_properties(FmiImage *sample,FmiImage *target){
  //register int h;
  if (target->width!=sample->width){
    //    h=target->coord_overflow_handler_x;
    //if ((h!=WRAP)&&(h!=TILE)){
    fmi_debug(2,"check_image_properties: unequal widths");
    return 0;
    //}
  }
  if (target->height!=sample->height){
    // h=target->coord_overflow_handler_y;
    // if ((h!=WRAP)&&(h!=TILE)){
    fmi_debug(2,"check_image_properties: unequal heights");
    return 0;
    //}
  }
  if (target->channels!=sample->channels){
    fmi_debug(2,"check_image_properties: unequal channel counts");
    return 0;}

  target->max_value=sample->max_value;
  target->area=target->width*target->height;
  target->volume=target->channels*target->area;
  return 1;
}

int canonize_image(FmiImage *sample,FmiImage *target){

	fmi_debug(2,"canonize_image?");
	if (!check_image_properties(sample,target)){
		fmi_debug(2,"canonize_image: YES");
		copy_image_properties(sample,target);
		initialize_image(target);
	}
	//target->type=TRUE_IMAGE; // VIOLATING SOMETHING?
	fmi_debug(2,"canonize_image END");
	return 1;
}

void reset_image(FmiImage *image){
  switch (image->type){
  case TRUE_IMAGE:
    free (image->array);
  case NULL_IMAGE:
  case LINK_IMAGE:
    image->width=0;
    image->height=0;
    image->channels=0;
    image->max_value=0;
    image->type=NULL_IMAGE;
    return;
  default:
    fprintf(stderr," IMAGE TYPE:%d\n",image->type);
    fmi_error("reset_image: unknown image type");
  }
  //  free (image->channel_mapping);
}


int legal_coords(FmiImage *img,int x,int y){
  if (x<0) return 0;
  if (y<0) return 0;
  if (x>=img->width) return 0;
  if (y>=img->height) return 0;
  return 1;
}

void handle_coord_overflow(FmiImage *img,int *x,int *y){
  // WRAP = quick TILE "ONCE"
  if (*x<0)
    switch (img->coord_overflow_handler_x){
    case MIRROR: *x=-*x; break;
    case WRAP  : *x=*x+img->width; break;
    case TILE  : *x=*x%img->width; break;
    case BORDER: *x=0; break;
    default: fmi_error("get: coord x underflow");}
  if (*x>=img->width)
    switch (img->coord_overflow_handler_x){
    case MIRROR: *x=2*img->width-*x; break;
    case WRAP  : *x=*x-img->width; break;
    case TILE  : *x=*x%img->width; break;
    case BORDER: *x=img->width-1; break;
    default: fmi_error("get: coord x overflow");}
  if (*y<0)
    switch (img->coord_overflow_handler_y){
    case MIRROR: *y=-*y; break;
    case WRAP  : *y=*y+img->height; break;
    case TILE  : *y=*y%img->height; break;
    case BORDER: *y=0; break;
    default: fmi_error("get: coord y underflow");}
  if (*y>=img->height)
    switch (img->coord_overflow_handler_y){
    case MIRROR: *y=2*img->height-*y; break;
    case WRAP  : *y=*y-img->height; break;
    case TILE  : *y=*y%img->height; break;
    case BORDER: *y=img->height-1; break;
    default: fmi_error("get: coord y overflow");}
}

Byte get_pixel_direct(FmiImage *img,int address){
  return img->array[address];
}

void put_pixel(FmiImage *img,int x,int y,int channel,Byte c){
  handle_coord_overflow(img,&x,&y);
  img->array[channel*img->area + y*img->width + x]=c;
  //  return 1;
}

void put_pixel_direct(FmiImage *img,int address,Byte c){
  img->array[address]=c;
}

void put_pixel_direct_inc(FmiImage *img,int address){
  if (img->array[address]<255)
    ++(img->array[address]);
}

void put_pixel_or(FmiImage *img,int x,int y,int channel,Byte c){
  static Byte *location;
  handle_coord_overflow(img,&x,&y);
  location=&(img->array[channel*img->area + y*img->width + x]);
  *location=*location|c;
}

void put_pixel_and(FmiImage *img,int x,int y,int channel,Byte c){
  static Byte *location;
  handle_coord_overflow(img,&x,&y);
  location=&(img->array[channel*img->area + y*img->width + x]);
  *location=*location&c;
}

void put_pixel_min(FmiImage *img,int x,int y,int channel,Byte c){
  static Byte *location;
  handle_coord_overflow(img,&x,&y);
  location=&(img->array[channel*img->area + y*img->width + x]);
  if (c<*location)  *location=c;
}

void put_pixel_max(FmiImage *img,int x,int y,int channel,Byte c){
  static Byte *location;
  handle_coord_overflow(img,&x,&y);
  location=&(img->array[channel*img->area + y*img->width + x]);
  if (c>*location)  *location=c;
}

Byte get_pixel(FmiImage *img,int x,int y,int channel){
  /*  return (img->array[y*img->width+x][channel]); */
  handle_coord_overflow(img,&x,&y);
  return (img->array[channel*img->area + y*img->width + x]); 
}

void fill_image(FmiImage *img,Byte c){
  register int i;
  img->area=img->width*img->height;
  img->volume=img->area*img->channels;
  for (i=0;i<img->volume;i++)
    img->array[i]=c;
}

void image_fill_random(FmiImage *img,Byte mean,Byte amplitude){
  register int i;
  img->area=img->width*img->height;
  img->volume=img->area*img->channels;
  for (i=0;i<img->volume;i++)
    img->array[i]=mean+(amplitude*(rand()&255))/256;
}

void invert_image(FmiImage *img){
  register int i;
  img->area=img->width*img->height;
  img->volume=img->area*img->channels;
  for (i=0;i<img->volume;i++)
    img->array[i]=img->array[i]^255;
}

void limit_image_intensities(FmiImage *img,Byte min,Byte max){
  register int i;
  img->area=img->width*img->height;
  img->volume=img->area*img->channels;
  for (i=0;i<img->volume;i++){
    if (img->array[i]<min)  img->array[i]=min;
    else if (img->array[i]>max)  img->array[i]=max;}
}

void translate_intensity(FmiImage *img,Byte from,Byte to){
  register int i;
  img->area=img->width*img->height;
  img->volume=img->area*img->channels;
  for (i=0;i<img->volume;i++)
    if (img->array[i]==from)  img->array[i]=to;
}


void add_image(FmiImage *source,FmiImage *source2,FmiImage *target){
  register int i;
  int sum;
  if (check_image_properties(source,source2)==0)
    fmi_error("subtract_image: incompatible images");
  canonize_image(source,target);
  for (i=0;i<target->volume;i++){
    sum=source->array[i]+source2->array[i];
    target->array[i]=(sum<=254 ? sum : 254);}
}

void subtract_image(FmiImage *source,FmiImage *source2,FmiImage *target){
  register int i;
  if (check_image_properties(source,source2)==0)
    fmi_error("subtract_image: incompatible images");
  canonize_image(source,target);
  for (i=0;i<target->volume;i++)
    target->array[i]=(source->array[i]>source2->array[i]) ? source->array[i]-source2->array[i] : 0;
}

void subtract_image128(FmiImage *source,FmiImage *source2,FmiImage *target){
  register int i;
  //  int sum;
  if (check_image_properties(source,source2)==0)
    fmi_error("subtract_image: incompatible images");
  canonize_image(source,target);
  for (i=0;i<target->volume;i++)
    target->array[i]=(255+source->array[i]-source2->array[i])/2;
}


void average_images(FmiImage *source,FmiImage *source2,FmiImage *target){
  register int i;
  int sum;
  if (check_image_properties(source,source2)==0)
    fmi_error("subtract_image: incompatible images");
  canonize_image(source,target);
  for (i=0;i<target->volume;i++){
    sum=(source->array[i]+source2->array[i])/2;
    target->array[i]=(sum<=254 ? sum : 254);}
}


void multiply_image255(FmiImage *source,FmiImage *source2,FmiImage *target){
  register int i,c;
  if (check_image_properties(source,source2)==0)
    fmi_error("multiply_image255: incompatible images");
  canonize_image(source,target);
  for (i=0;i<target->volume;i++){
    c=source->array[i]*source2->array[i]/255;
    target->array[i]=MIN(c,255);}
}

void multiply_image255_flex(FmiImage *source,FmiImage *source2,FmiImage *target){
  register int i,j,c;
  canonize_image(source,target);
  for (i=0;i<source->width;i++)
    for (j=0;j<source->height;j++){
      c=get_pixel(source,i,j,0)*get_pixel(source2,i,j,0)/255;
      put_pixel(target,i,j,0,MIN(c,255));
    }
}

void multiply_image255_sigmoid(FmiImage *source,FmiImage *source2,FmiImage *target){
  register int i;
  const int half_width=128*128;
  if (check_image_properties(source,source2)==0)
    fmi_error("multiply_image255_sigmoid: incompatible images");
  canonize_image(source,target);
  for (i=0;i<target->volume;i++)
    target->array[i]=pseudo_sigmoid(half_width,source->array[i]*source2->array[i]);
}


void max_image(FmiImage *source,FmiImage *source2,FmiImage *target){
  register int i;
  if (check_image_properties(source,source2)==0)
    fmi_error("subtract_image: incompatible images");
  canonize_image(source,target);
  for (i=0;i<target->volume;i++)
    target->array[i]=(source->array[i]>source2->array[i]) ? source->array[i] : source2->array[i];
}

void min_image(FmiImage *source,FmiImage *source2,FmiImage *target){
  register int i;
  if (check_image_properties(source,source2)==0)
    fmi_error("subtract_image: incompatible images");
  canonize_image(source,target);
  for (i=0;i<target->volume;i++)
    target->array[i]=(source->array[i]<source2->array[i]) ? source->array[i] : source2->array[i];
}

void multiply_image_scalar255(FmiImage *img,int coeff){
 register int i;
 int temp;
 for (i=0;i<img->volume;i++){
   temp=img->array[i]*coeff/255;
   img->array[i]=MIN(temp,255);}
}

void semisigmoid_image(FmiImage *source,int half_width){
 register int i;
 for (i=0;i<source->volume;i++)
   source->array[i]=pseudo_sigmoid(half_width,source->array[i]);
}

void semisigmoid_image_inv(FmiImage *source,int half_width){
 register int i;
 for (i=0;i<source->volume;i++)
   source->array[i]=255-pseudo_sigmoid(half_width,source->array[i]);
}

// slope = half-width
void sigmoid_image(FmiImage *source,int threshold,int slope){
 register int i;

 if (slope>0){
   for (i=0;i<source->volume;i++)
     source->array[i]=128+pseudo_sigmoid(slope,source->array[i]-threshold)/2;
   return;}

 if (slope<0){
   for (i=0;i<source->volume;i++)
     source->array[i]=128-pseudo_sigmoid(-slope,source->array[i]-threshold)/2;
   return;}

 if (slope==0){
   for (i=0;i<source->volume;i++)
     source->array[i]=(source->array[i]>threshold)*255;
   return;}

}

void gaussian_image(FmiImage *source,int mean,int half_width){
 register int i;
 for (i=0;i<source->volume;i++)
   source->array[i]=pseudo_gauss(half_width,source->array[i]-mean);
}

void copy_image(FmiImage *source,FmiImage *target){
  register int i;
  fmi_debug(4," copy_image");
  canonize_image(source,target);
  for (i=0;i<source->volume;i++)
    target->array[i]=source->array[i];
}


void extract_channel(FmiImage *source,int channel,FmiImage *target){
  register int i,j;
  fmi_debug(3,__FILE__);
  fmi_debug(3," extract_channel");
  fmi_debug(4," extract_channel: initialize");
  //  fmi_debug(1,__LINE__);
  if (channel>=source->channels) fmi_error("extracting nonexistent channel");
  copy_image_properties(source,target);
  target->channels=1;
  initialize_image(target);
  fmi_debug(4," extract_channel: extraction");
   for (i=0;i<source->width;i++)
     for (j=0;j<source->height;j++)
             put_pixel(target,i,j,0,get_pixel(source,i,j,channel));
       //       put_pixel(target,i,j,0,85); 
   //get_pixel(source,i,j,channel); T�M� SOI
  fmi_debug(4," extract_channel: ...extracted");
}

void write_channel(FmiImage *source,int channel,FmiImage *target){
  register int i,j;
  fmi_debug(2,__FILE__);
  fmi_debug(2," write_channel");
  //  fmi_debug(1,__LINE__);
  if (channel>=target->channels) fmi_error("extracting nonexistent channel");
  //  copy_image_properties(source,target);
  //target->channels=1;
  //initialize_image(target);
   for (i=0;i<source->width;i++)
     for (j=0;j<source->height;j++)
       put_pixel(target,i,j,channel,get_pixel(source,i,j,0));
}


void insert(FmiImage *source,FmiImage *target,int i0, int j0){
  register int i,j,k;
  fmi_debug(2,"insert");
  if (source->channels==target->channels)
    for (k=0;k<source->channels;k++)
      for (i=0;i<source->width;i++)
	for (j=0;j<source->height;j++)
	  put_pixel(target,i0+i,j0+j,k,get_pixel(source,i,j,k));
  else 
    if (source->channels==1)
      for (k=0;k<target->channels;k++)
	for (i=0;i<source->width;i++)
	  for (j=0;j<source->height;j++)
	  put_pixel(target,i0+i,j0+j,k,get_pixel(source,i,j,0));
    else{
      fprintf(stderr,"channels = %d vs %d",source->channels,target->channels);
      fmi_error("insert: channel number conflict");}
}

void compose2x2(FmiImage *source_ul,FmiImage *source_ur,FmiImage *source_ll,FmiImage *source_lr,FmiImage *target){
  //register int i,j;
  int t1,t2,tw,th;

  fmi_debug(1,"compose2x2");
  t1=source_ul->width+source_ur->width;
  t2=source_ll->width+source_lr->width;
  target->width=MAX(t1,t2); 
  tw=target->width;

  t1=source_ul->height+source_ll->height;
  t2=source_ur->height+source_lr->height;
  target->height=MAX(t1,t2); 
  th=target->height;

  target->channels=MAX(source_ul->channels,source_ur->channels);
  target->channels=MAX(target->channels,source_ll->channels);
  target->channels=MAX(target->channels,source_lr->channels);
  initialize_image(target);
  //  fmi_debug(1,"compose2x2 target:"); 
  if (FMI_DEBUG(3)) image_info(target);
  fill_image(target,251);

  insert(source_ul,target,0,0);
  insert(source_ur,target,tw-source_ur->width,0);
  insert(source_ll,target,0,th-source_ll->height);
  insert(source_lr,target,tw-source_lr->width,th-source_lr->height);
}

void compose3x2(FmiImage *source_ul,FmiImage *source_um,FmiImage *source_ur,FmiImage *source_ll,FmiImage *source_lm,FmiImage *source_lr,FmiImage *target){
  int t1,t2,t3,tw,th;

  fmi_debug(1,"compose2x2");
  t1=source_ul->width+source_um->width+source_ur->width;
  t2=source_ll->width+source_lm->width+source_lr->width;
  target->width=MAX(t1,t2); 
  tw=target->width;

  t1=source_ul->height+source_ll->height;
  t2=source_um->height+source_lm->height;
  t3=source_ur->height+source_lr->height;
  t2=MAX(t1,t2); 
  target->height=MAX(t2,t3); 
  th=target->height;

  target->channels=MAX(source_ul->channels,source_ur->channels);
  target->channels=MAX(target->channels,source_um->channels);
  target->channels=MAX(target->channels,source_ll->channels);
  target->channels=MAX(target->channels,source_lm->channels);
  target->channels=MAX(target->channels,source_lr->channels);
  initialize_image(target);
  fmi_debug(1,"compose3x2 target:"); 
  if (FMI_DEBUG(3)) image_info(target);
  fill_image(target,251);

  insert(source_ul,target,0,0);
  insert(source_um,target,source_ul->width,0);
  insert(source_ur,target,tw-source_ur->width,0);
  insert(source_ll,target,0,th-source_ll->height);
  insert(source_lm,target,source_ll->width,th-source_lr->height);
  insert(source_lr,target,tw-source_lr->width,th-source_lr->height);
}

#define STRLENGTH 256

/*
  FmiImageFormat detect_image_format_by_extension(char *filename){
  return PPM_RAW;
  }
*/

FmiImageFormat process_image_header(FmiImage *img,FILE *fp){
  int c;
  char str[MAX_COMMENT_LENGTH];
  //  FmiImageFormat format;
  /* DETECT FILE FORMAT FROM MAGIC NUMBER */
  c=getc(fp);
  if ((char)c=='P'){
    c=getc(fp);
    img->format=(FmiImageFormat)(c-'0');
    //    if (img->channels==0)
      switch (img->format){
      case PBM_ASC:
      case PBM_RAW:
      case PGM_ASC:
      case PGM_RAW:
	img->channels=1;
	break;
      case PPM_ASC:
      case PPM_RAW:
	img->channels=3; 
	break;
      default: fmi_error("Not a PNM image");}

    if (FMI_DEBUG(2)) fprintf(stderr,"Image type: P%c (%d)\n",c,img->format);

    /* scan to next line */
    do 
      c=getc(fp);
    while ((c==' ')||(c=='\t')||(c=='\n'));
    ungetc(c,fp);

    /* READ (OPTIONAL) COMMENTS */
    //    strp=comment_string;
    //l=0;
    img->comment_string[0]='\0';
    while ((c=getc(fp))=='#'){
	/*    getline(str,256,fp);
	      printf(str);
	 */
    	// fgets(str,MAX_COMMENT_LENGTH-strlen(img->comment_string),fp);
    	// strcat(img->comment_string,str);
    	fgets(str,MAX_COMMENT_LENGTH-1,fp);
    	if (strlen(img->comment_string)+strlen(str)<MAX_COMMENT_LENGTH)
    		strcat(img->comment_string,str);
    	else {
    		fmi_debug(1,"read_image, image comment overflow");
    		if (FMI_DEBUG(2))
    			fprintf(stderr,"%s\n",str);
    	}
    }
    //    printf(img->comment_string);
    //printf("channels %d\n",img->channels);
    //printf("%c\n",c);
    //    fmi_debug(0,"kaUnimplemented image format");
    ungetc(c,fp);
    // fmi_debug(0," ungetc");

    /* READ WIDTH, HEIGHT, MAXVAL  */
    fscanf(fp,"%d %d",&(img->width),&(img->height));
    if ((img->format==PBM_RAW)||(img->format==PBM_ASC)) 
      img->max_value=1;
    else
      fscanf(fp,"%d",&(img->max_value));

    if (FMI_DEBUG(3)) image_info(img);

    /* scan to next line */
    do 
      c=getc(fp);
    while ((c==' ')||(c=='\t')||(c=='\n'));
    ungetc(c,fp);}
  else 
    fmi_error("Unimplemented image format");

  return img->format;
}


void read_image(char *filename,FmiImage *img){
  //int i,j,k,c;
  //char str[STRLENGTH];
  FmiImageFormat format;
  FILE *fp;

  if (FMI_DEBUG(2)){
    fprintf(stderr," reading file: ");fflush(stderr);
    fprintf(stderr,"%s\n",filename);
  }

  if (strcmp(filename,"-")==0)
    fp = stdin;
  else
    fp = fopen(filename,"r");
  if (fp==NULL) fmi_error("read_image: file error");

  format=process_image_header(img,fp);
  img->format=format;
  initialize_image(img);
  switch (format){
  case PGM_ASC:
  case PBM_ASC:
  case PPM_ASC:
  case PGM_RAW:
  case PBM_RAW:
  case PPM_RAW:
    read_pnm_image(fp,img,format);
    break;
  default:
    fmi_error("read_image: unimplemented image format");}
}

void read_pnm_image(FILE *fp,FmiImage *img,FmiImageFormat format){
  register int i,j,k;
  Byte c;
  int l;
  switch (format){
  case PBM_RAW:
    for (j=0;j<img->height;j++){
      c=getc(fp);
      k=128;	  
      for (i=0;i<img->width;i++){
	if (k<1){
	  k=128;
	  c=getc(fp);}
      	put_pixel(img,i,j,0,((c&k)>0)?1:254);
      //	put_pixel(img,i,j,0,(i&j>0)?55:22);
	//	put_pixel(img,i,j,0,(i&j>0)?55:22);
	k=k>>1;}
    }
    break;
  case PGM_RAW:
    for (j=0;j<img->height;j++)
      for (i=0;i<img->width;i++)
	put_pixel(img,i,j,0,getc(fp));
    break;
  case PGM_ASC:
    for (j=0;j<img->height;j++)
      for (i=0;i<img->width;i++){
	fscanf(fp,"%d",&l);
	//printf("%d\n",l);
	put_pixel(img,i,j,0,(Byte)l);
      }
    break;
  case PPM_RAW:
    for (j=0;j<img->height;j++)
      for (i=0;i<img->width;i++)
	for (k=0;k<3;k++){
	  put_pixel(img,i,j,k,getc(fp));
	}
    break;
  case PPM_ASC:
    for (j=0;j<img->height;j++)
      for (i=0;i<img->width;i++)
	for (k=0;k<3;k++){
	  fscanf(fp,"%d",&l);
	  //printf("%d\n",l);
	  put_pixel(img,i,j,0,(Byte)k);
	}
    break;
  default:
    fmi_error("Error: unimplemented image file type");}
  fclose(fp);
}

void write_image(char *filename,FmiImage *img,FmiImageFormat format){
  char *actual_filename;
  FILE *fp;
  fmi_debug(1,"write_image: ");
  fmi_debug(2,filename);
  //  if (strcmp(filename,"-")==0)
  //  fp = stderr;...?

  if (file_extension(filename)!=NULL)
    actual_filename=filename;
  else {
    actual_filename=(char *)malloc(strlen(filename)+5);
    strcpy(actual_filename,filename);
    strcat(actual_filename,".");
    strncat(actual_filename,FmiImageFormatExtension[format],4);
  }
  fmi_debug(1,actual_filename); 
  fp = fopen(actual_filename,"w");
  if (fp){
    if ((format>=PGM_ASC)&&(format<=PPM_RAW))
      write_pnm_image(fp,img,format);
    else
      fmi_error("Unsupported image format.");
    fclose(fp);
    fmi_debug(2,"write_image: ok");
  }
  else
    fmi_error("Failed opening file for writing.");
  free(actual_filename);  // segmentation fault? UUSI!
}

void dump_comments(FILE *fp,char *comment,char *begin_code,char *end_code,int line_length){
  int i,l,len;
  char c;
  if (comment==NULL)
    return;
  len=strlen(comment);
  l=0;
  for (i=0;i<=len;i++){
    if (l==0)
      fprintf(fp,begin_code);
    c=comment[i];
    if (l==line_length-1){
      fprintf(fp,"%c\\%s",c,end_code);
      l=0;}
    else
      if ((c=='\n')||(i==len)){
	//    if ((l==line_length-1)||(i==len-1)){
	fprintf(fp,end_code);
	l=0;}
      else {
	fputc(c,fp);
	l++;}
  }
}

void write_pnm_image(FILE *fp,FmiImage *img,FmiImageFormat format){
  register int i,j,k;
  unsigned char c;
  /* MAGIC NUMBER, see man pnm */
  fprintf(fp,"P%d\n",format);

  /* COMMENTS */
  trchr(fmi_util_comment,'\\','\n');
  dump_comments(fp,fmi_util_comment,   "# ","\n",1023);
  if (strlen(img->comment_string)>0)
    dump_comments(fp,img->comment_string,"# [","]\n",1023);
  dump_comments(fp,fmi_util_command_line,   "# ","\n",1023);
  if (FMI_IMAGE_COMMENT)
    //    fprintf(fp,"# CREATOR: %s (c) Markus.Peura@fmi.fi\n",FMI_IMAGE_VER);
    fprintf(fp,"# CREATOR: %s\n",FMI_IMAGE_VER);
  //  if (fmi_util_command_line[0]!='\0')
  //  fprintf(fp,"# %s\n",fmi_util_command_line);


  /* IMAGE DIMENSIONS */
  fprintf(fp,"%d %d\n",img->width,img->height);
  fprintf(fp,"%d\n",img->max_value);

  switch (format){ 
  case PBM_RAW:
	  if (img->channels!=1)
		  fmi_error("write_pnm_image: PBM implies 1 channel");
	  for (j=0;j<img->height;j++){
		  c=0;
		  k=128;
		  for (i=0;i<img->width;i++){
			  c=(c|(((get_pixel(img,i,j,0)&128)==0)?0:k));
			  if (k==1){
				  fputc(c,fp);
				  c=0;
				  k=128;}
			  k=k>>1;}
	  }
	  break;
  case PGM_RAW:
	  if (img->channels!=1)
		  fmi_error("write_pnm_image: PGM implies 1 channel");
	  for (j=0;j<img->height;j++)
		  for (i=0;i<img->width;i++)
			  fputc(get_pixel(img,i,j,0),fp);
	  break;
  case PPM_RAW:
	  if (img->channels!=3)
		  fmi_error("write_pnm_image: PPM implies 3 channels");
	  for (j=0;j<img->height;j++)
		  for (i=0;i<img->width;i++)
			  for (k=0;k<3;k++)
				  fputc(get_pixel(img,i,j,k),fp);
	  break;
  default:
	  break;
  }
  return 1;
}


void image_info(FmiImage *img){
  if (FMI_DEBUG(1)) {
    fprintf(stderr," type: ");
    switch (img->type){
      case NULL_IMAGE: fprintf(stderr,"NULL_IMAGE\n"); break;
      case TRUE_IMAGE: fprintf(stderr,"TRUE_IMAGE\n"); break;
      case LINK_IMAGE: fprintf(stderr,"LINK_IMAGE\n"); break;
    }
    fprintf(stderr," channels: %d\n",img->channels);
    fprintf(stderr," width:    %d\n",img->width);
    fprintf(stderr," height:   %d\n",img->height);
    fprintf(stderr," area:     %d\n",img->area);
    fprintf(stderr," volume:   %d\n",img->volume);
    fprintf(stderr," max_val:  %d\n",img->max_value);
    fflush(stderr);
  }
}




/* TRANSFORMS */
void expand_channel_to_rgb(FmiImage *source,int channel,FmiImage *target){
  register int i,j,k;
  Byte c;
  fmi_debug(1,"fmi_image.c: expand_channel_to_rgb");

  copy_image_properties(source,target);
  target->channels=3;
  initialize_image(target);
  
  for (i=0;i<source->width;i++)
    for (j=0;j<source->height;j++){
      c=get_pixel(source,i,j,channel);
      for (k=0;k<3;k++)
	put_pixel(target,i,j,k,c);
    }
}

void map_channel_to_colors(FmiImage *source,int channel,FmiImage *target,int map_size,ColorMap map){
  register int i,j,k;
  Byte l,c;
  fmi_debug(4,"fmi_image.c: map_channel_to_channels");
  if (map_size>255) fmi_error("map_channel_to_channels: too many levels");
  
  /*
    for (l=0;l<map_size;l++) {
    printf(" %d %d \n",l,map[l][0]); fflush(stdout);}
  */
  
  copy_image_properties(source,target);
  target->channels=3;
  initialize_image(target);
  
  for (i=0;i<source->width;i++)
    for (j=0;j<source->height;j++){
      c=get_pixel(source,i,j,channel);
      for (l=0;l<map_size;l++){
	if (map[l][0]>=c){
	  for (k=0;k<3;k++)
	    put_pixel(target,i,j,k,map[l][k+1]);
	  break;}
      }
    }
}

void map_channel_to_256_colors(FmiImage *source,int channel,FmiImage *target,ColorMap256 map){
  register int i,j,k;
  //Byte l,c;
  fmi_debug(4,"fmi_image.c: map_channel_to_256_colors");
  
  /*
    for (l=0;l<map_size;l++) {
    printf(" %d %d \n",l,map[l][0]); fflush(stdout);}
  */

  copy_image_properties(source,target);
  target->channels=3;
  initialize_image(target);

  for (i=0;i<source->width;i++)
    for (j=0;j<source->height;j++)
      for (k=0;k<3;k++)
	put_pixel(target,i,j,k,map[get_pixel(source,i,j,channel)][k]);

}

void map_256_colors_to_gray(FmiImage *source,FmiImage *target,ColorMap256 map){
  register int i,j,l;
  Byte r,g,b;
  fmi_debug(4,"fmi_image.c: map_256_colors_to_gray");
  copy_image_properties(source,target);
  target->channels=1;
  initialize_image(target);
  fill_image(target,0);

  for (i=0;i<source->width;i++)
    for (j=0;j<source->height;j++){
      r=get_pixel(source,i,j,0);
      g=get_pixel(source,i,j,1);
      b=get_pixel(source,i,j,2);
      for (l=0;l<256;l++)
	if ((map[l][0]==r)&&(map[l][1]==g)&&(map[l][2]==b)){
	  put_pixel(target,i,j,0,l);
	  break;}
    }
  
}

void read_colormap256(char *filename,ColorMap256 map){
  FILE *fp;
  int c;
  fmi_debug(4,"read_colormap256");
  fmi_debug(5,filename);
  fp = fopen(filename,"r");
  if (!fp) fmi_error("read_colormap256: file read error");

  if ((c=getc(fp))=='#'){
    do {
      //  printf("koe%c",c);
      while ((char)(c=getc(fp))!='\n'){
	// printf("%c",c);
      if (c==EOF) fmi_error("read_colormap256: only comments?");
      }
    } while ((c=getc(fp))=='#');
    //    printf("Comment: %c\n",c);
    // fflush(stdout);
    ungetc(c,fp);
  }

  fmi_debug(5,"read_colormap256: colors");
  // MAIN LOOP
  while (fscanf(fp,"%d",&c)!=EOF)
    fscanf(fp,"%d %d %d\n",&map[c][0],&map[c][1],&map[c][2]);
  fclose(fp);
  fmi_debug(5,"read_colormap256 complete");
}


/*
  int to_rgb(FmiImage *source,FmiImage *target){
  int i,j,dbz,r,g,b;
  
      target->width=source->width;
      target->height=source->height;
      target->max_value=255;
      target->channels=3;
      initialize_image(target);
      
      

      for (i=0;i<source->width;i++)
      for (j=0;j<source->height;j++){
      byte_to_rgb(get_pixel(source,i,j,0),&r,&g,&b);
      //byte_to_rgb(255*i/source->width,&r,&g,&b);
      put_pixel(target,i,j,0,r);
      put_pixel(target,i,j,1,g);
      put_pixel(target,i,j,2,b);}
}
*/

void to_cart(FmiImage *source,FmiImage *target,Byte outside_fill){
  int i,j,k,i0,j0,max_radius,di,dj,radius,a;
  target->width=source->width;
  target->height=source->width; /* yes, width, -> square */
  //target->height=source->height; 
  target->max_value=255;
  target->channels=source->channels;
  initialize_image(target);
  //  initialize_image(source);
  

  fmi_debug(3,"to_cart");
  /*  printf("area %d\n",source->area); fflush(stdout); */
  
  max_radius=target->width;
  i0=target->width/2;
  j0=target->height/2; /* =i0 */

  for (i=0;i<target->width;i++){
    di=i-i0;
    for (j=0;j<target->height;j++){
      dj=j-j0;
      radius=2*(int)sqrt((double)(di*di+dj*dj));
      for (k=0;k<target->channels;k++){
	if (radius<=max_radius){
	  if (radius>0){
	    a=180*atan2(di,-dj)/3.14;
	    if (a<0) a+=360;
	    put_pixel(target,i,j,k,get_pixel(source,radius,a,k));
	    //	    put_pixel(target,i,j,k,get_pixel(source,radius,a,k));
	  }
	} else put_pixel(target,i,j,k,255); 
      }
    }
  }
}


void clear_histogram(Histogram hist){
  register int i;
  for (i=0;i<HISTOGRAM_SIZE;i++)  hist[i]=0;
}

void calc_histogram(FmiImage *source,Histogram hist){
  register int i;
  clear_histogram(hist);
  for (i=0;i<source->volume;i++)
    ++hist[source->array[i]];
}

void output_histogram(FILE *fp,Histogram hist){
  register int i;
  for (i=0;i<HISTOGRAM_SIZE;i++)  
    //    fprintf(fp,"%d\n",hist[i]);
    fprintf(fp,"%d\t%d\n",i,hist[i]);
}

void write_histogram(char *filename,Histogram hist){
  FILE *fp;
  fp=fopen(filename,"w");
  output_histogram(fp,hist);
  fclose(fp);
}

void dump_histogram(Histogram hist){
  output_histogram(stdout,hist);
}



//Mask mask_speck1={3,3,"koke"};       NO ERROR
//Mask mask_speck1={3,3, {'k','o','k','e','\0'} }; ERROR!
