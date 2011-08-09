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


#ifndef __FMI_IMAGE__
#define __FMI_IMAGE__
#define FMI_IMAGE_VER "fmi_image \t v2.2\t Jul 2002 (c) Markus.Peura@fmi.fi"

#include <stdio.h>

/*
  typedef Coord short int;
  typedef Byte unsigned char;
*/

/*
enum fmi_image_format {
  PBM_ASC=1,
  PGM_ASC=2,
  PPM_ASC=3,
  PBM_RAW=4,
  PGM_RAW=5,
  PPM_RAW=6};
*/

typedef enum {
  UNDEFINED=0,
  PBM_ASC=1,
  PGM_ASC=2,
  PPM_ASC=3,
  PBM_RAW=4,
  PGM_RAW=5,
  PPM_RAW=6
} FmiImageFormat;

typedef enum {
  NULL_IMAGE,
  TRUE_IMAGE,
  LINK_IMAGE
} FmiImageType;

extern char FmiImageFormatExtension[7][4];
/*char FmiImageFormatExtension[][]; */

typedef enum {
  ZERO,
  MAX,
  MIRROR,
  BORDER,
  WRAP,
  TILE
} CoordOverflowHandler;

/*typedef signed char Dbz; */
typedef unsigned char Byte;
typedef int Celsius;

#define HISTOGRAM_SIZE (256+15)
/*typedef unsigned long Histogram[HISTOGRAM_SIZE];   */
typedef signed long Histogram[HISTOGRAM_SIZE];  /* signed is needed for AREA */
/* this list contains special entries which are not very useful */
/* in standard histogram usage */

enum histogram_special_data {
  HIST_SIZE = 256,
  HIST_SUM,
  HIST_MIN,
  HIST_AREA,
  HIST_MAX,
  HIST_PERIMx3,
  HIST_SUM_I,
  HIST_SUM_J,
  HIST_SUM_II,
  HIST_SUM_JJ,
  HIST_SUM_IJ,
  HIST_MIN_I,
  HIST_MIN_J,
  HIST_MAX_I,
  HIST_MAX_J
};
/* lis�� my�s dump_stats:iin */

/*
#define HIST_SIZE    256
#define HIST_SUM     257
#define HIST_MIN     258
#define HIST_AREA    259
#define HIST_MAX     260
#define HIST_PERIMx2 261
#define HIST_SUM_I   262
#define HIST_SUM_J   263
#define HIST_SUM_II  264
#define HIST_SUM_JJ  265
#define HIST_SUM_IJ  266
#define HIST_MIN_I   267
#define HIST_MIN_J   268
#define HIST_MAX_I   269
#define HIST_MAX_J   270
*/


typedef unsigned char ColorMap[][4];
typedef unsigned char ColorMap256[256][3];

#define MAX_COMMENT_LENGTH 1024

extern int FMI_IMAGE_COMMENT;


struct fmi_image {

  /*
    width = bins
    height = rays
  */
  int width,height,channels;
  /*  int *channel_mapping; */
  int area,volume;
  int max_value;

  int sweep_count;
  int *heights;
  
  float bin_depth;
  float elevation_angle;

  /* depth ? */
  /* unsigned char **array;*/
  Byte *array;
  CoordOverflowHandler coord_overflow_handler_x, coord_overflow_handler_y;
  /*  unsigned char *stream;*/
  char comment_string[MAX_COMMENT_LENGTH];
  FmiImageFormat format;
  FmiImageType type;
};

typedef struct fmi_image FmiImage;

/*! Checks if coords are inside the image area. */
int legal_coords(FmiImage *img,int x,int y);

/* BASIC OPERATIONS OFR SETTING AND RESETTING IMAGES */
/*int set(FmiImage *img,int width,int height,int channels);*/
FmiImage *new_image(int sweep_count); /* Allocator */
int initialize_image(FmiImage *img); /* constructor */

void reset_image(FmiImage *img);

/* virtual images */
int link_image_segment(FmiImage *reference_img,int channel_start,int channel_count,FmiImage *linked_img);
void link_image_channel(FmiImage *source,int channel,FmiImage *linked);

void release_image(FmiImage *img); /* destructor */

void split_to_link_array(FmiImage *source,int segments,FmiImage *target);
void split_to_channels(FmiImage *source,int channels);

int check_image_properties(FmiImage *sample,FmiImage *target);
/*
 int split_to_channels(FmiImage *img,int channels);
 int restore_to_single_channel(FmiImage *img);
 int reclaim_channels(FmiImage *source,int channels);
 int channels_to_link_array(FmiImage *source,FmiImage *target);
*/

void concatenate_images_vert(FmiImage *source,int count,FmiImage *target);
/*int convert_to_linkarray(FmiImage *source,FmiImage *target); */

int copy_image_properties(FmiImage *sample,FmiImage *target);
int check_image_integrity(FmiImage *sample,FmiImage *target);
int canonize_image(FmiImage *sample,FmiImage *target);


Byte get_pixel(FmiImage *img,int x,int y,int channel);
/*#define get_pixel(img,x,y,channel) get(img,x,y,channel)  */

Byte get_pixel_direct(FmiImage *img,int i);

void put_pixel(FmiImage *img,int x,int y,int channel,Byte c);
/*#define put_pixel(img,x,y,channel,c) put(img,x,y,channel,c)  */
void put_pixel_direct(FmiImage *img,int address,Byte c);
void put_pixel_direct_inc(FmiImage *img,int address);

void put_pixel_min(FmiImage *img,int x,int y,int channel,Byte c);
void put_pixel_max(FmiImage *img,int x,int y,int channel,Byte c);
void put_pixel_or(FmiImage *img,int x,int y,int channel,Byte c);
void put_pixel_and(FmiImage *img,int x,int y,int channel,Byte c);

void fill_image(FmiImage *img,Byte c);
void image_fill_random(FmiImage *img,Byte mean,Byte amplitude);
void invert_image(FmiImage *img);
void translate_intensity(FmiImage *img,Byte from,Byte to);
void limit_image_intensities(FmiImage *img,Byte min,Byte max);

void add_image(FmiImage *source,FmiImage *source2,FmiImage *target);
void average_images(FmiImage *source,FmiImage *source2,FmiImage *target);
void subtract_image(FmiImage *source,FmiImage *source2,FmiImage *target);
void subtract_image128(FmiImage *source,FmiImage *source2,FmiImage *target);
void multiply_image255(FmiImage *source,FmiImage *source2,FmiImage *target);
void multiply_image255_flex(FmiImage *source,FmiImage *source2,FmiImage *target);
void multiply_image255_sigmoid(FmiImage *source,FmiImage *source2,FmiImage *target);
void max_image(FmiImage *source,FmiImage *source2,FmiImage *target);
void min_image(FmiImage *source,FmiImage *source2,FmiImage *target);

/* intensity mappings */
void multiply_image_scalar255(FmiImage *img,int coeff);
void semisigmoid_image(FmiImage *source,int half_width); /* scale */
void semisigmoid_image_inv(FmiImage *source,int half_width); /* scale */
void sigmoid_image(FmiImage *source,int threshold,int slope); /* soft threshold */
void gaussian_image(FmiImage *source,int mean,int half_width);/* soft threshold */

void copy_image(FmiImage *source,FmiImage *target);
void insert(FmiImage *source,FmiImage *target,int i0,int j0);
void compose2x2(FmiImage *source_ul,FmiImage *source_ur,FmiImage *source_ll,FmiImage *source_lr,FmiImage *target);
void compose3x2(FmiImage *source_ul,FmiImage *source_um,FmiImage *source_ur,FmiImage *source_ll,FmiImage *source_lm,FmiImage *source_lr,FmiImage *target);

int legal_coord(FmiImage *img,int x,int y);

/* INPUT/OUTPUT */
/* int write(FmiImage *img,char *filename,int fmi_image_format);*/
void write_image(char *filename,FmiImage *img,FmiImageFormat format);
void read_image(char *filename,FmiImage *img);
void image_info(FmiImage *img);

void read_pnm_image(FILE *fp,FmiImage *img,FmiImageFormat format);
void write_pnm_image(FILE *fp,FmiImage *img,FmiImageFormat format);


/* BASIC TRANSFORMS (further tricks in other *.c files) */
void extract_channel(FmiImage *source,int channel,FmiImage *target);
void write_channel(FmiImage *source,int channel,FmiImage *target);

/* TRANSFORMS */
/* intensity */
void expand_channel_to_rgb(FmiImage *source,int channel,FmiImage *target);
void map_channel_to_colors(FmiImage *source,int channel,FmiImage *target,int map_size,ColorMap map);
void map_channel_to_256_colors(FmiImage *source,int channel,FmiImage *target,ColorMap256 map);
void map_256_colors_to_gray(FmiImage *source,FmiImage *target,ColorMap256 map);
void read_colormap256(char *filename,ColorMap256 map);
/* int to_rgb(FmiImage *source,FmiImage *target); */


/*! coordinates */
void to_cart(FmiImage *source,FmiImage *target,Byte outside_fill);



/* filters */
/*  typedef struct {int width;int height;int *array;} Mask; */
/*
  typedef struct {int width;int height;char *name;} Mask;
  Mask mask_speck1;
  typedef struct {int width;int height;int array[3][3];} Mask;
  char *stri;
*/

void calc_histogram(FmiImage *source,Histogram hist);
void clear_histogram(Histogram hist);
void write_histogram(char *filename,Histogram hist);
void dump_histogram(Histogram hist);
/*initialize_vert_stripe */
int initialize_vert_stripe(FmiImage *img,int height);


#endif
