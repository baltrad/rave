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


/* THIS LIBRARY CONTAINS IMAGE PROCESSING OPERATIONS FOR GENERAL PURPOSE */
/* radar images, satellite images why not ... */

// #define UNVISITED 255
#define VISITED 1

/*
  #include "fmi_image_filter_speck.h"
  #include "fmi_image_filter_line.h"
  #include "fmi_image_filter_texture.h"
*/

void detect_horz_gradient(FmiImage *source,FmiImage *trace);
void detect_vert_gradient(FmiImage *source,FmiImage *trace);

/* LOCAL MAXIMA */
void detect_horz_maxima(FmiImage *source,FmiImage *trace);
void detect_horz_maxima2(FmiImage *source,FmiImage *trace); // normalized
void detect_horz_edges(FmiImage *source,FmiImage *trace);

/// Finds local maxima, magnitude defined as the lower difference to neighbouring intensities.
void detect_vert_maxima(FmiImage *source,FmiImage *trace); // MP changed 2010: zeros also written.

void detect_vert_maxima2(FmiImage *source,FmiImage *trace);
void detect_vert_edges(FmiImage *source,FmiImage *trace);

/* INFINITE INPULSE RESPONSE FILTERS */
/* simple */
void iir_left(FmiImage *source,FmiImage *trace,int promille);
void iir_right(FmiImage *source,FmiImage *trace,int promille);
void iir_up(FmiImage *source,FmiImage *trace,int promille);
void iir_down(FmiImage *source,FmiImage *trace,int promille);

/* In source, change pixels (i,j) with mask(i,j) < threshold to c */
void mask_image(FmiImage *source,FmiImage *mask,Byte threshold,Byte c);

void threshold_image(FmiImage *source,FmiImage *target,Byte threshold);
void binarize_image(FmiImage *source,FmiImage *target,Byte threshold);
//void high_boost(FmiImage *source,FmiImage *target,Byte threshold);

/* Propagate = spread MAX information within a speck or segment */
/*  source = information, as byte values; or NULL -> byte=0  */
/*  domain = area {(x,y)|f(x,y)>0} inside which information is spread */
/*  target = target image */
/*  slope  = slope for increasing/decreasing byte value */
/* Typically, threshold is applied first. */

void propagate_right(FmiImage *source,FmiImage *domain,FmiImage *target, signed char slope,void (* put_func)(FmiImage *,int,int,int,Byte));
void propagate_left(FmiImage *source,FmiImage *domain,FmiImage *target, signed char slope,void (* put_func)(FmiImage *,int,int,int,Byte));
void propagate_horz(FmiImage *source,FmiImage *domain,FmiImage *target, signed char slope,void (* put_func)(FmiImage *,int,int,int,Byte));
void propagate_up(FmiImage *source,FmiImage *domain,FmiImage *target, signed char slope,void (* put_func)(FmiImage *,int,int,int,Byte));
void propagate_down(FmiImage *source,FmiImage *domain,FmiImage *target, signed char slope,void (* put_func)(FmiImage *,int,int,int,Byte));

void horz_seg_lengths(FmiImage *source,FmiImage *target);
void vert_seg_lengths(FmiImage *source,FmiImage *target);

void row_statistics(FmiImage *source,Byte *nonzero,Byte *sum,Byte *sum2);
void col_statistics(FmiImage *source,Byte *nonzero,Byte *sum,Byte *sum2);
