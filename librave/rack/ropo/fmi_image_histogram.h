/**

    Copyright 2001 - 2010  Seppo Pulkkinen & Markus Peura, 
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


/* HISTOGRAM as such defined already in fmi_image */

/* convolution windows */
/* EI TOIMI */
void convolve(FmiImage *source,FmiImage *target,int mask_width,int mask_height,int **mask,int divisor);


void clear_histogram_full(Histogram hist);


/* histogram windows */
void initialize_histogram_int(FmiImage *source,Histogram h,int hrad,int vrad,int i,int j,void (* hist_func)(Histogram));
void    up(FmiImage *source,Histogram h,int hrad,int vrad,int *i,int *j);
void  down(FmiImage *source,Histogram h,int hrad,int vrad,int *i,int *j);
void  left(FmiImage *source,Histogram h,int hrad,int vrad,int *i,int *j);
void right(FmiImage *source,Histogram h,int hrad,int vrad,int *i,int *j);

Histogram histogram_weights;
Histogram histogram_sine;
Histogram histogram_cosine;
//void initialize_histogram_trigon();

int histogram_sample_count; //  histogram_median2
int histogram_threshold; //  histogram_variance_rot

FmiImage *histogram_weight_image;

int histogram_median_biased(Histogram h,int count);
int histogram_median_biased_top(Histogram h,int count);

int histogram_size(Histogram h);
int histogram_sum(Histogram h);
int histogram_area(Histogram h);
int histogram_area_inv255(Histogram h);
int histogram_area2(Histogram h); /* sigmoid-moderated */
int histogram_area2_inv255(Histogram h); /* sigmoid-moderated */
int histogram_perimeter(Histogram h);
int histogram_perimeter2(Histogram h);
//int histogram_perimeter_normalized(Histogram h);
int histogram_compactness(Histogram h);
int histogram_min(Histogram h);
int histogram_max(Histogram h);
int histogram_range(Histogram h);
int histogram_median(Histogram h); /* dont use this! recalcs "count"*/

int histogram_median2(Histogram h); /* use this! with precalc'd "count"*/
int histogram_median2_top(Histogram h);
int histogram_mean(Histogram h);
int histogram_mean2(Histogram h);   /* use this! with precalc'd "count"*/
int histogram_mean_nonzero(Histogram h);
//Histogram histogram_weighted_mean2_weights;
//int histogram_weighted_mean(Histogram h,Histogram weights);
//int histogram_weighted_mean2(Histogram h);
int histogram_mean_weighted(Histogram h);
int (* histogram_mean_weighted_pyramid)(Histogram h);

int histogram_variance_rot(Histogram h);

int histogram_dom(Histogram h);
int histogram_dom_nonzero(Histogram h);
int histogram_principal_component_ratio(Histogram h);
int histogram_smoothness(Histogram h);

void histogram_dump_stats(Histogram h);
void histogram_dump_nonzero(Histogram h);

int histogram_meanX(Histogram h);
int histogram_meanY(Histogram h);

int histogram_scaling_parameter;
int (* histogram_scaling_function)(int param, int value);
int histogram_semisigmoid(int a, int x);
int histogram_semisigmoid_inv(int a, int x);

void pipeline_process(FmiImage *source,FmiImage *target,int horz_rad,int vert_rad,int (* histogram_function)(Histogram));
