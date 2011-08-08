/**

    Copyright 2001 - 2010  Markus Peura,
    Finnish Meteorological Institute (First.Last@fmi.fi)
    Copyright 2011 Martin Raspaud, SMHI <martin.raspaud@smhi.se>


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


#ifndef __FMI_RADAR_IMAGE__
#define __FMI_RADAR_IMAGE__
#define FMI_RADAR_IMAGE_VER "fmi_radar_image\t v2.3\t Jan 2002 (c) Markus.Peura@fmi.fi"

typedef signed char Dbz;

#include "fmi_image.h"
#include "fmi_radar_codes.h"


#define FMI_RADAR_SWEEP_COUNT 20  // ACTUALLY: MAXIMUM  2009

extern float fmi_radar_sweep_angles[FMI_RADAR_SWEEP_COUNT];

/* defaults */
#define FMI_RADAR_BIN_DEPTH 500.0 // 500m
#define FMI_RADAR_BIN_COUNT 500   // 500 pixels
//#define FMI_RADAR_RAY_COUNT 360

#define EARTH_RADIUS43 (EARTH_RADIUS*4/3) // 500m

// TRANSFORMS
// intensity transforms
int abs_dbz_to_byte(Dbz); 
int rel_dbz_to_byte(Dbz); 
int abs_dbz_to_int(Dbz); 
int rel_dbz_to_int(Dbz); 

int byte_to_abs_dbz(Byte);
int byte_to_rel_dbz(Byte);
// geometric transforms
int bin_to_metre(int bin);
int metre_to_bin(int metres);

int bin_to_altitude(int sweep_bin,float sweep_angle); // metres
int altitude_to_bin(int metres,float sweep_angle); 

int sweep_to_ground(int sweep_bin,float sweep_angle);
int ground_to_bin(int ground_metres,float sweep_angle);

int bin_to_bin(int sweep_bin,float sweep_angle,float target_sweep_angle);

int bin_altitude(int sweep_bin,float sweep_angle);
int altitude_to_altitude(int altitude,float sweep_angle);

void xy_to_polar(int i,int j,int *theta,int *radius);

void volume_to_cappi(FmiImage *volume,int altitude,FmiImage *cappi); // metres

//void detect_ground_echo(FmiImage *source,int ppi_count,FmiImage *prob,int intensity_diff,int half_altitude);
void detect_ground_echo_mingrad(FmiImage *source,int ppi_count,FmiImage *prob,int intensity_diff,int half_altitude);
void detect_ground_echo_minnetgrad(FmiImage *source,int ppi_count,FmiImage *prob,int intensity_diff,int half_altitude);

void detect_emitters(FmiImage *source,FmiImage *trace,int min_intensity,int min_length);
void detect_emitters2(FmiImage *source,FmiImage *trace,int min_intensity,int min_length,int max_width);

void detect_emitters2old(FmiImage *source,FmiImage *trace,int min_intensity,int min_length,int max_width);

//void detect_emitters4(FmiImage *source,FmiImage *trace,int min_intensity,int min_length,int max_width);

//void detect_emitters5(FmiImage *source,FmiImage *trace,int min_intensity,int min_length,int max_width);

void enhance_horz_lines2(FmiImage *trace,int weight);
void enhance_vert_lines(FmiImage *trace,int weight);


//initialize_vert_stripe
int initialize_vert_stripe(FmiImage *img,int height);
void initialize_histogram(FmiImage *source,Histogram histogram,
		int hrad,int vrad,int i,int j,int (* hist_func)(Histogram));

/* ANOMALIA FILTERING */
/* delete segments with maxtop < intensity */
void detect_ships(FmiImage *source,FmiImage *prob,int min_intensity,int min_area);
//void remove_ships2(FmiImage *source,int min_intensity,int min_area);

// geom. correction
//void compensate_radially(FmiImage *image,int slope);
void distance_compensation_mul(FmiImage *image,int slope);
void distance_compensation_div(FmiImage *image,int slope);


/* remove distant emitter/solar disturbance lines */ 
void remove_thin_horz_lines(FmiImage *target,int min_elevation,int weight);
void remove_horz_lines(FmiImage *target,int min_length,int min_elevation,int weight);
//void remove_thin_horz_lines(FmiImage *target,int min_elevation,int weight);

void detect_insect_band(FmiImage *source,FmiImage *prob,int start_intensity,int radius,int weight);

void detect_biomet(FmiImage *source,FmiImage *prob,int intensity_max,int intensity_delta,int altitude_max,int altitude_delta);

//void detect_sun(FmiImage *source,FmiImage *trace,int min_intensity,int typical_width);
void detect_sun(FmiImage *source,FmiImage *trace,int min_intensity,int min_length,int typical_width);

void detect_sun2(FmiImage *source,FmiImage *trace,int min_intensity,int min_length,int typical_width,int azimuth,int elevation);

void detect_too_warm(FmiImage *source,FmiImage *prob,FmiImage *meteosat,Celsius c50,Celsius c75,int min_intensity,int min_size);

void detect_doppler_anomaly(FmiImage *source,FmiImage *target,int width, int height,int threshold);



/* CONVERSIONS */


/* display and visualization */
//int byte_to_rgb(FmiImage *source,FmiImage *target); 
void pgm_to_ppm_radar(FmiImage *source,FmiImage *target);
void pgm_to_ppm_radar_iris(FmiImage *source,FmiImage *target);
void pgm_to_redgreen(FmiImage *source,FmiImage *target);
void pgm_to_pgm_print(FmiImage *source,FmiImage *target);
void pgm_to_pgm_print2(FmiImage *source,FmiImage *target);


/* ODDITIES */
int histogram_dominate_anaprop(Histogram h);
void virtual_rhi(FmiImage *volume);
void gradient_rgb(FmiImage *volume);

#endif
