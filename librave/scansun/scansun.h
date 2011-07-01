/* --------------------------------------------------------------------
Copyright (C) 2010 Royal Netherlands Meteorological Institute, KNMI and
                   Swedish Meteorological and Hydrological Institute, SMHI,

This is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with HLHDF.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/

/** Header file for KNMI's sun scanning functionality
 * @file
 * @author Original algorithm and code: Iwan Holleman, KNMI, and Integration: Daniel Michelson, SMHI
 * @date 2010-10-29
 */
#ifndef SCANSUN_H
#define SCANSUN_H
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "rave_io.h"
#include "polarscan.h"
#include "polarvolume.h"
#include "rave_object.h"
#include "rave_types.h"
#include "rave_attribute.h"
#include "rave_alloc.h"
#include "rave_list.h"

/******************************************************************************/
/*Definition of analysis parameters.                                          */
/******************************************************************************/

#define HEIGMIN   (6.0)           /*Minimum height for analyses in km.*/
#define RANGMIN   (50.0)          /*Minimum range for analyses in km.*/
#define FRACDATA  (0.90)          /*Fraction of data above threshold in ray.*/
#define ANGLEDIF  (2.0)           /*Maximum dev. from calculated sun in deg.*/ 
#define GASATTN   (0.016)         /*Two-way gaseous attenuation in dB/km.*/
#define CWIDTH    (1.2)           /*Factor between bandwidth and 1/pulselength.*/

/******************************************************************************/
/*Definition of standard parameters.                                          */
/******************************************************************************/

#define DEG2RAD    DEG_TO_RAD     /*Degrees to radians. From PROJ.4*/
#define RAD2DEG    RAD_TO_DEG     /*Radians to degrees. From PROJ.4*/
#define LSTR       (128)          /*Length of all strings used.*/
#define NELEVX     (64)           /*Maximum number of elevations.*/
#define RADIUS43   (8495.0)       /*Earth radius used for height calculations.*/

/******************************************************************************/
/*Definition of parameters for refraction correction.                         */
/******************************************************************************/

#define REFPRES    (1013.0)       /*Pressure for refraction correction in hPa.*/
#define REFTEMP    (288.15)       /*Temperature for refraction correction in K.*/

/******************************************************************************/
/*Structure for containing SCAN metadata:                                     */
/******************************************************************************/

struct scanmeta {
	long date;             /**<Date of scan data in YYYYMMDD.*/
	long time;             /**<Time of scan data in HHMMSS.*/
	double elev;           /*Elevation of scan in deg.*/
	long nrang;            /*Number of range bins in scan.*/
	long nazim;            /*Number of azimuth rays in scan.*/
	double rscale;         /*Size of range bins in scan in km.*/
	double ascale;         /*Size of azimuth steps in scan in deg.*/
	long azim0;            /*Ray number with which radar scan started.*/
	double zoffset;        /*Offset value of quantity contained by scan.*/
	double zscale;         /*Scale of value of quantity contained by scan.*/
	double nodata;         /*Nodata value of quantity contained by scan.*/
	double PRFh;           /*High PRF used for scan in Hz.*/
	double PRFl;           /*Low PRF used for scan in Hz.*/
	double pulse;          /*Pulse length in microsec.*/
	double radcnst;        /*Radar constant in dB.*/
	double txnom;          /*Nominal maximum TX power in kW.*/
	double antvel;         /*Antenna velocity in deg/s.*/
};
typedef struct scanmeta SCANMETA;

/*
 * Structure for containing output values:
 */
struct rvals {
	long date;				/* Date of scan data in YYYYMMDD */
	long time;				/* Time of scan data in HHMMSS */
	double Elev;			/* Elevation of scan in deg. */
	double Azimuth;			/* Azimuth of scan in deg. */
	double ElevSun;			/* Elevation angle of the sun in deg. */
	double AzimSun;         /* Azimuth angle of the sun in deg. */
	double dBmSun;          /* Sun's reflectivity in dBm */
	double dBmStdd;         /* Standard deviation of the sun's reflectivity in dBm */
	double RelevSun;		/* Refraction-corrected (perceived) elevation angle of the sun in deg. */
};
typedef struct rvals RVALS;

/******************************************************************************/
/*Prototypes of local functions:                                              */
/******************************************************************************/

/**
 * Returns a double attribute value from any polar object type.
 * @param[in] obj - a polar volume, scan, or scan parameter
 * @param[in] aname - a string of the attribute to retrieve
 * @param[in] tmpd - the double value to retrieve
 * @returns 1 on success or 0 if the attribute doesn't exist
 */
int getDoubleAttribute(RaveCoreObject* obj, const char* aname, double* tmpd);

/**
 * Reads metadata into the SCANMETA structure from volume, scan, param.
 * @param[in] scan - PolarScan_t object containing the sweep
 * @param[in] dbzh - PolarScanParam_t object containing the sweep's parameter
 * @param[in] meta - SCANMETA struct that will be filled with metadata
 * @returns 1 on success, otherwise 0
 */
int fill_meta(PolarScan_t* scan, PolarScanParam_t* dbzh, SCANMETA *meta);

/**
 * This function calculates the height and range from the Radar corresponding
 * to a point with a known elevation and on-ground distance from the Radar.
 * The formulae used are exact. For details see lab book.
 * @param[in] elev - Elevation angle as a double
 * @param[in] heig - Height (above the radar) in meters
 * @returns Range from the radar in meters, as a double
 */
double ElevHeig2Rang(double elev,float heig);

/**
 * This function returns the date/time of (date1,time1) and 'ss' later. The
 * date should be given in yyyymmhh format and the time in hhmmss format.
 * The time lapse between the input and output date/time is given in seconds.
 * @param[in] date1 - long int of the input date
 * @param[in] time1 - long int of the input time
 * @param[in] date2 - long int of the output date
 * @param[in] time2 - long int of the output time
 * @returns nothing
 */
void datetime(long date1, long time1, long ss, long *date2, long *time2);

/**
 * This function calculates the refraction correction of the solar position.
 * The function uses the true elevation, i.e., the calculated elevation of the
 * sun in degrees.
 * Reference: D. Sonntag, 1989, Abh. des Met. Dienstes der DDR Nr 143: Formeln
 * verschiedenen Genauigkeitsgrades zur Berechnung der Sonnenkoordinaten.
 * @param[in] elev - The sun's real elevation angle.
 * @returns a double containing the value of the refracted (perceived) elevation angle.
 */
double refraction(double* elev);

/**
 * This function calculates the solar elevation and azimuth using the
 * geographical position, date, and time. The equations and constants are taken
 * from the WMO guide on Meteorological Instruments and Methods of Observations
 * (CIMO, WMO no. 8), annex 7.D. The equations have been slightly modified and
 * extended to include the calculation of both the sine and cosine of the
 * azimuth.
 * Modified slightly further to include the refracted (perceived) elevation angle.
 * @param[in] lon - double containing the longitude position
 * @param[in] lat - double containing the latitude position
 * @param[in] yyyymmdd - year-month-day as a long
 * @param[in] hhmmss - hour-minute-second as a long
 * @param[out] elev - elevation angle above the horizon in degrees, as a pointer to a double
 * @param[out] azim - azimuth angle clockwise from true north, as a pointer to a double
 * @param[out] relev - refracted elevation angle, based on elev, as a pointer to a double
 * @returns nothing
 */
void solar_elev_azim(double lon, double lat, long yyyymmdd, long hhmmss, double *elev, double *azim, double *relev);

/**
 * Gregorian Calender adopted on October 15. 1582.
 * In this routine date2julday returns the Julian Day Number that begins at
 * noon of the calendar date specified by month mm, day id, and year iyyy, all
 * integer variables. Positive year signifies A.D.; negative, B.C. Remember that
 * the year after 1 B.C. was 1 A.D.
 * @param[in] yyyymmdd - year-month-day as a long
 * @returns the Julian day (of the year) as a long
 */
long date2julday(long yyyymmdd);

/**
 * Inverse of the function date2julday given above. Here julian is input as a
 * Julian Day Number, and the routine outputs mm,id, and iyyy as the month,
 * day, and year on which the specified Julian Day started at noon.
 * @param[in] julian - the Julian day (of the year) as a long
 * @returns the day of the year in YYYYMMDD (year-month-day) format, as a long
*/
long julday2date(long julian);

/**
 * @param[in] filename - string containing the name (and path if somewhere else) of the file to process
 * @param[out] list - RaveList_t object for holding one or more sets of return values
 * @param[out] source - string containing the value of /what/source
 * @returns 1 upon success, otherwise 0
 */
int scansun(const char* filename, RaveList_t* list, char** source);

#endif
