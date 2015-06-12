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
 * @date 2015-05-11
 */
#ifndef SCANSUN_H
#define SCANSUN_H
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "rave_debug.h"
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

#define HEIGMIN1   (8.0)          /*Minimum height for first analyses in km.*/
#define HEIGMIN2   (4.0)          /*Minimum height for second analyses in km.*/
#define RAYMIN     (20.0)         /*Minimum length of rays for analyses in km.*/
#define FRACDATA   (0.70)         /*Fraction of rangebins passing the analyses.*/
#define DBDIFX     (2.0)          /*Maximum dev. of power from 1st fit in dB.*/
#define ANGLEDIF   (5.0)          /*Maximum dev. from calculated sun in deg.*/
#define GASATTN    (0.016)        /*Two-way gaseous attenuation in dB/km.*/
#define CWIDTH     (1.2)          /*Factor between bandwidth and 1/pulselength.*/

/******************************************************************************/
/*Definition of radar parameters for solar analysis:                          */
/******************************************************************************/

#define WAVELENGTH (0.053)        /*Default wavelength in meters*/
#define ANTVEL     (18.0)         /*Default antenna velocity in deg/s*/
#define ANTGAIN    (45.0)         /*Default antenna gain in dB*/
#define ASTART     (0.0)          /*Default azimuthal offset from 0 in degrees*/
#define PULSEWIDTH (2.0)          /*Default pulse width in microseconds*/
#define RXLOSS     (0.0)          /*Default receiver chain loss in dB*/
#define RADCNST    (64.0)         /*Super-arbitrary default radar constant*/

#define ONEPOL     (3.01)         /*Losses due to one-pol RX only in dB.*/
#define AVGATTN    (1.39)         /*Averaging and overlap losses in dB for 1deg.*/

/******************************************************************************/
/*Definition of standard parameters.                                          */
/******************************************************************************/

#define DEG2RAD    (0.017453293)  /*Degrees to radians.*/
#define RAD2DEG    (57.29578)     /*Radians to degrees.*/
#define LSTR       (128)          /*Length of all strings used.*/
#define NELEVX     (64)           /*Maximum number of elevations.*/
#define RADIUS43   (8495.0)       /*Earth radius used for height calculations.*/

/******************************************************************************/
/*Definition of parameters for refraction correction.                         */
/******************************************************************************/

#define REFPRES    (1013.0)       /*Pressure for refraction correction in hPa.*/
#define REFTEMP    (288.15)       /*Temperature for refraction correction in K.*/

/**
 * Different value types for whether ZDR is its own quantity or must be calculated.
 */
typedef enum ZdrType {
  ZdrType_None = 0,      /**< Non-existant */
  ZdrType_READ = 1,      /**< ZDR is in its own quantity and can be read as such */
  ZdrType_CALCULATE = 2  /**< Vertical reflectivity is read and ZDR needs to be calculated */
} ZdrType;

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
	double astart;         /*Azimuthal offset in degrees from 0*/
	double pulse;          /*Pulse length in microsec.*/
	double radcnst;        /*Radar constant in dB.*/
	double antvel;         /*Antenna velocity in deg/s.*/
  double lon;            /*Longitude of radar in deg.*/
  double lat;            /*Latitude of radar in deg.*/
  double RXLoss;         /*Total losses between reference and feed (dB).*/
  double AntGain;        /*Antenna gain (dB), will end up being linearized*/
  double AntArea;        /*Effective antenna area (m2)*/
  double wavelength;     /*Wavelength in meters*/
  const char* quant1;    /*what/quantity for the given scan's parameter, either TH or DBZH*/
  const char* quant2;    /*ZDR or either TV or DBZV*/
  ZdrType Zdr;           /*Flag used to note presence of ZDR, either read or calculated*/
};
typedef struct scanmeta SCANMETA;

/*
 * Structure for containing output values:
 */
struct rvals {
	long date;        /* Date of scan data in YYYYMMDD */
	long time;        /* Time of scan data in HHMMSS */
	double Elev;      /* Elevation of scan in deg. */
	double Azimuth;   /* Azimuth of scan in deg. */
	double ElevSun;   /* Elevation angle of the sun in deg. */
	double AzimSun;   /* Azimuth angle of the sun in deg. */
	double SunMean;   /* Sun's reflectivity in dBm */
	double SunStdd;   /* Standard deviation of the sun's reflectivity in dBm */
	double dBSunFlux; /* Sun flux in dB */
	double ZdrMean;   /* Zdr mean */
	double ZdrStdd;   /* Zdr standard deviation */
  char* quant1;     /* what/quantity for the given scan's parameter: TH, DBZH, TV, or DBZV */
  char* quant2;     /* ZDR or either TV or DBZV */
	double RelevSun;  /* Refraction-corrected (perceived) elevation angle of the sun in deg. */
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
 * Returns a double attribute array from a polar scan object.
 * @param[in] obj - a polar scan
 * @param[in] aname - a string of the attribute to retrieve
 * @param[in] array - the double array to retrieve
 * @returns 1 on success or 0 if the attribute doesn't exist
 */
int getDoubleArrayAttribute(PolarScan_t* scan, const char* aname, double** array);

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
 * Finds sun hits in reflectivity data.
 * @param[in] scan - polar scan object
 * @param[in] meta - internal metadata structure
 * @param[in] list - RAVE list object containing hits
 */
void processData(PolarScan_t* scan, SCANMETA* meta, RaveList_t* list);

/**
 * Helper function that calls @ref processReflectivity for each of a number of
 * given parameters/quantities. Called internally by @ref scansun.
 * @param[in] scan - polar scan object
 * @param[in] meta - internal metadata structure
 * @param[in] list - RAVE list object containing hits
  */
void processScan(PolarScan_t* scan, SCANMETA* meta, RaveList_t* list);

/**
 * Debug function that writes metadata to file.
 * @param[in] meta - internal metadata structure
 */
void outputMeta(SCANMETA* meta);

/**
 * Masterminds the scanning of polar data and determination of sun hits.
 * @param[in] filename - string containing the name (and path if somewhere else) of the file to process
 * @param[out] list - RaveList_t object for holding one or more sets of return values
 * @param[out] source - string containing the value of /what/source
 * @returns 1 upon success, otherwise 0
 */
int scansun(const char* filename, RaveList_t* list, char** source);

#endif
