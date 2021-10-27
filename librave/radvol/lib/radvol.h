/* --------------------------------------------------------------------
Copyright (C) 2012 Institute of Meteorology and Water Management -
National Research Institute, IMGW-PIB

This file is part of Radvol-QC package.

Radvol-QC is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Radvol-QC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Radvol-QC.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/
/**
 * Radvol-QC general structures and algorithms.
 * @file radvol.h
 * @author Katarzyna Osrodka (Institute of Meteorology and Water Management, IMGW-PIB)
 * @date 2012-12-20
 */

#ifndef RADVOL_H
#define RADVOL_H
#include "rave_object.h"
#include "rave_simplexml.h"
#include "polarvolume.h"
#include "polarscan.h"
#include <math.h>

/** the best QI value */
#define QI_GOOD 1.0
/** the worst QI value */
#define QI_BAD 0.0
/** value for double no data */
#define cNull 9999.0
/** equivalent earth's radius [km] */
#define cEer 8493
/** Radius at the equator */
#define DEFAULT_EQUATOR_RADIUS 6378160.0
/** Radius to the poles */
#define DEFAULT_POLE_RADIUS 6356780.0

/**
 * Represents an elevation
 */
typedef struct Elevation_t {
    int nbin;         /**< number of bins */
    int nray;         /**< number of rays */
    double rscale;    /**< resolution of bins along the ray [km] */
    double elangle;   /**< elevation angle [rad] */
    double gain;      /**< gain */
    double offset;    /**< offset */
    double nodata;    /**< nodata */
    double undetect;  /**< undetect */
    double* ReflElev; /**< reflectivity data */
    double* QIElev;   /**< quality data */
}Elevation_t;

/**
 * Represents the Radvol
 */
struct _Radvol_t {
  RAVE_OBJECT_HEAD      /** Always on top */
  Elevation_t *TabElev; /**< elevation data */
  int nele;             /**< number of elevations in TabElev*/
  double beamwidth;     /**< ray width [rad] */
  double wavelength;    /**< length of wave [cm] */
  double pulselength;   /**< half of radar pulse length */
  double Eer;           /**< equivalent earth's radius [km] */
  int altitude;         /**< altitude of antenna */
  char* name;           /**< radar name what->source->NOD */
  char* task_name;      /**< task name to be saved in *.h5 file */
  char* task_args;      /**< task arguments to be saved in *.h5 file */
  int QIOn;             /**< 1 if QI is calculated, 0 otherwise */
  int QCOn;             /**< 1 if QC is on, 0 otherwise */
  int DBZHtoTH;         /**< 1 if to copy unprocessed DBZH into TH if TH does not exist, 0 otherwise */
};

/**
 * Represents argument parameters for Radvol's algorithms.
 * See each algorithm's documentation.
 */
struct _Radvol_params_t {
  int    DBZHtoTH;
  int    BROAD_QIOn;
  int    BROAD_QCOn;
  double BROAD_LhQI1;
  double BROAD_LhQI0;
  double BROAD_LvQI1;
  double BROAD_LvQI0;
  double BROAD_Pulse;
  int    SPIKE_QIOn;
  int    SPIKE_QCOn;
  double SPIKE_QI;
  double SPIKE_QIUn;
  double SPIKE_ACovFrac;
  int    SPIKE_AAzim;
  int    SPIKE_AVarAzim;
  int    SPIKE_ABeam;
  int    SPIKE_AVarBeam;
  double SPIKE_AFrac;
  double SPIKE_BDiff;
  int    SPIKE_BAzim;
  double SPIKE_BFrac;
  int    NMET_QIOn;
  int    NMET_QCOn;
  double NMET_QI;
  double NMET_QIUn;
  double NMET_AReflMin;
  double NMET_AReflMax;
  double NMET_AAltMin;
  double NMET_AAltMax;
  double NMET_ADet;
  double NMET_BAlt;
  int    SPECK_QIOn;
  int    SPECK_QCOn;
  double SPECK_QI;
  double SPECK_QIUn;
  double SPECK_AGrid;
  double SPECK_ANum;
  double SPECK_AStep;
  double SPECK_BGrid;
  double SPECK_BNum;
  double SPECK_BStep;
  int    BLOCK_QIOn;
  int    BLOCK_QCOn;
  double BLOCK_MaxElev;
  double BLOCK_dBLim;
  double BLOCK_GCQI;
  double BLOCK_GCQIUn;
  double BLOCK_GCMinPbb;
  double BLOCK_PBBQIUn;
  double BLOCK_PBBMax;
  double ATT_a;
  double ATT_b;
  int    ATT_QIOn;
  int    ATT_QCOn;
  double ATT_ZRa;
  double ATT_ZRb;
  double ATT_QIUn;
  double ATT_QI1;
  double ATT_QI0;
  double ATT_Refl;
  double ATT_Last;
  double ATT_Sum;
};

typedef struct _Radvol_t Radvol_t;
typedef struct _Radvol_params_t Radvol_params_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType Radvol_TYPE;

/**
 * Reads radar node name (NOD) into self->name
 * @param self - self
 * @param source - what/source from PVOL or SCAN
 */
void Radvol_getName(Radvol_t* self, const char* source);

/**
 * Reads attribute value from scan
 * @param scan - polar scan
 * @param name - attribute name
 * @param value - read value
 * @returns 1 on success, otherwise 0
 */
int Radvol_getAttrDouble_scan(PolarScan_t* scan, char* name, double* value);

/**
 * Reads attribute value from volume
 * @param pvol - polar volume
 * @param name - attribute name
 * @param value - read value
 * @returns 1 on success, otherwise 0
 */
int Radvol_getAttrDouble_pvol(PolarVolume_t* pvol, char* name, double* value);

/**
 * Sets name of task
 * @param self - self
 * @param task_name - name of task
 * @returns 1 on success, otherwise 0 
 */
int Radvol_setTaskName(Radvol_t* self, const char* task_name);

/**
 * Sets arguments of task
 * @param self - self
 * @param task_args - task arguments
 * @returns 1 on success, otherwise 0
 */
int Radvol_setTaskArgs(Radvol_t* self, const char* task_args);

/**
 * Reads polar scan into radvolqc structure
 * @param self - self
 * @param scan - polar scan
 * @returns 1 on success, otherwise 0
 */
int Radvol_load_scan(Radvol_t* self, PolarScan_t* scan);

/**
 * Reads polar volume into radvolqc structure
 * @param self - self
 * @param pvol - polar volume
 * @returns 1 on success, otherwise 0
 */
int Radvol_load_pvol(Radvol_t* self, PolarVolume_t* pvol);

/**
 * Writes data from radvolqc into polar scan
 * @param self - self
 * @param scan - polar scan
 * @returns 1 on success, otherwise 0
 */
int Radvol_save_scan(Radvol_t* self, PolarScan_t* scan);

/**
 * Writes data from radvolqc into polar volume
 * @param self - self
 * @param pvol - polar volume
 * @returns 1 on success, otherwise 0
 */
int Radvol_save_pvol(Radvol_t* self, PolarVolume_t* pvol);

/**
 * Reads xml child for a specific radar and factor/algorithm from xml file
 * @param self - self
 * @param aFileName - xml filename
 * @param aFactorName - factor/algorithm name
 * @param IsDefault - 1 if returned child is default, 0 otherwise
 * @returns related or default xml child on success, NULL otherwise
 */
SimpleXmlNode_t *Radvol_getFactorChild(Radvol_t* self, char* aFileName, char* aFactorName, int* IsDefault);

/**
 * Returns value of a specific parameter as double from xml child
 * @param node - xml child
 * @param aParamName - parameter name
 * @param value - parameter value
 * @returns 1 on success, otherwise 0
 */
int Radvol_getParValueDouble(SimpleXmlNode_t* node, char* aParamName, double* value);

/**
 * Returns value of a specific parameter as int from xml child
 * @param node - xml child
 * @param aParamName - parameter name
 * @param value - parameter value
 * @returns 1 on success, otherwise 0
 */
int Radvol_getParValueInt( SimpleXmlNode_t* node, char* aParamName, int* value); 

/**
 * Estimates equivalent Earth radius based on radar site latitude
 * @param self - self
 * @param lat - radar site latitude
 */
void Radvol_setEquivalentEarthRadius(Radvol_t* self, double lat);

/**
 * Returns height of a particular bin in the scan resulting from Earth curvature
 * @param self - self
 * @param ele - elevation number
 * @param aBin - bin number
 * @returns altitude [km]
 */
double Radvol_getCurvature(Radvol_t* self, int ele, int aBin);

/**
 * Returns quality index value for linear relationship
 * @param x - quality factor value
 * @param a - lower threshold in linear relationship
 * @param b - upper threshold in linear relationship
 * @returns quality index value
 */
double Radvol_getLinearQuality(double x, double a, double b);

#endif

/** 1 if two values are the same (max acceptable difference 0.001), 0 otherwise */
#ifndef SameValue
  #define SameValue( a, b)  ( (fabs(a - b) < 0.001) ? 1 : 0)
#endif

