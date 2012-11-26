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
 * @date 2012-07-12
 */

#ifndef RADVOL_H
#define RADVOL_H
#include "rave_object.h"
#include "rave_simplexml.h"
#include "polarvolume.h"

/** the best QI value */
#define QI_GOOD	1.0
/** the worst QI value */
#define QI_BAD 0.0
/** value for no rain */
#define cNoRain -32.0   
/** value for no data */
#define cNull 9999.0
/** equivalent earth's radius [km] */
#define cEer 8493
/** equivalent earth's radius squared [km] */
#define cEer2 8493*8493

/**
 * Represents an elevation
 */
typedef struct Elevation_t {
    int nbin		/**< number of bins */;
    int nray		/**< number of rays */;
    double rscale	/**< resolution of bins along the ray [km] */;
    double elangle	/**< ray width [rad] */;
    double gain		/**< gain */;
    double offset	/**< offset */;
    double nodata	/**< nodata */;
    double undetect	/**< undetect */;
    double* ReflElev	/**< reflectivity data */;
    double* QIElev	/**< quality data */;
}Elevation_t;

/**
 * Represents the Radvol
 */
struct _Radvol_t {
  RAVE_OBJECT_HEAD	/** Always on top */
  Elevation_t *TabElev;	/**< elevation data */
  int nele;		/**< number of elevations in TabElev*/
  double beam;		/**< ray width [rad] */
  int altitude;		/**< altitude of antenna */
  char* task_name;	/**< task name to be saved in *.h5 file */
  char* task_args;	/**< task arguments to be saved in *.h5 file */
  int QIOn;		/**< 1 if QI is calculated, 0 otherwise */
  int QCOn;		/**< 1 if QC is on, 0 otherwise */
};

typedef struct _Radvol_t Radvol_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType Radvol_TYPE;

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
 * Reads polar volume into radvolqc structure
 * @param self - self
 * @param pvol - polar volume
 * @returns 1 on success, otherwise 0
 */
int Radvol_loadVol(Radvol_t* self, PolarVolume_t* pvol);

/**
 * Writes data from radvolqc into polar volume
 * @param self - self
 * @param pvol - polar volume
 * @returns 1 on success, otherwise 0
 */
int Radvol_saveVol(Radvol_t* self, PolarVolume_t* pvol);

/**
 * Reads xml child for a specific factor/algorithm from xml file 
 * @param aFileName - xml filename
 * @param aFactorName - factor/algorithm name
 * @returns related xml child on success, NULL otherwise 
 */
SimpleXmlNode_t *Radvol_getFactorChild( char* aFileName, char* aFactorName);

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
 * Returns altitude of a particular bin in elevation
 * @param aElev - elevation
 * @param aBin - bin number
 * @returns altitude [m]
 */
int Radvol_getAltitude(Elevation_t aElev, int aBin);

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

