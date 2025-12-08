/* --------------------------------------------------------------------
Copyright (C) 2025 Swedish Meteorological and Hydrological Institute, SMHI,

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/
/**
 * 
 * This object does support \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-12-02
 */
#ifndef RAVE_PIA_H
#define RAVE_PIA_H
#include "rave_object.h"
#include "rave_field.h"
#include "polarscan.h"

/**
 * Defines PIA
 */
typedef struct _RavePIA_t RavePIA_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType RavePIA_TYPE;

const char* RavePIA_getHowTaskName();

/**
 * Sets the coefficient of Z-k power law
 * @param[in] self - self
 * @param[in] c - the coefficient
 */
void RavePIA_setZkPowerCoefficient(RavePIA_t* self, double c);

/**
 * Returns the coefficient of Z-k power law
 * @param[in] self - self
 */
double RavePIA_getZkPowerCoefficient(RavePIA_t* self);

/**
 * Sets the exponent of Z-k power law
 * @param[in] self - self
 * @param[in] d - the exponent
 */
void RavePIA_setZkPowerExponent(RavePIA_t* self, double d);

/**
 * Returns the coefficient of Z-k power law
 * @param[in] self - self
 */
double RavePIA_getZkPowerExponent(RavePIA_t* self);

/**
 * Sets the PIA max value
 * @param[in] self - self
 * @param[in] maxv - the max value
 */
void RavePIA_setPiaMax(RavePIA_t* self, double maxv);

/**
 * Returns the PIA max value
 * @param[in] self - self
 * @return the PIA max value
 */
double RavePIA_getPiaMax(RavePIA_t* self);


/**
 * Sets the range resolution in km.
 * @param[in] self - self
 * @param[in] rr - the range resolution in km
 */
void RavePIA_setRangeResolution(RavePIA_t* self, double rr);

/**
 * Returns the range resolution in km.
 * @param[in] self - self
 * @return the range resolution
 */
double RavePIA_getRangeResolution(RavePIA_t* self);


/**
 * Calculates PIA for specified quantity in scan
 * @param[in] self - self
 * @param[in] scan - the scna
 * @param[in] quantity - the quantity
 * @param[out] outDr - if outDr != NULL the used range resolution will be set here.
 * @return the rave field
 */
RaveField_t* RavePIA_calculatePIA(RavePIA_t* self, PolarScan_t* scan, const char* quantity, double* outDr);

/**
 * Creates the PIA parameter
 * @param[in] self - self
 * @param[in] scan - the scan
 * @param[in] quantity - the quantity
 * @param[out] outPIA - if outPIA != NULL the resulting PIA will be copied here.
 * @param[out] outDr - if outDr != NULL the used range resolution will be set here.
 * @return the polar scan param
 */
PolarScanParam_t* RavePIA_createPIAParameter(RavePIA_t* self, PolarScan_t* scan, const char* quantity, RaveField_t** outPIA, double* outDr);

/**
 * Performs the processing and adds quality flag / PIA parameter and adjusts DBZH field.
 * @param[in] self - self
 * @param[in] scan - the scan on which the PIA calculation should be performed
 * @param[in] quantity - the quantity of the parameter
 * @param[in] addparam - if the PIA parameter should be added to the scan
 * @param[in] reprocessquality - if True (1), then the quality field is created even if the quality field with same how/task name already exists.
 * @param[in] apply - if the DBZH parameter should be adjusted with the PIA field
 * @return 1 on success
 */
int RavePIA_process(RavePIA_t* self, PolarScan_t* scan, const char* quantity, int addparam, int reprocessquality, int apply);

#endif /* RAVE_GRA_H */
