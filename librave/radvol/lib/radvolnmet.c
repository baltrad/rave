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
 * Radvol-QC algorithms for non-meteorological echoes removal.
 * @file radvolnmet.c
 * @author Katarzyna Osrodka (Institute of Meteorology and Water Management, IMGW-PIB)
 * @date 2013
 */

#include "radvolnmet.h"
#include "radvol.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include <string.h>

/**
 * Represents the RadvolNmet algorithm
 */
struct _RadvolNmet_t {
  RAVE_OBJECT_HEAD      /** Always on top */
  Radvol_t* radvol;     /**< volume of reflectivity and QI */
  double NMET_QI;       /**< QI<sub>NMET</sub> value for non-meteorological echoes */
  double NMET_QIUn;     /**< QI<sub>NMET</sub> value for uncorrected non-meteorological echoes */
  double NMET_AReflMin; /**< Minimum value of reflectivity in NMET echo detector (algorithm A) */
  double NMET_AReflMax; /**< Maximum value of reflectivity in NMET echo detector (algorithm A) */
  double NMET_AAltMin;  /**< Minimum value of height in NMET echo detector (algorithm A) */
  double NMET_AAltMax;  /**< Maximum value of height in NMET echo detector (algorithm A) */
  double NMET_ADet;     /**< Threshold for NMET echo detector (algorithm A) */
  double NMET_BAlt;     /**< Threshold for meteorological echo altitude (algorithm B)  */
};

/*@{ Private functions */
/**
 * Constructor
 */
static int RadvolNmet_constructor(RaveCoreObject* obj)
{
  RadvolNmet_t* self = (RadvolNmet_t*)obj;
  self->radvol = RAVE_OBJECT_NEW(&Radvol_TYPE);
  if (self->radvol == NULL) {
    goto error;
  }
  self->NMET_QI = 0.75;
  self->NMET_QIUn = 0.3;
  self->NMET_AReflMin = -15.0;
  self->NMET_AReflMax = 5.0;
  self->NMET_AAltMin = 1.0;
  self->NMET_AAltMax = 3.0;
  self->NMET_ADet = 0.3;
  self->NMET_BAlt = 20.0;
  return 1;

error:
  return 0;
}

/**
 * Copy constructor
 */
static int RadvolNmet_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RadvolNmet_t* this = (RadvolNmet_t*)obj;
  RadvolNmet_t* src = (RadvolNmet_t*)srcobj;
  this->radvol = RAVE_OBJECT_CLONE(src->radvol);
  if (this->radvol == NULL) {
    goto error;
  }
  this->NMET_QI = src->NMET_QI;
  this->NMET_QIUn = src->NMET_QIUn;
  this->NMET_AReflMin = src->NMET_AReflMin;
  this->NMET_AReflMax = src->NMET_AReflMax;
  this->NMET_AAltMin = src->NMET_AAltMin;
  this->NMET_AAltMax = src->NMET_AAltMax;
  this->NMET_ADet = src->NMET_ADet;
  this->NMET_BAlt = src->NMET_BAlt;
  return 1;

error:
  return 0;
}

/**
 * Destructor
 */
static void RadvolNmet_destructor(RaveCoreObject* obj)
{
  RadvolNmet_t* self = (RadvolNmet_t*)obj;
  RAVE_OBJECT_RELEASE(self->radvol);
}

/**
 * Reads algorithm parameters if xml file exists
 * @param self - self
 * @param params - struct containing algorithm-specific parameter settings
 * @param paramFileName - name of xml file with parameters
 * @returns 1 if all parameters were read, otherwise 0
 */
static int RadvolNmetInternal_readParams(RadvolNmet_t* self, Radvol_params_t* params, char* paramFileName)
{
  int result = 0;
  int IsDefaultChild;
  SimpleXmlNode_t* node = NULL;

  if (paramFileName == NULL) {
      self->radvol->QCOn =     params->ATT_QCOn;
      self->radvol->QIOn =     params->ATT_QIOn;
      self->radvol->DBZHtoTH = params->DBZHtoTH;
      self->NMET_QI =          params->NMET_QI;
      self->NMET_QIUn =        params->NMET_QIUn;
      self->NMET_AReflMin =    params->NMET_AReflMin;
      self->NMET_AReflMax =    params->NMET_AReflMax;
      self->NMET_AAltMin =     params->NMET_AAltMin;
      self->NMET_AAltMax =     params->NMET_AAltMax;
      self->NMET_ADet =        params->NMET_ADet;
      self->NMET_BAlt =        params->NMET_BAlt;
      result = 1;
  }
  else if ((paramFileName != NULL) && ((node = Radvol_getFactorChild(self->radvol, paramFileName, "NMET_QIOn", &IsDefaultChild)) != NULL)) {
    result = 1;
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "NMET_QIOn", &self->radvol->QIOn));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "NMET_QCOn", &self->radvol->QCOn));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "DBZHtoTH", &self->radvol->DBZHtoTH));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "NMET_QI", &self->NMET_QI));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "NMET_QIUn", &self->NMET_QIUn));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "NMET_AReflMin", &self->NMET_AReflMin));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "NMET_AReflMax", &self->NMET_AReflMax));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "NMET_AAltMin", &self->NMET_AAltMin));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "NMET_AAltMax", &self->NMET_AAltMax));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "NMET_ADet", &self->NMET_ADet));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "NMET_BAlt", &self->NMET_BAlt));
    RAVE_OBJECT_RELEASE(node);
    }
  return result;
}

/**
 * Prepares algorithm parameters as task_args
 * @param self - self
 * @returns 1 on success, otherwise 0
 */
static int RadvolNmetInternal_addTaskArgs(RadvolNmet_t* self)
{
  int result = 0;
  char task_args[1000];

  sprintf(task_args, "NMET: NMET_QI=%4.2f, NMET_QIUn=%4.2f, NMET_AReflMin=%5.2f, NMET_AReflMax=%5.2f, NMET_AAltMin=%4.1f, NMET_AAltMax=%4.1f, NMET_ADet=%4.2f, NMET_BAlt=%4.1f",
          self->NMET_QI, self->NMET_QIUn, self->NMET_AReflMin, self->NMET_AReflMax, self->NMET_AAltMin, self->NMET_AAltMax, self->NMET_ADet, self->NMET_BAlt);
  if (Radvol_setTaskArgs(self->radvol, task_args)) {
    result = 1;
  }
  return result;
}

/**
 * Calculates number of the gate placed above the given gate at the higher elevation
 * @param radvol - self
 * @param aEle - elevation number
 * @param aRay - ray number of input gate
 * @param aBin - bin number of input gate
 * @returns number of the higher gate if exists, otherwise -1
 */
static long int RadvolNmetInternal_NearestHigher(Radvol_t* radvol, int aEle, int aRay, int aBin) {
  int bBin, bRay;
  long int result = -1;

  if (aEle < radvol->nele - 1) {
    bRay = (int) rint(aRay * radvol->TabElev[aEle + 1].nray / radvol->TabElev[aEle].nray);
    bBin = (int) rint(cos(radvol->TabElev[aEle].elangle) * aBin * radvol->TabElev[aEle].rscale / cos(radvol->TabElev[aEle + 1].elangle) / radvol->TabElev[aEle + 1].rscale);
    if ((bRay >= 0) && (bRay < radvol->TabElev[aEle + 1].nray) && (bBin >= 0) && (bBin < radvol->TabElev[aEle + 1].nbin)) {
      result = bRay * radvol->TabElev[aEle + 1].nbin + bBin;
    }
  }
  return result;
}

/**
 * Algorithm for non-meteorological echoes removal and quality characterization
 * @param self - self
 */
static void RadvolNmetInternal_nmetRemoval(RadvolNmet_t* self) {
  int aEle, aRay, aBin;
  int nray, nbin;
  long int l, lHigher;
  double QI;
  double EchoMaxHeight;
  double height;

  EchoMaxHeight = self->NMET_BAlt - self->radvol->altitude / 1000.0;
  QI = self->radvol->QCOn ? self->NMET_QI : self->NMET_QIUn;
  for (aEle = self->radvol->nele - 1; aEle >= 0; aEle--) {
    nbin = self->radvol->TabElev[aEle].nbin;
    nray = self->radvol->TabElev[aEle].nray;
    for (aBin = 0; aBin < nbin; aBin++) {
      height = Radvol_getCurvature(self->radvol, aEle, aBin);
      for (aRay = 0; aRay < nray; aRay++) {
        l = aRay * nbin + aBin;
        if (!SameValue(self->radvol->TabElev[aEle].ReflElev[l], self->radvol->TabElev[aEle].offset)
                && !SameValue(self->radvol->TabElev[aEle].ReflElev[l], cNull)) {
          lHigher = RadvolNmetInternal_NearestHigher(self->radvol, aEle, aRay, aBin);
          if ((height >= EchoMaxHeight)
                  || ((Radvol_getLinearQuality(height, self->NMET_AAltMin, self->NMET_AAltMax) * Radvol_getLinearQuality(self->radvol->TabElev[aEle].ReflElev[l], self->NMET_AReflMin, self->NMET_AReflMax) > self->NMET_ADet)
                  && ((aEle + 1 == self->radvol->nele) || ((lHigher>-1) && SameValue(self->radvol->TabElev[aEle + 1].ReflElev[lHigher], self->radvol->TabElev[aEle + 1].offset))))) {
            if (self->radvol->QCOn) {
              self->radvol->TabElev[aEle].ReflElev[l] = self->radvol->TabElev[aEle].offset;
            }
            if (self->radvol->QCOn) {
              self->radvol->TabElev[aEle].QIElev[l] = QI;
            }
          }
        }
      }
    }
  }
}

/*@} End of Private functions */

/*@{ Interface functions */

int RadvolNmet_nmetRemoval_scan(PolarScan_t* scan, Radvol_params_t* params, char* paramFileName)
{
  RadvolNmet_t* self = RAVE_OBJECT_NEW(&RadvolNmet_TYPE);
  int retval = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (scan == NULL) {
    RAVE_ERROR0("Polar scan == NULL");
    return retval;
  }
  Radvol_getName(self->radvol, PolarScan_getSource(scan));
  if (paramFileName == NULL || !RadvolNmetInternal_readParams(self, params, paramFileName)) {
    /* RAVE_WARNING0("Default parameter values"); */
  }
  if (self->radvol->QCOn || self->radvol->QIOn) {
    if (!Radvol_setTaskName(self->radvol,"pl.imgw.radvolqc.nmet")) {
      RAVE_ERROR0("Processing failed (setting task name)");
      goto done;
    }
    if (!RadvolNmetInternal_addTaskArgs(self)) {
      RAVE_ERROR0("Processing failed (setting task args)");
      goto done;
    }
    Radvol_setEquivalentEarthRadius(self->radvol, PolarScan_getLatitude(scan));
    if (!Radvol_load_scan(self->radvol, scan)) {
      RAVE_ERROR0("Processing failed (loading volume)");
      goto done;
    }
    RadvolNmetInternal_nmetRemoval(self);
    if (!Radvol_save_scan(self->radvol, scan)) {
      RAVE_ERROR0("Processing failed (saving scan)");
      goto done;
    }
    retval = 1;
  } else {
    RAVE_WARNING0("Processing stopped because QC and QI switched off");
  }

done:
  RAVE_OBJECT_RELEASE(self);
  return retval;
}

int RadvolNmet_nmetRemoval_pvol(PolarVolume_t* pvol, Radvol_params_t* params, char* paramFileName)
{
  RadvolNmet_t* self = RAVE_OBJECT_NEW(&RadvolNmet_TYPE);
  int retval = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (pvol == NULL) {
    RAVE_ERROR0("Polar volume == NULL");
    return retval;
  }
  Radvol_getName(self->radvol, PolarVolume_getSource(pvol));
  if (paramFileName == NULL || !RadvolNmetInternal_readParams(self, params, paramFileName)) {
     /* RAVE_WARNING0("Default parameter values"); */
  }
  if (self->radvol->QCOn || self->radvol->QIOn) {
    if (!Radvol_setTaskName(self->radvol,"pl.imgw.radvolqc.nmet")) {
      RAVE_ERROR0("Processing failed (setting task name)");
      goto done;
    }
    if (!RadvolNmetInternal_addTaskArgs(self)) {
      RAVE_ERROR0("Processing failed (setting task args)");
      goto done;
    }
    Radvol_setEquivalentEarthRadius(self->radvol, PolarVolume_getLatitude(pvol));
    if (!Radvol_load_pvol(self->radvol, pvol)) {
      RAVE_ERROR0("Processing failed (loading volume)");
      goto done;
    }
    RadvolNmetInternal_nmetRemoval(self);
    if (!Radvol_save_pvol(self->radvol, pvol)) {
      RAVE_ERROR0("Processing failed (saving volume)");
      goto done;
    }
    retval = 1;
  } else {
    RAVE_WARNING0("Processing stopped because QC and QI switched off");
  }

done:
  RAVE_OBJECT_RELEASE(self);
  return retval;
}

/*@} End of Interface functions */

RaveCoreObjectType RadvolNmet_TYPE = {
  "RadvolNmet",
  sizeof(RadvolNmet_t),
  RadvolNmet_constructor,
  RadvolNmet_destructor,
  RadvolNmet_copyconstructor
};

