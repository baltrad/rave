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
 * Radvol-QC algorithms for assessment of distance to radar related effects.
 * @file radvolbroad.c
 * @author Katarzyna Osrodka (Institute of Meteorology and Water Management, IMGW-PIB)
 * @date 2012-12-20
 */
#include "radvolbroad.h"
#include "radvol.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include <string.h>

/**
 * Represents the RadvolBroad algorithm
 */
struct _RadvolBroad_t {
  RAVE_OBJECT_HEAD    /** Always on top */
  Radvol_t* radvol;   /**< volume of reflectivity and QI */
  double BROAD_LhQI1; /**< Maximum LH for which QI<sub>LH</sub> = 1 */
  double BROAD_LhQI0; /**< Minimum LH for which QI<sub>LH</sub> = 0 */
  double BROAD_LvQI1; /**< Maximum LV for which QI<sub>LV</sub> = 1 */
  double BROAD_LvQI0; /**< Minimum LV for which QI<sub>LV</sub> = 0 */
  double BROAD_Pulse; /**< Pulse length */
};

/*@{ Private functions */
/**
 * Constructor
 */
static int RadvolBroad_constructor(RaveCoreObject* obj)
{
  RadvolBroad_t* self = (RadvolBroad_t*)obj;
  self->radvol = RAVE_OBJECT_NEW(&Radvol_TYPE);
  if (self->radvol == NULL) {
    goto error;
  }
  self->BROAD_LhQI1 = 1.1;
  self->BROAD_LhQI0 = 2.5;  
  self->BROAD_LvQI1 = 1.6;
  self->BROAD_LvQI0 = 4.3;
  self->BROAD_Pulse = 0.3;
  return 1;
  
error:
  return 0;
}

/**
 * Copy constructor
 */
static int RadvolBroad_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RadvolBroad_t* this = (RadvolBroad_t*)obj;
  RadvolBroad_t* src = (RadvolBroad_t*)srcobj;
  this->radvol = RAVE_OBJECT_CLONE(src->radvol);
  if (this->radvol == NULL) {
    goto error;
  }
  this->BROAD_LhQI1 = src->BROAD_LhQI1;
  this->BROAD_LhQI0 = src->BROAD_LhQI0;  
  this->BROAD_LvQI1 = src->BROAD_LvQI1;   
  this->BROAD_LvQI0 = src->BROAD_LvQI0;    
  this->BROAD_Pulse = src->BROAD_Pulse;    
  return 1;

error:
  return 0;
}

/**
 * Destructor
 */
static void RadvolBroad_destructor(RaveCoreObject* obj)
{
  RadvolBroad_t* self = (RadvolBroad_t*)obj;
  RAVE_OBJECT_RELEASE(self->radvol);
}

/**
 * Reads algorithm parameters if xml file exists
 * @param self - self
 * @param paramFileName - name of xml file with parameters
 * @returns 1 if all parameters were read, otherwise 0
 */
static int RadvolBroadInternal_readParams(RadvolBroad_t* self, char* paramFileName)
{
  int result = 0;
  int IsDefaultChild;
  SimpleXmlNode_t* node = NULL;

  if ((paramFileName != NULL) && ((node = Radvol_getFactorChild(self->radvol, paramFileName, "BROAD_QIOn", &IsDefaultChild)) != NULL)) {
    result = 1;
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "BROAD_QIOn", &self->radvol->QIOn));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "DBZHtoTH", &self->radvol->DBZHtoTH));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "BROAD_LhQI1", &self->BROAD_LhQI1));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "BROAD_LhQI0", &self->BROAD_LhQI0));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "BROAD_LvQI1", &self->BROAD_LvQI1));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "BROAD_LvQI0", &self->BROAD_LvQI0));
    if (!((!IsDefaultChild) && (Radvol_getParValueDouble(node, "BROAD_Pulse", &self->BROAD_Pulse)))) {
      if (!SameValue(self->radvol->pulselength, cNull)) {
        self->BROAD_Pulse = self->radvol->pulselength;
      } else
        if ((IsDefaultChild) && (Radvol_getParValueDouble(node, "BROAD_Pulse", &self->BROAD_Pulse))) {
      } else {
        result = 0;
      }
    }
    RAVE_OBJECT_RELEASE(node);
    if (self->BROAD_LhQI1 > self->BROAD_LhQI0) {
      result = 0;
      self->BROAD_LhQI1 = 1.1;
      self->BROAD_LhQI0 = 2.5;
      RAVE_WARNING0("Error in parameter values - BROAD_LhQI1 must not exceed BROAD_LhQI0");
    }
    if (self->BROAD_LvQI1 > self->BROAD_LvQI0) {
      result = 0;
      self->BROAD_LvQI1 = 1.5;
      self->BROAD_LvQI0 = 3.2;
      RAVE_WARNING0("Error in parameter values - BROAD_LvQI1 must not exceed BROAD_LvQI0");
    }
  }
  return result;
}


/**
 * Prepares algorithm parameters as task_args
 * @param self - self
 * @returns 1 on success, otherwise 0
 */
static int RadvolBroadInternal_addTaskArgs(RadvolBroad_t* self)
{
  int result = 0;
  char task_args[1000];
  
  sprintf(task_args, "BROAD: BROAD_LhQI1=%3.1f, BROAD_LhQI0=%3.1f, BROAD_LvQI1=%3.1f, BROAD_LvQI0=%3.1f, BROAD_Pulse=%4.2f", 
          self->BROAD_LhQI1, self->BROAD_LhQI0, self->BROAD_LvQI1, self->BROAD_LvQI0, self->BROAD_Pulse);
  if (Radvol_setTaskArgs(self->radvol, task_args)) {
    result = 1;
  }
  return result;
}

/**
 * Algorithm for assessment of distance to radar related effects
 * @param self - self
 * @returns 1 on success, otherwise 0
 */
static int RadvolBroadInternal_broadAssessment(RadvolBroad_t* self)
{
  int aEle;
  int aRay, aBin;
  double sin1, sin2, cos1, cos2, qi;
  
  for (aEle = 0; aEle < self->radvol->nele; aEle++) {
    sin1 = sin(self->radvol->TabElev[aEle].elangle + self->radvol->beamwidth / 2.0);
    sin2 = sin(self->radvol->TabElev[aEle].elangle - self->radvol->beamwidth / 2.0);
    cos1 = cos(self->radvol->TabElev[aEle].elangle - self->radvol->beamwidth / 2.0);
    cos2 = cos(self->radvol->TabElev[aEle].elangle + self->radvol->beamwidth / 2.0);
    for (aBin = 0; aBin < self->radvol->TabElev[aEle].nbin; aBin++) {
      qi = Radvol_getLinearQuality(((aBin + 1) * self->radvol->TabElev[aEle].rscale + self->BROAD_Pulse / 2.0) * sin1 - ((aBin + 1) * self->radvol->TabElev[aEle].rscale - self->BROAD_Pulse / 2.0) * sin2, self->BROAD_LvQI1, self->BROAD_LvQI0) * Radvol_getLinearQuality(((aBin + 1) * self->radvol->TabElev[aEle].rscale + self->BROAD_Pulse / 2.0) * cos1 - ((aBin + 1) * self->radvol->TabElev[aEle].rscale - self->BROAD_Pulse / 2.0) * cos2, self->BROAD_LhQI1, self->BROAD_LhQI0);
      for (aRay = 0; aRay < self->radvol->TabElev[aEle].nray; aRay++) {
         if (!SameValue(self->radvol->TabElev[aEle].QIElev[aRay * self->radvol->TabElev[aEle].nbin + aBin], QI_BAD)) {
           self->radvol->TabElev[aEle].QIElev[aRay * self->radvol->TabElev[aEle].nbin + aBin]=qi;
         }
      }
    }
  }
  return 1;
}

/*@} End of Private functions */

/*@{ Interface functions */

int RadvolBroad_broadAssessment_scan(PolarScan_t* scan, char* paramFileName)
{
  RadvolBroad_t* self = RAVE_OBJECT_NEW(&RadvolBroad_TYPE);
  int retval = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (scan == NULL) {
    RAVE_ERROR0("Polar scan == NULL");
    return retval;
  }
  Radvol_getName(self->radvol, PolarScan_getSource(scan));
  Radvol_getAttrDouble_scan(scan, "how/pulsewidth", &self->radvol->pulselength);
  if (paramFileName == NULL || !RadvolBroadInternal_readParams(self, paramFileName)) {
    RAVE_WARNING0("Default parameter values");
  }
  if (self->radvol->QIOn) {
    if (!Radvol_setTaskName(self->radvol,"pl.imgw.radvolqc.broad")) {
      RAVE_ERROR0("Processing failed (setting task name)");
      goto done;
    }
    if (!RadvolBroadInternal_addTaskArgs(self)) {
      RAVE_ERROR0("Processing failed (setting task args)");
      goto done;
    }
    self->radvol->QCOn = 0;
    if (!Radvol_load_scan(self->radvol, scan)) {
      RAVE_ERROR0("Processing failed (loading scan)");
      goto done;
    }
    if (!RadvolBroadInternal_broadAssessment(self)) {
      RAVE_ERROR0("Processing failed (broadning assessment)");
      goto done;
    }
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

int RadvolBroad_broadAssessment_pvol(PolarVolume_t* pvol, char* paramFileName)
{
  RadvolBroad_t* self = RAVE_OBJECT_NEW(&RadvolBroad_TYPE);
  int retval = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (pvol == NULL) {
    RAVE_ERROR0("Polar volume == NULL");
    return retval;
  }
  Radvol_getName(self->radvol, PolarVolume_getSource(pvol));
  Radvol_getAttrDouble_pvol(pvol, "how/pulsewidth", &self->radvol->pulselength);
  if (paramFileName == NULL || !RadvolBroadInternal_readParams(self, paramFileName)) {
    RAVE_WARNING0("Default parameter values");
  }
  if (self->radvol->QIOn) {
    if (!Radvol_setTaskName(self->radvol, "pl.imgw.radvolqc.broad")) {
      RAVE_ERROR0("Processing failed (setting task name)");
      goto done;
    }
    if (!RadvolBroadInternal_addTaskArgs(self)) {
      RAVE_ERROR0("Processing failed (setting task args)");
      goto done;
    }
    self->radvol->QCOn = 0;
    if (!Radvol_load_pvol(self->radvol, pvol)) {
      RAVE_ERROR0("Processing failed (loading volume)");
      goto done;
    }
    if (!RadvolBroadInternal_broadAssessment(self)) {
      RAVE_ERROR0("Processing failed (broadning assessment)");
      goto done;
    }
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

RaveCoreObjectType RadvolBroad_TYPE = {
  "RadvolBroad",
  sizeof(RadvolBroad_t),
  RadvolBroad_constructor,
  RadvolBroad_destructor,
  RadvolBroad_copyconstructor
};
