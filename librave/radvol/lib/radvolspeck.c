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
 * Radvol-QC algorithms for speck removal.
 * @file radvolspeck.c
 * @author Katarzyna Osrodka (Institute of Meteorology and Water Management, IMGW-PIB)
 * @date 2012-12-20
 */
#include "radvolspeck.h"
#include "radvol.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include <string.h>

/**
 * Represents the RadvolSpeck algorithm
 */
struct _RadvolSpeck_t {
  RAVE_OBJECT_HEAD    /** Always on top */
  Radvol_t* radvol;   /**< volume of reflectivity and QI */
  double SPECK_QI;    /**< QI<sub>SPECK</sub> value for speck */
  double SPECK_QIUn;  /**< QI<sub>SPECK</sub> value for uncorrected speck */
  int    SPECK_AGrid; /**< Reverse speck vicinity (algorithm A) */
  int    SPECK_ANum;  /**< Maximum number of non-rainy gates (algorithm A) */
  int    SPECK_AStep; /**< Number of reverse speck removal cycles (algorithm A) */
  int    SPECK_BGrid; /**< Speck vicinity (algorithm B) */
  int    SPECK_BNum;  /**< Maximum number of rainy gates (algorithm B) */
  int    SPECK_BStep; /**< Number of speck removal cycles (algorithm B) */
};

/*@{ Private functions */
/**
 * Constructor
 */
static int RadvolSpeck_constructor(RaveCoreObject* obj)
{
  RadvolSpeck_t* self = (RadvolSpeck_t*)obj;
  self->radvol = RAVE_OBJECT_NEW(&Radvol_TYPE);
  if (self->radvol == NULL) {
    goto error;
  }
  self->SPECK_QI = 0.9;
  self->SPECK_QIUn = 0.5;  
  self->SPECK_AGrid = 1;   
  self->SPECK_ANum = 2;    
  self->SPECK_AStep = 1; 
  self->SPECK_BGrid = 1;   
  self->SPECK_BNum = 2;    
  self->SPECK_BStep = 2; 
  return 1;
  
error:
  return 0;
}

/**
 * Copy constructor
 */
static int RadvolSpeck_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RadvolSpeck_t* this = (RadvolSpeck_t*)obj;
  RadvolSpeck_t* src = (RadvolSpeck_t*)srcobj;
  this->radvol = RAVE_OBJECT_CLONE(src->radvol);
  if (this->radvol == NULL) {
    goto error;
  }
  this->SPECK_QI = src->SPECK_QI;
  this->SPECK_QIUn = src->SPECK_QIUn;  
  this->SPECK_AGrid = src->SPECK_AGrid;   
  this->SPECK_ANum = src->SPECK_ANum;    
  this->SPECK_AStep = src->SPECK_AStep; 
  this->SPECK_BGrid = src->SPECK_BGrid;   
  this->SPECK_BNum = src->SPECK_BNum;    
  this->SPECK_BStep = src->SPECK_BStep; 
  return 1;

error:
  return 0;
}

/**
 * Destructor
 */
static void RadvolSpeck_destructor(RaveCoreObject* obj)
{
  RadvolSpeck_t* self = (RadvolSpeck_t*)obj;
  RAVE_OBJECT_RELEASE(self->radvol);
}

/**
 * Reads algorithm parameters if xml file exists
 * @param self - self
 * @param paramFileName - name of xml file with parameters
 * @returns 1 if all parameters were read, otherwise 0
 */
static int RadvolSpeckInternal_readParams(RadvolSpeck_t* self, char* paramFileName)
{
  int result = 0;
  int IsDefaultChild;
  SimpleXmlNode_t* node = NULL;
  
  if ((paramFileName != NULL) && ((node = Radvol_getFactorChild(self->radvol, paramFileName, "SPECK_QIOn", &IsDefaultChild)) != NULL)) {
    result = 1;
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "SPECK_QIOn", &self->radvol->QIOn));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "SPECK_QCOn", &self->radvol->QCOn));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "DBZHtoTH", &self->radvol->DBZHtoTH));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "SPECK_QI", &self->SPECK_QI));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "SPECK_QIUn", &self->SPECK_QIUn));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "SPECK_AGrid", &self->SPECK_AGrid));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "SPECK_ANum", &self->SPECK_ANum));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "SPECK_AStep", &self->SPECK_AStep));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "SPECK_BGrid", &self->SPECK_BGrid));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "SPECK_BNum", &self->SPECK_BNum));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "SPECK_BStep", &self->SPECK_BStep));
    RAVE_OBJECT_RELEASE(node);
  } 
  return result;
}

/**
 * Prepares algorithm parameters as task_args
 * @param self - self
 * @returns 1 on success, otherwise 0
 */
static int RadvolSpeckInternal_addTaskArgs(RadvolSpeck_t* self)
{
  int result = 0;
  char task_args[1000];
  
  sprintf(task_args, "SPECK: SPECK_QI=%3.1f, SPECK_QIUn=%3.1f, SPECK_AGrid=%d, SPECK_ANum=%d, SPECK_AStep=%d, SPECK_BGrid=%d, SPECK_BNum=%d, SPECK_BStep=%d", 
          self->SPECK_QI, self->SPECK_QIUn, self->SPECK_AGrid, self->SPECK_ANum, self->SPECK_AStep, self->SPECK_BGrid, self->SPECK_BNum, self->SPECK_BStep);
  if (Radvol_setTaskArgs(self->radvol, task_args)) {
    result = 1;
  }
  return result;
}

/**
 * Removal of reverse speckles from elevation 
 * @param self - self
 * @param ele - elevation number
 * @param aTabElev - auxiliary input array
 * @param bTabElev - auxiliary output array
 * @param aQI - quality index value
 */
static void RadvolSpeckInternal_ElevRevSpecleRemoval(RadvolSpeck_t* self, int ele, double *aTabElev, double *bTabElev, double aQI)
{
  int aBin, aRay;
  int bRay, bBin;
  int sum, sum1;
  long int l1, l2;
  double aver;

  for (aRay = 0; aRay < self->radvol->TabElev[ele].nray; aRay++) {
    for (aBin = 0; aBin < self->radvol->TabElev[ele].nbin; aBin++) {
      l2 = aRay * self->radvol->TabElev[ele].nbin;
      if (SameValue(aTabElev[l2 + aBin], self->radvol->TabElev[ele].offset)) {
        sum = 0;
        sum1 = 0;
        aver = 0.0;
        for (bRay = aRay - self->SPECK_AGrid; bRay <= aRay + self->SPECK_AGrid; bRay++) {
          l1 = ((bRay + self->radvol->TabElev[ele].nray) % self->radvol->TabElev[ele].nray) * self->radvol->TabElev[ele].nbin;
          for (bBin = RAVEMAX(0, aBin - self->SPECK_AGrid); bBin <= RAVEMIN(aBin + self->SPECK_AGrid, self->radvol->TabElev[ele].nbin - 1); bBin++) {
            if (SameValue(aTabElev[l1 + bBin], self->radvol->TabElev[ele].offset)) {
              sum++;
            } else if (!SameValue(aTabElev[l1 + aBin], cNull)) {
              aver += aTabElev[l1 + bBin];
              sum1++;
            }
          }
        }
        if (sum <= self->SPECK_ANum) {
          bTabElev[l2 + aBin] = aver / sum1;
          if (self->radvol->QIOn) {
            self->radvol->TabElev[ele].QIElev[l2 + aBin] = aQI;
          }
        } else {
          bTabElev[l2 + aBin] = aTabElev[l2 + aBin];
        }
      } else {
        bTabElev[l2 + aBin] = aTabElev[l2 + aBin];
      }
    }
  }
}

/**
 * Removal of speckles from elevation 
 * @param self - self
 * @param ele - elevation number
 * @param aTabElev - auxiliary input array
 * @param bTabElev - auxiliary output array
 * @param aQI - quality index value
 */
static void RadvolSpeckInternal_ElevSpecleRemoval(RadvolSpeck_t* self, int ele, double *aTabElev, double *bTabElev, double aQI)
{
  int aBin;
  int aRay;
  int sum;
  long int l1, l2;
  int bRay;
  int bBin;

  for (aRay = 0; aRay < self->radvol->TabElev[ele].nray; aRay++) {
    for (aBin = 0; aBin < self->radvol->TabElev[ele].nbin; aBin++) {
      l2 = aRay * self->radvol->TabElev[ele].nbin;
      if (!SameValue(aTabElev[l2 + aBin], self->radvol->TabElev[ele].offset) && !SameValue(aTabElev[l2 + aBin], cNull)) {
        sum = 0;
        for (bRay = aRay - self->SPECK_BGrid; bRay <= aRay + self->SPECK_BGrid; bRay++) {
          l1 = ((bRay + self->radvol->TabElev[ele].nray) % self->radvol->TabElev[ele].nray) * self->radvol->TabElev[ele].nbin;
          for (bBin = RAVEMAX(0, aBin - self->SPECK_BGrid); bBin <= RAVEMIN(aBin + self->SPECK_BGrid, self->radvol->TabElev[ele].nbin - 1); bBin++) {
            if (!SameValue(aTabElev[l1 + bBin], self->radvol->TabElev[ele].offset) && !SameValue(aTabElev[l1 + bBin], cNull)) {
              sum++;
            }
          }
        }
        if (sum <= self->SPECK_BNum) {
          bTabElev[l2 + aBin] = self->radvol->TabElev[ele].offset;
          if (self->radvol->QIOn) {
            self->radvol->TabElev[ele].QIElev[l2 + aBin] = aQI;
          }
        } else {
          bTabElev[l2 + aBin] = aTabElev[l2 + aBin];
        }
      } else {
        bTabElev[l2 + aBin] = aTabElev[l2 + aBin];
      }
    }
  }
}

/**
 * Algorithm for speck removal and quality characterization
 * @param self - self
 * @returns 1 on success, otherwise 0
 */
static int RadvolSpeckInternal_speckRemoval(RadvolSpeck_t* self)
{
  int aEle;
  int nray, nbin;
  int step;
  long int l1;
  double *TabElev1 = NULL, *TabElev2 = NULL;
  double QI;
  
  QI = self->radvol->QCOn ? self->SPECK_QI : self->SPECK_QIUn;
  for (aEle = 0; aEle < self->radvol->nele; aEle++) {
    nbin = self->radvol->TabElev[aEle].nbin;
    nray = self->radvol->TabElev[aEle].nray;
    TabElev1 = RAVE_MALLOC(sizeof(double) * nbin * nray);
    if (TabElev1 == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory");
      goto error;
    }
    if ((self->SPECK_AStep > 1) || (self->SPECK_BStep > 1)) {
      TabElev2 = RAVE_MALLOC(sizeof (double) * nbin * nray);
      if (TabElev2 == NULL) {
        RAVE_CRITICAL0("Failed to allocate memory");
        goto error;
      }
    }
    
    //algorithm A - reverse specle removal
    RadvolSpeckInternal_ElevRevSpecleRemoval(self, aEle, self->radvol->TabElev[aEle].ReflElev, TabElev1, QI);
    for (step = 2; step <= self->SPECK_AStep; step++) {
      if (step % 2) {
        RadvolSpeckInternal_ElevRevSpecleRemoval(self, aEle, TabElev2, TabElev1, QI);
      } else {
        RadvolSpeckInternal_ElevRevSpecleRemoval(self, aEle, TabElev1, TabElev2, QI);
      }
    }
    if (self->radvol->QCOn) {
      if (self->SPECK_AStep % 2) {
        for (l1 = 0; l1 < nbin * nray; l1++) {
          self->radvol->TabElev[aEle].ReflElev[l1] = TabElev1[l1];
        }
      } else {
        for (l1 = 0; l1 < nbin * nray; l1++) {
          self->radvol->TabElev[aEle].ReflElev[l1] = TabElev2[l1];
        }
      }
    }

   //algorithm B - specle removal
    RadvolSpeckInternal_ElevSpecleRemoval(self, aEle, self->radvol->TabElev[aEle].ReflElev, TabElev1, QI);
    for (step = 2; step <= self->SPECK_BStep; step++) {
      if (step % 2) {
        RadvolSpeckInternal_ElevSpecleRemoval(self, aEle, TabElev2, TabElev1, QI);
      } else {
        RadvolSpeckInternal_ElevSpecleRemoval(self, aEle, TabElev1, TabElev2, QI);
      }
    }
    if (self->radvol->QCOn) {
      if (self->SPECK_BStep % 2) {
        for (l1 = 0; l1 < nbin * nray; l1++) {
          self->radvol->TabElev[aEle].ReflElev[l1] = TabElev1[l1];
        }
      } else {
        for (l1 = 0; l1 < nbin * nray; l1++) {
          self->radvol->TabElev[aEle].ReflElev[l1] = TabElev2[l1];
        }
      }
    }

   RAVE_FREE(TabElev1);
   if ((self->SPECK_AStep > 1) || (self->SPECK_BStep > 1)) {
     RAVE_FREE(TabElev2);
   }
  }
  return 1;
  
error:
   RAVE_FREE(TabElev1);
   if ((self->SPECK_AStep > 1) || (self->SPECK_BStep > 1)) {
     RAVE_FREE(TabElev2);
   }
  return 0;
}

/*@} End of Private functions */

/*@{ Interface functions */

int RadvolSpeck_speckRemoval_scan(PolarScan_t* scan, char* paramFileName)
{
  RadvolSpeck_t* self = RAVE_OBJECT_NEW(&RadvolSpeck_TYPE);
  int retval = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (scan == NULL) {
    RAVE_ERROR0("Polar scan == NULL");
    return retval;
  }
  Radvol_getName(self->radvol, PolarScan_getSource(scan));
  if (paramFileName == NULL || !RadvolSpeckInternal_readParams(self, paramFileName)) {
    RAVE_WARNING0("Default parameter values");
  }
  if (self->radvol->QCOn || self->radvol->QIOn) {
    if (!Radvol_setTaskName(self->radvol,"pl.imgw.radvolqc.speck")) {
      RAVE_ERROR0("Processing failed (setting task name)");
      goto done;    
    }
    if (!RadvolSpeckInternal_addTaskArgs(self)) {
      RAVE_ERROR0("Processing failed (setting task args)");
      goto done;    
    } 
    if (!Radvol_load_scan(self->radvol, scan)) {
      RAVE_ERROR0("Processing failed (loading scan)");
      goto done;
    }
    if (!RadvolSpeckInternal_speckRemoval(self)) {
      RAVE_ERROR0("Processing failed (speck removal)");
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

int RadvolSpeck_speckRemoval_pvol(PolarVolume_t* pvol, char* paramFileName)
{
  RadvolSpeck_t* self = RAVE_OBJECT_NEW(&RadvolSpeck_TYPE);
  int retval = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (pvol == NULL) {
    RAVE_ERROR0("Polar volume == NULL");
    return retval;
  }
  Radvol_getName(self->radvol, PolarVolume_getSource(pvol));
  if (paramFileName == NULL || !RadvolSpeckInternal_readParams(self, paramFileName)) {
    RAVE_WARNING0("Default parameter values");
  }
  if (self->radvol->QCOn || self->radvol->QIOn) {
    if (!Radvol_setTaskName(self->radvol,"pl.imgw.radvolqc.speck")) {
      RAVE_ERROR0("Processing failed (setting task name)");
      goto done;
    }
    if (!RadvolSpeckInternal_addTaskArgs(self)) {
      RAVE_ERROR0("Processing failed (setting task args)");
      goto done;
    }
    if (!Radvol_load_pvol(self->radvol, pvol)) {
      RAVE_ERROR0("Processing failed (loading volume)");
      goto done;
    }
    if (!RadvolSpeckInternal_speckRemoval(self)) {
      RAVE_ERROR0("Processing failed (speck removal)");
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

RaveCoreObjectType RadvolSpeck_TYPE = {
  "RadvolSpeck",
  sizeof(RadvolSpeck_t),
  RadvolSpeck_constructor,
  RadvolSpeck_destructor,
  RadvolSpeck_copyconstructor
};
