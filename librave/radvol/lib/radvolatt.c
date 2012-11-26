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
 * Radvol-QC algorithms of correction for attenuation in rain.
 * @file radvolatt.c
 * @author Katarzyna Osrodka (Institute of Meteorology and Water Management, IMGW-PIB)
 * @date 2012-07-12
 */
#include "radvolatt.h"
#include "radvol.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include <string.h>

/**
 * Represents the RadvolAtt algorithm
 */
struct _RadvolAtt_t {
  RAVE_OBJECT_HEAD 	/** Always on top */
  Radvol_t* radvol;	/**< volume of reflectivity and QI */
  double ATT_QI1;	/**< Maximum correction for which QI<sub>ATT</sub> = 1 */
  double ATT_QI0;	/**< Minimum correction for which QI<sub>ATT</sub> = 0 */
  double ATT_QIUn;	/**< Multiplier of QI<sub>ATT</sub> value for uncorrected attenuation */
  double ATT_a;		/**< Coefficient a in attenuation formula */
  double ATT_b;		/**< Coefficient b in attenuation formula */
  double ATT_MPa;	/**< Coefficient a in Marshall-Palmer formula */
  double ATT_MPb;	/**< Coefficient b in Marshall-Palmer formula */
  double ATT_Refl;	/**< Minimum reflectivity to apply the correction */
  double ATT_Last;	/**< Maximum correction within last km */
  double ATT_Sum;	/**< Maximum summarized correction */
};

/*@{ Private functions */
/**
 * Constructor
 */
static int RadvolAtt_constructor(RaveCoreObject* obj)
{
  RadvolAtt_t* self = (RadvolAtt_t*)obj;
  self->radvol = RAVE_OBJECT_NEW(&Radvol_TYPE);
  if (self->radvol == NULL) {
    goto error;
  }
  self->ATT_QI1 = 1.0;
  self->ATT_QI0 = 5.0;  
  self->ATT_QIUn = 0.9;   
  self->ATT_a = 0.0044;    
  self->ATT_b = 1.17; 
  self->ATT_MPa = 200.0;    
  self->ATT_MPb = 1.6; 
  self->ATT_Refl = 4.0;   
  self->ATT_Last = 1.0;    
  self->ATT_Sum = 5.0; 
  return 1;
  
error:
  return 0;
}

/**
 * Copy constructor
 */
static int RadvolAtt_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RadvolAtt_t* this = (RadvolAtt_t*)obj;
  RadvolAtt_t* src = (RadvolAtt_t*)srcobj;
  this->radvol = RAVE_OBJECT_CLONE(src->radvol);
  if (this->radvol == NULL) {
    goto error;
  }
  this->ATT_QI1 = src->ATT_QI1;
  this->ATT_QI0 = src->ATT_QI0;  
  this->ATT_QIUn = src->ATT_QIUn;   
  this->ATT_a = src->ATT_a;    
  this->ATT_b = src->ATT_b; 
  this->ATT_MPa = src->ATT_MPa;    
  this->ATT_MPb = src->ATT_MPb; 
  this->ATT_Refl = src->ATT_Refl;   
  this->ATT_Last = src->ATT_Last;    
  this->ATT_Sum = src->ATT_Sum; 
  return 1;

error:
  return 0;
}

/**
 * Destructor
 */
static void RadvolAtt_destructor(RaveCoreObject* obj)
{
  RadvolAtt_t* self = (RadvolAtt_t*)obj;
  RAVE_OBJECT_RELEASE(self->radvol);
}

/**
 * Reads algorithm parameters if xml file exists
 * @param self - self
 * @param paramFileName - name of xml file with parameters
 * @returns 1 if all parameters were read, otherwise 0
 */
static int RadvolAttInternal_readParams(RadvolAtt_t* self, char* paramFileName)
{
  int result = 0;
  SimpleXmlNode_t* node = NULL;
  
  node = Radvol_getFactorChild(paramFileName, "ATT");
  if (node != NULL) {
    result = 1;
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "QIOn", &self->radvol->QIOn));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "QCOn", &self->radvol->QCOn));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "QI1", &self->ATT_QI1));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "QI0", &self->ATT_QI0));  
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "QIUn", &self->ATT_QIUn));  
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "a", &self->ATT_a));   
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "b", &self->ATT_b));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "MPa", &self->ATT_MPa));   
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "MPb", &self->ATT_MPb));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "Refl", &self->ATT_Refl));  
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "Last", &self->ATT_Last));  
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "Sum", &self->ATT_Sum));
    RAVE_OBJECT_RELEASE(node);
  } 
  return result;
}

/**
 * Prepares algorithm parameters as task_args
 * @param self - self
 * @returns 1 on success, otherwise 0
 */
static int RadvolAttInternal_addTaskArgs(RadvolAtt_t* self)
{
  int result = 0;
  char task_args[1000];
 
  sprintf(task_args, "ATT: ATT_QI1=%3.1f, ATT_QI0=%3.1f, ATT_QIUn=%3.1f, ATT_a=%7.4f, ATT_b=%5.2f, ATT_MPa=%4.1f, ATT_MPb=%4.1f, ATT_Refl=%4.1f, ATT_Last=%4.1f, ATT_Sum=%4.1f", 
	  self->ATT_QI1, self->ATT_QI0, self->ATT_QIUn, self->ATT_a, self->ATT_b, self->ATT_MPa, self->ATT_MPb, self->ATT_Refl, self->ATT_Last, self->ATT_Sum);
  if (Radvol_setTaskArgs(self->radvol, task_args)) {
    result = 1;
  }
  return result;
}


/**
 * Algorithm for speck removal and quality characterization
 * @param self - self
 * @returns 1 on success, otherwise 0
 */
static int RadvolAttInternal_attCorrection(RadvolAtt_t* self)
{
  int aEle;
  int aBin, aRay;
  double QI;
  long int l;
  double R, R1, dBZ1;
  double AttSum, AttSpec;
  
  for (aEle = 0; aEle < self->radvol->nele; aEle++) {
    for (aRay = 0; aRay < self->radvol->TabElev[aEle].nray; aRay++) {
      l = aRay * self->radvol->TabElev[aEle].nbin;
      AttSum = 0.0;
      QI = 1;
      for (aBin = 0; aBin < self->radvol->TabElev[aEle].nbin; aBin++) {
	if (SameValue(self->radvol->TabElev[aEle].ReflElev[l + aBin], cNull)) {
	  if (self->radvol->QIOn) {
	    self->radvol->TabElev[aEle].QIElev[l + aBin] = 0.0;
	  }
	} else if (SameValue(self->radvol->TabElev[aEle].ReflElev[l + aBin], cNoRain)) {
	  if (self->radvol->QIOn) {
	    self->radvol->TabElev[aEle].QIElev[l + aBin] = QI;
	  }
	} else  if ((self->radvol->TabElev[aEle].ReflElev[l + aBin] < self->ATT_Refl) || (AttSum + 0.001 > self->ATT_Sum)) {
	  if (self->radvol->QIOn) {
	    self->radvol->TabElev[aEle].QIElev[l + aBin] = QI;
	  }
	  if (self->radvol->QCOn) {
	    self->radvol->TabElev[aEle].ReflElev[l + aBin] += AttSum;
	  }
	} else {
	  R = dBZ2R(self->radvol->TabElev[aEle].ReflElev[l + aBin], self->ATT_MPa, self->ATT_MPb);
	  AttSpec = self->ATT_a * pow(R, self->ATT_b);
	  dBZ1 = self->radvol->TabElev[aEle].ReflElev[l + aBin] + AttSum + AttSpec;
	  R1 = dBZ2R(dBZ1, self->ATT_MPa, self->ATT_MPb);
	  AttSpec = self->ATT_a * pow(R1, self->ATT_b);
	  if (AttSpec > self->ATT_Last) {
	    printf("Threshold for correction in 1km exceeded - elevation %d, ray %d, bin %d, corr=%f\n", aEle, aRay, aBin, AttSpec);
	    AttSpec = self->ATT_Last;
	  }
	  if (AttSum + AttSpec > self->ATT_Sum) {
	    printf("Threshold for total correction exceeded - elevation %d, ray %d, bin %d, corr=%f\n",  aEle, aRay, aBin, AttSum + AttSpec);
	    AttSum = self->ATT_Sum;
	  } else {
	    AttSum += AttSpec;
	  }
	  if (self->radvol->QCOn) {
	    self->radvol->TabElev[aEle].ReflElev[l + aBin] += AttSum;
	  }
	  if (self->radvol->QIOn) {
	    QI = Radvol_getLinearQuality(AttSum, self->ATT_QI1, self->ATT_QI0);
	    if (!self->radvol->QCOn) {
	      QI *= self->ATT_QIUn;
	    }
	    self->radvol->TabElev[aEle].QIElev[l + aBin] = QI;
	  }    
	}
      }
    }
  }
  return 1;
}

/*@} End of Private functions */

/*@{ Interface functions */

int RadvolAtt_attCorrection(PolarVolume_t* pvol, char* paramFileName)
{
  RadvolAtt_t* self = RAVE_OBJECT_NEW(&RadvolAtt_TYPE);
  int retval = 0;
  
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (pvol == NULL) {
    RAVE_ERROR0("Polar volume == NULL");
    return retval;
  }
  if (paramFileName == NULL || !RadvolAttInternal_readParams(self, paramFileName)) {
    RAVE_WARNING0("Default parameter values");
  }
  if (self->radvol->QCOn || self->radvol->QIOn) {
    if (!Radvol_setTaskName(self->radvol,"pl.imgw.radvolqc.att")) {
      RAVE_ERROR0("Processing failed (setting task name)");
      goto done;    
    }
    if (!RadvolAttInternal_addTaskArgs(self)) {
      RAVE_ERROR0("Processing failed (setting task args)");
      goto done;    
    }
    // self->radvol->QCOn = 0;
    if (!Radvol_loadVol(self->radvol, pvol)) {
      RAVE_ERROR0("Processing failed (loading volume)");
      goto done;
    }
    if (!RadvolAttInternal_attCorrection(self)) {
      RAVE_ERROR0("Processing failed (attenuation correction)");
      goto done;
    }
    if (!Radvol_saveVol(self->radvol, pvol)) {
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

RaveCoreObjectType RadvolAtt_TYPE = {
  "RadvolAtt",
  sizeof(RadvolAtt_t),
  RadvolAtt_constructor,
  RadvolAtt_destructor,
  RadvolAtt_copyconstructor
};
