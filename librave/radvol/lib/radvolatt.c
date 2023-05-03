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
 * @date 2012-12-20
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
  RAVE_OBJECT_HEAD  /** Always on top */
  Radvol_t* radvol; /**< volume of reflectivity and QI */
  double ATT_QI1;   /**< Maximum correction for which QI<sub>ATT</sub> = 1 */
  double ATT_QI0;   /**< Minimum correction for which QI<sub>ATT</sub> = 0 */
  double ATT_QIUn;  /**< Multiplier of QI<sub>ATT</sub> value for uncorrected attenuation */
  double ATT_a;     /**< Coefficient a in attenuation formula */
  double ATT_b;     /**< Coefficient b in attenuation formula */
  double ATT_ZRa;   /**< Coefficient a in ZR relationship */
  double ATT_ZRb;   /**< Coefficient b in ZR relationship */
  double ATT_Refl;  /**< Minimum reflectivity to apply the correction */
  double ATT_Last;  /**< Maximum correction within last km */
  double ATT_Sum;   /**< Maximum summarized correction */
};

/** parameter values deduced from wavelength */
#define ValuesFromWavelength 0
/** parameter values read from xml file */
#define ValuesFromXml 1
/** parameter values from memory, passed from RAVE */
#define ValuesFromRave 2
/** no parameter values - error */
#define NoValues 3

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
  self->ATT_ZRa = 200.0;
  self->ATT_ZRb = 1.6;
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
  this->ATT_ZRa = src->ATT_ZRa;
  this->ATT_ZRb = src->ATT_ZRb;
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
 * @param params - struct containing algorithm-specific parameter settings
 * @param paramFileName - name of xml file with parameters
 * @returns 1 if all parameters were read, 0 if ATT_a and ATT_b were deduced from wavelength, 2 if ATT_a and ATT_b unknown
 */
static int RadvolAttInternal_readParams(RadvolAtt_t* self, Radvol_params_t* params, char* paramFileName)
{
  int result = ValuesFromRave;
  int IsDefaultChild = 1;
  int result_att = 0;
  SimpleXmlNode_t* node = NULL;
  
  if (paramFileName == NULL) {
    self->radvol->QCOn =     params->ATT_QCOn;
    self->radvol->QIOn =     params->ATT_QIOn;
    self->radvol->DBZHtoTH = params->DBZHtoTH;
    self->ATT_QI1 =          params->ATT_QI1;
    self->ATT_QI0 =          params->ATT_QI0;
    self->ATT_QIUn =         params->ATT_QIUn;
    self->ATT_ZRa =          params->ATT_ZRa;
    self->ATT_ZRb =          params->ATT_ZRb;
    self->ATT_Refl =         params->ATT_Refl;
    self->ATT_Last =         params->ATT_Last;
    self->ATT_Sum =          params->ATT_Sum;
    self->ATT_a =            params->ATT_a;
    self->ATT_b =            params->ATT_b;
    result_att = 1;
  }
  else if ((paramFileName != NULL) &&  ((node = Radvol_getFactorChild(self->radvol, paramFileName, "ATT_QIOn", &IsDefaultChild)) != NULL)) {
    result = ValuesFromXml;
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "ATT_QIOn", &self->radvol->QIOn));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "ATT_QCOn", &self->radvol->QCOn));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "DBZHtoTH", &self->radvol->DBZHtoTH));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "ATT_QI1", &self->ATT_QI1));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "ATT_QI0", &self->ATT_QI0));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "ATT_QIUn", &self->ATT_QIUn));
    if (!IsDefaultChild) {
      result_att = Radvol_getParValueDouble(node, "ATT_a", &self->ATT_a);
      result_att = RAVEMIN(result_att, Radvol_getParValueDouble(node, "ATT_b", &self->ATT_b));
    }
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "ATT_ZRa", &self->ATT_ZRa));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "ATT_ZRb", &self->ATT_ZRb));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "ATT_Refl", &self->ATT_Refl));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "ATT_Last", &self->ATT_Last));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "ATT_Sum", &self->ATT_Sum));
    RAVE_OBJECT_RELEASE(node);
  }
  if (!result_att) {
    if (SameValue(self->radvol->wavelength, cNull) || (self->radvol->wavelength < 2.5) || (self->radvol->wavelength > 15.0)) {
      result = NoValues;
    } else {
      result = ValuesFromWavelength;
      if (self->radvol->wavelength < 3.75) { //X-band
        self->ATT_a = 0.0148;
        self->ATT_b = 1.31;
      } else if (self->radvol->wavelength < 7.5) { //C-band
        self->ATT_a = 0.0044;
        self->ATT_b = 1.17;
      } else { //S-band
        self->ATT_a = 0.0006;
        self->ATT_b = 1.0;
      }
    }
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
 
  snprintf(task_args, 1000, "ATT: ATT_QI1=%3.1f, ATT_QI0=%3.1f, ATT_QIUn=%3.1f, ATT_a=%7.4f, ATT_b=%5.2f, ATT_ZRa=%4.1f, ATT_ZRb=%4.1f, ATT_Refl=%4.1f, ATT_Last=%4.1f, ATT_Sum=%4.1f",
          self->ATT_QI1, self->ATT_QI0, self->ATT_QIUn, self->ATT_a, self->ATT_b, self->ATT_ZRa, self->ATT_ZRb, self->ATT_Refl, self->ATT_Last, self->ATT_Sum);
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
        } else if (SameValue(self->radvol->TabElev[aEle].ReflElev[l + aBin], self->radvol->TabElev[aEle].offset)) {
          if (self->radvol->QIOn) {
            self->radvol->TabElev[aEle].QIElev[l + aBin] = QI;
          }
        } else if ((self->radvol->TabElev[aEle].ReflElev[l + aBin] < self->ATT_Refl) || (AttSum + 0.001 > self->ATT_Sum)) {
          if (self->radvol->QIOn) {
            self->radvol->TabElev[aEle].QIElev[l + aBin] = QI;
          }
          if (self->radvol->QCOn) {
            self->radvol->TabElev[aEle].ReflElev[l + aBin] += AttSum;
          }
        } else {
          R = dBZ2R(self->radvol->TabElev[aEle].ReflElev[l + aBin], self->ATT_ZRa, self->ATT_ZRb);
          AttSpec = self->ATT_a * pow(R, self->ATT_b) * self->radvol->TabElev[aEle].rscale;
          dBZ1 = self->radvol->TabElev[aEle].ReflElev[l + aBin] + AttSum + AttSpec;
          R1 = dBZ2R(dBZ1, self->ATT_ZRa, self->ATT_ZRb);
          AttSpec = self->ATT_a * pow(R1, self->ATT_b) * self->radvol->TabElev[aEle].rscale;
          if (AttSpec > self->ATT_Last * self->radvol->TabElev[aEle].rscale) {
            AttSpec = self->ATT_Last * self->radvol->TabElev[aEle].rscale;
          }
          if (AttSum + AttSpec > self->ATT_Sum) {
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

int RadvolAtt_attCorrection_scan(PolarScan_t* scan, Radvol_params_t* params, char* paramFileName)
{
  RadvolAtt_t* self = RAVE_OBJECT_NEW(&RadvolAtt_TYPE);
  int retval = 0;
  int paramsAtt = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (scan == NULL) {
    RAVE_ERROR0("Polar scan == NULL");
    return retval;
  }
  Radvol_getName(self->radvol, PolarScan_getSource(scan));
  Radvol_getAttrDouble_scan(scan, "how/wavelength", &self->radvol->wavelength);
  paramsAtt = RadvolAttInternal_readParams(self, params, paramFileName);
  if (paramsAtt == ValuesFromWavelength) {
     /* RAVE_WARNING0("Default parameter values for wavelength"); */
  } else if (paramsAtt == NoValues) {
    RAVE_ERROR0("Processing stopped because of lack of correct params in xml file and how/wavelength in input file");
    goto done;
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
    if (!Radvol_load_scan(self->radvol, scan)) {
      RAVE_ERROR0("Processing failed (loading scan)");
      goto done;
    }
    if (!RadvolAttInternal_attCorrection(self)) {
      RAVE_ERROR0("Processing failed (attenuation correction)");
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

int RadvolAtt_attCorrection_pvol(PolarVolume_t* pvol, Radvol_params_t* params, char* paramFileName)
{
  RadvolAtt_t* self = RAVE_OBJECT_NEW(&RadvolAtt_TYPE);
  int retval = 0;
  int paramsAtt = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (pvol == NULL) {
    RAVE_ERROR0("Polar volume == NULL");
    goto done;
  }
  Radvol_getName(self->radvol, PolarVolume_getSource(pvol));
  Radvol_getAttrDouble_pvol(pvol, "how/wavelength", &self->radvol->wavelength);
  paramsAtt = RadvolAttInternal_readParams(self, params, paramFileName);
  if (paramsAtt == ValuesFromWavelength) {
    /* RAVE_WARNING0("Default parameter values for wavelength"); */
  } else if (paramsAtt == NoValues) {
    RAVE_ERROR0("Processing stopped because of lack of correct params in xml file and how/wavelength in input file");
    goto done;
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
    if (!Radvol_load_pvol(self->radvol, pvol)) {
      RAVE_ERROR0("Processing failed (loading volume)");
      goto done;
    }
    if (!RadvolAttInternal_attCorrection(self)) {
      RAVE_ERROR0("Processing failed (attenuation correction)");
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

RaveCoreObjectType RadvolAtt_TYPE = {
  "RadvolAtt",
  sizeof(RadvolAtt_t),
  RadvolAtt_constructor,
  RadvolAtt_destructor,
  RadvolAtt_copyconstructor
};
