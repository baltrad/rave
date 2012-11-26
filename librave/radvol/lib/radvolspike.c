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
 * Radvol-QC algorithms for spike removal.
 * @file radvolspike.c
 * @author Katarzyna Osrodka (Institute of Meteorology and Water Management, IMGW-PIB)
 * @date 2012-07-12
 */
#include "radvolspike.h"
#include "radvol.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include <string.h>

/**
 * Represents the RadvolSpike algorithm
 */
struct _RadvolSpike_t {
  RAVE_OBJECT_HEAD 	/** Always on top */
  Radvol_t* radvol;	/**< volume of reflectivity and QI */
  double SPIKE_QI;	/**< QI<sub>SPIKE</sub> value for external interference signals */
  double SPIKE_QIUn;	/**< QI<sub>SPIKE</sub> value for uncorrected external interference signals */      
  double SPIKE_ACovFrac;/**< Maximum fraction of echo cover to apply the correction (algorithm A) */       
  int    SPIKE_AAzim;	/**< Number of azimuths to find variance across beam (algorithm A) */     
  double SPIKE_AVarAzim;/**< Threshold for variance across beam (algorithm A) */  
  int    SPIKE_ABeam;	/**< Number of pixels along beam to find variance (algorithm A) */     
  double SPIKE_AVarBeam;/**< Threshold for variance along beam (algorithm A) */  
  double SPIKE_AFrac;	/**< Minimum fraction of potential spike gates in beam (algorithm A) */     
  double SPIKE_BDiff;	/**< Minimum difference between the potential spike and vicinity (algorithm B) */     
  int    SPIKE_BAzim;	/**< Number of azimuths to find reflectivity gradient (algorithm B) */     
  double SPIKE_BFrac;	/**< Minimum fraction of potential spike gates in beam (algorithm B) */     
  double SPIKE_CAlt;	/**< Threshold for meteorological echo altitude (algorithm C)  */      
};

/** value for non-spike bin */
#define NoSpike 0
/** value for potential spike A */
#define PotentialASpike -1
/** value for detected spike A */
#define DetectedASpike -2   
/** value for detected spike B */
#define DetectedBSpike -3
/** value for interpolated spike */
#define InterpolatedSpike -4

/*@{ Private functions */
/**
 * Constructor
 */
static int RadvolSpike_constructor(RaveCoreObject* obj)
{
  RadvolSpike_t* self = (RadvolSpike_t*)obj;
  self->radvol = RAVE_OBJECT_NEW(&Radvol_TYPE);
  if (self->radvol == NULL) {
    goto error;
  }
  self->SPIKE_QI = 0.5;       
  self->SPIKE_QIUn = 0.3;      
  self->SPIKE_ACovFrac = 0.9;      
  self->SPIKE_AAzim = 3; 
  self->SPIKE_AVarAzim = 1000.0; 
  self->SPIKE_ABeam = 15; 
  self->SPIKE_AVarBeam = 5.0;
  self->SPIKE_AFrac = 0.45;      
  self->SPIKE_BDiff = 10.0;     
  self->SPIKE_BAzim = 3;  
  self->SPIKE_BFrac = 0.25;     
  self->SPIKE_CAlt = 20.0;     
  return 1;
  
error:
  return 0;
}

/**
 * Copy constructor
 */
static int RadvolSpike_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RadvolSpike_t* this = (RadvolSpike_t*)obj;
  RadvolSpike_t* src = (RadvolSpike_t*)srcobj;
  this->radvol = RAVE_OBJECT_CLONE(src->radvol);
  if (this->radvol == NULL) {
    goto error;
  }
  this->SPIKE_QI = src->SPIKE_QI;
  this->SPIKE_QIUn = src->SPIKE_QIUn;
  this->SPIKE_ACovFrac = src->SPIKE_ACovFrac;
  this->SPIKE_AAzim = src->SPIKE_AAzim;
  this->SPIKE_AVarAzim = src->SPIKE_AVarAzim;
  this->SPIKE_ABeam = src->SPIKE_ABeam;
  this->SPIKE_AVarBeam = src->SPIKE_AVarBeam;
  this->SPIKE_AFrac = src->SPIKE_AFrac;
  this->SPIKE_BDiff = src->SPIKE_BDiff;
  this->SPIKE_BAzim = src->SPIKE_BAzim;
  this->SPIKE_BFrac = src->SPIKE_BFrac;
  this->SPIKE_CAlt = src->SPIKE_CAlt;
  return 1;

error:
  return 0;
}

/**
 * Destructor
 */
static void RadvolSpike_destructor(RaveCoreObject* obj)
{
  RadvolSpike_t* self = (RadvolSpike_t*)obj;
  RAVE_OBJECT_RELEASE(self->radvol);
}

/**
 * Reads algorithm parameters if xml file exists
 * @param self - self
 * @param paramFileName - name of xml file with parameters
 * @returns 1 if all parameters were read, otherwise 0
 */
static int RadvolSpikeInternal_readParams(RadvolSpike_t* self, char* paramFileName)
{
  int result = 0;
  SimpleXmlNode_t* node = NULL;
  
  node = Radvol_getFactorChild(paramFileName, "SPIKE");
  if (node != NULL) {
    result = 1;
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "QIOn", &self->radvol->QIOn));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "QCOn", &self->radvol->QCOn));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "QI", &self->SPIKE_QI));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "QIUn", &self->SPIKE_QIUn));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "ACovFrac", &self->SPIKE_ACovFrac));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "AAzim", &self->SPIKE_AAzim));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "AVarAzim", &self->SPIKE_AVarAzim));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "ABeam", &self->SPIKE_ABeam));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "AVarBeam", &self->SPIKE_AVarBeam));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "AFrac", &self->SPIKE_AFrac));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "BDiff", &self->SPIKE_BDiff));
    result = RAVEMIN(result, Radvol_getParValueInt(node,    "BAzim", &self->SPIKE_BAzim));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "BFrac", &self->SPIKE_BFrac));
    result = RAVEMIN(result, Radvol_getParValueDouble(node, "CAlt", &self->SPIKE_CAlt));
    RAVE_OBJECT_RELEASE(node);
  } 
  return result;
}

/**
 * Provides echo fraction
 * @param aElev - elevation data
 * @returns fraction of echo
 */
static double RadvolSpikeInternal_echoFraction(Elevation_t aElev) {
  long int count = 0;
  int j, k;
  
  for (j = 0; j < aElev.nray; j++) {
    for (k = 0; k < aElev.nbin; k++) {
      if (!SameValue(aElev.ReflElev[j * aElev.nbin + k], cNoRain) && !SameValue(aElev.ReflElev[j * aElev.nbin + k], cNull)) {
	count++;
      }
    }
  }
  return (double) count / aElev.nbin / aElev.nray;
}

/**
 * Checks variances along and across beam for A-type spikes
 * @param aRay - ray number
 * @param aBin - bin number
 * @param aElev - elevation data
 * @param aTabCount - auxillary array with number of potential spike bins in the ray
 * @param aTabVol - auxillary array with flags for particular bins
 */
static void RadvolSpikeInternal_checkVar(RadvolSpike_t* self, int aRay, int aBin, Elevation_t aElev, int *aTabCount, signed char *aTabVol) 
{
  int j;
  long int l1;
  double s = 0.0, s1 = 0.0;
  int count = 0;
  double varL, varAz;
  
  //variance along the beam
  l1 = aRay * aElev.nbin;
  for (j = RAVEMAX(0, aBin - self->SPIKE_ABeam); j <= RAVEMIN(aBin + self->SPIKE_ABeam, aElev.nbin - 1); j++) {
    if (!SameValue(aElev.ReflElev[l1 + j], cNoRain) && !SameValue(aElev.ReflElev[l1 + j], cNull)) {
      s += aElev.ReflElev[l1 + j];
      s1 += pow(aElev.ReflElev[l1 + j], 2);
      count++;
    }
  }
  if (count > 1) {
    varL = (count * s1 - pow(s, 2)) / count / (count - 1);
  } else {
    varL = cNoRain;
  }
  //variance across the beam
  s = 0.0;
  s1 = 0.0;
  count = 0;
  for (j = -self->SPIKE_AAzim; j <= self->SPIKE_AAzim; j++) {
    l1 = ((aRay + j + aElev.nray) % aElev.nray) * aElev.nbin + aBin;
    if (!SameValue(aElev.ReflElev[l1], cNoRain) && !SameValue(aElev.ReflElev[l1], cNull)) {
      s += pow(10.0, aElev.ReflElev[l1] / 10.0);
      s1 += pow(pow(10.0, aElev.ReflElev[l1 ] / 10.0), 2.0);
      count++;
    }
  }
  if (count > 1) {
    varAz = (count * s1 - pow(s, 2.0)) / count / (count - 1);
  } else {
    varAz = cNoRain;
  }
  if ((varAz > self->SPIKE_AVarAzim) && (varL < self->SPIKE_AVarBeam) && (varL > cNoRain)) {
    aTabCount[aRay]++;
    aTabVol[aRay * aElev.nbin + aBin] = PotentialASpike;
  }
}
 
/**
 * Removal of B-type spikes
 * @param aWidth - spike width
 * @param aElev - elevation data
 * @param aTabCount - auxillary array with number of potential spike bins in the ray
 * @param aTabVol - auxillary array with flags for particular bins
 */
static void RadvolSpikeInternal_elevSpikeRemoval(RadvolSpike_t* self,int aWidth, Elevation_t aElev, int *aTabCount, signed char *aTabVol) 
{
  int aBin;
  int aRay;
  long int l, left, right;
  
  for (aRay = 0; aRay < aElev.nray; aRay++) {
    l = aRay * aElev.nbin;
    left = ((aRay - aWidth + aElev.nray) % aElev.nray) * aElev.nbin;
    right = ((aRay + aWidth) % aElev.nray) * aElev.nbin;
    for (aBin = 0; aBin < aElev.nbin; aBin++) {
      if (!SameValue(aElev.ReflElev[l + aBin], cNoRain) && !SameValue(aElev.ReflElev[l + aBin], cNull)
	&&(((aElev.ReflElev[l + aBin] - aElev.ReflElev[left + aBin] > self->SPIKE_BDiff) && SameValue(aElev.ReflElev[left + aBin], cNoRain))
	|| (((aTabVol[((aRay - aWidth + aElev.nray) % aElev.nray) * aElev.nbin + aBin] > aWidth) || (aTabVol[((aRay - aWidth + aElev.nray) % aElev.nray) * aElev.nbin + aBin] == DetectedASpike))))
	&&(((aElev.ReflElev[l + aBin] - aElev.ReflElev[right + aBin] > self->SPIKE_BDiff) && SameValue(aElev.ReflElev[right + aBin], cNoRain))
	|| (((aTabVol[((aRay + aWidth) % aElev.nray) * aElev.nbin + aBin] > aWidth) || (aTabVol[((aRay + aWidth) % aElev.nray) * aElev.nbin + aBin] == DetectedASpike))))
	&& (aTabVol[aRay * aElev.nbin + aBin] <= 0)) {
	aTabCount[aRay]++;
        aTabVol[aRay * aElev.nbin + aBin ] = aWidth;
	}
    }
  }
}

/**
 * Prepares algorithm parameters as task_args
 * @param self - self
 * @returns 1 on success, otherwise 0
 */
static int RadvolSpikeInternal_addTaskArgs(RadvolSpike_t* self)
{
  int result = 0;
  char task_args[1000];
  
  sprintf(task_args, "SPIKE: SPIKE_QI=%3.1f, SPIKE_QIUn=%3.1f, SPIKE_ACovFrac=%3.1f, SPIKE_AAzim=%d, SPIKE_AVarAzim=%8.1f, SPIKE_ABeam=%d, SPIKE_AVarBeam=%3.1f, SPIKE_AFrac=%4.2f, SPIKE_BDiff=%4.1f, SPIKE_BAzim=%d, SPIKE_BFrac=%4.2f, SPIKE_CAlt=%4.1f", 
	  self->SPIKE_QI, self->SPIKE_QIUn, self->SPIKE_ACovFrac, self->SPIKE_AAzim, self->SPIKE_AVarAzim, self->SPIKE_ABeam, self->SPIKE_AVarBeam, self->SPIKE_AFrac, self->SPIKE_BDiff, self->SPIKE_BAzim, self->SPIKE_BFrac, self->SPIKE_CAlt );
  if (Radvol_setTaskArgs(self->radvol, task_args)) {
    result = 1;
  }
  return result;
}

/**
 * Algorithm for spike removal and quality characterization
 * @param self - self
 * @returns 1 on success, otherwise 0
 */
static int RadvolSpikeInternal_spikeRemoval(RadvolSpike_t* self)
{
  int *TabCount = NULL;
  signed char *TabVol = NULL;
  int aEle, aWidth;
  int aRay, aBin;
  int nray, nbin;
  double EchoFrac;
  long int l;
  int SpikeAB, SpikeC;
  int left, right, width;
  double z, z1;
  double QI;
  double EchoMaxHeight;
  
  EchoMaxHeight = self->SPIKE_CAlt * 1000 + self->radvol->altitude;
  QI = self->radvol->QCOn ? self->SPIKE_QI : self->SPIKE_QIUn;
  for (aEle = 0; aEle < self->radvol->nele; aEle++) {
    nbin = self->radvol->TabElev[aEle].nbin;
    nray = self->radvol->TabElev[aEle].nray;
    
    //removal of spike type A
    TabVol = RAVE_MALLOC(sizeof(signed char) * nbin * nray);
    if (TabVol == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory");
      goto error;
    }
    TabCount = RAVE_MALLOC(sizeof(int) * nbin * nray);
    if (TabCount == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory");
      goto error;
    }
    EchoFrac = RadvolSpikeInternal_echoFraction(self->radvol->TabElev[aEle]);
    if (EchoFrac < self->SPIKE_ACovFrac) {
      for (aRay = 0; aRay < nray * nbin; aRay++) {
	TabVol[aRay] = NoSpike;
	TabCount[aRay] = 0;
      }
      for (aRay = 0; aRay < nray; aRay++) {
	for (aBin = 0; aBin < nbin; aBin++) {
	  RadvolSpikeInternal_checkVar(self, aRay, aBin, self->radvol->TabElev[aEle], TabCount, TabVol);
	}  
	if ((double) TabCount[aRay] / nbin > self->SPIKE_AFrac) {
	  l = aRay * nbin;
	  for (aBin = 0; aBin < nbin; aBin++) {
	    if (TabVol[l + aBin] < NoSpike) {
	      TabVol[l + aBin] = DetectedASpike;
	    }
	  }
	}
      }
    }
    //removal of spike type B
    for (aRay = 0; aRay < nray * nbin; aRay++) {
      TabCount[aRay] = 0;
    }
    for (aWidth = self->SPIKE_BAzim; aWidth > 0; aWidth--) {
      RadvolSpikeInternal_elevSpikeRemoval(self, aWidth, self->radvol->TabElev[aEle], TabCount, TabVol);
    }
    for (aRay = 0; aRay < nray; aRay++) {
      if ((double) TabCount[aRay] / nbin > self->SPIKE_BFrac) {
	l = aRay * nbin;
	for (aBin = 0; aBin < nbin; aBin++) {
	  if (TabVol[l + aBin] > NoSpike)
	    TabVol[l + aBin] = DetectedBSpike;
	}
      }
    }
    //interpolation
    for (aRay = 0; aRay < nray; aRay++) {
      SpikeC = 0;
      SpikeAB = 0;
      for (aBin = 0; aBin < nbin; aBin++) {
	if (TabVol[aRay * nbin + aBin]<-1) {
	  SpikeAB = 1;
	  if ((TabVol[aRay * nbin + aBin] == DetectedASpike) || (TabVol[aRay * nbin + aBin] == DetectedBSpike)) {
	    TabVol[aRay * nbin + aBin] = InterpolatedSpike;
	    left = 1;
	    while (TabVol[((aRay - left + nray) % nray) * nbin + aBin]<-1) {
	      TabVol[((aRay - left + nray) % nray) * nbin + aBin] = InterpolatedSpike;
	      left++;
	    }
	    z = self->radvol->TabElev[aEle].ReflElev[((aRay - left + nray) % nray) * nbin + aBin];
	    right = 1;
	    while (TabVol[((aRay + right) % nray) * nbin + aBin]<-1) {
	      TabVol[((aRay + right) % nray) * nbin + aBin] = InterpolatedSpike;
	      right++;
	    }
	    if (self->radvol->QCOn) {
	      if (SameValue(z, cNull)) {
		z = cNoRain;
	      }
	      z1 = self->radvol->TabElev[aEle].ReflElev[((aRay + right) % nray) * nbin + aBin];
	      if (SameValue(z1, cNull)) {
		z1 = cNoRain;
	      }
	      if ((!SameValue(z, cNoRain) || !SameValue(z1, cNoRain)) && (Radvol_getAltitude(self->radvol->TabElev[aEle], aBin) < EchoMaxHeight)) {
		z = (z + self->radvol->TabElev[aEle].ReflElev[((aRay + right) % nray) * nbin + aBin]) / 2;
		for (width = -left + 1; width < right; width++) {
		  self->radvol->TabElev[aEle].ReflElev[((aRay + width + nray) % nray) * nbin + aBin]= z;
		}
		
	      } else {
		for (width = -left + 1; width < right; width++) {
		  self->radvol->TabElev[aEle].ReflElev[((aRay + width + nray) % nray) * nbin + aBin]= cNoRain;
		}
	      }
	    }
	  }
	} else if ((Radvol_getAltitude(self->radvol->TabElev[aEle], aBin) >= EchoMaxHeight)  
	  && !SameValue(self->radvol->TabElev[aEle].ReflElev[aRay * nbin + aBin],cNoRain) 
	  && !SameValue(self->radvol->TabElev[aEle].ReflElev[aRay * nbin + aBin],cNull)) { 
	  //removal of spike type C
	  SpikeC = 1;
	  if (self->radvol->QCOn) {
	    self->radvol->TabElev[aEle].ReflElev[aRay * nbin + aBin] = cNoRain;
	  }
	}
      }
      if (SpikeAB || SpikeC) {
	l = aRay * self->radvol->TabElev[aEle].nbin;
	if (self->radvol->QIOn) {
	  for (aBin = 0; aBin < nbin; aBin++) {
	    self->radvol->TabElev[aEle].QIElev[l + aBin] = QI;
	  }
	}
      }
    }
    RAVE_FREE(TabCount);
    RAVE_FREE(TabVol);
  }
  return 1;
  
error:
  RAVE_FREE(TabCount);
  RAVE_FREE(TabVol);
  return 0;
}

/*@} End of Private functions */

/*@{ Interface functions */

int RadvolSpike_spikeRemoval(PolarVolume_t* pvol, char* paramFileName)
{
  RadvolSpike_t* self = RAVE_OBJECT_NEW(&RadvolSpike_TYPE);
  int retval = 0;
  
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (pvol == NULL) {
    RAVE_ERROR0("Polar volume == NULL");
    return retval;
  }
  if (paramFileName == NULL || !RadvolSpikeInternal_readParams(self, paramFileName)) {
    RAVE_WARNING0("Default parameter values");
  }
  if (self->radvol->QCOn || self->radvol->QIOn) {
    if (!Radvol_setTaskName(self->radvol,"pl.imgw.radvolqc.spike")) {
      RAVE_ERROR0("Processing failed (setting task name)");
      goto done;    
    }
    if (!RadvolSpikeInternal_addTaskArgs(self)) {
      RAVE_ERROR0("Processing failed (setting task args)");
      goto done;    
    } 
    if (!Radvol_loadVol(self->radvol, pvol)) {
      RAVE_ERROR0("Processing failed (loading volume)");
      goto done;
    }
    if (!RadvolSpikeInternal_spikeRemoval(self)) {
      RAVE_ERROR0("Processing failed (spike removal)");
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

RaveCoreObjectType RadvolSpike_TYPE = {
  "RadvolSpike",
  sizeof(RadvolSpike_t),
  RadvolSpike_constructor,
  RadvolSpike_destructor,
  RadvolSpike_copyconstructor
};
