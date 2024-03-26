/* --------------------------------------------------------------------
Copyright (C) 2019 The Crown (i.e. Her Majesty the Queen in Right of Canada)

This file is an add-on to RAVE.

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
 * Functionality for deriving depolarization ratio and using it to quality
 * control other quantities
 * @file dr_qc.c
 * @author Daniel Michelson, Environment and Climate Change Canada
 * @date 2019-04-11
 */
#include "dr_qc.h"


double drCalculate(double ZDR, double RHOHV, double zdr_offset) {
  if (zdr_offset != 0.0) ZDR = ZDR + zdr_offset;
  double zdr = pow(10.0,((double)(ZDR/10.0)));  /* linearize ZDR */
  double rhohv = MY_MIN(RHOHV, MAX_RHOHV);      /* Sanity check on RHOHV */
  double NUM = zdr + 1 - 2 * pow(zdr, 0.5) * rhohv;
  double DEN = zdr + 1 + 2 * pow(zdr, 0.5) * rhohv;
  double DR = NUM / DEN;

  return (double)10 * log10(DR);
}


int drDeriveParameter(PolarScan_t *scan, double zdr_offset) {
  PolarScanParam_t *ZDR = NULL;
  PolarScanParam_t *RHOHV = NULL;
  PolarScanParam_t *DR = NULL;

  int nrays, nbins, ray, bin;
  double scaled;

  if ( (PolarScan_hasParameter(scan, "ZDR")) && 
       (PolarScan_hasParameter(scan, "RHOHV")) ) {

    nrays = (int)PolarScan_getNrays(scan);
    nbins = (int)PolarScan_getNbins(scan);
    ZDR = PolarScan_getParameter(scan, "ZDR");
    RHOHV = PolarScan_getParameter(scan, "RHOHV");

    /* Create a new parameter to store depolarization ratio */
    DR = RAVE_OBJECT_NEW(&PolarScanParam_TYPE);
    PolarScanParam_setGain(DR, DR_GAIN);
    PolarScanParam_setOffset(DR, DR_OFFSET);
    scaled = (DR_NODATA - DR_OFFSET) / DR_GAIN;
    PolarScanParam_setNodata(DR, scaled);
    scaled = (DR_UNDETECT - DR_OFFSET) / DR_GAIN;
    PolarScanParam_setUndetect(DR, scaled);
    PolarScanParam_setQuantity(DR, "DR");
    PolarScanParam_createData(DR, (long)nbins, (long)nrays, RaveDataType_UCHAR);

    for (ray=0; ray<nrays; ray++) {
      for (bin=0; bin<nbins; bin++) {
	double ZDRval, RHOHVval;
	double DRdb = 0.0;
	RaveValueType ZDRvtype, RHOHVvtype;
	ZDRvtype = PolarScanParam_getConvertedValue(ZDR, bin, ray, &ZDRval);
	RHOHVvtype = PolarScanParam_getConvertedValue(RHOHV, bin, ray, &RHOHVval);
	/* Normally, we expect that parameters match up, but we know there are
	   cases where they don't, so we have to manage such situations. */

	/* Valid ZDR and RHOHV */
	if ( (ZDRvtype == RaveValueType_DATA) && 
	     (RHOHVvtype == RaveValueType_DATA) ) {
	  DRdb = drCalculate(ZDRval, RHOHVval, zdr_offset);
	} else if 
	   /* No ZDR but valid RHOHV, assume ZDR==0 when calculating DR.
	      It is preferable to threshold RHOHV only, but we can't represent
	      the result rationally with the DR parameter. The likelihood of it 
	      happening should be minimal, so we won't make the effort. */
	   ( (ZDRvtype == RaveValueType_UNDETECT) && 
	     (RHOHVvtype == RaveValueType_DATA) ) {
	  DRdb = drCalculate(0.0, RHOHVval, zdr_offset);
	} else if 
	   /* Valid ZDR but no RHOHV, cannot do anything meaningful */
	   ( (ZDRvtype == RaveValueType_DATA) && 
	     (RHOHVvtype == RaveValueType_UNDETECT) ) {
	  DRdb = DR_NODATA;
	} else {
	  DRdb = DR_UNDETECT;
	}
	if ( (DRdb > DR_OFFSET) && (DRdb < (DR_OFFSET + DR_GAIN)) ) {
	  DRdb = DR_OFFSET + DR_GAIN;
	}
	scaled = (DRdb - DR_OFFSET) / DR_GAIN;
	PolarScanParam_setValue(DR, bin, ray, round(scaled));
      }
    }
    PolarScan_addParameter(scan, DR);  /* Add DR to the scan */

    RAVE_OBJECT_RELEASE(ZDR);
    RAVE_OBJECT_RELEASE(RHOHV);
    RAVE_OBJECT_RELEASE(DR);
  } else {
    /* alert: required parameter(s) missing */
    return 0;
  }
  return 1;
}


int drSpeckleFilter(PolarScan_t *scan, char* param_name, int kernely, int kernelx, double param_thresh, double dr_thresh) {
  PolarScanParam_t *param, *DR, *cloned;
  int nrays, nbins, ray, bin;  /* dimensions and iterators */
  int n, c, w, u, y, Y, x, X;  /* more iterators and counters */
  double undetect;
  RaveValueType pType, drType;  /* DATA, UNDETECT, or NODATA */
  double pv, drv;               /* Actual parameter and DR values */

  if ( (!PolarScan_hasParameter(scan, param_name)) || 
       (!PolarScan_hasParameter(scan, "DR")) ) {
    return 0;
  }

  param = PolarScan_removeParameter(scan, param_name);
  cloned = RAVE_OBJECT_CLONE(param);
  DR = PolarScan_getParameter(scan, "DR");
  nrays = (int)PolarScan_getNrays(scan);
  nbins = (int)PolarScan_getNbins(scan);
  undetect = PolarScanParam_getUndetect(param);

  for (ray=0;ray<nrays;ray++) {
    for (bin=0;bin<nbins;bin++) {
      n = 0;  /* total sample size */
      c = 0;  /* non-met counts */
      w = 0;  /* met counts */
      u = 0;  /* undetect counts */
      pType = PolarScanParam_getConvertedValue(param, bin, ray, &pv);
      if (pType != RaveValueType_NODATA) {

	for (y=-kernely;y<kernely+1;y++) {
	  if ((ray+y) < 0) {
	    Y = nrays + ray+y;
	  } else if ((ray+y) > (nrays-1)) {
	    Y = ray+y - nrays;
	  } else {
	    Y = ray + y;
	  }

	  for (x=-kernelx;x<kernelx+1;x++) {
	    if ( ((bin+x) >= 0) && ((bin+x) < nbins) ) {
	      X = bin + x;

	      /* The filter */
	      pType = PolarScanParam_getConvertedValue(param, X, Y, &pv);
	      drType = PolarScanParam_getConvertedValue(DR, X, Y, &drv);

	      if ( (pType == RaveValueType_DATA) && 
		   (drType == RaveValueType_DATA) ) {
		if ( ((pv<param_thresh) && (drv<=dr_thresh)) ||
		     (pv>=param_thresh) ) {
		  w += 1;  /* weather */
		} else if ( (pv<param_thresh) && (drv>dr_thresh) ) {
		  c += 1;  /* nonmet */
		}

	      /* Count as valid weather cases where DR could not be determined */
	      } else if ( (pType == RaveValueType_DATA) && 
			  (drType == RaveValueType_NODATA) ) {
		w += 1;

	      /* Can only be undetect */
	      } else if (pType == RaveValueType_UNDETECT) {
		/* no echo, only if param==UNDETECT
		   There will be cases where DR==UNDETECT, that should be 
		   ignored. This will lower the total sample, but that's OK. */
		u += 1;
	      }
	    }
	  }
	}
      }
      n = w + c + u;
      if (n) {
	double nonmet = (double)c / (double)n;
	double wx = (double)w / (double)n;
	double mostly_clear = (double)u / (double)n;

	if ( (nonmet > wx) || (mostly_clear > wx) ) {
	  pType = PolarScanParam_getConvertedValue(param, bin, ray, &pv);
	  if (pType == RaveValueType_DATA) {
	    PolarScanParam_setValue(cloned, bin, ray, undetect);
	  }
	}
      }
    }
  }
  PolarScan_addParameter(scan, cloned);
  RAVE_OBJECT_RELEASE(param);
  RAVE_OBJECT_RELEASE(cloned);
  RAVE_OBJECT_RELEASE(DR);

  return 1;
}
