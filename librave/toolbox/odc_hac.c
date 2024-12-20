/* --------------------------------------------------------------------
Copyright (C) 2013 Swedish Meteorological and Hydrological Institute, SMHI

This is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with HLHDF.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/

/** Functionality for performing hit-accumulation clutter filtering.
 *  Added Z-diff operation too.
 * @file
 * @author Daniel Michelson, SMHI
 * @date 2013-01-23
 */

#include "odc_hac.h"
#include "polarscan.h"
#include "rave_attribute.h"
#include "rave_field.h"
#include "rave_types.h"
#include <stdio.h>

int hacFilter(PolarScan_t* scan, RaveField_t* hac, char* quant) {
  PolarScanParam_t* param = NULL;
  RaveField_t* qind = NULL;
  RaveAttribute_t* attr = NULL;
  RaveValueType rvt;
  int retval = 0;
  int ir, ib;
  long nrays, nbins, N;
  double nodata, ni, Pi, val, thresh;
  
  nbins = PolarScan_getNbins(scan);
  nrays = PolarScan_getNrays(scan);

  if (PolarScan_hasParameter(scan, quant)) {
    param = PolarScan_getParameter(scan, quant);
    qind = PolarScan_getQualityFieldByHowTask(scan, "eu.opera.odc.hac");
    nodata = PolarScanParam_getNodata(param);

    if (qind == NULL) {
      qind = RAVE_OBJECT_NEW(&RaveField_TYPE);
      if (qind != NULL) { // (scan.nrays, scan.nbins)
        if (!RaveField_createData(qind, nrays, nbins, RaveDataType_UCHAR)) {
          RAVE_ERROR0("Failed to create data field");
          goto done;
        }
        attr = RaveAttributeHelp_createString("how/task", "eu.opera.odc.hac");
        if (attr == NULL) {
          RAVE_ERROR0("Failed to add how/task to quality field");
          goto done;
        }
        if (!RaveField_addAttribute(qind, attr)) {
          RAVE_ERROR0("Failed to add attribute to qfield");
          goto done;
        }
        RAVE_OBJECT_RELEASE(attr);
        if (!PolarScan_addQualityField(scan, qind)) {
          RAVE_ERROR0("Failed to add hac quality field to scan");
          goto done;
        }
      } else {
        RAVE_ERROR0("Failed to create rave data field");
        goto done;
      }
    }

    attr = RaveField_getAttribute(qind, "how/task_args");
    if (attr == NULL) {
      thresh = 60.0;
      attr = RaveAttributeHelp_createDouble("how/task_args", thresh);
      if (attr == NULL || !RaveField_addAttribute(qind, attr)) {
        RAVE_ERROR0("Failed to add how/task_args to quality field");
        goto done;
      }
    } else {
      RaveAttribute_getDouble(attr, &thresh);
    }
    RAVE_OBJECT_RELEASE(attr);
    
    attr = RaveAttributeHelp_createDouble("what/offset", 0.0);
    if (attr == NULL || !RaveField_addAttribute(qind, attr)) {
      RAVE_ERROR0("Failed to add what/offset to quality field");
      goto done;
    }
    RAVE_OBJECT_RELEASE(attr);

    attr = RaveAttributeHelp_createDouble("what/gain", 1.0/255.0);
    if (attr == NULL || !RaveField_addAttribute(qind, attr)) {
      RAVE_ERROR0("Failed to add what/gain to quality field");
      goto done;
    }
    RAVE_OBJECT_RELEASE(attr);

    attr = RaveField_getAttribute(hac, "how/count");
    RaveAttribute_getLong(attr, &N);
      
    for (ir=0; ir<nrays; ir++) {
      for (ib=0; ib<nbins; ib++) {
        rvt = PolarScanParam_getValue(param, ib, ir, &val);
        RaveField_setValue(qind, ib, ir, 255);
        if (rvt==RaveValueType_DATA) {
          RaveField_getValue(hac, ib, ir, &ni);
          Pi = 100 * (ni/(double)N);
          if (Pi > thresh) {
            PolarScanParam_setValue(param, ib, ir, nodata);
            RaveField_setValue(qind, ib, ir, 0);
          }
        }
      }
    }

    retval = 1;
  }
done:
  RAVE_OBJECT_RELEASE(param);
  RAVE_OBJECT_RELEASE(qind);
  RAVE_OBJECT_RELEASE(attr);
  return retval;
}


int hacIncrement(PolarScan_t* scan, RaveField_t* hac, char* quant) {
  PolarScanParam_t* param = NULL;
  RaveAttribute_t* attr = NULL;
  RaveValueType rvt;
  int retval = 0;
  int ir, ib;
  long nrays, nbins, N;
  double val, ni;
  
  nbins = PolarScan_getNbins(scan);
  nrays = PolarScan_getNrays(scan);

  if (PolarScan_hasParameter(scan, quant)) {
     param = PolarScan_getParameter(scan, quant);

     attr = RaveField_getAttribute(hac, "how/count");
     RaveAttribute_getLong(attr, &N);
     N+=1;
     RaveAttribute_setLong(attr, N);
     
     for (ir=0; ir<nrays; ir++) {
       for (ib=0; ib<nbins; ib++) {
         rvt = PolarScanParam_getValue(param, ib, ir, &val);

         if (rvt==RaveValueType_DATA) {
           RaveField_getValue(hac, ib, ir, &ni);
           ni+=1;
           RaveField_setValue(hac, ib, ir, ni);
         }
       }
     }

     retval = 1;
  }
  RAVE_OBJECT_RELEASE(param);
  RAVE_OBJECT_RELEASE(attr);
  return retval;
}


int zdiff(PolarScan_t* scan, double thresh) {
  PolarScanParam_t* dbzu = NULL;
  PolarScanParam_t* dbzc = NULL;
  RaveField_t* field = NULL;
  RaveValueType rvtu, rvtc;
  int retval = 0;
  int ir, ib;
  long nrays, nbins;
  double uval, cval, diff, quality;
  double gain = 1/255.0;

  nbins = PolarScan_getNbins(scan);
  nrays = PolarScan_getNrays(scan);

  if ( (PolarScan_hasParameter(scan, "TH")) && (PolarScan_hasParameter(scan, "DBZH")) ) {
    dbzu = PolarScan_getParameter(scan, "TH");
    dbzc = PolarScan_getParameter(scan, "DBZH");
    field = PolarScan_getQualityFieldByHowTask(scan, "eu.opera.odc.zdiff");

    for (ir=0; ir<nrays; ir++) {
      for (ib=0; ib<nbins; ib++) {
        rvtu = PolarScanParam_getConvertedValue(dbzu, ib, ir, &uval);
        rvtc = PolarScanParam_getConvertedValue(dbzc, ib, ir, &cval);
        diff = 0.0;
        if ( (rvtu==RaveValueType_DATA) && (rvtc==RaveValueType_DATA) ) {
          diff = uval - cval;
        } else if ( (rvtu==RaveValueType_DATA) && (rvtc==RaveValueType_UNDETECT) ) {
          diff = uval;
        }
        if (diff > thresh) diff = thresh;
        quality = (1.0 - (diff / thresh)) / gain;  /* scale directly to 8 bit */
        RaveField_setValue(field, (long)ir, (long)ib, quality);
      }
    }
    retval = 1;
  }

  RAVE_OBJECT_RELEASE(field);
  RAVE_OBJECT_RELEASE(dbzu);
  RAVE_OBJECT_RELEASE(dbzc);
  return retval;
}
