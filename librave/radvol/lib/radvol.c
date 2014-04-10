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
/*! \mainpage
 *  Radvol-QC is software developed in IMGW-PIB (Poland) for corrections 
 *  and generation of quality information for volumes of weather radar 
 *  data. The work has been performed in the frame of the BALTRAD Project.
 *  
 *  At present the following algorithms are included in the Radvol-QC package:
 *   - BROAD: Assessment of distance to radar related effects (for quality characterization),
 *   - SPIKE: Removal of geometrically shaped non-meteorological echoes (from sun,emitters, etc.) (for data correction and quality characterization),
 *   - NMET: Removal of non-meteorological echoes (for data correction and quality characterization),
 *   - SPECK: Removal of measurement noise (specks) (for data correction and quality characterization),
 *   - [ BLOCK: Beam blockage correction (for data correction and quality characterization) - included into beamb package ],
 *   - ATT: Correction for attenuation in rain (for data correction and quality characterization).
 *
 * Total quality characterization can be estimated applying QI_TOTAL algorithm.
 */

/**
 * Radvol-QC general structures and algorithms.
 * @file radvol.c
 * @author Katarzyna Osrodka (Institute of Meteorology and Water Management, IMGW-PIB)
 * @date 2012-12-20
 */
#include "radvol.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include <string.h>
#include <math.h>


/*@{ Private functions */
/**
 * Constructor
 */
static int Radvol_constructor(RaveCoreObject* obj)
{
  Radvol_t* self = (Radvol_t*)obj;
  self->nele = 0;
  self->TabElev = NULL;
  self->beamwidth = cNull;
  self->wavelength = cNull;
  self->pulselength = cNull;
  self->Eer = cEer;
  self->name = NULL;
  self->task_name = NULL;
  self->task_args = NULL;
  self->QIOn = 1;
  self->QCOn = 1;
  self->DBZHtoTH = 1;
  return 1;
}

/**
 * Sets name of radar
 * @param self - self
 * @param aname - name of radar
 * @returns 1 on success, otherwise 0
 */
static int RadvolInternal_setName(Radvol_t* self, const char* aname)
{
  char* tmp = NULL;
  int result = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");

  if (aname != NULL) {
    tmp = RAVE_STRDUP(aname);
    if (tmp == NULL) {
      goto done;
    }
  }
  RAVE_FREE(self->name);
  self->name = tmp;
  tmp = NULL;
  result = 1;

done:
  RAVE_FREE(tmp);
  return result;
}

static int Radvol_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  int i = 0;
  long j = 0;
  
  Radvol_t* this = (Radvol_t*)obj;
  Radvol_t* src = (Radvol_t*)srcobj;
  this->nele = src->nele;
  if (this->nele > 0) {
    this->TabElev = RAVE_MALLOC(sizeof(Elevation_t) * this->nele);
    if (this->TabElev == NULL) {
      goto error;
    }
    for (i = 0; i < this->nele; i++) {
      this->TabElev[i].nbin = src->TabElev[i].nbin;
      this->TabElev[i].nray = src->TabElev[i].nray;
      this->TabElev[i].rscale = src->TabElev[i].rscale;
      this->TabElev[i].elangle = src->TabElev[i].elangle;
      this->TabElev[i].gain = src->TabElev[i].gain;   
      this->TabElev[i].offset = src->TabElev[i].offset;
      this->TabElev[i].nodata = src->TabElev[i].nodata;
      this->TabElev[i].undetect = src->TabElev[i].undetect;
      this->TabElev[i].ReflElev = RAVE_MALLOC(sizeof(double) * this->TabElev[i].nbin * this->TabElev[i].nray);
      this->TabElev[i].QIElev = RAVE_MALLOC(sizeof(double) * this->TabElev[i].nbin * this->TabElev[i].nray);
      if (this->TabElev[i].ReflElev == NULL ||  this->TabElev[i].QIElev == NULL) {
        for (j = 0; j <= i; j++) {
          RAVE_FREE(this->TabElev[j].ReflElev);
          RAVE_FREE(this->TabElev[j].QIElev);
        }
        goto error;
      }
      for (j = 0; j < this->TabElev[i].nbin * this->TabElev[i].nray; j++) {
        this->TabElev[i].ReflElev[j] = src->TabElev[i].ReflElev[j];
        this->TabElev[i].QIElev[j] = src->TabElev[i].QIElev[j];
        }
    }
  } else {
    this->TabElev = NULL;
  }
  this->beamwidth = src->beamwidth;
  this->wavelength = src->wavelength;
  this->pulselength = src->pulselength;
  this->Eer = src->Eer;
  this->altitude = src->altitude;
  this->name = NULL;
  if (!RadvolInternal_setName(this, src->name)) {
    goto error;
  }
  this->task_name = NULL;
  if (!Radvol_setTaskName(this, src->task_name)) {
    goto error;    
  }
  this->task_args = NULL;
  if (!Radvol_setTaskArgs(this, src->task_args)) {
    goto error;    
  }
  this->QIOn = src->QIOn;
  this->QCOn = src->QCOn;
  this->DBZHtoTH = src->DBZHtoTH;
  return 1;
  
error:
  RAVE_FREE(this->TabElev);
  RAVE_FREE(this->name);
  RAVE_FREE(this->task_name);
  RAVE_FREE(this->task_args);
  return 0;
}

/**
 * Destructor
 */
static void Radvol_destructor(RaveCoreObject* obj)
{
  int i = 0;

  Radvol_t* self = (Radvol_t*)obj;

  if (self->nele > 0) {
    for (i = 0; i < self->nele; i++) {
      RAVE_FREE(self->TabElev[i].ReflElev);
      RAVE_FREE(self->TabElev[i].QIElev);
    }
    RAVE_FREE(self->TabElev);
  }  
  RAVE_FREE(self->name);
  RAVE_FREE(self->task_name);
  RAVE_FREE(self->task_args);
}

/**
 * In PolarScanParameter updates attribute with separator and new_value
 * @param param - PolarScanParameter to be updated
 * @param name - attribute name to be updated
 * @param new_value - value to be added
 * @param sep - separator
 * @returns 1 on success, otherwise 0
 */
static int RadvolInternal_updateAttribute(PolarScanParam_t* param, const char* name, const char* new_value, const char* sep)
{
  int result = 0;
  RaveAttribute_t* attribute = NULL;
  char* value = NULL;
  char task[1000];
  
  RAVE_ASSERT((param != NULL), "param == NULL");
  RAVE_ASSERT((name != NULL), "name == NULL");
  RAVE_ASSERT((new_value != NULL), "new_value == NULL");

  attribute = PolarScanParam_getAttribute(param, name);
  if (attribute == NULL) {
    attribute = RaveAttributeHelp_createString(name, new_value);
  } else {
    RaveAttribute_getString(attribute, &value);
    if (value == NULL) {
      attribute = RaveAttributeHelp_createString(name, new_value);
    } else {  
      strcpy(task, value);
      if (value != NULL) {
        strcat(task, sep);
      }
      strcat(task, new_value);
      attribute = RaveAttributeHelp_createString(name, task);
    }
  }
  if  (!(attribute == NULL) && PolarScanParam_addAttribute(param, attribute)) {
    result = 1;
  }
  RAVE_OBJECT_RELEASE(attribute);
  return result;
}

/**
 * Reads particular polar scan into radvolqc structure
 * @param self - self
 * @param scan - polar scan
 * @param aEle - elevation number
 * @returns 1 on success, otherwise 0
 */
static int RadvolInternal_loadScanData(Radvol_t* self, PolarScan_t* scan, int aEle) 
{
  PolarScanParam_t* param_in = NULL;
  PolarScanParam_t* param_out = NULL;
  int bi, ri;
  double v;
  int result = 0;

  if (PolarScan_getDataType(scan) == RaveDataType_UCHAR) {
    self->TabElev[aEle].elangle = PolarScan_getElangle(scan);
    self->TabElev[aEle].nbin = PolarScan_getNbins(scan);
    self->TabElev[aEle].nray = PolarScan_getNrays(scan);
    self->TabElev[aEle].rscale = PolarScan_getRscale(scan) / 1000.0;

    if ((!PolarScan_hasParameter(scan, "TH")) && (PolarScan_hasParameter(scan, "DBZH"))) {
      param_out = PolarScan_getParameter(scan, "DBZH");
      if ((self->QCOn) && (self->DBZHtoTH)) {
        param_in = RAVE_OBJECT_CLONE(param_out);
        PolarScanParam_setQuantity(param_in, "TH");
        PolarScan_addParameter(scan, param_in);
      }
    } else if ((PolarScan_hasParameter(scan, "TH")) && (!PolarScan_hasParameter(scan, "DBZH"))) {
      param_in = PolarScan_getParameter(scan, "TH");
      param_out = RAVE_OBJECT_CLONE(param_in);
      PolarScanParam_setQuantity(param_out, "DBZH");
      if (self->QCOn) {
        PolarScan_addParameter(scan, param_out);
      }
    } else if ((PolarScan_hasParameter(scan, "TH")) && (PolarScan_hasParameter(scan, "DBZH"))) {
      param_out = PolarScan_getParameter(scan, "DBZH");
    }
    if (param_out == NULL) {
      RAVE_ERROR0("Incorrect input file - processing stopped");
      goto done;
    }

    self->TabElev[aEle].gain = PolarScanParam_getGain(param_out);
    self->TabElev[aEle].offset = PolarScanParam_getOffset(param_out);
    self->TabElev[aEle].nodata = PolarScanParam_getNodata(param_out);
    self->TabElev[aEle].undetect = PolarScanParam_getUndetect(param_out);

    self->TabElev[aEle].ReflElev = RAVE_MALLOC(sizeof (double) * self->TabElev[aEle].nbin * self->TabElev[aEle].nray);
    self->TabElev[aEle].QIElev = RAVE_MALLOC(sizeof (double) * self->TabElev[aEle].nbin * self->TabElev[aEle].nray);
    if (self->TabElev[aEle].ReflElev == NULL || self->TabElev[aEle].QIElev == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory");
      goto done;
    }

    for (ri = 0; ri < self->TabElev[aEle].nray; ri++) {
      for (bi = 0; bi < self->TabElev[aEle].nbin; bi++) {
        self->TabElev[aEle].QIElev[ri * self->TabElev[aEle].nbin + bi] = QI_GOOD;
        PolarScanParam_getValue(param_out, bi, ri, &v);
        if (SameValue(v, self->TabElev[aEle].undetect)) {
          self->TabElev[aEle].ReflElev[ri * self->TabElev[aEle].nbin + bi] = self->TabElev[aEle].offset;
        } else if (SameValue(v, self->TabElev[aEle].nodata)) {
          self->TabElev[aEle].ReflElev[ri * self->TabElev[aEle].nbin + bi] = cNull;
          self->TabElev[aEle].QIElev[ri * self->TabElev[aEle].nbin + bi] = QI_BAD;
        } else {
          self->TabElev[aEle].ReflElev[ri * self->TabElev[aEle].nbin + bi] = (v * self->TabElev[aEle].gain) + self->TabElev[aEle].offset;
        }
      }
    }
    RAVE_OBJECT_RELEASE(param_in);
    RAVE_OBJECT_RELEASE(param_out);
  } else {
    RAVE_ERROR0("Incorrect scan - processing stopped");
    goto done;
  }
  result = 1;

done:
  RAVE_OBJECT_RELEASE(param_in);
  RAVE_OBJECT_RELEASE(param_out);
  return result;
}

/**
 * Writes data for particular elevation from radvolqc into polar scan
 * @param self - self
 * @param scan - polar scan
 * @param aEle - elevation number
 * @returns 1 on success, otherwise 0
 */
static int RadvolInternal_saveScanData(Radvol_t* self, PolarScan_t* scan, int aEle) {
  PolarScanParam_t* parameter = NULL;
  RaveAttribute_t* attribute = NULL;
  RaveField_t* field = NULL;
  int bi, ri;
  int result = 0;

  //update reflectivity
  if ((self->TabElev[aEle].ReflElev != NULL) && (PolarScan_hasParameter(scan, "DBZH")) && self->QCOn && (self->task_name != NULL) && (self->task_args != NULL)) {
    for (ri = 0; ri < self->TabElev[aEle].nray; ri++) {
      for (bi = 0; bi < self->TabElev[aEle].nbin; bi++) {
        if (SameValue(self->TabElev[aEle].ReflElev[ri * self->TabElev[aEle].nbin + bi], self->TabElev[aEle].offset)) {
          PolarScan_setParameterValue(scan, "DBZH", bi, ri, (int) self->TabElev[aEle].undetect);
        } else if (SameValue(self->TabElev[aEle].ReflElev[ri * self->TabElev[aEle].nbin + bi], cNull)) {
          PolarScan_setParameterValue(scan, "DBZH", bi, ri, (int) self->TabElev[aEle].nodata);
        } else {
          PolarScan_setParameterValue(scan, "DBZH", bi, ri, ((self->TabElev[aEle].ReflElev[ri * self->TabElev[aEle].nbin + bi] - self->TabElev[aEle].offset) / self->TabElev[aEle].gain));
        }
      }
    }
    parameter = PolarScan_getParameter(scan, "DBZH");
    if (!RadvolInternal_updateAttribute(parameter, "how/task", self->task_name, "; ")) {
      RAVE_ERROR0("Failed to add task name to DBZH");
      goto error;
    }
    if (!RadvolInternal_updateAttribute(parameter, "how/task_args", self->task_args, ";\n")) {
      RAVE_ERROR0("Failed to add task name to DBZH");
      goto error;
    }
    RAVE_OBJECT_RELEASE(parameter);
  }
  //save quality
  if ((self->TabElev[aEle].QIElev != NULL) && self->QIOn && (self->task_name != NULL) && (self->task_args != NULL)) {
    field = RAVE_OBJECT_NEW(&RaveField_TYPE);
    attribute = RaveAttributeHelp_createString("how/task", self->task_name);
    if ((attribute == NULL) || !RaveField_addAttribute(field, attribute)) {
      RAVE_ERROR0("Failed to add quality field");
      goto error;
    }
    RAVE_OBJECT_RELEASE(attribute);
    attribute = RaveAttributeHelp_createString("how/task_args", self->task_args);
    if ((attribute == NULL) || !RaveField_addAttribute(field, attribute)) {
      RAVE_ERROR0("Failed to add quality field");
      goto error;
    }
    RAVE_OBJECT_RELEASE(attribute);
    attribute = RaveAttributeHelp_createString("what/quantity", "QIND");
    if ((field == NULL) || (attribute == NULL) || !RaveField_createData(field, self->TabElev[aEle].nbin, self->TabElev[aEle].nray, RaveDataType_UCHAR) || !RaveField_addAttribute(field, attribute)) {
      RAVE_ERROR0("Failed to add quality field");
      goto error;
    }
    RAVE_OBJECT_RELEASE(attribute);
    attribute = RaveAttributeHelp_createDouble("what/gain", 0.003937);
    if ((attribute == NULL) || !RaveField_addAttribute(field, attribute)) {
      RAVE_ERROR0("Failed to add quality field");
      goto error;
    }
    RAVE_OBJECT_RELEASE(attribute);
    attribute = RaveAttributeHelp_createDouble("what/nodata", 0.0);
    if ((attribute == NULL) || !RaveField_addAttribute(field, attribute)) {
      RAVE_ERROR0("Failed to add quality field");
      goto error;
    }
    RAVE_OBJECT_RELEASE(attribute);
    attribute = RaveAttributeHelp_createDouble("what/undetect", 0.0);
    if ((attribute == NULL) || !RaveField_addAttribute(field, attribute)) {
      RAVE_ERROR0("Failed to add quality field");
      goto error;
    }
    RAVE_OBJECT_RELEASE(attribute);
    attribute = RaveAttributeHelp_createDouble("what/offset", -0.003937);
    for (ri = 0; ri < self->TabElev[aEle].nray; ri++) {
      for (bi = 0; bi < self->TabElev[aEle].nbin; bi++) {
        RaveField_setValue(field, bi, ri, (self->TabElev[aEle].QIElev[ri * self->TabElev[aEle].nbin + bi] + 1 / 254.0) * 254.0);
      }
    }
    if ((attribute == NULL) || !RaveField_addAttribute(field, attribute) || !PolarScan_addQualityField(scan, field)) {
      RAVE_ERROR0("Failed to add quality field to the cartesian product");
      goto error;
    }
    RAVE_OBJECT_RELEASE(field);
  }
  result = 1;

error:
  RAVE_OBJECT_RELEASE(field);
  RAVE_OBJECT_RELEASE(attribute);
  RAVE_OBJECT_RELEASE(parameter);
  return result;
}
/*@} End of Private functions */

/*@{ Interface functions */

void Radvol_getName(Radvol_t* self, const char* source) {
  char* tmp = NULL;
  char* asource = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (source != NULL) {
    asource = RAVE_STRDUP(source);
    if (asource != NULL) {
      tmp = strtok(asource, ",");
      while (tmp != NULL) {
        if (strspn(tmp, "NOD") == 3) {
          tmp = strtok(tmp, ":");
          if (tmp != NULL) {
            tmp = strtok(NULL, "");
            if (tmp != NULL) {
              RadvolInternal_setName(self, tmp);
            }
          }
          break;
        }
        tmp = strtok(NULL, ",");
      }
    }
    RAVE_FREE(asource);
  }
}

int Radvol_getAttrDouble_scan(PolarScan_t* scan, char* name, double* value)
{
  RaveAttribute_t* attribute = NULL;
  int result = 0;

  RAVE_ASSERT((scan != NULL), "scan == NULL");
  attribute = PolarScan_getAttribute(scan, name);
  result = (attribute != NULL) && RaveAttribute_getDouble(attribute, value);
  RAVE_OBJECT_RELEASE(attribute);
  return result;
}

int Radvol_getAttrDouble_pvol(PolarVolume_t* pvol, char* name, double* value)
{
  RaveAttribute_t* attribute = NULL;
  int result = 0;

  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  attribute = PolarVolume_getAttribute(pvol, name);
  result = (attribute != NULL) && RaveAttribute_getDouble(attribute, value);
  RAVE_OBJECT_RELEASE(attribute);
  return result;
}

int Radvol_setTaskName(Radvol_t* self, const char* task_name)
{
 char* tmp = NULL;
  int result = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");

  if (task_name != NULL) {
    tmp = RAVE_STRDUP(task_name);
    if (tmp == NULL) {
      goto done;
    }
  }
  RAVE_FREE(self->task_name);
  self->task_name = tmp;
  tmp = NULL;
  result = 1;

done:
  RAVE_FREE(tmp);
  return result;
}

int Radvol_setTaskArgs(Radvol_t* self, const char* task_args)
{
  char* tmp = NULL;
  int result = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");

  if (task_args != NULL) {
    tmp = RAVE_STRDUP(task_args);
    if (tmp == NULL) {
      goto done;
    }
  }
  RAVE_FREE(self->task_args);
  self->task_args = tmp;
  tmp = NULL;
  result = 1;

done:
  RAVE_FREE(tmp);
  return result;
}

int Radvol_load_scan(Radvol_t* self, PolarScan_t* scan)
{

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  self->nele = 1;
  self->beamwidth = PolarScan_getBeamwidth(scan);
  self->altitude = PolarScan_getHeight(scan);
  self->TabElev = RAVE_MALLOC(sizeof(Elevation_t) * self->nele);
  if (self->TabElev == NULL) {
    RAVE_CRITICAL0("Failed to allocate memory");
  }
  return RadvolInternal_loadScanData(self, scan, 0);
}

int Radvol_load_pvol(Radvol_t* self, PolarVolume_t* pvol)
{
  int result = 0;
  PolarScan_t* scan = NULL;
  int aEle;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");
  
  self->nele = PolarVolume_getNumberOfScans(pvol);
  self->beamwidth = PolarVolume_getBeamwidth(pvol);
  self->altitude = PolarVolume_getHeight(pvol);
  self->TabElev = RAVE_MALLOC(sizeof (Elevation_t) * self->nele);
  if (self->TabElev == NULL) {
    RAVE_CRITICAL0("Failed to allocate memory");
    goto done;
  }
  
  if (!PolarVolume_isAscendingScans(pvol)) {
    PolarVolume_sortByElevations(pvol, 1);
  }
  for (aEle = 0; aEle < self->nele; aEle++) {
    scan = PolarVolume_getScan(pvol, aEle);
    if ((scan == NULL) || (!RadvolInternal_loadScanData(self, scan, aEle))) {
      RAVE_ERROR0("Error in reading scan - processing stopped");
      goto done;
    }
    RAVE_OBJECT_RELEASE(scan);
  }
  result = 1;
  
done:
  RAVE_OBJECT_RELEASE(scan);
  return result;
}

int Radvol_save_scan(Radvol_t* self, PolarScan_t* scan) {

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((scan != NULL), "scan == NULL");

  if (RadvolInternal_saveScanData(self, scan, 0)) {
    return 1;
  } else {
    RAVE_ERROR0("Incorrect scan - processing stopped");
    return 0;
  }
}

int Radvol_save_pvol(Radvol_t* self, PolarVolume_t* pvol)
{
  PolarScan_t* scan = NULL;
  int aEle;
  int result = 0;
  
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((pvol != NULL), "pvol == NULL");

  for (aEle = 0; aEle < self->nele; aEle++) {
    scan = PolarVolume_getScan(pvol, aEle);
    if ((scan != NULL) && RadvolInternal_saveScanData(self, scan, aEle)) {
      RAVE_OBJECT_RELEASE(scan);
    } else {
      RAVE_ERROR0("Incorrect scan - processing stopped");
      goto done;
    }
  }
  return 1;
  
done:
  RAVE_OBJECT_RELEASE(scan);
  return result;
}

SimpleXmlNode_t* Radvol_getFactorChild(Radvol_t* self, char* aFileName, char* aFactorName, int* IsDefault)
{
  SimpleXmlNode_t* node = NULL;
  SimpleXmlNode_t* result = NULL;

  RAVE_ASSERT((aFileName != NULL), "filename == NULL");
  RAVE_ASSERT((aFactorName != NULL), "FactorName == NULL");
  *IsDefault = 0;
  node = SimpleXmlNode_parseFile(aFileName);
  if (node == NULL) {
    goto done;
  }
  if ((self == NULL) || (self->name == NULL) || ((result = SimpleXmlNode_getChildByName(node, self->name)) == NULL) || (SimpleXmlNode_getChildByName(result, aFactorName) == NULL)) {
    result = SimpleXmlNode_getChildByName(node, "default");
    *IsDefault = 1;
  }
  
done:
  RAVE_OBJECT_RELEASE(node);
  return result;
}

int Radvol_getParValueDouble(SimpleXmlNode_t* node, char* aParamName, double* value) 
{
  SimpleXmlNode_t* child = NULL;
  const char* attr = NULL;
  int result = 0;
  
  RAVE_ASSERT((node != NULL), "node == NULL");
  RAVE_ASSERT((aParamName != NULL), "aParamName == NULL");

  child = SimpleXmlNode_getChildByName(node,aParamName);
  if (child != NULL) {
    attr = SimpleXmlNode_getText(child);
    if (attr != NULL) {
      *value = atof(attr);
      result = 1;
    }
  }
  RAVE_OBJECT_RELEASE(child);
  return result;
}

int Radvol_getParValueInt(SimpleXmlNode_t* node, char* aParamName, int* value) 
{
  SimpleXmlNode_t* child = NULL;
  const char* attr = NULL;
  int result = 0;
  
  RAVE_ASSERT((node != NULL), "node == NULL");
  RAVE_ASSERT((aParamName != NULL), "aParamName == NULL");
  
  child = SimpleXmlNode_getChildByName(node,aParamName);
  if (child != NULL) {
    attr = SimpleXmlNode_getText(child);
    if (attr != NULL) {
      *value = atoi(attr);
      result = 1;
    }
  }
  RAVE_OBJECT_RELEASE(child);
  return result;
}

void Radvol_setEquivalentEarthRadius(Radvol_t* self, double lat) {
  double radius = 0L;
  double a = 0L;
  double b = 0L;

  a = sin(lat) * DEFAULT_EQUATOR_RADIUS;
  b = cos(lat) * DEFAULT_POLE_RADIUS;
  radius = sqrt(a * a + b * b) / 1000;
  self->Eer = 1.0 / ((1.0 / radius) - 3.9e-5);
}

double Radvol_getCurvature(Radvol_t* self, int ele, int aBin)
{
  return (pow(pow((aBin + 1) * self->TabElev[ele].rscale, 2)
  + self->Eer * self->Eer
  + 2 * (aBin + 1) * self->TabElev[ele].rscale * self->Eer * sin(self->TabElev[ele].elangle), 0.5)
  - self->Eer);
}

double Radvol_getLinearQuality(double x, double a, double b)
{
  if (x > b) {
    return QI_BAD;
  } else if (x < a) {
    return QI_GOOD;
  } else if (SameValue(a, b)) {
     return QI_GOOD;
  } else {
    return (b - x) / (b - a);
  }
}

/*@} End of Interface functions */

RaveCoreObjectType Radvol_TYPE = {
    "Radvol",
    sizeof(Radvol_t),
    Radvol_constructor,
    Radvol_destructor,
    Radvol_copyconstructor
};
