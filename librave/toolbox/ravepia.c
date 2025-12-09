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
 * This object does support \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-12-03
 */
#include "ravepia.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "raveutil.h"
#include <string.h>
#include <math.h>
#include <stdio.h>


#define DEFAULT_MAX_PIA 10.0

#define CBAND_CO_ZK_POWER   7.34e5                                  /* Coefficient of Z-k power law of C-band */
#define CBAND_EXP_ZK_POWER  1.344                                   /* Exponent of Z-k power law of C-band */

#define XBAND_CO_ZK_POWER   9.25e4                                  /* Coefficient of Z-k power law of X-band */
#define XBAND_EXP_ZK_POWER  1.25                                    /* Exponent of Z-k power law of X-band */

#define HOW_TASK_NAME "se.smhi.qc.hitschfeld-bordan"

/**
 * Represents the gra applier
 */
struct _RavePIA_t {
  RAVE_OBJECT_HEAD /** Always on top */
  double coeff_zk_power; /**< coefficient fot the z/k power relation */
  double exp_zk_power;   /**< exponent fot the z/k power relation */
  double max_pia;        /**< max allowed PIA adjustment */
  double rr;             /**< the range resolution in km */
};

/*@{ Private functions */

/**
 * Constructor
 */
static int RavePIA_constructor(RaveCoreObject* obj)
{
  RavePIA_t* self = (RavePIA_t*)obj;
  self->coeff_zk_power = CBAND_CO_ZK_POWER;
  self->exp_zk_power = CBAND_EXP_ZK_POWER;
  self->max_pia = DEFAULT_MAX_PIA;
  self->rr = 0.0;
  return 1;
}

/**
 * Copy constructor
 */
static int RavePIA_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  RavePIA_t* self = (RavePIA_t*)obj;
  RavePIA_t* src = (RavePIA_t*)obj;
  self->coeff_zk_power = src->coeff_zk_power;
  self->exp_zk_power = src->exp_zk_power;
  self->max_pia = src->max_pia;
  self->rr = src->rr;
  return 1;
}

/**
 * Destructor
 */
static void RavePIA_destructor(RaveCoreObject* obj)
{
}

/*@} End of Private functions */

/*@{ Interface functions */

const char* RavePIA_getHowTaskName()
{
  return HOW_TASK_NAME;
}

void RavePIA_setZkPowerCoefficient(RavePIA_t* self, double c)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->coeff_zk_power = c;
}

double RavePIA_getZkPowerCoefficient(RavePIA_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->coeff_zk_power;
}

void RavePIA_setZkPowerExponent(RavePIA_t* self, double d)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->exp_zk_power = d;
}

double RavePIA_getZkPowerExponent(RavePIA_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->exp_zk_power;
}

void RavePIA_setPiaMax(RavePIA_t* self, double maxv)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->max_pia = maxv;
}

double RavePIA_getPiaMax(RavePIA_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->max_pia;
}

void RavePIA_setRangeResolution(RavePIA_t* self, double rr)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->rr = rr;
}

double RavePIA_getRangeResolution(RavePIA_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->rr;
}

RaveField_t* RavePIA_calculatePIA(RavePIA_t* self, PolarScan_t* scan, const char* quantity, double* outDr)
{
  PolarScanParam_t* param = NULL;
  RaveData2D_t* pdata = NULL;
  RaveField_t *field = NULL, *result = NULL;

  double rscale = 0.0, dr = 0.0;
  long nrays = 0, nbins = 0, ri = 0, bi = 0;
  double kacumsum_factor = 0.0, kacumsum_limit = 0.0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (scan == NULL || !PolarScan_hasParameter(scan, quantity)) {
    RAVE_ERROR1("Must provide scan with correct quantity: %s", quantity);
    return NULL;
  }
  param = PolarScan_getParameter(scan, quantity);
  if (param == NULL) {
    RAVE_ERROR0("Runtime error.... this should not be possible...");
    goto fail;
  }
  pdata = PolarScanParam_getData2D(param); /* We want data2d since there are multiple matrix operations in there, like cumsum.... */
  if (pdata == NULL) {
    RAVE_ERROR0("Memory error");
    goto fail;
  }

  nbins = PolarScanParam_getNbins(param);
  nrays = PolarScanParam_getNrays(param);

  field = RAVE_OBJECT_NEW(&RaveField_TYPE);
  if (field == NULL || !RaveField_createData(field, nbins, nrays, RaveDataType_DOUBLE)) {
    RAVE_ERROR0("Failed to allocate rave field");
    goto fail;
  }

  rscale = PolarScan_getRscale(scan);

  dr = rscale / 1000.0;
  if (self->rr != 0.0) {
    dr = self->rr;
  }

  kacumsum_factor = 0.2 * dr  * log(10.0) / self->exp_zk_power;
  kacumsum_limit = 1.0 / kacumsum_factor;

  for (ri = 0; ri < nrays; ri++) {
    double kacumsum = 0.0, pia = 0.0;
    for (bi = 0; bi < nbins; bi++) {
      double v = 0.0, nv = 0.0;
      RaveValueType vt = PolarScanParam_getConvertedValue(param, bi, ri, &v);
      if (vt == RaveValueType_DATA) {
        nv = pow((pow(10, 0.1 * v) / self->coeff_zk_power), (1.0 / self->exp_zk_power));
        kacumsum += nv;
        if (kacumsum > kacumsum_limit) {
          kacumsum = kacumsum_limit - 1e-06;
        }
        pia = -10.0 * self->exp_zk_power * log10(1 - kacumsum_factor * kacumsum);
        if (pia > self->max_pia) {
          pia = self->max_pia;
        }
        RaveField_setValue(field, bi, ri, pia);
      } else {
        RaveField_setValue(field, bi, ri, pia);
      }
    }
  }
  if (outDr != NULL) {
    *outDr = dr;
  }
  result = RAVE_OBJECT_COPY(field);
fail:
  RAVE_OBJECT_RELEASE(param);
  RAVE_OBJECT_RELEASE(pdata);
  RAVE_OBJECT_RELEASE(field);
  return result;
}

PolarScanParam_t* RavePIA_createPIAParameter(RavePIA_t* self, PolarScan_t* scan, const char* quantity, RaveField_t** outPIA, double* outDr)
{
  RaveField_t* PIA = NULL;
  PolarScanParam_t *param = NULL, *result = NULL, *dbzhparam = NULL;
  double gain = 0.0, offset = 0.0, nodata = 0.0, undetect = 0.0;
  long nrays = 0, nbins = 0, ri = 0, bi = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");

  if (scan == NULL || !PolarScan_hasParameter(scan, quantity)) {
    RAVE_ERROR1("Must provide scan with correct quantity: %s", quantity);
    return NULL;
  }
  dbzhparam = PolarScan_getParameter(scan, quantity);
  if (dbzhparam == NULL) {
    RAVE_ERROR0("Runtime error.... this should not be possible...");
    goto fail;
  }

  gain = PolarScanParam_getGain(dbzhparam);
  offset = PolarScanParam_getOffset(dbzhparam);
  nodata = PolarScanParam_getNodata(dbzhparam);
  undetect = PolarScanParam_getUndetect(dbzhparam);
  nbins = PolarScanParam_getNbins(dbzhparam);
  nrays = PolarScanParam_getNrays(dbzhparam);

  if (gain == 0.0) {
    RAVE_ERROR0("Gain = 0, changing to 1....");
    gain = 1.0;
  }

  PIA = RavePIA_calculatePIA(self, scan, quantity, outDr);
  if (PIA != NULL) {
    param = RAVE_OBJECT_NEW(&PolarScanParam_TYPE);
    if (param == NULL || !PolarScanParam_createData(param, nbins, nrays, RaveDataType_DOUBLE)) {
      goto fail;
    }
    if (!PolarScanParam_setQuantity(param, "PIA")) {
      goto fail;
    }
    PolarScanParam_setGain(param, gain);
    PolarScanParam_setOffset(param, offset);
    PolarScanParam_setNodata(param, nodata);
    PolarScanParam_setUndetect(param, undetect);
    for (ri = 0; ri < nrays; ri++) {
      for (bi = 0; bi < nbins; bi++) {
        double v = 0.0;
        RaveField_getValue(PIA, bi, ri, &v);
        PolarScanParam_setValue(param, bi, ri, (v - offset) / gain);
      }
    }
  }
  if (outPIA != NULL) {
    *outPIA = RAVE_OBJECT_COPY(PIA);
  }
  result = RAVE_OBJECT_COPY(param);
fail:
  RAVE_OBJECT_RELEASE(dbzhparam);
  RAVE_OBJECT_RELEASE(PIA);
  RAVE_OBJECT_RELEASE(param);

  return result;
}

int RavePIA_process(RavePIA_t* self, PolarScan_t* scan, const char* quantity, int addparam, int reprocessquality, int apply)
{
  RaveField_t* PIA = NULL;
  PolarScanParam_t *param = NULL, *dbzh = NULL;
  RaveField_t* qfield = NULL;
  RaveAttribute_t* attr = NULL;

  int result = 0;
  double dr = 0.0;

  RAVE_ASSERT((self != NULL), "self == NULL");

  param = RavePIA_createPIAParameter(self, scan, quantity, &PIA, &dr);
  if (param == NULL) {
    goto fail;
  }

  if (addparam && !PolarScan_addParameter(scan, param)) {
    RAVE_ERROR0("Could not add parameter");
    goto fail;
  }

  if (reprocessquality || !PolarScan_getQualityFieldByHowTask(scan, HOW_TASK_NAME)) {
    qfield = PolarScanParam_toField(param);
    if (qfield != NULL) {
      attr = RaveAttributeHelp_createString("how/task", HOW_TASK_NAME);
      if (attr == NULL || !RaveField_addAttribute(qfield, attr)) {
        goto fail;
      }
      RAVE_OBJECT_RELEASE(attr);
      attr = RaveAttributeHelp_createStringFmt("how/task_args", "param_name=PIA c_ZK=%.2g d_ZK=%.4g PIAmax=%.2g dr=%g", self->coeff_zk_power, self->exp_zk_power, self->max_pia, dr);
      if (attr == NULL || !RaveField_addAttribute(qfield, attr)) {
        goto fail;
      }
      RAVE_OBJECT_RELEASE(attr);

      if (!PolarScan_addOrReplaceQualityField(scan, qfield)) {
        goto fail;
      }

      RAVE_OBJECT_RELEASE(qfield);
    }
  }

  if (apply) {
    long nrays = 0, nbins = 0, ri = 0, bi = 0;
    double gain = 1.0, offset = 0.0;
    dbzh = PolarScan_getParameter(scan, quantity);
    if (dbzh == NULL) {
      RAVE_ERROR0("Failed to acquire dbzh parameter");
      goto fail;
    }

    nbins = PolarScanParam_getNbins(dbzh);
    nrays = PolarScanParam_getNrays(dbzh);
    gain = PolarScanParam_getGain(dbzh);
    offset = PolarScanParam_getOffset(dbzh);
    
    for (ri = 0; ri < nrays; ri++) {
      for (bi = 0; bi < nbins; bi++) {
        double v = 0.0, pv = 0.0;
        RaveValueType vt = PolarScanParam_getConvertedValue(dbzh, bi, ri, &v);
        if (vt == RaveValueType_DATA) {
          RaveField_getValue(PIA, bi, ri, &pv);
          PolarScanParam_setValue(dbzh, bi, ri, ((v + pv - offset)/gain));
        }
      }
    }
  }

  result = 1;
fail:
  RAVE_OBJECT_RELEASE(attr);
  RAVE_OBJECT_RELEASE(qfield);
  RAVE_OBJECT_RELEASE(PIA);
  RAVE_OBJECT_RELEASE(param);
  RAVE_OBJECT_RELEASE(dbzh);
  return result;
}


/*@} End of Interface functions */

RaveCoreObjectType RavePIA_TYPE = {
    "RavePIA",
    sizeof(RavePIA_t),
    RavePIA_constructor,
    RavePIA_destructor,
    RavePIA_copyconstructor
};
