/* --------------------------------------------------------------------
Copyright (C) 2009 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Defines the functions available when working with polar scans
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-15
 */
#include "polarscan.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>

/**
 * Represents one scan in a volume.
 */
struct _PolarScan_t {
  long ps_refCount;

  // Where
  double elangle; /**< elevation of scan */
  long nbins; /**< number of bins */
  double rscale; /**< scale */
  RaveDataType type; /**< data type */
  long nrays; /**< number of rays / scan */
  double rstart; /**< start of ray */
  long a1gate; /**< something */

  // How
  double beamwidth;

  // What
  char quantity[64]; /**< what does this data represent */
  double gain; /**< gain when scaling */
  double offset; /**< offset when scaling */
  double nodata; /**< nodata */
  double undetect; /**< undetect */

  // Data
  void* data; /**< data ptr */

  // Miscellaneous data that is useful
  void* voidPtr; /**< a pointer for pointing to miscellaneous data */

  // Debugging
  int debug; /**< indicates if debugging should be active or not */
};

/*@{ Private functions */
/**
 * Destroys the scan
 * @param[in] scan - the scan to destroy
 */
static void PolarScan_destroy(PolarScan_t* scan)
{
  if (scan != NULL) {
    RAVE_FREE(scan->data);
    RAVE_FREE(scan);
  }
}
/*@} End of Private functions */

/*@{ Interface functions */
PolarScan_t* PolarScan_new(void)
{
  PolarScan_t* result = NULL;
  result = RAVE_MALLOC(sizeof(PolarScan_t));
  if (result != NULL) {
    result->elangle = 0.0;
    result->nbins = 0;
    result->rscale = 0.0;
    result->type = RaveDataType_UNDEFINED;
    result->nrays = 0;
    result->rstart = 0.0;
    result->a1gate = 0;
    result->beamwidth = 0.0;
    strcpy(result->quantity, "");
    result->gain = 0.0;
    result->offset = 0.0;
    result->nodata = 0.0;
    result->undetect = 0.0;
    result->data = NULL;

    result->ps_refCount = 1;
    result->voidPtr = NULL;
    result->debug = 0;
  }
  return result;
}

void PolarScan_release(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->ps_refCount--;
  if (scan->ps_refCount <= 0) {
    PolarScan_destroy(scan);
  }
}

PolarScan_t* PolarScan_copy(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->ps_refCount++;
  return scan;
}

void PolarScan_setElangle(PolarScan_t* scan, double elangle)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->elangle = elangle;
}

double PolarScan_getElangle(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->elangle;
}

void PolarScan_setNbins(PolarScan_t* scan, long nbins)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->nbins = nbins;
}

long PolarScan_getNbins(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->nbins;
}

void PolarScan_setRscale(PolarScan_t* scan, double rscale)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->rscale = rscale;
}

double PolarScan_getRscale(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->rscale;
}

void PolarScan_setNrays(PolarScan_t* scan, long nrays)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->nrays = nrays;
}

long PolarScan_getNrays(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->nrays;
}

void PolarScan_setRstart(PolarScan_t* scan, double rstart)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->rstart = rstart;
}

double PolarScan_getRstart(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->rstart;
}

int PolarScan_setDataType(PolarScan_t* scan, RaveDataType type)
{
  int result = 0;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  if (type >= RaveDataType_UNDEFINED && type < RaveDataType_LAST) {
    scan->type = type;
    result = 1;
  }
  return result;
}

RaveDataType PolarScan_getDataType(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->type;
}

void PolarScan_setA1gate(PolarScan_t* scan, long a1gate)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->a1gate = a1gate;
}

long PolarScan_getA1gate(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->a1gate;
}

void PolarScan_setBeamWidth(PolarScan_t* scan, long beamwidth)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->beamwidth = beamwidth;
}

double PolarScan_getBeamWidth(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->beamwidth;
}

void PolarScan_setQuantity(PolarScan_t* scan, const char* quantity)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  if (quantity != NULL) {
    strcpy(scan->quantity, quantity);
  }
}

const char* PolarScan_getQuantity(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return (const char*)scan->quantity;
}

void PolarScan_setGain(PolarScan_t* scan, double gain)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->gain = gain;
}

double PolarScan_getGain(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->gain;
}

void PolarScan_setOffset(PolarScan_t* scan, double offset)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->offset = offset;
}

double PolarScan_getOffset(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->offset;
}

void PolarScan_setNodata(PolarScan_t* scan, double nodata)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->nodata = nodata;
}

double PolarScan_getNodata(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->nodata;
}

void PolarScan_setUndetect(PolarScan_t* scan, double undetect)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->undetect = undetect;
}

double PolarScan_getUndetect(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->undetect;
}

int PolarScan_setData(PolarScan_t* scan, long nbins, long nrays, void* data, RaveDataType type)
{
  long sz = 0;
  long nbytes = 0;
  void* ptr = NULL;
  int result = 0;

  RAVE_ASSERT((scan != NULL), "scan was NULL");

  sz = get_ravetype_size(type);
  nbytes = nbins*nrays*sz;
  ptr = RAVE_MALLOC(nbytes);

  if (ptr == NULL) {
    RAVE_CRITICAL1("Failed to allocate memory (%d bytes)", (int)nbytes);
    goto fail;
  }
  memcpy(ptr, data, nbytes);
  RAVE_FREE(scan->data);
  scan->data = ptr;
  PolarScan_setNbins(scan, nbins);
  PolarScan_setNrays(scan, nrays);
  PolarScan_setDataType(scan, type);
  result = 1;
fail:
  return result;
}

void* PolarScan_getData(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->data;
}

int PolarScan_getRangeIndex(PolarScan_t* scan, double r)
{
  int result = -1;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  RAVE_ASSERT((scan->nbins > 0), "nbins must be > 0");
  RAVE_ASSERT((scan->rscale > 0.0), "rscale must be > 0.0");

  if (r <= 0.0) {
    result = 0;
  } else {
    result = (int)rint((r/scan->rscale));
  }
  if (result >= scan->nbins || result < 0) {
    result = -1;
  }
  return result;
}

int PolarScan_getAzimuthIndex(PolarScan_t* scan, double a)
{
  int result = -1;
  double azOffset = 0.0L;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  RAVE_ASSERT((scan->nrays > 0), "nrays must be > 0");

  azOffset = 2*M_PI/scan->nrays;
  result = (int)rint(a/azOffset);
  if (result >= scan->nrays) {
    result -= scan->nrays;
  } else if (result < 0) {
    result += scan->nrays;
  }
  return result;
}

RaveValueType PolarScan_getValueAtIndex(PolarScan_t* scan, int ai, int ri, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  RAVE_ASSERT((v != NULL), "v was NULL");
  *v = scan->nodata;
  if (ai >= 0 && ai < scan->nrays && ri >= 0 && ri < scan->nbins) {
    result = RaveValueType_DATA;
    *v = get_array_item_2d(scan->data, ri, ai, scan->type, scan->nbins);
    if (*v == scan->nodata) {
      result = RaveValueType_NODATA;
    } else if (*v == scan->undetect) {
      result = RaveValueType_UNDETECT;
    }
  }
  return result;
}

RaveValueType PolarScan_getValueAtAzimuthAndRange(PolarScan_t* scan, double a, double r, double* v)
{
  RaveValueType result = RaveValueType_NODATA;
  int ai = 0, ri = 0;
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  RAVE_ASSERT((v != NULL), "v was NULL");
  *v = scan->nodata;
  ai = PolarScan_getAzimuthIndex(scan, a);
  if (ai < 0) {
    goto done;
  }
  ri = PolarScan_getRangeIndex(scan, r);
  if (ri < 0) {
    goto done;
  }

  result = PolarScan_getValueAtIndex(scan, ai, ri, v);
done:
  return result;
}

void PolarScan_setVoidPtr(PolarScan_t* scan, void* ptr)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->voidPtr = ptr;
}

void* PolarScan_getVoidPtr(PolarScan_t* scan)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  return scan->voidPtr;
}

void PolarScan_setDebug(PolarScan_t* scan, int enable)
{
  RAVE_ASSERT((scan != NULL), "scan was NULL");
  scan->debug = enable;
}
/*@} End of Interface functions */
