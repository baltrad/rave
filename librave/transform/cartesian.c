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
 * Defines the functions available when working with cartesian products
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-16
 */
#include "cartesian.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>

/**
 * Represents one scan in a volume.
 */
struct _Cartesian_t {
  long ps_refCount;

  // Where
  long xsize;
  long ysize;
  double xscale;
  double yscale;
  RaveDataType type; /**< data type */

  double llX;
  double llY;
  double urX;
  double urY;

  // What
  char quantity[64]; /**< what does this data represent */
  double gain; /**< gain when scaling */
  double offset; /**< offset when scaling */
  double nodata; /**< nodata */
  double undetect; /**< undetect */

  Projection_t* projection;

  // Data
  void* data; /**< data ptr */

  // Miscellaneous data that is useful
  void* voidPtr; /**< a pointer for pointing to miscellaneous data */
};

/*@{ Private functions */
/**
 * Destroys the cartesian product
 * @param[in] scan - the cartesian product to destroy
 */
static void Cartesian_destroy(Cartesian_t* cartesian)
{
  if (cartesian != NULL) {
    Projection_release(cartesian->projection);
    RAVE_FREE(cartesian->data);
    RAVE_FREE(cartesian);
  }
}
/*@} End of Private functions */

/*@{ Interface functions */
Cartesian_t* Cartesian_new(void)
{
  Cartesian_t* result = NULL;
  result = RAVE_MALLOC(sizeof(Cartesian_t));
  if (result != NULL) {
    result->ps_refCount = 1;
    result->type = RaveDataType_UNDEFINED;
    result->xsize = 0;
    result->ysize = 0;
    result->xscale = 0.0;
    result->yscale = 0.0;
    result->llX = 0.0;
    result->llY = 0.0;
    result->urX = 0.0;
    result->urY = 0.0;
    strcpy(result->quantity, "");
    result->gain = 0.0;
    result->offset = 0.0;
    result->nodata = 0.0;
    result->undetect = 0.0;
    result->projection = NULL;
    result->data = NULL;

    result->voidPtr = NULL;
  }
  return result;
}

void Cartesian_release(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->ps_refCount--;
  if (cartesian->ps_refCount <= 0) {
    Cartesian_destroy(cartesian);
  }
}

Cartesian_t* Cartesian_copy(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->ps_refCount++;
  return cartesian;
}

void Cartesian_setXSize(Cartesian_t* cartesian, long xsize)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->xsize = xsize;
}

long Cartesian_getXSize(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->xsize;
}

void Cartesian_setYSize(Cartesian_t* cartesian, long ysize)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->ysize = ysize;
}

long Cartesian_getYSize(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->ysize;
}

void Cartesian_setAreaExtent(Cartesian_t* cartesian, double llX, double llY, double urX, double urY)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->llX = llX;
  cartesian->llY = llY;
  cartesian->urX = urX;
  cartesian->urY = urY;
}

void Cartesian_getAreaExtent(Cartesian_t* cartesian, double* llX, double* llY, double* urX, double* urY)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  if (llX != NULL) {
    *llX = cartesian->llX;
  }
  if (llY != NULL) {
    *llY = cartesian->llY;
  }
  if (urX != NULL) {
    *urX = cartesian->urX;
  }
  if (urY != NULL) {
    *urY = cartesian->urY;
  }
}

void Cartesian_setXScale(Cartesian_t* cartesian, double xscale)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->xscale = xscale;
}

double Cartesian_getXScale(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->xscale;
}

void Cartesian_setYScale(Cartesian_t* cartesian, double yscale)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->yscale = yscale;
}

double Cartesian_getYScale(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->yscale;
}

double Cartesian_getLocationX(Cartesian_t* cartesian, long x)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->llX + cartesian->xscale * x;
}

double Cartesian_getLocationY(Cartesian_t* cartesian, long y)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->urY - cartesian->yscale * y;
}

int Cartesian_setDataType(Cartesian_t* cartesian, RaveDataType type)
{
  int result = 0;
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  if (type >= RaveDataType_UNDEFINED && type < RaveDataType_LAST) {
    cartesian->type = type;
    result = 1;
  }
  return result;
}

RaveDataType Cartesian_getDataType(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->type;
}

void Cartesian_setQuantity(Cartesian_t* cartesian, const char* quantity)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  if (quantity != NULL) {
    strcpy(cartesian->quantity, quantity);
  }
}

const char* Cartesian_getQuantity(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return (const char*)cartesian->quantity;
}

void Cartesian_setGain(Cartesian_t* cartesian, double gain)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->gain = gain;
}

double Cartesian_getGain(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->gain;
}

void Cartesian_setOffset(Cartesian_t* cartesian, double offset)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->offset = offset;
}

double Cartesian_getOffset(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->offset;
}

void Cartesian_setNodata(Cartesian_t* cartesian, double nodata)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->nodata = nodata;
}

double Cartesian_getNodata(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->nodata;
}

void Cartesian_setUndetect(Cartesian_t* cartesian, double undetect)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->undetect = undetect;
}

double Cartesian_getUndetect(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->undetect;
}

void Cartesian_setProjection(Cartesian_t* cartesian, Projection_t* projection)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  Projection_release(cartesian->projection);
  cartesian->projection = NULL;
  if (projection != NULL) {
    cartesian->projection = Projection_copy(projection);
  }
}

Projection_t* Cartesian_getProjection(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  if (cartesian->projection != NULL) {
    return Projection_copy(cartesian->projection);
  }
  return NULL;
}

int Cartesian_setData(Cartesian_t* cartesian, long xsize, long ysize, void* data, RaveDataType type)
{
  long sz = 0;
  long nbytes = 0;
  void* ptr = NULL;
  int result = 0;

  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");

  sz = get_ravetype_size(type);
  nbytes = xsize*ysize*sz;
  ptr = RAVE_MALLOC(nbytes);

  if (ptr == NULL) {
    RAVE_CRITICAL1("Failed to allocate memory (%d bytes)", (int)nbytes);
    goto fail;
  }
  memcpy(ptr, data, nbytes);
  RAVE_FREE(cartesian->data);
  cartesian->data = ptr;
  Cartesian_setXSize(cartesian, xsize);
  Cartesian_setYSize(cartesian, ysize);
  Cartesian_setDataType(cartesian, type);
  result = 1;
fail:
  return result;
}

void Cartesian_setVoidPtr(Cartesian_t* cartesian, void* ptr)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  cartesian->voidPtr = ptr;
}

void* Cartesian_getVoidPtr(Cartesian_t* cartesian)
{
  RAVE_ASSERT((cartesian != NULL), "cartesian was NULL");
  return cartesian->voidPtr;
}
/*@} End of Interface functions */
