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
 * Functions for performing rave related IO operations, mostly ODIM-formatted HDF5 files.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-11-12
 */
#include "rave_io.h"
#include "rave_debug.h"
#include "rave_alloc.h"
//#include "hlhdf.h"

/**
 * Defines the structure for the RaveIO in a volume.
 */
struct _RaveIO_t {
  long ps_refCount;

};

/*@{ Private functions */
/**
 * Destroys the RaveIO instance
 * @param[in] scan - the cartesian product to destroy
 */
static void RaveIO_destroy(RaveIO_t* raveio)
{
  if (raveio != NULL) {
    RAVE_FREE(raveio);
  }
}
/*@} End of Private functions */

RaveIO_t* RaveIO_new(void)
{
  RaveIO_t* result = NULL;
  result = RAVE_MALLOC(sizeof(RaveIO_t));
  if (result != NULL) {
    result->ps_refCount = 1;
  }
  return result;
}

void RaveIO_release(RaveIO_t* raveio)
{
  if (raveio != NULL) {
    raveio->ps_refCount--;
    if (raveio->ps_refCount <= 0) {
      RaveIO_destroy(raveio);
    }
  }
}

RaveIO_t* RaveIO_copy(RaveIO_t* raveio)
{
  if (raveio != NULL) {
    raveio->ps_refCount++;
  }
  return raveio;
}

PolarVolume_t* RaveIO_loadVolume(RaveIO_t* raveio, const char* filename)
{
  PolarVolume_t* result = NULL;
/*
  HL_NodeList* nodelist = NULL;
  nodelist = HLNodeList_read(filename);
  if (nodelist == NULL) {
    RAVE_ERROR1("Failed to load hdf5 file %s\n", filename);
    goto done;
  }

  //result = RaveH5_buildVolumeFromNodelist(nodelist);
*/
done:
  return result;
}
/*
PolarVolume_t* RaveIO_buildVolumeFromNodelist(RaveIO_t* raveio, HL_NodeList* nodelist)
{
  PolarVolume_t* result = NULL;
  HL_Node* node = NULL;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  if ((node = HLNodeList_fetchNode(nodelist, "/Conventions")) == NULL) {
    RAVE_ERROR0("Could not find /Conventions in file");
    goto done;
  }


done:
  return result;
}
*/
