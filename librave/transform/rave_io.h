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
#ifndef RAVE_IO_H
#define RAVE_IO_H

#include "polarvolume.h"
#include "cartesian.h"

//(PVOL|CVOL|SCAN|RAY|AZIM|IMAGE|COMP|XSEC|VP|PIC)
//(SCAN|PPI|CAPPI|PCAPPI|ETOP|MAX|RR|VIL|COMP|VP|RHI|XSEC|VSP|HSP|RAY|AZIM|QUAL)

typedef enum RaveIO_ObjectType {
  RaveIO_ObjectType_UNDEFINED = -1,
  RaveIO_ObjectType_PVOL = 0,
  RaveIO_ObjectType_CVOL = 1,
  RaveIO_ObjectType_SCAN,
  RaveIO_ObjectType_RAY,
  RaveIO_ObjectType_AZIM,
  RaveIO_ObjectType_IMAGE,
  RaveIO_ObjectType_COMP,
  RaveIO_ObjectType_XSEC,
  RaveIO_ObjectType_VP,
  RaveIO_ObjectType_PIC
} RaveIO_ObjectType;

typedef enum RaveIO_ODIM_Version {
  RaveIO_ODIM_Version_UNDEFINED = -1,
  RaveIO_ODIM_Version_2_0 = 0,
} RaveIO_ODIM_Version;

/**
 * Defines a Rave IO instance
 */
typedef struct _RaveIO_t RaveIO_t;

/**
 * Creates a new RaveIO instance.
 * @return a new instance or NULL on failure
 */
RaveIO_t* RaveIO_new(void);

/**
 * Releases the responsibility for the rave IO instance, it is not certain that
 * it will be deleted though if there still are references existing
 * to this rave io.
 * @param[in] transform - the transformer
 */
void RaveIO_release(RaveIO_t* raveio);

/**
 * Copies the reference to this instance by increasing a
 * reference counter.
 * @param[in] transform - the transformer to be copied
 * @return a pointer to the scan
 */
RaveIO_t* RaveIO_copy(RaveIO_t* raveio);

/**
 * Closes the HDF5 file but will keep the RaveIO instance.
 * @param[in] raveio - the rave IO instance
 */
void RaveIO_close(RaveIO_t* raveio);

/**
 * Opens a HDF5 file.
 * @param[in] filename - the file that should be opened
 * @returns a RaveIO_t instance on success, otherwise NULL
 */
RaveIO_t* RaveIO_open(const char* filename);

/**
 * Loads a HDF5 file into the RaveIO instance.
 * @param[in] raveio - the Rave IO instance
 * @param[in] filename - the HDF5 file to open
 * @returns 0 on failure, otherwise 1
 */
int RaveIO_openFile(RaveIO_t* raveio, const char* filename);

/**
 * Returns if the HDF5 nodelist has been read into memory or not.
 * @param[in] raveio- the Rave IO instance
 * @returns 1 if nodelist is loaded, otherwise 0
 */
int RaveIO_isOpen(RaveIO_t* raveio);

/**
 * Loads a scan at the specified index where the format for the node will
 * be /dataset<dsindex>/data<dindex>.
 * @param[in] raveio - the Rave IO instance
 * @param[in] dsindex - the dataset index
 * @param[in] dindex  - the data index
 * @returns the loaded scan or NULL on failure.
 */
PolarScan_t* RaveIO_loadScanIndex(RaveIO_t* raveio, const int dsindex, const int dindex);

/**
 * Loads a polar volume.
 * @param[in] raveio - the Rave IO instance
 * @returns a Polar Volume on success, otherwise NULL
 */
PolarVolume_t* RaveIO_loadVolume(RaveIO_t* raveio);

/**
 * Verifies if the currently opened file is supported
 * by the RaveIO interface.
 * @param[in] raveio - the Rave IO instance
 * @returns 1 if it is supported, otherwise 0
 */
int RaveIO_isSupported(RaveIO_t* raveio);

/**
 * Returns the object type for the currently opened file.
 * @param[in] raveio - the Rave IO instance
 * @returns the object type or RaveIO_ObjectType_UNDEFINED on error.
 */
RaveIO_ObjectType RaveIO_getObjectType(RaveIO_t* raveio);

/**
 * Returns the ODIM version for the opened file.
 * @param[in] raveio - the Rave IO instance
 * @returns a valid ODIM version or RaveIO_ODIM_Version_UNDEFINED.
 */
RaveIO_ODIM_Version RaveIO_getOdimVersion(RaveIO_t* raveio);

#endif
