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
 * Defines the functions available when transforming between different
 * types of products
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-10-20
 */
#ifndef TRANSFORM_H
#define TRANSFORM_H
#include "rave_transform.h"
#include "polarvolume.h"
#include "cartesian.h"

/**
 * Defines a transformer
 */
typedef struct _Transform_t Transform_t;

/**
 * Creates a new transform instance. Default method is NEAREST.
 * @return a new instance or NULL on failure
 */
Transform_t* Transform_new(void);

/**
 * Releases the responsibility for the scan, it is not certain that
 * it will be deleted though if there still are references existing
 * to this scan.
 * @param[in] transform - the transformer
 */
void Transform_release(Transform_t* transform);

/**
 * Copies the reference to this instance by increasing a
 * reference counter.
 * @param[in] transform - the transformer to be copied
 * @return a pointer to the scan
 */
Transform_t* Transform_copy(Transform_t* transform);

/**
 * Sets the transformation method to be used, like NEAREST, BILINEAR, ...
 * @param[in] transform - the transformer
 * @param[in] method - the transformation method
 * @return 0 if method is not known, otherwise the method was set
 */
int Transform_setMethod(Transform_t* transform, RaveTransformationMethod method);

/**
 * Returns the transformation method.
 * @param[in] transform - the transformer
 * @return the transformation method
 */
RaveTransformationMethod Transform_getMethod(Transform_t* transform);

/**
 * Creates a ppi from a polar scan.
 * @param[in] transform - the transformer
 * @param[in] scan - the polar scan
 * @param[in] cartesian - the cartesian product
 * @returns 0 on failure, otherwise success
 */
int Transform_ppi(Transform_t* transform, PolarScan_t* scan, Cartesian_t* cartesian);

/**
 * Creates a cappi from a polar volume
 * @param[in] transform - the transformer
 * @param[in] pvol - the polar volume
 * @param[in] cartesian - the cartesian product
 * @param[in] height - the height of the cappi
 * @returns 0 on failure, otherwise success
 */
int Transform_cappi(Transform_t* transform, PolarVolume_t* pvol, Cartesian_t* cartesian, double height);

/**
 * Creates a pseudo-cappi from a polar volume
 * @param[in] transform - the transformer
 * @param[in] pvol - the polar volume
 * @param[in] cartesian - the cartesian product
 * @param[in] height - the height of the cappi
 * @returns 0 on failure, otherwise success
 */
int Transform_pcappi(Transform_t* transform, PolarVolume_t* pvol, Cartesian_t* cartesian, double height);

#endif
