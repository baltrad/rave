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
 * Type definitions for RAVE
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-15
 */
#ifndef RAVE_TYPES_H
#define RAVE_TYPES_H

/**
 * Object types that defines the /what/object in the ODIM format.
 */
typedef enum Rave_ObjectType {
  Rave_ObjectType_UNDEFINED = -1,
  Rave_ObjectType_PVOL = 0,       /**< Polar volume */
  Rave_ObjectType_CVOL = 1,       /**< Cartesian volume */
  Rave_ObjectType_SCAN,           /**< Polar scan */
  Rave_ObjectType_RAY,            /**< Single polar ray */
  Rave_ObjectType_AZIM,           /**< Azimuthal object */
  Rave_ObjectType_IMAGE,          /**< 2-D cartesian image */
  Rave_ObjectType_COMP,           /**< Cartesian composite image(s) */
  Rave_ObjectType_XSEC,           /**< 2-D vertical cross section(s) */
  Rave_ObjectType_VP,             /**< 1-D vertical profile */
  Rave_ObjectType_PIC,            /**< Embedded graphical image */
  Rave_ObjectType_ENDOFTYPES      /**< Last entry */
} Rave_ObjectType;

/**
 * Product types that defines the <datasetX>/what/product in the ODIM format.
 */
typedef enum Rave_ProductType {
  Rave_ProductType_UNDEFINED = -1, /**< Undefined */
  Rave_ProductType_SCAN = 0, /**< A scan of polar data */
  Rave_ProductType_PPI,      /**< Plan position indicator */
  Rave_ProductType_CAPPI,    /**< Constant altitude PPI */
  Rave_ProductType_PCAPPI,   /**< Pseudo-CAPPI */
  Rave_ProductType_ETOP,     /**< Echo top */
  Rave_ProductType_MAX,      /**< Maximum */
  Rave_ProductType_RR,       /**< Accumulation */
  Rave_ProductType_VIL,      /**< Vertically integrated liquid water */
  Rave_ProductType_COMP,     /**< Composite */
  Rave_ProductType_VP,       /**< Vertical profile */
  Rave_ProductType_RHI,      /**< Range height indicator */
  Rave_ProductType_XSEC,     /**< Arbitrary vertical slice */
  Rave_ProductType_VSP,      /**< Vertical side panel */
  Rave_ProductType_HSP,      /**< Horizontal side panel */
  Rave_ProductType_RAY,      /**< Ray */
  Rave_ProductType_AZIM,     /**< Azimuthal type product */
  Rave_ProductType_QUAL,     /**< Quality metric */
  Rave_ProductType_ENDOFTYPES /**< Last entry */
} Rave_ProductType;

#endif /* RAVE_TYPES_H */
