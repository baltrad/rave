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
#include "rave_utilities.h"
#include "rave_data2d.h"
#include "hlhdf.h"
#include "hlhdf_alloc.h"
#include "hlhdf_debug.h"
#include "string.h"
#include "stdarg.h"
#include "raveobject_hashtable.h"
#include "raveobject_list.h"
#include "polarvolume.h"
#include "cartesianvolume.h"
#include "rave_field.h"
#include "rave_hlhdf_utilities.h"
#include "cartesian_odim_io.h"
#include "polar_odim_io.h"
#include "vp_odim_io.h"

#ifdef RAVE_BUFR_SUPPORTED
#include "rave_bufr_io.h"
#endif


/**
 * Defines the structure for the RaveIO in a volume.
 */
struct _RaveIO_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveCoreObject* object;                 /**< the object */
  RaveIO_ODIM_Version version;            /**< the odim version */
  RaveIO_ODIM_H5rad_Version h5radversion; /**< the h5rad object version */
  RaveIO_ODIM_FileFormat fileFormat;      /**< the file format */
  char* filename;                         /**< the filename */
  HL_Compression* compression;            /**< the compression to use */
  HL_FileCreationProperty* property;       /**< the file creation properties */
  char* bufrTableDir;                      /**< the bufr table dir */
};

/*@{ Constants */
static const char RaveIO_ODIM_Version_2_0_STR[] = "ODIM_H5/V2_0";
static const char RaveIO_ODIM_Version_2_1_STR[] = "ODIM_H5/V2_1";

static const char RaveIO_ODIM_H5rad_Version_2_0_STR[] = "H5rad 2.0";
static const char RaveIO_ODIM_H5rad_Version_2_1_STR[] = "H5rad 2.1";

/*@} End of Constants */

/*@{ Private functions */
static int RaveIO_constructor(RaveCoreObject* obj)
{
  RaveIO_t* raveio = (RaveIO_t*)obj;
  int result = 0;
  raveio->object = NULL;
  raveio->version = RaveIO_ODIM_Version_2_0;
  raveio->h5radversion = RaveIO_ODIM_H5rad_Version_2_0;
  raveio->fileFormat = RaveIO_ODIM_FileFormat_UNDEFINED;
  raveio->filename = NULL;
  raveio->compression = HLCompression_new(CT_ZLIB);
  raveio->property = HLFileCreationProperty_new();
  raveio->bufrTableDir = NULL;
  if (raveio->compression == NULL || raveio->property == NULL) {
    RAVE_ERROR0("Failed to create compression or file creation properties");
    goto done;
  }
  raveio->compression->level = (int)6;
  raveio->property->userblock = (hsize_t)0;
  raveio->property->sizes.sizeof_size = (size_t)4;
  raveio->property->sizes.sizeof_addr = (size_t)4;
  raveio->property->sym_k.ik = (int)1;
  raveio->property->sym_k.lk = (int)1;
  raveio->property->istore_k = (long)1;
  raveio->property->meta_block_size = (long)0;

  result = 1;
done:
  if (result == 0) {
    HLCompression_free(raveio->compression);
    HLFileCreationProperty_free(raveio->property);
    raveio->compression = NULL;
    raveio->property = NULL;
  }
  return result;
}

/**
 * Destroys the RaveIO instance
 * @param[in] scan - the cartesian product to destroy
 */
static void RaveIO_destructor(RaveCoreObject* obj)
{
  RaveIO_t* raveio = (RaveIO_t*)obj;
  if (raveio != NULL) {
    RaveIO_close(raveio);
  }
  HLCompression_free(raveio->compression);
  HLFileCreationProperty_free(raveio->property);
  RAVE_FREE(raveio->bufrTableDir);
}

/**
 * Returns the ODIM version from the /Conventions field in the nodelist.
 * @param[in] nodelist - the hlhdf nodelist
 */
static RaveIO_ODIM_Version RaveIOInternal_getOdimVersion(HL_NodeList* nodelist)
{
  RaveIO_ODIM_Version result = RaveIO_ODIM_Version_UNDEFINED;
  char* version = NULL;
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  if (!RaveHL_getStringValue(nodelist, &version, "/Conventions")) {
    RAVE_ERROR0("Failed to read attribute /Conventions");
    goto done;
  }

  if (strcmp(RaveIO_ODIM_Version_2_0_STR, version) == 0) {
    result = RaveIO_ODIM_Version_2_0;
  } else if (strcmp(RaveIO_ODIM_Version_2_1_STR, version) == 0) {
    result = RaveIO_ODIM_Version_2_1;
  }
done:
  return result;
}

/**
 * Returns the H5rad version from the /what/version field in the nodelist.
 * @param[in] nodelist - the hlhdf nodelist
 */
static RaveIO_ODIM_H5rad_Version RaveIOInternal_getH5radVersion(HL_NodeList* nodelist)
{
  RaveIO_ODIM_H5rad_Version result = RaveIO_ODIM_H5rad_Version_UNDEFINED;
  char* version = NULL;
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  if (!RaveHL_getStringValue(nodelist, &version, "/what/version")) {
    RAVE_ERROR0("Failed to read attribute /what/version");
    goto done;
  }

  if (strcmp(RaveIO_ODIM_H5rad_Version_2_0_STR, version) == 0) {
    result = RaveIO_ODIM_H5rad_Version_2_0;
  } else if (strcmp(RaveIO_ODIM_H5rad_Version_2_1_STR, version) == 0) {
    result = RaveIO_ODIM_H5rad_Version_2_1;
  }
done:
  return result;
}

/**
 * Returns the object type for provided file.
 * @param[in] nodelist - the hlhdf nodelist
 * @returns the object type or Rave_ObjectType_UNDEFINED on error.
 */
static Rave_ObjectType RaveIOInternal_getObjectType(HL_NodeList* nodelist)
{
  Rave_ObjectType result = Rave_ObjectType_UNDEFINED;
  char* objectType = NULL;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  if (!RaveHL_getStringValue(nodelist, &objectType, "/what/object")) {
    RAVE_ERROR0("Failed to read attribute /what/object");
    goto done;
  }

  result = RaveTypes_getObjectTypeFromString(objectType);
done:
  return result;
}

///////////////////////////////////////////////////////////////////
///// POLAR SPECIFIC FUNCTIONS
///////////////////////////////////////////////////////////////////

/**
 * Loads a individual polar scan
 * @param[in] nodelist - the node list
 * @param[in] fmt - the varargs name of the scan to load
 * @returns a polar scan on success otherwise NULL
 */
static PolarScan_t* RaveIOInternal_loadScan(HL_NodeList* nodelist)
{
  PolarScan_t* result = NULL;
  PolarOdimIO_t* odimio = RAVE_OBJECT_NEW(&PolarOdimIO_TYPE);
  if (odimio != NULL) {
    PolarScan_t* scan = RAVE_OBJECT_NEW(&PolarScan_TYPE);
    if (scan != NULL) {
      if (PolarOdimIO_readScan(odimio, nodelist, scan)) {
        result = RAVE_OBJECT_COPY(scan);
      }
    }
    RAVE_OBJECT_RELEASE(scan);
  }
  RAVE_OBJECT_RELEASE(odimio);
  return result;
}

/**
 * Loads a polar volume.
 * @param[in] nodelist - the node list
 * @returns a polar volume on success otherwise NULL
 */
static PolarVolume_t* RaveIOInternal_loadPolarVolume(HL_NodeList* nodelist)
{
  PolarVolume_t* result = NULL;
  PolarOdimIO_t* odimio = RAVE_OBJECT_NEW(&PolarOdimIO_TYPE);
  if (odimio != NULL) {
    PolarVolume_t* volume = RAVE_OBJECT_NEW(&PolarVolume_TYPE);
    if (volume != NULL) {
      if (PolarOdimIO_readVolume(odimio, nodelist, volume)) {
        result = RAVE_OBJECT_COPY(volume);
      }
    }
    RAVE_OBJECT_RELEASE(volume);
  }
  RAVE_OBJECT_RELEASE(odimio);
  return result;
}

/**
 * Adds a volume to a node list.
 * @param[in] object - the volume to translate into hlhdf nodes
 * @param[in] nodelist - the nodelist the nodes should be added to
 * @returns 1 on success otherwise 0
 */
static int RaveIOInternal_addPolarVolumeToNodeList(PolarVolume_t* object, HL_NodeList* nodelist)
{
  int result = 0;
  PolarOdimIO_t* odimio = RAVE_OBJECT_NEW(&PolarOdimIO_TYPE);
  if (odimio != NULL) {
    result = PolarOdimIO_fillVolume(odimio, object, nodelist);
  }
  RAVE_OBJECT_RELEASE(odimio);
  return result;
}

/**
 * Adds a scan to a node list.
 * @param[in] object - the scan to translate into hlhdf nodes
 * @param[in] nodelist - the nodelist the nodes should be added to
 * @returns 1 on success otherwise 0
 */
static int RaveIOInternal_addScanToNodeList(PolarScan_t* object, HL_NodeList* nodelist)
{
  int result = 0;
  PolarOdimIO_t* odimio = RAVE_OBJECT_NEW(&PolarOdimIO_TYPE);
  if (odimio != NULL) {
    result = PolarOdimIO_fillScan(odimio, object, nodelist);
  }
  RAVE_OBJECT_RELEASE(odimio);
  return result;
}


/**
 * Loads and returns a cartesian object.
 * @param[in] nodelist - the hlhdf nodelist
 * @returns a cartesian object or NULL on failure
 */
static Cartesian_t* RaveIOInternal_loadCartesian(HL_NodeList* nodelist)
{
  Cartesian_t* result = NULL;
  CartesianOdimIO_t* odimio = RAVE_OBJECT_NEW(&CartesianOdimIO_TYPE);
  if (odimio != NULL) {
    Cartesian_t* cartesian = RAVE_OBJECT_NEW(&Cartesian_TYPE);
    if (cartesian != NULL) {
      if (CartesianOdimIO_readCartesian(odimio, nodelist, cartesian)) {
        result = RAVE_OBJECT_COPY(cartesian);
      }
    }
    RAVE_OBJECT_RELEASE(cartesian);
  }
  RAVE_OBJECT_RELEASE(odimio);
  return result;
}

static RaveCoreObject* RaveIOInternal_loadCartesianVolume(HL_NodeList* nodelist)
{
  CartesianVolume_t* result = NULL;
  CartesianOdimIO_t* odimio = RAVE_OBJECT_NEW(&CartesianOdimIO_TYPE);
  if (odimio != NULL) {
    CartesianVolume_t* volume = RAVE_OBJECT_NEW(&CartesianVolume_TYPE);
    if (volume != NULL) {
      if (CartesianOdimIO_readVolume(odimio, nodelist, volume)) {
        result = RAVE_OBJECT_COPY(volume);
      }
    }
    RAVE_OBJECT_RELEASE(volume);
  }
  RAVE_OBJECT_RELEASE(odimio);
  return (RaveCoreObject*)result;
}

/**
 * Adds a cartesian volume to a node list.
 * @param[in] cvol - the cartesian volume to be added to a node list
 * @param[in] nodelist - the nodelist the nodes should be added to
 * @returns 1 on success otherwise 0
 */
static int RaveIOInternal_addCartesianVolumeToNodeList(CartesianVolume_t* cvol, HL_NodeList* nodelist)
{
  int result = 0;
  CartesianOdimIO_t* odimio = NULL;

  odimio = RAVE_OBJECT_NEW(&CartesianOdimIO_TYPE);
  if (odimio != NULL) {
    result = CartesianOdimIO_fillVolume(odimio, nodelist, cvol);
  }

  RAVE_OBJECT_RELEASE(odimio);

  return result;
}

/**
 * Adds a separate cartesian  to a node list.
 * @param[in] image - the cartesian image to be added to a node list
 * @param[in] nodelist - the nodelist the nodes should be added to
 * @returns 1 on success otherwise 0
 */
static int RaveIOInternal_addCartesianToNodeList(Cartesian_t* image, HL_NodeList* nodelist)
{
  int result = 0;
  CartesianOdimIO_t* odimio = NULL;

  odimio = RAVE_OBJECT_NEW(&CartesianOdimIO_TYPE);
  if (odimio != NULL) {
    result = CartesianOdimIO_fillImage(odimio, nodelist, image);
  }

  RAVE_OBJECT_RELEASE(odimio);

  return result;
}

static RaveCoreObject* RaveIOInternal_loadVP(HL_NodeList* nodelist)
{
  VerticalProfile_t* result = NULL;
  VpOdimIO_t* odimio = RAVE_OBJECT_NEW(&VpOdimIO_TYPE);
  if (odimio != NULL) {
    VerticalProfile_t* vp = RAVE_OBJECT_NEW(&VerticalProfile_TYPE);
    if (vp != NULL) {
      if (VpOdimIO_read(odimio, nodelist, vp)) {
        result = RAVE_OBJECT_COPY(vp);
      }
    }
    RAVE_OBJECT_RELEASE(vp);
  }
  RAVE_OBJECT_RELEASE(odimio);
  return (RaveCoreObject*)result;
}

/**
 * Adds a vertical profile  to a node list.
 * @param[in] vp - the vertical profile to be added to a node list
 * @param[in] nodelist - the nodelist the nodes should be added to
 * @returns 1 on success otherwise 0
 */
static int RaveIOInternal_addVPToNodeList(VerticalProfile_t* vp, HL_NodeList* nodelist)
{
  int result = 0;
  VpOdimIO_t* odimio = NULL;

  odimio = RAVE_OBJECT_NEW(&VpOdimIO_TYPE);
  if (odimio != NULL) {
    result = VpOdimIO_fill(odimio, vp, nodelist);
  }

  RAVE_OBJECT_RELEASE(odimio);

  return result;
}

static int RaveIOInternal_loadHDF5(RaveIO_t* raveio)
{
  HL_NodeList* nodelist = NULL;
  Rave_ObjectType objectType = Rave_ObjectType_UNDEFINED;
  RaveCoreObject* object = NULL;
  int result = 0;
  RaveIO_ODIM_Version version = RaveIO_ODIM_Version_UNDEFINED;
  RaveIO_ODIM_H5rad_Version h5radversion = RaveIO_ODIM_H5rad_Version_UNDEFINED;

  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  RAVE_ASSERT((raveio->filename != NULL), "filename == NULL");

  nodelist = HLNodeList_read(raveio->filename);
  if (nodelist == NULL) {
    RAVE_ERROR1("Failed to load hdf5 file '%s'", raveio->filename);
    goto done;
  }

  HLNodeList_selectAllNodes(nodelist);
  if (!HLNodeList_fetchMarkedNodes(nodelist)) {
    RAVE_ERROR1("Failed to load hdf5 file '%s'", raveio->filename);
    goto done;
  }

  version = RaveIOInternal_getOdimVersion(nodelist);
  h5radversion = RaveIOInternal_getH5radVersion(nodelist);

  objectType = RaveIOInternal_getObjectType(nodelist);
  if (objectType == Rave_ObjectType_CVOL || objectType == Rave_ObjectType_COMP) {
    object = (RaveCoreObject*)RaveIOInternal_loadCartesianVolume(nodelist);
  } else if (objectType == Rave_ObjectType_IMAGE) {
    object = (RaveCoreObject*)RaveIOInternal_loadCartesian(nodelist);
  } else if (objectType == Rave_ObjectType_PVOL) {
    object = (RaveCoreObject*)RaveIOInternal_loadPolarVolume(nodelist);
  } else if (objectType == Rave_ObjectType_SCAN) {
    object = (RaveCoreObject*)RaveIOInternal_loadScan(nodelist);
  } else if (objectType == Rave_ObjectType_VP) {
    object = (RaveCoreObject*)RaveIOInternal_loadVP(nodelist);
  } else {
    RAVE_ERROR1("Currently, RaveIO does not support the object type as defined by '%s'", raveio->filename);
    goto done;
  }

  if (object != NULL) {
    RAVE_OBJECT_RELEASE(raveio->object);
    raveio->object = RAVE_OBJECT_COPY(object);
    raveio->version = version;
    raveio->h5radversion = h5radversion;
    raveio->fileFormat = RaveIO_ODIM_FileFormat_HDF5;
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(object);
  HLNodeList_free(nodelist);
  return result;
}

#ifdef RAVE_BUFR_SUPPORTED
static int RaveIOInternal_loadBUFR(RaveIO_t* raveio)
{
  RaveBufrIO_t* bufrio = NULL;
  int result = 0;

  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  RAVE_ASSERT((raveio->filename != NULL), "filename == NULL");

  bufrio = RAVE_OBJECT_NEW(&RaveBufrIO_TYPE);
  if (bufrio != NULL) {
    RaveCoreObject* obj = RaveBufrIO_read(bufrio, raveio->filename);
    if (obj != NULL) {
      RAVE_OBJECT_RELEASE(raveio->object);
      raveio->object = RAVE_OBJECT_COPY(obj);
      raveio->h5radversion = RaveIO_ODIM_H5rad_Version_UNDEFINED;
      raveio->version = RaveIO_ODIM_Version_UNDEFINED;
      raveio->fileFormat = RaveIO_ODIM_FileFormat_BUFR;
      RAVE_OBJECT_RELEASE(obj);
    } else {
      goto done;
    }
    result = 1;
  }

done:
  RAVE_OBJECT_RELEASE(bufrio);
  return result;
}
#endif

/*@} End of Private functions */
void RaveIO_close(RaveIO_t* raveio)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  RAVE_FREE(raveio->filename);
  RAVE_OBJECT_RELEASE(raveio->object);
  raveio->h5radversion = RaveIO_ODIM_H5rad_Version_2_0;
  raveio->version = RaveIO_ODIM_Version_2_0;
}

RaveIO_t* RaveIO_open(const char* filename)
{
  RaveIO_t* result = NULL;

  if (filename == NULL) {
    goto done;
  }

  result = RAVE_OBJECT_NEW(&RaveIO_TYPE);
  if (result == NULL) {
    RAVE_CRITICAL0("Failed to create raveio instance");
    goto done;
  }

  if (!RaveIO_setFilename(result, filename)) {
    RAVE_CRITICAL0("Failed to set filename");
    RAVE_OBJECT_RELEASE(result);
    goto done;
  }

  if (!RaveIO_load(result)) {
    RAVE_WARNING0("Failed to load file");
    RAVE_OBJECT_RELEASE(result);
    goto done;
  }

done:
  return result;
}

int RaveIO_load(RaveIO_t* raveio)
{
  int result = 0;

  RAVE_ASSERT((raveio != NULL), "raveio == NULL");

  if (raveio->filename == NULL) {
    RAVE_ERROR0("Atempting to load a file even though no filename has been specified");
    goto done;
  }

  if(HL_isHDF5File(raveio->filename)) {
    result = RaveIOInternal_loadHDF5(raveio);
#ifdef RAVE_BUFR_SUPPORTED
  } else if (RaveBufrIO_isBufr(raveio->filename)) {
    result = RaveIOInternal_loadBUFR(raveio);
#endif
  } else {
    RAVE_ERROR1("Atempting to load '%s', but file format does not seem to be supported by rave", raveio->filename);
    goto done;
  }

done:
  return result;
}

int RaveIO_save(RaveIO_t* raveio, const char* filename)
{
  int result = 0;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");

  if (filename != NULL) {
    if (!RaveIO_setFilename(raveio, filename)) {
      RAVE_ERROR0("Failed to set filename before saving");
      return 0;
    }
  }

  if (raveio->filename == NULL) {
    RAVE_ERROR0("Atempting to save an object without a filename");
    return 0;
  }
  if (raveio->object != NULL) {
    if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &Cartesian_TYPE) ||
        RAVE_OBJECT_CHECK_TYPE(raveio->object, &PolarVolume_TYPE) ||
        RAVE_OBJECT_CHECK_TYPE(raveio->object, &CartesianVolume_TYPE) ||
        RAVE_OBJECT_CHECK_TYPE(raveio->object, &PolarScan_TYPE) ||
        RAVE_OBJECT_CHECK_TYPE(raveio->object, &VerticalProfile_TYPE)) {
      HL_NodeList* nodelist = HLNodeList_new();

      if (nodelist != NULL) {
        result = RaveHL_createStringValue(nodelist, RaveIO_ODIM_Version_2_1_STR, "/Conventions");
        if (result == 1) {
          if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &PolarVolume_TYPE)) {
            result = RaveIOInternal_addPolarVolumeToNodeList((PolarVolume_t*)raveio->object, nodelist);
          } else if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &CartesianVolume_TYPE)) {
            result = RaveIOInternal_addCartesianVolumeToNodeList((CartesianVolume_t*)raveio->object, nodelist);
          } else if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &Cartesian_TYPE)) {
            result = RaveIOInternal_addCartesianToNodeList((Cartesian_t*)raveio->object, nodelist);
          } else if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &PolarScan_TYPE)) {
            result = RaveIOInternal_addScanToNodeList((PolarScan_t*)raveio->object, nodelist);
          } else if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &VerticalProfile_TYPE)) {
            result = RaveIOInternal_addVPToNodeList((VerticalProfile_t*)raveio->object, nodelist);
          } else {
            RAVE_ERROR0("No io support for provided object");
            result = 0;
          }
        }

        if (result == 1) {
          result = HLNodeList_setFileName(nodelist, raveio->filename);
        }

        if (result == 1) {
          result = HLNodeList_write(nodelist, raveio->property, raveio->compression);
        }
      }
      HLNodeList_free(nodelist);
    }
  }

  return result;
}

void RaveIO_setObject(RaveIO_t* raveio, RaveCoreObject* object)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  RAVE_OBJECT_RELEASE(raveio->object);
  raveio->object = RAVE_OBJECT_COPY(object);
}

RaveCoreObject* RaveIO_getObject(RaveIO_t* raveio)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  return RAVE_OBJECT_COPY(raveio->object);
}

int RaveIO_setFilename(RaveIO_t* raveio, const char* filename)
{
  int result = 0;
  char* tmp = NULL;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  if (filename != NULL) {
    tmp = RAVE_STRDUP(filename);
    if (tmp != NULL) {
      RAVE_FREE(raveio->filename);
      raveio->filename = tmp;
      result = 1;
    }
  } else {
    RAVE_FREE(raveio->filename);
    result = 1;
  }
  return result;
}

const char* RaveIO_getFilename(RaveIO_t* raveio)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  return (const char*)raveio->filename;
}

Rave_ObjectType RaveIO_getObjectType(RaveIO_t* raveio)
{
  Rave_ObjectType result = Rave_ObjectType_UNDEFINED;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  if (raveio->object != NULL) {
    if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &Cartesian_TYPE)) {
      result = Cartesian_getObjectType((Cartesian_t*)raveio->object);
    } else if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &PolarVolume_TYPE)) {
      result = Rave_ObjectType_PVOL;
    } else if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &PolarScan_TYPE)) {
      result = Rave_ObjectType_SCAN;
    } else if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &CartesianVolume_TYPE)) {
      result = CartesianVolume_getObjectType((CartesianVolume_t*)raveio->object);
    }
  }
  return result;
}

int RaveIO_setOdimVersion(RaveIO_t* raveio, RaveIO_ODIM_Version version)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  if (version != RaveIO_ODIM_Version_2_0) {
    return 0;
  }
  raveio->version = version;
  return 1;
}

RaveIO_ODIM_Version RaveIO_getOdimVersion(RaveIO_t* raveio)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  return raveio->version;
}

int RaveIO_setH5radVersion(RaveIO_t* raveio, RaveIO_ODIM_H5rad_Version version)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  if (version != RaveIO_ODIM_H5rad_Version_2_0) {
    return 0;
  }
  raveio->h5radversion = version;
  return 1;
}

RaveIO_ODIM_H5rad_Version RaveIO_getH5radVersion(RaveIO_t* raveio)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  return raveio->h5radversion;
}

RaveIO_ODIM_FileFormat RaveIO_getFileFormat(RaveIO_t* raveio)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  return raveio->fileFormat;
}

void RaveIO_setCompressionLevel(RaveIO_t* raveio, int lvl)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  if (lvl >= 0 && lvl <= 9) {
    raveio->compression->level = lvl;
  }
}

int RaveIO_getCompressionLevel(RaveIO_t* raveio)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  return raveio->compression->level;
}

void RaveIO_setUserBlock(RaveIO_t* raveio, unsigned long long userblock)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  raveio->property->userblock = (hsize_t)userblock;
}

unsigned long long RaveIO_getUserBlock(RaveIO_t* raveio)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  return (unsigned long long)raveio->property->userblock;
}

void RaveIO_setSizes(RaveIO_t* raveio, size_t sz, size_t addr)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  raveio->property->sizes.sizeof_size = sz;
  raveio->property->sizes.sizeof_addr = addr;
}

void RaveIO_getSizes(RaveIO_t* raveio, size_t* sz, size_t* addr)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  if (sz != NULL) {
    *sz = raveio->property->sizes.sizeof_size;
  }
  if (addr != NULL) {
    *addr = raveio->property->sizes.sizeof_addr;
  }
}

void RaveIO_setSymk(RaveIO_t* raveio, int ik, int lk)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  raveio->property->sym_k.ik = ik;
  raveio->property->sym_k.lk = lk;
}

void RaveIO_getSymk(RaveIO_t* raveio, int* ik, int* lk)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  if (ik != NULL) {
    *ik = raveio->property->sym_k.ik;
  }
  if (lk != NULL) {
    *lk = raveio->property->sym_k.lk;
  }
}

void RaveIO_setIStoreK(RaveIO_t* raveio, long k)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  raveio->property->istore_k = k;
}

long RaveIO_getIStoreK(RaveIO_t* raveio)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  return raveio->property->istore_k;
}

void RaveIO_setMetaBlockSize(RaveIO_t* raveio, long sz)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  raveio->property->meta_block_size = sz;
}

long RaveIO_getMetaBlockSize(RaveIO_t* raveio)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  return raveio->property->meta_block_size;
}

int RaveIO_setBufrTableDir(RaveIO_t* raveio, const char* dname)
{
  char* tmp = NULL;
  int result = 0;

  RAVE_ASSERT((raveio != NULL), "raveio == NULL");

  if (dname != NULL) {
    tmp = RAVE_STRDUP(dname);
    if (tmp == NULL) {
      goto done;
    }
  }
  RAVE_FREE(raveio->bufrTableDir);
  raveio->bufrTableDir = tmp;
  tmp = NULL; /* Not responsible any longer */
  result = 1;
done:
  RAVE_FREE(tmp);
  return result;
}

const char* RaveIO_getBufrTableDir(RaveIO_t* raveio)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  return (const char*)raveio->bufrTableDir;
}

int RaveIO_supports(RaveIO_ODIM_FileFormat format)
{
  int result = 0;
  if (format == RaveIO_ODIM_FileFormat_HDF5) {
    result = 1;
#ifdef RAVE_BUFR_SUPPORTED
  } else if (format == RaveIO_ODIM_FileFormat_BUFR) {
    result = 1;
#endif
  } else {
    result = 0;
  }
  return result;
}

/*@} End of Interface functions */

RaveCoreObjectType RaveIO_TYPE = {
    "RaveIO",
    sizeof(RaveIO_t),
    RaveIO_constructor,
    RaveIO_destructor
};
