/* --------------------------------------------------------------------
Copyright (C) 2024 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Functions for managing io cache files like static files.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-02-05
 */
#include <lazy_nodelist_reader.h>
#include "rave_iocache.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_types.h"
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
#include "odim_io_utilities.h"

/**
 * Defines the structure for the RaveIOCache.
 */
struct _RaveIOCache_t {
  RAVE_OBJECT_HEAD /** Always on top */
  HL_Compression* compression;            /**< the compression to use */
  HL_FileCreationProperty* property;       /**< the file creation properties */
  char error_message[1024];                /**< if an error occurs during writing an error message might give you the reason */
};

/*@{ Constants */

/*@} End of Constants */

/*@{ Private functions */
static int RaveIOCache_constructor(RaveCoreObject* obj)
{
  RaveIOCache_t* iocache = (RaveIOCache_t*)obj;
  int result = 0;
  iocache->compression = HLCompression_new(CT_ZLIB);
  iocache->property = HLFileCreationProperty_new();
  strcpy(iocache->error_message, "");
  if (iocache->compression == NULL || iocache->property == NULL) {
    RAVE_ERROR0("Failed to create compression or file creation properties");
    goto done;
  }
  iocache->compression->level = (int)6;
  iocache->property->userblock = (hsize_t)0;
  iocache->property->sizes.sizeof_size = (size_t)4;
  iocache->property->sizes.sizeof_addr = (size_t)4;
  iocache->property->sym_k.ik = (int)1;
  iocache->property->sym_k.lk = (int)1;
  iocache->property->istore_k = (long)1;
  iocache->property->meta_block_size = (long)0;

  result = 1;
done:
  if (result == 0) {
    HLCompression_free(iocache->compression);
    HLFileCreationProperty_free(iocache->property);
    iocache->compression = NULL;
    iocache->property = NULL;
  }
  return result;
}

/**
 * Destroys the RaveIOCache instance
 * @param[in] obj - the instance to destroy
 */
static void RaveIOCache_destructor(RaveCoreObject* obj)
{
  RaveIOCache_t* iocache = (RaveIOCache_t*)obj;
  HLCompression_free(iocache->compression);
  HLFileCreationProperty_free(iocache->property);
}

/**
 * Adds a separate cartesian  to a node list.
 * @param[in] image - the cartesian image to be added to a node list
 * @param[in] nodelist - the nodelist the nodes should be added to
 * @param[in] version - the version we want to write the ODIM file as
 * @returns 1 on success otherwise 0
 */
static int RaveIOCacheInternal_addRaveFieldToNodeList(RaveIOCache_t* raveio, RaveField_t* field, HL_NodeList* nodelist)
{
  int result = 0;

  result = OdimIoUtilities_addRaveField(field, nodelist, RaveIO_ODIM_Version_UNDEFINED, "/field1");

  return result;
}

// static RaveCoreObject* RaveIOInternal_loadVP(LazyNodeListReader_t* lazyReader, RaveIO_ODIM_Version version)
// {
//   VerticalProfile_t* result = NULL;
//   VpOdimIO_t* odimio = RAVE_OBJECT_NEW(&VpOdimIO_TYPE);
//   if (odimio != NULL) {
//     VerticalProfile_t* vp = RAVE_OBJECT_NEW(&VerticalProfile_TYPE);
//     VpOdimIO_setVersion(odimio, version);
//     if (vp != NULL) {
//       if (VpOdimIO_read(odimio, lazyReader, vp)) {
//         result = RAVE_OBJECT_COPY(vp);
//       }
//     }
//     RAVE_OBJECT_RELEASE(vp);
//   }
//   RAVE_OBJECT_RELEASE(odimio);
//   return (RaveCoreObject*)result;
// }

// /**
//  * Adds a vertical profile  to a node list.
//  * @param[in] vp - the vertical profile to be added to a node list
//  * @param[in] nodelist - the nodelist the nodes should be added to
//  * @returns 1 on success otherwise 0
//  */
// static int RaveIOInternal_addVPToNodeList(RaveIO_t* raveio, VerticalProfile_t* vp, HL_NodeList* nodelist, RaveIO_ODIM_Version version)
// {
//   int result = 0;
//   VpOdimIO_t* odimio = NULL;

//   odimio = RAVE_OBJECT_NEW(&VpOdimIO_TYPE);
//   if (odimio != NULL) {
//     VpOdimIO_setVersion(odimio, version);
//     VpOdimIO_setStrict(odimio, raveio->strict);
//     result = VpOdimIO_fill(odimio, vp, nodelist);
//   }

//   RAVE_OBJECT_RELEASE(odimio);

//   return result;
// }

// static int RaveIOInternal_loadHDF5(RaveIO_t* raveio, int lazyLoading, const char* preloadQuantities)
// {
//   HL_NodeList* nodelist = NULL;
//   LazyNodeListReader_t* lazyReader = NULL;

//   Rave_ObjectType objectType = Rave_ObjectType_UNDEFINED;
//   RaveCoreObject* object = NULL;
//   int result = 0;
//   RaveIO_ODIM_Version version = RaveIO_ODIM_Version_UNDEFINED;
//   RaveIO_ODIM_H5rad_Version h5radversion = RaveIO_ODIM_H5rad_Version_UNDEFINED;

//   RAVE_ASSERT((raveio != NULL), "raveio == NULL");
//   RAVE_ASSERT((raveio->filename != NULL), "filename == NULL");

//   lazyReader = LazyNodeListReader_read(raveio->filename);
//   if (lazyReader == NULL) {
//     RAVE_ERROR1("Failed to load hdf5 file '%s'", raveio->filename);
//     goto done;
//   }

//   if (lazyLoading) {
//     if (preloadQuantities != NULL) {
//       if (!LazyNodeListReader_preloadQuantities(lazyReader, preloadQuantities)) {
//         RAVE_ERROR2("Preloading of quantities (%s) failed: %s", preloadQuantities, raveio->filename);
//       }
//     }
//   } else {
//     if (!LazyNodeListReader_preload(lazyReader)) {
//       RAVE_ERROR1("Preloading of file failed: %s", raveio->filename);
//       goto  done;
//     }
//   }

//   nodelist = LazyNodeListReader_getHLNodeList(lazyReader);

//   version = RaveIOInternal_getOdimVersion(nodelist);
//   h5radversion = RaveIOInternal_getH5radVersion(nodelist);
//   objectType = RaveIOInternal_getObjectType(nodelist);

//   if (objectType == Rave_ObjectType_CVOL || objectType == Rave_ObjectType_COMP) {
//     object = (RaveCoreObject*)RaveIOInternal_loadCartesianVolume(lazyReader, version);
//   } else if (objectType == Rave_ObjectType_IMAGE) {
//     object = (RaveCoreObject*)RaveIOInternal_loadCartesian(lazyReader, version);
//   } else if (objectType == Rave_ObjectType_PVOL) {
//     object = (RaveCoreObject*)RaveIOInternal_loadPolarVolume(lazyReader, version);
//   } else if (objectType == Rave_ObjectType_SCAN) {
//     object = (RaveCoreObject*)RaveIOInternal_loadScan(lazyReader, version);
//   } else if (objectType == Rave_ObjectType_VP) {
//     object = (RaveCoreObject*)RaveIOInternal_loadVP(lazyReader, version);
//   } else {
//     RAVE_ERROR1("Currently, RaveIO does not support the object type as defined by '%s'", raveio->filename);
//     goto done;
//   }

//   if (object != NULL) {
//     RAVE_OBJECT_RELEASE(raveio->object);
//     raveio->object = RAVE_OBJECT_COPY(object);
//     raveio->version = RaveIO_ODIM_Version_2_4;
//     raveio->read_version = version;
//     raveio->h5radversion = h5radversion;
//     raveio->fileFormat = RaveIO_ODIM_FileFormat_HDF5;
//   } else {
//     goto done;
//   }

//   result = 1;
// done:
//   RAVE_OBJECT_RELEASE(object);
//   RAVE_OBJECT_RELEASE(lazyReader);
//   return result;
// }

/*@} End of Private functions */
RaveField_t* RaveIOCache_loadField(RaveIOCache_t* iocache, const char* filename, const char* fieldname)
{
  RaveField_t* result = NULL;
  HL_NodeList* nodelist = NULL;
  LazyNodeListReader_t* lazyReader = NULL;
  char sfieldname[512];

  RAVE_ASSERT((iocache != NULL), "iocache == NULL");

  if (filename == NULL) {
    RAVE_ERROR0("Atempting to load a file even though no filename has been specified");
    goto done;
  }

  if(HL_isHDF5File(filename)) {
    lazyReader = LazyNodeListReader_read(filename);
    if (lazyReader == NULL) {
      RAVE_ERROR1("Failed to load hdf5 file '%s'", filename);
      goto done;
    }

    if (!LazyNodeListReader_preload(lazyReader)) {
      RAVE_ERROR1("Preloading of file failed: %s", filename);
      goto  done;
    }

    if (fieldname == NULL) {
      strcpy(sfieldname, "/field1");
    } else {
      strcpy(sfieldname, fieldname);
    }

    result = OdimIoUtilities_loadField(lazyReader, RaveIO_ODIM_Version_UNDEFINED, sfieldname);
  } else {
    RAVE_ERROR1("Atempting to load '%s', but file format does not seem to be supported by rave", filename);
    goto done;
  }

done:
  RAVE_OBJECT_RELEASE(lazyReader);
  return result;
}

int RaveIOCache_saveField(RaveIOCache_t* iocache, RaveField_t* field, const char* filename)
{
  int result = 0;
  HL_NodeList* nodelist = NULL;

  RAVE_ASSERT((iocache != NULL), "iocache == NULL");

  strcpy(iocache->error_message, "");

  if (filename == NULL) {
    RAVE_ERROR0("Atempting to save an object without a filename");
    strcpy(iocache->error_message, "Atempting to save an object without a filename");
    goto done;
  }

  nodelist = HLNodeList_new();

  result = RaveIOCacheInternal_addRaveFieldToNodeList(iocache, field, nodelist);
  if (result == 1) {
    result = HLNodeList_setFileName(nodelist, filename);
  }

  if (result == 1) {
    result = HLNodeList_write(nodelist, iocache->property, iocache->compression);
  }

done:
  if (nodelist != NULL) {
    HLNodeList_free(nodelist);
  }
  return result;
}

void RaveIOCache_setCompressionLevel(RaveIOCache_t* iocache, int lvl)
{
  RAVE_ASSERT((iocache != NULL), "iocache == NULL");
  if (lvl >= 0 && lvl <= 9) {
    iocache->compression->level = lvl;
  }
}

int RaveIOCache_getCompressionLevel(RaveIOCache_t* iocache)
{
  RAVE_ASSERT((iocache != NULL), "iocache == NULL");
  return iocache->compression->level;
}

void RaveIOCache_setUserBlock(RaveIOCache_t* iocache, unsigned long long userblock)
{
  RAVE_ASSERT((iocache != NULL), "iocache == NULL");
  iocache->property->userblock = (hsize_t)userblock;
}

unsigned long long RaveIOCache_getUserBlock(RaveIOCache_t* iocache)
{
  RAVE_ASSERT((iocache != NULL), "iocache == NULL");
  return (unsigned long long)iocache->property->userblock;
}

void RaveIOCache_setSizes(RaveIOCache_t* iocache, size_t sz, size_t addr)
{
  RAVE_ASSERT((iocache != NULL), "iocache == NULL");
  iocache->property->sizes.sizeof_size = sz;
  iocache->property->sizes.sizeof_addr = addr;
}

void RaveIOCache_getSizes(RaveIOCache_t* iocache, size_t* sz, size_t* addr)
{
  RAVE_ASSERT((iocache != NULL), "iocache == NULL");
  if (sz != NULL) {
    *sz = iocache->property->sizes.sizeof_size;
  }
  if (addr != NULL) {
    *addr = iocache->property->sizes.sizeof_addr;
  }
}

void RaveIOCache_setSymk(RaveIOCache_t* iocache, int ik, int lk)
{
  RAVE_ASSERT((iocache != NULL), "iocache == NULL");
  iocache->property->sym_k.ik = ik;
  iocache->property->sym_k.lk = lk;
}

void RaveIOCache_getSymk(RaveIOCache_t* iocache, int* ik, int* lk)
{
  RAVE_ASSERT((iocache != NULL), "iocache == NULL");
  if (ik != NULL) {
    *ik = iocache->property->sym_k.ik;
  }
  if (lk != NULL) {
    *lk = iocache->property->sym_k.lk;
  }
}

void RaveIOCache_setIStoreK(RaveIOCache_t* iocache, long k)
{
  RAVE_ASSERT((iocache != NULL), "iocache == NULL");
  iocache->property->istore_k = k;
}

long RaveIOCache_getIStoreK(RaveIOCache_t* iocache)
{
  RAVE_ASSERT((iocache != NULL), "iocache == NULL");
  return iocache->property->istore_k;
}

void RaveIOCache_setMetaBlockSize(RaveIOCache_t* iocache, long sz)
{
  RAVE_ASSERT((iocache != NULL), "iocache == NULL");
  iocache->property->meta_block_size = sz;
}

long RaveIOCache_getMetaBlockSize(RaveIOCache_t* iocache)
{
  RAVE_ASSERT((iocache != NULL), "iocache == NULL");
  return iocache->property->meta_block_size;
}

const char* RaveIOCache_getErrorMessage(RaveIOCache_t* iocache)
{
  RAVE_ASSERT((iocache != NULL), "iocache == NULL");
  return (const char*)iocache->error_message;
}

/*@} End of Interface functions */

RaveCoreObjectType RaveIOCache_TYPE = {
    "RaveIOCache",
    sizeof(RaveIOCache_t),
    RaveIOCache_constructor,
    RaveIOCache_destructor
};
