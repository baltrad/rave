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

/**
 * Defines the structure for the RaveIO in a volume.
 */
struct _RaveIO_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveCoreObject* object;                 /**< the object */
  RaveIO_ODIM_Version version;            /**< the odim version */
  RaveIO_ODIM_H5rad_Version h5radversion; /**< the h5rad object version */
  char* filename;                         /**< the filename */
  HL_Compression* compression;            /**< the compression to use */
  HL_FileCreationProperty* property;       /**< the file creation properties */
};

/*@{ Constants */
static const char RaveIO_ODIM_Version_2_0_STR[] = "ODIM_H5/V2_0";

static const char RaveIO_ODIM_H5rad_Version_2_0_STR[] = "H5rad 2.0";

/*@} End of Constants */

/*@{ Macros */
/**
 * Quick access function for reading one atomic value from a
 * HLHDF node.
 *
 * @param[in] vt - the type for the read data
 * @param[in] nn - the node
 * @param[in] ss - the size of the data type
 * @param[in] ov - the output value
 * @param[in] ot - the type of the output value where assignment will be done.
 */
#define RAVEIO_GET_ATOMIC_NODEVALUE(vt, nn, ss, ot, ov) \
{ \
  vt v; \
  memcpy(&v, HLNode_getData(nn), ss); \
  ov = (ot)v; \
}

/*@} End of Macros */

/*@{ Private functions */
static int RaveIO_constructor(RaveCoreObject* obj)
{
  RaveIO_t* raveio = (RaveIO_t*)obj;
  int result = 0;
  raveio->object = NULL;
  raveio->version = RaveIO_ODIM_Version_2_0;
  raveio->h5radversion = RaveIO_ODIM_H5rad_Version_2_0;
  raveio->filename = NULL;
  raveio->compression = HLCompression_new(CT_ZLIB);
  raveio->property = HLFileCreationProperty_new();
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
}

/**
 * Fetches the node with the provided name and returns the
 * value. Note, the value will be the internal pointer so
 * do not free it.
 * @param[in] nodelist - the hlhdf node list
 * @param[out] value - the string value
 * @param[in] fmt - the varargs format specifier
 * @param[in] ... - the varargs
 * @returns 0 on failure, otherwise 1
 */
static int RaveIOInternal_getStringValue(HL_NodeList* nodelist, char** value, const char* fmt, ...)
{
  int result = 0;
  va_list ap;
  char nodeName[1024];
  int n = 0;
  HL_Node* node = NULL;
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n < 0 || n >= 1024) {
    goto done;
  }

  node = HLNodeList_getNodeByName(nodelist, nodeName);
  if (node == NULL) {
    RAVE_ERROR1("Could not read %s", nodeName);
    goto done;
  }

  if (HLNode_getFormat(node) != HLHDF_STRING) {
    RAVE_ERROR1("%s is not of type HLHDF_STRING", nodeName);
    goto done;
  }

  *value = (char*)HLNode_getData(node);

  result = 1;
done:
  return result;
}

/**
 * Creates a rave attribute from a HLHDF node value.
 * Node must contain data that can be translated to long, double or strings otherwise
 * NULL will be returned. Note, the name will not be set on the attribute and has to
 * be set after this function has been called.
 * @param[in] node - the HLHDF node
 * @returns the rave attribute on success, otherwise NULL.
 */
static RaveAttribute_t* RaveIOInternal_getAttribute(HL_Node* node)
{
  size_t sz = 0;
  HL_FormatSpecifier format = HLHDF_UNDEFINED;
  RaveAttribute_t* result = NULL;

  RAVE_ASSERT((node != NULL), "node == NULL");

  result = RAVE_OBJECT_NEW(&RaveAttribute_TYPE);
  if (result == NULL) {
    goto done;
  }
  format = HLNode_getFormat(node);
  sz = HLNode_getDataSize(node);
  if (format >= HLHDF_SCHAR && format <= HLHDF_ULLONG) {
    long value = 0;
    if (sz == sizeof(char)) {
      RAVEIO_GET_ATOMIC_NODEVALUE(char, node, sz, long, value);
    } else if (sz == sizeof(short)) {
      RAVEIO_GET_ATOMIC_NODEVALUE(short, node, sz, long, value);
    } else if (sz == sizeof(int)) {
      RAVEIO_GET_ATOMIC_NODEVALUE(int, node, sz, long, value);
    } else if (sz == sizeof(long)) {
      RAVEIO_GET_ATOMIC_NODEVALUE(long, node, sz, long, value);
    } else if (sz == sizeof(long long)) {
      RAVEIO_GET_ATOMIC_NODEVALUE(long long, node, sz, long, value);
    }
    RaveAttribute_setLong(result, value);
  } else if (format >= HLHDF_FLOAT && format <= HLHDF_LDOUBLE) {
    double value = 0.0;
    if (sz == sizeof(float)) {
      RAVEIO_GET_ATOMIC_NODEVALUE(float, node, sz, double, value);
    } else if (sz == sizeof(double)) {
      RAVEIO_GET_ATOMIC_NODEVALUE(double, node, sz, double, value);
    } else if (sz == sizeof(long double)) {
      RAVEIO_GET_ATOMIC_NODEVALUE(long double, node, sz, double, value);
    }
    RaveAttribute_setDouble(result, value);
  } else if (format == HLHDF_STRING) {
    RaveAttribute_setString(result, (char*)HLNode_getData(node));
  } else {
    RAVE_WARNING0("Node does not contain value conformant to rave_attribute");
    RAVE_OBJECT_RELEASE(result);
  }
done:
  return result;
}

/**
 * Verifies if the file contains a node with the name as specified by the variable
 * argument list.
 * @param[in] nodelist - the hlhdf nodelist
 * @param[in] fmt    - the variable argument format specifier
 * @param[in] ...    - the variable argument list
 * @returns 1 if the node could be found, otherwise 0
 */
static int RaveIOInternal_hasNodeByName(HL_NodeList* nodelist, const char* fmt, ...)
{
  va_list ap;
  char nodeName[1024];
  int n = 0;
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n >= 0 && n < 1024) {
    return HLNodeList_hasNodeByName(nodelist, nodeName);
  }
  return 0;
}

/**
 * Loads the attributes from the name into the RaveCoreObject. I.e.
 * name/how/..., name/where/... and name/what/...
 * @param[in] nodelist - the hlhdf list
 * @param[in] object - the object to fill
 * @param[in] fmt - the varargs name of the object
 * @param[in] ... - the varargs
 * @return 1 on success otherwise 0
 */
static int RaveIOInternal_loadAttributesAndDataForObject(HL_NodeList* nodelist, RaveCoreObject* object, const char* fmt, ...)
{
  int result = 1;
  int n = 0;
  int i = 0;
  int nameLength = 0;

  va_list ap;
  char name[1024];
  int nName = 0;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((object != NULL), "object == NULL");

  va_start(ap, fmt);
  nName = vsnprintf(name, 1024, fmt, ap);
  va_end(ap);
  if (nName < 0 || nName >= 1024) {
    RAVE_ERROR0("NodeName would evaluate to more than 1024 characters.");
    result = 0;
  } else {
    nameLength = strlen(name);
  }

  n = HLNodeList_getNumberOfNodes(nodelist);
  for (i = 0; result == 1 && i < n; i++) {
    HL_Node* node = HLNodeList_getNodeByIndex(nodelist, i);
    const char* nodeName = HLNode_getName(node);
    int nNameLength = strlen(nodeName);
    if (nNameLength>nameLength && strncasecmp(nodeName, name, nameLength)==0) {
      if (nodeName[nameLength]=='/') {
        char* tmpptr = (char*)nodeName+(nameLength + 1);
        if (HLNode_getType(node) == ATTRIBUTE_ID &&
            (strncasecmp(tmpptr, "how/", 4)==0 ||
             strncasecmp(tmpptr, "what/", 5)==0 ||
             strncasecmp(tmpptr, "where/", 6)==0)) {
          RaveAttribute_t* attribute = RaveIOInternal_getAttribute(node);
          if (attribute != NULL) {
            result = RaveAttribute_setName(attribute, tmpptr);
            if (result == 1) {
              if (RAVE_OBJECT_CHECK_TYPE(object, &PolarScanParam_TYPE)) {
                result = PolarScanParam_addAttribute((PolarScanParam_t*)object, attribute);
              } else if (RAVE_OBJECT_CHECK_TYPE(object, &PolarScan_TYPE)) {
                result = PolarScan_addAttribute((PolarScan_t*)object, attribute);
              } else if (RAVE_OBJECT_CHECK_TYPE(object, &PolarVolume_TYPE)) {
                result = PolarVolume_addAttribute((PolarVolume_t*)object, attribute);
              } else if (RAVE_OBJECT_CHECK_TYPE(object, &Cartesian_TYPE)) {
                result = Cartesian_addAttribute((Cartesian_t*)object, attribute);
              } else if (RAVE_OBJECT_CHECK_TYPE(object, &CartesianVolume_TYPE)) {
                result = CartesianVolume_addAttribute((CartesianVolume_t*)object, attribute);
              } else if (RAVE_OBJECT_CHECK_TYPE(object, &RaveObjectHashTable_TYPE)) {
                result = RaveObjectHashTable_put((RaveObjectHashTable_t*)object, tmpptr, (RaveCoreObject*)attribute);
              } else if (RAVE_OBJECT_CHECK_TYPE(object, &RaveObjectList_TYPE)) {
                result = RaveObjectList_add((RaveObjectList_t*)object, (RaveCoreObject*)attribute);
              } else if (RAVE_OBJECT_CHECK_TYPE(object, &RaveField_TYPE)) {
                result = RaveField_addAttribute((RaveField_t*)object, attribute);
              } else {
                RAVE_CRITICAL0("Unsupported type for load attributes");
                result = 0;
              }
            }
          }
          RAVE_OBJECT_RELEASE(attribute);
        } else if (HLNode_getType(node) == DATASET_ID &&
            strcasecmp(tmpptr, "data")==0) {
          hsize_t d0 = HLNode_getDimension(node, 0);
          hsize_t d1 = HLNode_getDimension(node, 1);
          RaveDataType dataType = RaveHL_hlhdfToRaveType(HLNode_getFormat(node));
          if (dataType != RaveDataType_UNDEFINED) {
            if (RAVE_OBJECT_CHECK_TYPE(object, &PolarScanParam_TYPE)) {
              result = PolarScanParam_setData((PolarScanParam_t*)object, d1, d0, HLNode_getData(node), dataType);
            } else if (RAVE_OBJECT_CHECK_TYPE(object, &Cartesian_TYPE)) {
              result = Cartesian_setData((Cartesian_t*)object,d1, d0,HLNode_getData(node), dataType);
            } else if (RAVE_OBJECT_CHECK_TYPE(object, &RaveField_TYPE)) {
              result = RaveField_setData((RaveField_t*)object, d1, d0, HLNode_getData(node), dataType);
            }
          } else {
            RAVE_ERROR0("Undefined datatype for dataset");
            result = 0;
          }
        }
      }
    }
  }

  return result;
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

  if (!RaveIOInternal_getStringValue(nodelist, &version, "/Conventions")) {
    RAVE_ERROR0("Failed to read attribute /Conventions");
    goto done;
  }

  if (strcmp(RaveIO_ODIM_Version_2_0_STR, version) == 0) {
    result = RaveIO_ODIM_Version_2_0;
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

  if (!RaveIOInternal_getStringValue(nodelist, &version, "/what/version")) {
    RAVE_ERROR0("Failed to read attribute /what/version");
    goto done;
  }

  if (strcmp(RaveIO_ODIM_H5rad_Version_2_0_STR, version) == 0) {
    result = RaveIO_ODIM_H5rad_Version_2_0;
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

  if (!RaveIOInternal_getStringValue(nodelist, &objectType, "/what/object")) {
    RAVE_ERROR0("Failed to read attribute /what/object");
    goto done;
  }

  result = RaveTypes_getObjectTypeFromString(objectType);
done:
  return result;
}

static int RaveIOInternal_addRaveFieldToNodeList(
  RaveField_t* field, HL_NodeList* nodelist, const char* fmt, ...)
{
  va_list ap;
  char nodeName[1024];
  int n = 0;
  int result = 0;
  RaveObjectList_t* attributes = NULL;

  RAVE_ASSERT((field != NULL), "field == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n < 0 || n >= 1024) {
    goto done;
  }
  if (!RaveIOInternal_hasNodeByName(nodelist, nodeName)) {
    if (!RaveHL_createGroup(nodelist, nodeName)) {
      goto done;
    }
  }

  attributes = RaveField_getAttributeValues(field);

  if (attributes == NULL || !RaveHL_addAttributes(nodelist, attributes, nodeName)) {
    goto done;
  }

  if (!RaveHL_addData(nodelist,
                      RaveField_getData(field),
                      RaveField_getXsize(field),
                      RaveField_getYsize(field),
                      RaveField_getDataType(field),
                      nodeName)) {
    goto done;
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(attributes);
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
 * Adds a scan parameter to a node list.
 * @param[in] object - the parameter to translate into hlhdf nodes
 * @param[in] nodelist - the nodelist the nodes should be added to
 * @param[in] fmt - the variable arguments format string that should be used to define the name of this parameter group
 * @returns 1 on success otherwise 0
 */
static int RaveIOInternal_addScanParamToNodeList(PolarScanParam_t* object, HL_NodeList* nodelist, const char* fmt, ...)
{
  int result = 0;
  char nodeName[1024];
  RaveObjectList_t* attributes = NULL;
  va_list ap;
  int n = 0;

  RAVE_ASSERT((object != NULL), "object == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n < 0 || n >= 1024) {
    RAVE_ERROR1("Failed to create parameter name from fmt=%s", fmt);
    goto done;
  }

  if (!RaveIOInternal_hasNodeByName(nodelist, nodeName)) {
    if (!RaveHL_createGroup(nodelist, nodeName)) {
      goto done;
    }
  }

  attributes = PolarScanParam_getAttributeValues(object);

  if (attributes == NULL || !RaveHL_addAttributes(nodelist, attributes, nodeName)) {
    goto done;
  }

  if (!RaveHL_addData(nodelist,
                      PolarScanParam_getData(object),
                      PolarScanParam_getNbins(object),
                      PolarScanParam_getNrays(object),
                      PolarScanParam_getDataType(object),
                      nodeName)) {
    goto done;
  }

  result = 1;

  if (PolarScanParam_getNumberOfQualityFields(object) > 0) {
    int nfields = PolarScanParam_getNumberOfQualityFields(object);
    int i = 0;
    for (i = 0; result == 1 && i < nfields; i++) {
      RaveField_t* field = PolarScanParam_getQualityField(object, i);
      if (field != NULL) {
        result = RaveIOInternal_addRaveFieldToNodeList(field, nodelist, "%s/quality%d", nodeName, (i+1));
      } else {
        result = 0;
      }
      RAVE_OBJECT_RELEASE(field);
    }
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(attributes);
  return result;
}

#ifdef KALLE
/**
 * Adds a scan to a node list.
 * @param[in] object - the parameter to translate into hlhdf nodes
 * @param[in] nodelist - the nodelist the nodes should be added to
 * @param[in] fmt - the variable arguments format string that should be used to define the name of this scan group
 * @returns 1 on success otherwise 0
 */
static int RaveIOInternal_addVolumeScanToNodeList(PolarScan_t* object, HL_NodeList* nodelist, const char* fmt, ...)
{
  int result = 0;
  char nodeName[1024];
  int nkeys = 0;
  RaveList_t* keys = NULL;
  RaveObjectList_t* attributes = NULL;
  va_list ap;
  int n = 0;

  RAVE_ASSERT((object != NULL), "object == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n < 0 || n >= 1024) {
    RAVE_ERROR1("Failed to create scan name from fmt=%s", fmt);
    goto done;
  }

  if (!RaveIOInternal_hasNodeByName(nodelist, nodeName)) {
    if (!RaveHL_createGroup(nodelist, nodeName)) {
      goto done;
    }
  }

  attributes = PolarScan_getAttributeValues(object, Rave_ObjectType_PVOL, 0);
  if (attributes != NULL) {
    if (!RaveUtilities_addStringAttributeToList(attributes,
           "what/product", RaveTypes_getStringFromProductType(Rave_ProductType_SCAN))) {
      goto done;
    }
    if (!RaveUtilities_addLongAttributeToList(attributes, "where/nbins", PolarScan_getNbins(object)) ||
        !RaveUtilities_addLongAttributeToList(attributes, "where/nrays", PolarScan_getNrays(object))) {
      goto done;
    }
  }

  if (attributes == NULL || !RaveHL_addAttributes(nodelist, attributes, nodeName)) {
    goto done;
  }

  result = 1; // Set result to 1 now and if adding of scans fails, it will become 0 again

  keys = PolarScan_getParameterNames(object);
  if (keys != NULL) {
    int i = 0;
    nkeys = RaveList_size(keys);
    for (i = 0; result == 1 && i < nkeys; i++) {
      const char* keyname = RaveList_get(keys, i);
      PolarScanParam_t* parameter = PolarScan_getParameter(object, keyname);
      if (parameter == NULL) {
        result = 0;
      }
      if (result == 1) {
        result = RaveIOInternal_addScanParamToNodeList(parameter, nodelist, "%s/data%d", nodeName, (i+1));
      }
      RAVE_OBJECT_RELEASE(parameter);
    }
  }

  if (result == 1 && PolarScan_getNumberOfQualityFields(object) > 0) {
    int nfields = PolarScan_getNumberOfQualityFields(object);
    int i = 0;
    for (i = 0; result == 1 && i < nfields; i++) {
      RaveField_t* field = PolarScan_getQualityField(object, i);
      if (field != NULL) {
        result = RaveIOInternal_addRaveFieldToNodeList(field, nodelist, "%s/quality%d", nodeName, (i+1));
      } else {
        result = 0;
      }
      RAVE_OBJECT_RELEASE(field);
    }
  }

done:
  RaveList_freeAndDestroy(&keys);
  RAVE_OBJECT_RELEASE(attributes);
  return result;
}
#endif

/**
 * Adds a volume to a node list.
 * @param[in] object - the parameter to translate into hlhdf nodes
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

#ifdef KALLE
  int result = 0;
  RaveObjectList_t* attributes = NULL;
  int nscans = 0;
  int i = 0;

  RAVE_ASSERT((object != NULL), "object == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  // First verify that no bogus data is entered into the system.
  if (!PolarVolume_isValid(object)) {
    goto done;
  }

  attributes = PolarVolume_getAttributeValues(object);
  if (attributes != NULL) {
    if (!RaveUtilities_addStringAttributeToList(attributes,
           "what/object", RaveTypes_getStringFromObjectType(Rave_ObjectType_PVOL)) ||
        !RaveUtilities_addStringAttributeToList(attributes, "what/version", RaveIO_ODIM_H5rad_Version_2_0_STR)) {
      RAVE_ERROR0("Failed to add what/object or what/version to attributes");
      goto done;
    }
  }

  if (attributes == NULL || !RaveHL_addAttributes(nodelist, attributes, "")) {
    goto done;
  }

  result = 1; // Set result to 1 now and if adding of scans fails, it will become 0 again

  nscans = PolarVolume_getNumberOfScans(object);
  for (i = 0; result == 1 && i < nscans; i++) {
    PolarScan_t* scan = PolarVolume_getScan(object, i);
    result = RaveIOInternal_addVolumeScanToNodeList(scan, nodelist, "/dataset%d", (i+1));
    RAVE_OBJECT_RELEASE(scan);
  }

done:
  RAVE_OBJECT_RELEASE(attributes);
  return result;
#endif
}

static int RaveIOInternal_addScanToNodeList(PolarScan_t* object, HL_NodeList* nodelist)
{
  int result = 0;
  PolarOdimIO_t* odimio = RAVE_OBJECT_NEW(&PolarOdimIO_TYPE);
  if (odimio != NULL) {
    result = PolarOdimIO_fillScan(odimio, object, nodelist);
  }
  RAVE_OBJECT_RELEASE(odimio);
  return result;

#ifdef KALLE
  int result = 0;
  RaveObjectList_t* attributes = NULL;
  RaveList_t* keys = NULL;

  RAVE_ASSERT((object != NULL), "object == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  attributes = PolarScan_getAttributeValues(object, Rave_ObjectType_SCAN, 1);
  if (attributes != NULL) {
    if (!RaveUtilities_addStringAttributeToList(attributes,
           "what/object", RaveTypes_getStringFromObjectType(Rave_ObjectType_SCAN)) ||
        !RaveUtilities_addStringAttributeToList(attributes, "what/version", RaveIO_ODIM_H5rad_Version_2_0_STR)) {
      RAVE_ERROR0("Failed to add what/object or what/version to attributes");
      goto done;
    }
  }

  if (attributes == NULL || !RaveHL_addAttributes(nodelist, attributes, "")) {
    goto done;
  }

  if (!RaveHL_createGroup(nodelist, "/dataset1")) {
    goto done;
  }

  RAVE_OBJECT_RELEASE(attributes);
  attributes = PolarScan_getAttributeValues(object, Rave_ObjectType_SCAN, 0);
  if (attributes == NULL || !RaveHL_addAttributes(nodelist, attributes, "/dataset1")) {
    goto done;
  }

  result = 1;

  keys = PolarScan_getParameterNames(object);
  if (keys != NULL) {
    int i = 0;
    int nkeys = RaveList_size(keys);
    for (i = 0; result == 1 && i < nkeys; i++) {
      const char* keyname = RaveList_get(keys, i);
      PolarScanParam_t* parameter = PolarScan_getParameter(object, keyname);
      if (parameter == NULL) {
        result = 0;
      }
      if (result == 1) {
        result = RaveIOInternal_addScanParamToNodeList(parameter, nodelist, "/dataset1/data%d", (i+1));
      }
      RAVE_OBJECT_RELEASE(parameter);
    }
  }

  if (result == 1 && PolarScan_getNumberOfQualityFields(object) > 0) {
    int nfields = PolarScan_getNumberOfQualityFields(object);
    int i = 0;
    for (i = 0; result == 1 && i < nfields; i++) {
      RaveField_t* field = PolarScan_getQualityField(object, i);
      if (field != NULL) {
        result = RaveIOInternal_addRaveFieldToNodeList(field, nodelist, "/dataset1/quality%d", (i+1));
      } else {
        result = 0;
      }
      RAVE_OBJECT_RELEASE(field);
    }
  }

done:
  RaveList_freeAndDestroy(&keys);
  RAVE_OBJECT_RELEASE(attributes);
  return result;
#endif
}

/**
 * Loads the 2-d data set with the specified name.
 * @param[in] nodelist - the hlhdf nodelist
 * @param[in] fmt - the varargs format string
 * @param[in] ... - the variable argument list
 * @returns the rave data 2d on success, otherwise NULL.
 */
static RaveData2D_t* RaveIOInternal_loadRave2dData(HL_NodeList* nodelist, const char* fmt, ...)
{
  char nodeName[1024];
  va_list ap;
  int n = 0;
  RaveData2D_t* result = NULL;
  HL_Node* node = NULL;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n < 0 || n >= 1024) {
    RAVE_ERROR1("Failed to create cartesian name from fmt=%s", fmt);
    goto done;
  }

  node = HLNodeList_getNodeByName(nodelist, nodeName);
  if (node != NULL &&
      HLNode_getType(node) == DATASET_ID) {
    result = RAVE_OBJECT_NEW(&RaveData2D_TYPE);
    if (result != NULL) {
      hsize_t d1 = HLNode_getDimension(node, 1);
      hsize_t d0 = HLNode_getDimension(node, 0);
      RaveDataType dataType = RaveHL_hlhdfToRaveType(HLNode_getFormat(node));
      if(!RaveData2D_setData(result, d1, d0, HLNode_getData(node), dataType)) {
        RAVE_ERROR0("Failed to set data into RaveData2D-field");
        RAVE_OBJECT_RELEASE(result);
        goto done;
      }
    }
  }

done:
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
  Cartesian_t* cartesian = NULL;

  Rave_ObjectType objectType = Rave_ObjectType_UNDEFINED;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  objectType = RaveIOInternal_getObjectType(nodelist);
  if (objectType != Rave_ObjectType_IMAGE) {
    RAVE_ERROR0("Can not load provided file as a cartesian product");
    return NULL;
  }
  cartesian = RAVE_OBJECT_NEW(&Cartesian_TYPE);
  if (cartesian == NULL) {
    RAVE_ERROR0("Failed to create cartesian object");
    goto done;
  }

  if (!Cartesian_setObjectType(cartesian, objectType)) {
    goto done;
  }

  if (!RaveIOInternal_loadAttributesAndDataForObject(nodelist, (RaveCoreObject*)cartesian, "")) {
    goto done;
  }

  if (!RaveIOInternal_loadAttributesAndDataForObject(nodelist, (RaveCoreObject*)cartesian, "/dataset1")) {
    goto done;
  }

  if (!RaveIOInternal_loadAttributesAndDataForObject(nodelist, (RaveCoreObject*)cartesian, "/dataset1/data1")) {
    goto done;
  }

  result = RAVE_OBJECT_COPY(cartesian);
done:
  RAVE_OBJECT_RELEASE(cartesian);
  return result;
}

static Cartesian_t* RaveIOInternal_loadCartesianForVolume(HL_NodeList* nodelist, const char* fmt, ...)
{
  char nodeName[1024];
  va_list ap;
  int n = 0;
  Cartesian_t* result = NULL;
  Cartesian_t* cartesian = NULL;
  RaveData2D_t* data2d = NULL;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n < 0 || n >= 1024) {
    RAVE_ERROR1("Failed to create cartesian name from fmt=%s", fmt);
    goto done;
  }

  if (!RaveIOInternal_hasNodeByName(nodelist, "%s/data1/data", nodeName)) {
    RAVE_ERROR1("No data in %s/data1\n",nodeName);
    goto done;
  }

  if (HLNodeList_hasNodeByName(nodelist, nodeName)) {
    cartesian = RAVE_OBJECT_NEW(&Cartesian_TYPE);
    if (cartesian == NULL) {
      goto done;
    }
    if (!RaveIOInternal_loadAttributesAndDataForObject(nodelist, (RaveCoreObject*)cartesian, nodeName)) {
      RAVE_OBJECT_RELEASE(cartesian);
      goto done;
    }
  }

  data2d = RaveIOInternal_loadRave2dData(nodelist, "%s/data1/data", nodeName);
  if (data2d != NULL) {
    if (!Cartesian_setData(cartesian,
                           RaveData2D_getXsize(data2d),
                           RaveData2D_getYsize(data2d),
                           RaveData2D_getData(data2d),
                           RaveData2D_getType(data2d))) {
      goto done;
    }
  }
  Cartesian_setObjectType(cartesian, Rave_ObjectType_IMAGE);

  result = RAVE_OBJECT_COPY(cartesian);
done:
  RAVE_OBJECT_RELEASE(cartesian);
  RAVE_OBJECT_RELEASE(data2d);
  return result;
}

static RaveCoreObject* RaveIOInternal_loadCartesianVolume(HL_NodeList* nodelist)
{
  int sindex = 1;
  CartesianVolume_t* result = NULL;
  CartesianVolume_t* cvol = NULL;
  Rave_ObjectType objectType = Rave_ObjectType_UNDEFINED;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  objectType = RaveIOInternal_getObjectType(nodelist);
  if (objectType != Rave_ObjectType_CVOL && objectType != Rave_ObjectType_COMP) {
    RAVE_ERROR0("Can not load provided file as a cartesian volume");
    return NULL;
  }
  cvol = RAVE_OBJECT_NEW(&CartesianVolume_TYPE);
  if (cvol == NULL) {
    RAVE_ERROR0("Failed to create cartesian volume");
    goto done;
  }

  if (!RaveIOInternal_loadAttributesAndDataForObject(nodelist, (RaveCoreObject*)cvol, "")) {
    goto done;
  }

  while (RaveIOInternal_hasNodeByName(nodelist, "/dataset%d", sindex)) {
    Cartesian_t* image = RaveIOInternal_loadCartesianForVolume(nodelist, "/dataset%d", sindex);
    if (image == NULL) {
      RAVE_ERROR0("Failed to read image");
      goto done;
    }

    if (!CartesianVolume_addImage(cvol, image)) {
      RAVE_ERROR0("Failed to add image to cartesian volume");
      RAVE_OBJECT_RELEASE(image);
      goto done;
    }
    RAVE_OBJECT_RELEASE(image);
    sindex++;
  }
  result = RAVE_OBJECT_COPY(cvol);
done:
  RAVE_OBJECT_RELEASE(cvol);
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
  HL_NodeList* nodelist = NULL;
  Rave_ObjectType objectType = Rave_ObjectType_UNDEFINED;
  RaveCoreObject* object = NULL;
  int result = 0;
  RaveIO_ODIM_Version version = RaveIO_ODIM_Version_UNDEFINED;
  RaveIO_ODIM_H5rad_Version h5radversion = RaveIO_ODIM_H5rad_Version_UNDEFINED;

  RAVE_ASSERT((raveio != NULL), "raveio == NULL");

  if (raveio->filename == NULL) {
    RAVE_ERROR0("Atempting to load a file even though no filename has been specified");
    goto done;
  }

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
  } else {
    RAVE_ERROR1("Currently, RaveIO does not support the object type as defined by '%s'", raveio->filename);
    goto done;
  }

  if (object != NULL) {
    RAVE_OBJECT_RELEASE(raveio->object);
    raveio->object = RAVE_OBJECT_COPY(object);
    raveio->version = version;
    raveio->h5radversion = h5radversion;
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(object);
  HLNodeList_free(nodelist);
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
        RAVE_OBJECT_CHECK_TYPE(raveio->object, &PolarScan_TYPE)) {
      HL_NodeList* nodelist = HLNodeList_new();

      if (nodelist != NULL) {
        result = RaveHL_createStringValue(nodelist, RaveIO_ODIM_Version_2_0_STR, "/Conventions");
        if (result == 1) {
          if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &PolarVolume_TYPE)) {
            result = RaveIOInternal_addPolarVolumeToNodeList((PolarVolume_t*)raveio->object, nodelist);
          } else if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &CartesianVolume_TYPE)) {
            result = RaveIOInternal_addCartesianVolumeToNodeList((CartesianVolume_t*)raveio->object, nodelist);
          } else if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &Cartesian_TYPE)) {
            result = RaveIOInternal_addCartesianToNodeList((Cartesian_t*)raveio->object, nodelist);
          } else if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &PolarScan_TYPE)) {
            result = RaveIOInternal_addScanToNodeList((PolarScan_t*)raveio->object, nodelist);
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

/*@} End of Interface functions */

RaveCoreObjectType RaveIO_TYPE = {
    "RaveIO",
    sizeof(RaveIO_t),
    RaveIO_constructor,
    RaveIO_destructor
};
