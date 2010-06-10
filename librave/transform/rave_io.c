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
#include "hlhdf.h"
#include "hlhdf_alloc.h"
#include "hlhdf_debug.h"
#include "string.h"
#include "stdarg.h"
#include "raveobject_hashtable.h"
#include "raveobject_list.h"

/**
 * Defines the structure for the RaveIO in a volume.
 */
struct _RaveIO_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveCoreObject* object;                 /**< the object */
  RaveIO_ODIM_Version version;            /**< the odim version */
  RaveIO_ODIM_H5rad_Version h5radversion; /**< the h5rad object version */
  char* filename;                         /**< the filename */
};

/*@{ Constants */
static const char RaveIO_ODIM_Version_2_0_STR[] = "ODIM_H5/V2_0";

static const char RaveIO_ODIM_H5rad_Version_2_0_STR[] = "H5rad 2.0";

#define RAVEIO_NR_TOKENS 32   /**< nr of tokens that can be filled */

#define RAVEIO_TOKEN_LENGTH 64 /**< length of each token */

// Object types

static const char RaveIO_ObjectType_UNDEFINED_STR[]= "UNDEFINED";
static const char RaveIO_ObjectType_PVOL_STR[]= "PVOL";
static const char RaveIO_ObjectType_CVOL_STR[]= "CVOL";
static const char RaveIO_ObjectType_SCAN_STR[]= "SCAN";
static const char RaveIO_ObjectType_RAY_STR[]= "RAY";
static const char RaveIO_ObjectType_AZIM_STR[]= "AZIM";
static const char RaveIO_ObjectType_IMAGE_STR[]= "IMAGE";
static const char RaveIO_ObjectType_COMP_STR[]= "COMP";
static const char RaveIO_ObjectType_XSEC_STR[]= "XSEC";
static const char RaveIO_ObjectType_VP_STR[]= "VP";
static const char RaveIO_ObjectType_PIC_STR[]= "PIC";

/**
 * Mapping between a object type and the corresponding string
 */
struct RaveIO_ObjectTypeMapping {
  Rave_ObjectType type;  /**< the object type */
  const char* str;       /**< the string representation */
};

/**
 * The mapping table.
 */
static const struct RaveIO_ObjectTypeMapping OBJECT_TYPE_MAPPING[] =
{
  {Rave_ObjectType_UNDEFINED, RaveIO_ObjectType_UNDEFINED_STR},
  {Rave_ObjectType_PVOL, RaveIO_ObjectType_PVOL_STR},
  {Rave_ObjectType_CVOL, RaveIO_ObjectType_CVOL_STR},
  {Rave_ObjectType_SCAN, RaveIO_ObjectType_SCAN_STR},
  {Rave_ObjectType_RAY, RaveIO_ObjectType_RAY_STR},
  {Rave_ObjectType_AZIM, RaveIO_ObjectType_AZIM_STR},
  {Rave_ObjectType_IMAGE, RaveIO_ObjectType_IMAGE_STR},
  {Rave_ObjectType_COMP, RaveIO_ObjectType_COMP_STR},
  {Rave_ObjectType_XSEC, RaveIO_ObjectType_XSEC_STR},
  {Rave_ObjectType_VP, RaveIO_ObjectType_VP_STR},
  {Rave_ObjectType_PIC, RaveIO_ObjectType_PIC_STR},
  {Rave_ObjectType_ENDOFTYPES, NULL}
};

/**
 * Mapping between hlhdf format and rave data type
 */
struct RaveToHlhdfTypeMap {
  HL_FormatSpecifier hlhdfFormat; /**< the hlhdf format */
  RaveDataType raveType;          /**< the rave data type */
};

/**
 * The mapping table
 */
static const struct RaveToHlhdfTypeMap RAVE_TO_HLHDF_MAP[] = {
  {HLHDF_UNDEFINED, RaveDataType_UNDEFINED},
  {HLHDF_CHAR, RaveDataType_CHAR},
  {HLHDF_SCHAR, RaveDataType_CHAR},
  {HLHDF_UCHAR, RaveDataType_UCHAR},
  {HLHDF_SHORT, RaveDataType_SHORT},
  {HLHDF_USHORT, RaveDataType_SHORT},
  {HLHDF_INT, RaveDataType_INT},
  {HLHDF_UINT, RaveDataType_INT},
  {HLHDF_LONG, RaveDataType_LONG},
  {HLHDF_ULONG, RaveDataType_LONG},
  {HLHDF_LLONG, RaveDataType_UNDEFINED},
  {HLHDF_ULLONG, RaveDataType_UNDEFINED},
  {HLHDF_FLOAT, RaveDataType_FLOAT},
  {HLHDF_DOUBLE, RaveDataType_DOUBLE},
  {HLHDF_LDOUBLE, RaveDataType_UNDEFINED},
  {HLHDF_HSIZE, RaveDataType_UNDEFINED},
  {HLHDF_HSSIZE, RaveDataType_UNDEFINED},
  {HLHDF_HERR, RaveDataType_UNDEFINED},
  {HLHDF_HBOOL, RaveDataType_UNDEFINED},
  {HLHDF_STRING, RaveDataType_UNDEFINED},
  {HLHDF_COMPOUND, RaveDataType_UNDEFINED},
  {HLHDF_ARRAY, RaveDataType_UNDEFINED},
  {HLHDF_END_OF_SPECIFIERS, RaveDataType_UNDEFINED}
};

/**
 * Product Type mappings
 */
static const char RaveIO_ProductType_UNDEFINED_STR[] = "UNDEFINED";
static const char RaveIO_ProductType_SCAN_STR[] = "SCAN";
static const char RaveIO_ProductType_PPI_STR[] = "PPI";
static const char RaveIO_ProductType_CAPPI_STR[] = "CAPPI";
static const char RaveIO_ProductType_PCAPPI_STR[] = "PCAPPI";
static const char RaveIO_ProductType_ETOP_STR[] = "ETOP";
static const char RaveIO_ProductType_MAX_STR[] = "MAX";
static const char RaveIO_ProductType_RR_STR[] = "RR";
static const char RaveIO_ProductType_VIL_STR[] = "VIL";
static const char RaveIO_ProductType_COMP_STR[] = "COMP";
static const char RaveIO_ProductType_VP_STR[] = "VP";
static const char RaveIO_ProductType_RHI_STR[] = "RHI";
static const char RaveIO_ProductType_XSEC_STR[] = "XSEC";
static const char RaveIO_ProductType_VSP_STR[] = "VSP";
static const char RaveIO_ProductType_HSP_STR[] = "HSP";
static const char RaveIO_ProductType_RAY_STR[] = "RAY";
static const char RaveIO_ProductType_AZIM_STR[] = "AZIM";
static const char RaveIO_ProductType_QUAL_STR[] = "QUAL";

/**
 * Mapping between a product type and the corresponding string
 */
struct RaveIO_ProductMapping {
  Rave_ProductType type; /**< the product type */
  const char* str;       /**< the string */
};

/**
 * The mapping table.
 */
static const struct RaveIO_ProductMapping PRODUCT_MAPPING[] =
{
  {Rave_ProductType_UNDEFINED, RaveIO_ProductType_UNDEFINED_STR},
  {Rave_ProductType_SCAN, RaveIO_ProductType_SCAN_STR},
  {Rave_ProductType_PPI, RaveIO_ProductType_PPI_STR},
  {Rave_ProductType_CAPPI, RaveIO_ProductType_CAPPI_STR},
  {Rave_ProductType_PCAPPI, RaveIO_ProductType_PCAPPI_STR},
  {Rave_ProductType_ETOP, RaveIO_ProductType_ETOP_STR},
  {Rave_ProductType_MAX, RaveIO_ProductType_MAX_STR},
  {Rave_ProductType_RR, RaveIO_ProductType_RR_STR},
  {Rave_ProductType_VIL, RaveIO_ProductType_VIL_STR},
  {Rave_ProductType_COMP, RaveIO_ProductType_COMP_STR},
  {Rave_ProductType_VP, RaveIO_ProductType_VP_STR},
  {Rave_ProductType_RHI, RaveIO_ProductType_RHI_STR},
  {Rave_ProductType_XSEC, RaveIO_ProductType_XSEC_STR},
  {Rave_ProductType_VSP, RaveIO_ProductType_VSP_STR},
  {Rave_ProductType_HSP, RaveIO_ProductType_HSP_STR},
  {Rave_ProductType_RAY, RaveIO_ProductType_RAY_STR},
  {Rave_ProductType_AZIM, RaveIO_ProductType_AZIM_STR},
  {Rave_ProductType_QUAL, RaveIO_ProductType_QUAL_STR},
  {Rave_ProductType_ENDOFTYPES, NULL},
};

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
  raveio->object = NULL;
  raveio->version = RaveIO_ODIM_Version_2_0;
  raveio->h5radversion = RaveIO_ODIM_H5rad_Version_2_0;
  raveio->filename = NULL;
  return 1;
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
}
/**
 * Translates a hlhdf format specified into a rave data type.
 * @param[in] format - the hlhdf format specified
 * @returns the RaveDataType
 */
static RaveDataType RaveIOInternal_hlhdfToRaveType(HL_FormatSpecifier format)
{
  int index = 0;
  RaveDataType result = RaveDataType_UNDEFINED;
  while (RAVE_TO_HLHDF_MAP[index].hlhdfFormat != HLHDF_END_OF_SPECIFIERS) {
    if (RAVE_TO_HLHDF_MAP[index].hlhdfFormat == format) {
      result = RAVE_TO_HLHDF_MAP[index].raveType;
      break;
    }
    index++;
  }
  return result;
}

/**
 * Translates a rave data type into a hlhdf format specifier
 * @param[in] format - the rave data type
 * @returns the hlhdf format specifier
 */
static HL_FormatSpecifier RaveIOInternal_raveToHlhdfType(RaveDataType format)
{
  int index = 0;
  HL_FormatSpecifier result = HLHDF_UNDEFINED;
  while (RAVE_TO_HLHDF_MAP[index].hlhdfFormat != HLHDF_END_OF_SPECIFIERS) {
    if (RAVE_TO_HLHDF_MAP[index].raveType == format) {
      result = RAVE_TO_HLHDF_MAP[index].hlhdfFormat;
      break;
    }
    index++;
  }
  return result;
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
 * Creates a string attribute node in the nodelist.
 * @param[in] nodelist - the list the node should be added to
 * @param[in] value - the string to be saved.
 * @param[in] fmt - the variable argument format
 * @param[in] ... - the arguments to the format
 * @returns 1 on success, otherwise 0
 */
static int RaveIOInternal_createStringValue(HL_NodeList* nodelist, const char* value, const char* fmt, ...)
{
  va_list ap;
  char nodeName[1024];
  int n = 0;
  int result = 0;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((fmt != NULL), "fmt == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n >= 0 && n < 1024) {
    HL_Node* node = NULL;
    node = HLNode_newAttribute(nodeName);
    if (node == NULL) {
      RAVE_CRITICAL1("Failed to create an attribute with name %s", nodeName);
      goto done;
    }
    if (!HLNode_setScalarValue(node, strlen(value) + 1, (unsigned char*)value, "string", -1)) {
      RAVE_ERROR1("Failed to set string value for %s", nodeName);
      HLNode_free(node);
      goto done;
    }
    if (!HLNodeList_addNode(nodelist, node)) {
      RAVE_ERROR1("Failed to add node %s to nodelist", nodeName);
      HLNode_free(node);
      goto done;
    }
    result = 1;
  }

done:
  if (result == 0) {
    RAVE_ERROR0("Failed to create string attribute node");
  }
  return result;
}

/**
 * Fetches the node with the provided name and returns the
 * value.
 * @param[in] nodelist - the hlhdf node list
 * @param[out] value - the double value
 * @param[in] fmt - the varargs format specifier
 * @param[in] ... - the varargs
 * @returns 0 on failure, otherwise 1
 */
static int RaveIOInternal_getDoubleValue(HL_NodeList* nodelist, double* value, const char* fmt, ...)
{
  int result = 0;
  va_list ap;
  char nodeName[1024];
  int n = 0;
  size_t sz = 0;
  HL_Node* node = NULL;
  HL_FormatSpecifier format = HLHDF_UNDEFINED;
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

  format = HLNode_getFormat(node);
  if (format < HLHDF_FLOAT || format > HLHDF_LDOUBLE) {
    RAVE_ERROR1("%s is not a float or double", nodeName);
    goto done;
  }
  sz = HLNode_getDataSize(node);

  if (sz == sizeof(float)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(float, node, sz, double, *value);
  } else if (sz == sizeof(double)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(double, node, sz, double, *value);
  } else if (sz == sizeof(long double)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(long double, node, sz, double, *value);
    RAVE_WARNING1("Node %s is of type long double, downcasting",nodeName);
  }

  result = 1;
done:
  return result;
}

/**
 * Creates a double attribute node.
 * @param[in] nodelist - the node list the node should be added to
 * @param[in] value - the value to be saved
 * @param[in] fmt - the variable argument format string representing the node name
 * @param[in] ... - the arguments
 * @returns 1 on success, otherwise 0
 */
int RaveIOInternal_createDoubleValue(HL_NodeList* nodelist, double value, const char* fmt, ...)
{
  va_list ap;
  char nodeName[1024];
  int n = 0;
  int result = 0;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((fmt != NULL), "fmt == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n >= 0 && n < 1024) {
    HL_Node* node = NULL;
    node = HLNode_newAttribute(nodeName);
    if (node == NULL) {
      RAVE_CRITICAL1("Failed to create an attribute with name %s", nodeName);
      goto done;
    }
    if (!HLNode_setScalarValue(node, sizeof(double), (unsigned char*)&value, "double", -1)) {
      RAVE_ERROR1("Failed to set double value for %s", nodeName);
      HLNode_free(node);
      goto done;
    }
    if (!HLNodeList_addNode(nodelist, node)) {
      RAVE_ERROR1("Failed to add node %s to nodelist", nodeName);
      HLNode_free(node);
      goto done;
    }
    result = 1;
  }

done:
  if (result == 0) {
    RAVE_ERROR0("Failed to create double attribute node");
  }
  return result;
}

/**
 * Fetches the node with the provided name and returns the
 * value.
 * @param[in] nodelist - the hlhdf node list
 * @param[out] value - the long value
 * @param[in] fmt - the varargs format specifier
 * @param[in] ... - the varargs
 * @returns 0 on failure, otherwise 1
 */
static int RaveIOInternal_getLongValue(HL_NodeList* nodelist, long* value, const char* fmt, ...)
{
  int result = 0;
  va_list ap;
  char nodeName[1024];
  int n = 0;
  size_t sz = 0;
  HL_Node* node = NULL;
  HL_FormatSpecifier format = HLHDF_UNDEFINED;
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

  format = HLNode_getFormat(node);
  if (format < HLHDF_SCHAR || format > HLHDF_ULLONG) {
    RAVE_ERROR1("%s is not a integer", nodeName);
    goto done;
  }
  sz = HLNode_getDataSize(node);
  if (sz == sizeof(char)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(char, node, sz, long, *value);
  } else if (sz == sizeof(short)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(short, node, sz, long, *value);
  } else if (sz == sizeof(int)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(int, node, sz, long, *value);
  } else if (sz == sizeof(long)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(long, node, sz, long, *value);
  } else if (sz == sizeof(long long)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(long long, node, sz, long, *value);
    RAVE_WARNING1("Node %s is of type long long, downcasting",nodeName);
  }

  result = 1;
done:
  return result;
}

/**
 * Creates a long attribute node in the nodelist
 * @param[in] nodelist - the list the node should be added to
 * @param[in] value - the value to save
 * @param[in] fmt - the variable argument format string representing the node name
 * @param[in] ... - the variable argument list
 * @returns 1 on success, otherwise 0
 */
static int RaveIOInternal_createLongValue(HL_NodeList* nodelist, long value, const char* fmt, ...)
{
  va_list ap;
  char nodeName[1024];
  int n = 0;
  int result = 0;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((fmt != NULL), "fmt == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n >= 0 && n < 1024) {
    HL_Node* node = NULL;
    node = HLNode_newAttribute(nodeName);
    if (node == NULL) {
      RAVE_CRITICAL1("Failed to create an attribute with name %s", nodeName);
      goto done;
    }
    if (!HLNode_setScalarValue(node, sizeof(long), (unsigned char*)&value, "long", -1)) {
      RAVE_ERROR1("Failed to set long value for %s", nodeName);
      HLNode_free(node);
      goto done;
    }
    if (!HLNodeList_addNode(nodelist, node)) {
      RAVE_ERROR1("Failed to add node %s to nodelist", nodeName);
      HLNode_free(node);
      goto done;
    }
    result = 1;
  }

done:
  if (result == 0) {
    RAVE_ERROR0("Failed to create long attribute node");
  }
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
 * Puts an attribute in the nodelist as a hlhdf node.
 * @param[in] nodelist - the node list
 * @param[in] attribute - the attribute
 * @param[in] name - the root name, e.g. /dataset1/data1
 * @returns 1 on success otherwise 0
 */
static int RaveIOInternal_addAttribute(
  HL_NodeList* nodelist, RaveAttribute_t* attribute, const char* name)
{
  const char* attrname = NULL;
  int result = 0;
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((attribute != NULL), "attribute == NULL");

  attrname = RaveAttribute_getName(attribute);
  if (attrname != NULL) {
    HL_Node* node = NULL;
    char nodeName[1024];
    sprintf(nodeName, "%s/%s", name, attrname);
    node = HLNode_newAttribute(nodeName);
    if (node == NULL) {
      RAVE_CRITICAL1("Failed to create an attribute with name %s", nodeName);
      goto done;
    }
    if (RaveAttribute_getFormat(attribute) == RaveAttribute_Format_Long) {
      long value;
      RaveAttribute_getLong(attribute, &value);
      result = HLNode_setScalarValue(node, sizeof(long), (unsigned char*)&value, "long", -1);
    } else if (RaveAttribute_getFormat(attribute) == RaveAttribute_Format_Double) {
      double value;
      RaveAttribute_getDouble(attribute, &value);
      result = HLNode_setScalarValue(node, sizeof(double), (unsigned char*)&value, "double", -1);
    } else if (RaveAttribute_getFormat(attribute) == RaveAttribute_Format_String) {
      char* value = NULL;
      RaveAttribute_getString(attribute, &value);
      if (value != NULL) {
        result = HLNode_setScalarValue(node, strlen(value)+1, (unsigned char*)value, "string", -1);
      } else {
        RAVE_WARNING1("Attribute %s is NULL and will be ignored", nodeName);
        HLNode_free(node);
        node = NULL;
        result = 1;
      }
    }
    if (result == 1 && node != NULL) {
      result = HLNodeList_addNode(nodelist, node);
      if (result == 0) {
        HLNode_free(node);
        node = NULL;
        RAVE_ERROR1("Could not add node %s", nodeName);
      }
    }
  }

done:
  return result;
}

/**
 * Returns the node with the name as specified by the variable argument
 * list.
 * @param[in] nodelist - the hlhdf nodelist
 * @param[in] fmt    - the variable argument format
 * @param[in] ...    - the argument list
 * @returns the found node or NULL if not found
 */
static HL_Node* RaveIOInternal_getNode(HL_NodeList* nodelist, const char* fmt, ...)
{
  va_list ap;
  char nodeName[1024];
  int n = 0;
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n >= 0 && n < 1024) {
    return HLNodeList_getNodeByName(nodelist, nodeName);
  }
  return NULL;
}

/**
 * Creates a group node in the node list.
 * @param[in] nodelist - the node list
 * @param[in] fmt - the variable argument format
 * @param[in] ... - the arguments.
 */
static int RaveIOInternal_createGroup(HL_NodeList* nodelist, const char* fmt, ...)
{
  va_list ap;
  char nodeName[1024];
  int n = 0;
  int result = 0;
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n >= 0 && n < 1024) {
    HL_Node* node = HLNode_newGroup(nodeName);
    if (node == NULL) {
      RAVE_CRITICAL1("Failed to create group with name %s", nodeName);
      goto done;
    }
    if (!HLNodeList_addNode(nodelist, node)) {
      RAVE_CRITICAL1("Failed to add group node with name %s", nodeName);
      HLNode_free(node);
      goto done;
    }
    result = 1;
  }
done:
  if (result == 0) {
    RAVE_CRITICAL0("Failed to add group node");
  }
  return result;
}

/**
 * Creates a dataset with the provided 2-dimensional array.
 * @param[in] nodelist - the node list
 * @param[in] data - the data
 * @param[in] xsize - the xsize
 * @param[in] ysize - the ysize
 * @param[in] dataType - the type of data
 * @param[in] fmt - the variable argument format
 * @param[in] ... - the arguments.
 * @returns 1 on success, otherwise 0
 */
static int RaveIOInternal_createDataset(HL_NodeList* nodelist, void* data, long xsize, long ysize, RaveDataType dataType, const char* fmt, ...)
{
  va_list ap;
  char nodeName[1024];
  int n = 0;
  int result = 0;
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n >= 0 && n < 1024) {
    HL_Node* node = HLNode_newDataset(nodeName);
    HL_FormatSpecifier specifier = RaveIOInternal_raveToHlhdfType(dataType);
    const char* hlhdfFormat = HL_getFormatSpecifierString(specifier);
    hsize_t dims[2];
    dims[1] = xsize;
    dims[0] = ysize;
    if (node == NULL) {
      RAVE_CRITICAL1("Failed to create dataset with name %s", nodeName);
      goto done;
    }

    if (!HLNode_setArrayValue(node,(size_t)get_ravetype_size(dataType),2,dims,data,hlhdfFormat,-1)) {
      HLNode_free(node);
      goto done;
    }

    if (!HLNodeList_addNode(nodelist, node)) {
      RAVE_CRITICAL1("Failed to add dataset node with name %s", nodeName);
      HLNode_free(node);
      goto done;
    }

    result = 1;
  }
done:
  if (result == 0) {
    RAVE_CRITICAL0("Failed to add dataset node");
  }
  return result;
}

/**
 * Returns the string representation of the product type.
 * @param[in] type - the product type
 * @returns the string representation or NULL if nothing could be found.
 */
static const char* RaveIOInternal_getProductString(Rave_ProductType type)
{
  int index = 0;
  while (PRODUCT_MAPPING[index].str != NULL) {
    if (type == PRODUCT_MAPPING[index].type) {
      return PRODUCT_MAPPING[index].str;
    }
    index++;
  }
  return NULL;
}

/**
 * Returns the string representation of the object type.
 * @param[in] type - the product type
 * @returns the string representation or NULL if nothing could be found.
 */
static const char* RaveIOInternal_getObjectTypeString(Rave_ObjectType type)
{
  int index = 0;
  while (OBJECT_TYPE_MAPPING[index].str != NULL) {
    if (type == OBJECT_TYPE_MAPPING[index].type) {
      return OBJECT_TYPE_MAPPING[index].str;
    }
    index++;
  }
  return NULL;
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
 * @param[in] name - the name of the object
 * @param[in] object - the object to fill
 */
static int RaveIOInternal_loadAttributesAndDataForObject(HL_NodeList* nodelist, const char* name, RaveCoreObject* object)
{
  int result = 1;
  int n = 0;
  int i = 0;
  int nameLength = 0;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((name != NULL), "nodelist == NULL");
  RAVE_ASSERT((object != NULL), "object == NULL");

  nameLength = strlen(name);

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
              } else if (RAVE_OBJECT_CHECK_TYPE(object, &RaveObjectHashTable_TYPE)) {
                result = RaveObjectHashTable_put((RaveObjectHashTable_t*)object, tmpptr, (RaveCoreObject*)attribute);
              } else if (RAVE_OBJECT_CHECK_TYPE(object, &RaveObjectList_TYPE)) {
                result = RaveObjectList_add((RaveObjectList_t*)object, (RaveCoreObject*)attribute);
              } else {
                RAVE_CRITICAL0("Unsupported type for load attributes");
                result = 0;
              }
            }
          }
          RAVE_OBJECT_RELEASE(attribute);
        } else if (HLNode_getType(node) == DATASET_ID &&
            strcasecmp(tmpptr, "data")==0) {
          hsize_t d1 = HLNode_getDimension(node, 1);
          hsize_t d0 = HLNode_getDimension(node, 0);
          RaveDataType dataType = RaveIOInternal_hlhdfToRaveType(HLNode_getFormat(node));
          if (dataType != RaveDataType_UNDEFINED) {
            if (RAVE_OBJECT_CHECK_TYPE(object, &PolarScanParam_TYPE)) {
              result = PolarScanParam_setData((PolarScanParam_t*)object, d1, d0, HLNode_getData(node), dataType);
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
 * Stores the attributes from the object into the nodelist
 * name/how/..., name/where/... and name/what/...
 * @param[in] nodelist - the hlhdf list
 * @param[in] name - the name of the object
 * @param[in] object - the object to fill
 */
static int RaveIOInternal_addAttributes(HL_NodeList* nodelist, RaveObjectList_t* attributes, const char* name)
{
  RaveAttribute_t* attribute = NULL;
  int result = 0;
  int nattrs = 0;
  int i = 0;
  int hashow = 0, haswhat = 0, haswhere = 0;
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((attributes != NULL), "attributes == NULL");

  hashow = RaveIOInternal_hasNodeByName(nodelist, "%s/how", name);
  haswhat = RaveIOInternal_hasNodeByName(nodelist, "%s/what", name);
  haswhere = RaveIOInternal_hasNodeByName(nodelist, "%s/where", name);

  nattrs = RaveObjectList_size(attributes);

  for (i = 0; i < nattrs; i++) {
    const char* attrname = NULL;
    RAVE_OBJECT_RELEASE(attribute);
    attribute = (RaveAttribute_t*)RaveObjectList_get(attributes, i);
    if (attribute == NULL) {
      RAVE_WARNING1("Failed to get attribute at index %d", i);
      goto done;
    }
    attrname = RaveAttribute_getName(attribute);
    if (attrname == NULL) {
      RAVE_ERROR1("Attribute at %d has no name set", i);
      goto done;
    }
    if (haswhat==0 && strncasecmp(attrname, "what/", 5)==0) {
      haswhat = RaveIOInternal_createGroup(nodelist, "%s/what",name);
      if (haswhat == 0) {
        RAVE_ERROR1("Failed to create group %s/what", name);
        goto done;
      }
    } else if (haswhere==0 && strncasecmp(attrname, "where/", 6)==0) {
      haswhere = RaveIOInternal_createGroup(nodelist, "%s/where",name);
      if (haswhere == 0) {
        RAVE_ERROR1("Failed to create group %s/where", name);
        goto done;
      }
    } else if (hashow==0 && strncasecmp(attrname, "how/", 4)==0) {
      hashow = RaveIOInternal_createGroup(nodelist, "%s/how",name);
      if (hashow == 0) {
        RAVE_ERROR1("Failed to create group %s/how", name);
        goto done;
      }
    } else {
      if (strncasecmp(attrname, "how/", 4) != 0 &&
          strncasecmp(attrname, "what/", 5) != 0 &&
          strncasecmp(attrname, "where/", 6) != 0) {
        RAVE_ERROR1("Unsupported attribute name %s", attrname);
        goto done;
      }
    }

    if (!RaveIOInternal_addAttribute(nodelist, attribute, name)) {
      RAVE_ERROR2("Failed to add attribute %s/%s to nodelist", name, attrname);
      goto done;
    }
  }
  result = 1;
done:
  RAVE_OBJECT_RELEASE(attribute);
  return result;
}

/**
 * Adds a data field to the node list according to ODIM H5. If data type is
 * UCHAR, the nessecary attributes for viewing in HdfView will also be added.
 * The name will always be <root>/data since that is according to ODIM H5
 * as well.
 * @param[in] nodelist - the node list that should get nodes added
 * @param[in] data - the array data
 * @param[in] xsize - the xsize
 * @param[in] ysize - the ysize
 * @param[in] dataType - type of data
 * @param[in] root - the root name
 * @returns 1 on success otherwise 0
 */
static int RaveIOInternal_addData(
  HL_NodeList* nodelist,
  void* data,
  long xsize,
  long ysize,
  RaveDataType dataType,
  const char* root)
{
  HL_Node* node = NULL;
  HL_FormatSpecifier specifier = HLHDF_UNDEFINED;
  hsize_t dims[2];
  int result = 0;
  const char* hlhdfFormat;
  char nodeName[1024];
  dims[1] = xsize;
  dims[0] = ysize;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((root != NULL), "root == NULL");
  if (data == NULL) {
    goto done;
  }
  sprintf(nodeName, "%s/data", root);
  node = HLNode_newDataset(nodeName);
  if (node == NULL) {
    RAVE_CRITICAL1("Failed to create dataset with name %s", nodeName);
    goto done;
  }
  specifier = RaveIOInternal_raveToHlhdfType(dataType);
  hlhdfFormat = HL_getFormatSpecifierString(specifier);

  if (!HLNode_setArrayValue(node,(size_t)get_ravetype_size(dataType),2,dims,data,hlhdfFormat,-1)) {
    goto done;
  }

  if (!HLNodeList_addNode(nodelist, node)) {
    RAVE_CRITICAL1("Failed to add dataset node with name %s", nodeName);
    goto done;
  }
  node = NULL; // Node has been added to list so release responsibility

  result = 1; // Set result to 1 now, if hdfview specific fails, result will be set back to 0.

  if (dataType == RaveDataType_UCHAR) {
    RaveAttribute_t* imgAttribute = RaveAttributeHelp_createString("CLASS", "IMAGE");
    RaveAttribute_t* verAttribute = RaveAttributeHelp_createString("IMAGE_VERSION", "1.2");
    if (imgAttribute == NULL || verAttribute == NULL) {
      result = 0;
    }
    if (result == 1) {
      result = RaveIOInternal_addAttribute(nodelist, imgAttribute, nodeName);
    }
    if (result == 1) {
      result = RaveIOInternal_addAttribute(nodelist, verAttribute, nodeName);
    }
    RAVE_OBJECT_RELEASE(imgAttribute);
    RAVE_OBJECT_RELEASE(verAttribute);
  }

done:
  HLNode_free(node);
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
  int index = 0;
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  if (!RaveIOInternal_getStringValue(nodelist, &objectType, "/what/object")) {
    RAVE_ERROR0("Failed to read attribute /what/object");
    goto done;
  }

  while (OBJECT_TYPE_MAPPING[index].str != NULL) {
    if (strcmp(OBJECT_TYPE_MAPPING[index].str, objectType) == 0) {
      result = OBJECT_TYPE_MAPPING[index].type;
      break;
    }
    index++;
  }
done:
  return result;
}

/**
 * Returns the product type for the specified varargs name.
 * @param[in] nodelist - the hlhdf node list
 * @param[in] fmt - the varargs format string
 * @param[in] ... - the varargs list
 * @returns a product type or Rave_ProductType_UNDEFINED on error
 */
static Rave_ProductType RaveIOInternal_getProductType(HL_NodeList* nodelist, const char* fmt, ...)
{
  Rave_ObjectType result = Rave_ObjectType_UNDEFINED;
  char* productType = NULL;
  int index = 0;
  va_list ap;
  char nodeName[1024];
  int n = 0;
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n < 0 || n >= 1024) {
    RAVE_CRITICAL0("Could not create node name for product type");
    goto done;
  }

  if (!RaveIOInternal_getStringValue(nodelist, &productType, nodeName)) {
    RAVE_ERROR1("Failed to read attribute %s", nodeName);
    goto done;
  }

  while (PRODUCT_MAPPING[index].str != NULL) {
    if (strcmp(PRODUCT_MAPPING[index].str, productType) == 0) {
      result = PRODUCT_MAPPING[index].type;
      break;
    }
    index++;
  }
done:
  return result;
}

///////////////////////////////////////////////////////////////////
///// POLAR SPECIFIC FUNCTIONS
///////////////////////////////////////////////////////////////////

/**
 * Validates that the scan is valid for storing as a part of a volume.
 * @param[in] scan - the scan to validate
 * @returns 1 if all is ok, otherwise 0
 */
static int RaveIOInternal_validateVolumeScan(PolarScan_t* scan)
{
  int result = 0;
  long nrays = 0, nbins = 0;
  RaveList_t* list = NULL;
  int nparams = 0;
  int i = 0;

  if (scan == NULL) {
    RAVE_INFO0("scan == NULL");
    goto done;
  }
  if (PolarScan_getTime(scan) == NULL) {
    RAVE_INFO0("time == NULL");
    goto done;
  }
  if (PolarScan_getDate(scan) == NULL) {
    RAVE_INFO0("date == NULL");
    goto done;
  }
  nrays = PolarScan_getNrays(scan);
  nbins = PolarScan_getNbins(scan);
  if (nrays <= 0) {
    RAVE_INFO0("nrays <= 0");
    goto done;
  }
  if (nbins <= 0) {
    RAVE_INFO0("nbins <= 0");
    goto done;
  }

  list = PolarScan_getParameterNames(scan);
  if (list == NULL) {
    RAVE_WARNING0("no parameter names");
    goto done;
  }
  nparams = RaveList_size(list);
  if (nparams == 0) {
    RAVE_WARNING0("no parameter names");
    goto done;
  }
  for (i = 0; i < nparams; i++) {
    char* name = RaveList_get(list, i);
    PolarScanParam_t* param = PolarScan_getParameter(scan, name);
    if (param == NULL) {
      goto done;
    }
    if (PolarScanParam_getQuantity(param)==NULL) {
      RAVE_INFO0("Parameter does not specify quantity");
      RAVE_OBJECT_RELEASE(param);
      goto done;
    }
    RAVE_OBJECT_RELEASE(param);
  }
  result = 1;
done:
  RaveList_freeAndDestroy(&list);
  return result;
}

/**
 * Validates that the volume not contains any bogus data before
 * atempting to store it.
 * @param[in] pvol - the volume to validate
 * @returns 1 if all is valid, otherwise 0.
 */
static int RaveIOInternal_validateVolume(PolarVolume_t* pvol)
{
  int result = 0;
  int nrScans = 0;
  int i = 0;

  if (PolarVolume_getTime(pvol) == NULL) {
    RAVE_INFO0("time == NULL");
    goto done;
  }
  if (PolarVolume_getDate(pvol) == NULL) {
    RAVE_INFO0("date == NULL");
    goto done;
  }
  if (PolarVolume_getSource(pvol) == NULL) {
    RAVE_INFO0("source == NULL");
    goto done;
  }

  nrScans = PolarVolume_getNumberOfScans(pvol);
  if (nrScans <= 0) {
    RAVE_INFO0("volume contains no scans");
    goto done;
  }

  for (i = 0; i < nrScans; i++) {
    PolarScan_t* scan = PolarVolume_getScan(pvol, i);
    if (!RaveIOInternal_validateVolumeScan(scan)) {
      RAVE_OBJECT_RELEASE(scan);
      goto done;
    }
    RAVE_OBJECT_RELEASE(scan);
  }

  result = 1;
done:
  return result;
}

/**
 * Loads a scan parameter.
 * @param[in] nodelist - the node list
 * @param[in] fmt - the varargs name of the parameter to load
 * @returns a parameter on success otherwise NULL
 */
static PolarScanParam_t* RaveIOInternal_loadScanParam(HL_NodeList* nodelist, const char* fmt, ...)
{
  va_list ap;
  char nodeName[1024];
  int n = 0;
  PolarScanParam_t* result = NULL;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n < 0 || n >= 1024) {
    goto done;
  }

  if (HLNodeList_hasNodeByName(nodelist, nodeName)) {
    result = RAVE_OBJECT_NEW(&PolarScanParam_TYPE);
    if (result == NULL) {
      goto done;
    }
    if (!RaveIOInternal_loadAttributesAndDataForObject(nodelist, nodeName, (RaveCoreObject*)result)) {
      RAVE_OBJECT_RELEASE(result);
    }
  }

done:
  return result;
}

/**
 * Loads a specific polar scan
 * @param[in] nodelist - the node list
 * @param[in] fmt - the varargs name of the scan to load
 * @returns a polar scan on success otherwise NULL
 */
static PolarScan_t* RaveIOInternal_loadSpecificScan(HL_NodeList* nodelist, const char* fmt, ...)
{
  va_list ap;
  char nodeName[1024];
  int pindex = 1;
  int n = 0;
  PolarScan_t* result = NULL;
  PolarScan_t* scan = NULL;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n < 0 || n >= 1024) {
    goto done;
  }

  if (HLNodeList_hasNodeByName(nodelist, nodeName)) {
    scan = RAVE_OBJECT_NEW(&PolarScan_TYPE);
    if (scan == NULL) {
      goto done;
    }
    if (!RaveIOInternal_loadAttributesAndDataForObject(nodelist, nodeName, (RaveCoreObject*)scan)) {
      RAVE_OBJECT_RELEASE(scan);
    }
  }

  while (RaveIOInternal_hasNodeByName(nodelist, "%s/data%d", nodeName, pindex)) {
    PolarScanParam_t* parameter = (PolarScanParam_t*)RaveIOInternal_loadScanParam(nodelist, "%s/data%d", nodeName, pindex);
    if (parameter == NULL) {
      RAVE_ERROR0("Failed to read parameter");
      goto done;
    }
    if (!PolarScan_addParameter(scan, parameter)) {
      RAVE_ERROR0("Failed to add parameter to scan");
      RAVE_OBJECT_RELEASE(parameter);
      goto done;
    }
    pindex++;
    RAVE_OBJECT_RELEASE(parameter);
  }

  result = RAVE_OBJECT_COPY(scan);
done:
  RAVE_OBJECT_RELEASE(scan);
  return result;
}

#ifdef KALLE
/**
 * Loads a individual polar scan
 * @param[in] nodelist - the node list
 * @param[in] fmt - the varargs name of the scan to load
 * @returns a polar scan on success otherwise NULL
 */
static PolarScan_t* RaveIOInternal_loadScan(HL_NodeList* nodelist)
{
  int sindex = 1;
  PolarScan_t* result = NULL;
  RaveObjectHashTable_t* rootattrs = NULL;
  RaveObjectHashTable_t* scanattrs = NULL;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  if (RaveIOInternal_getObjectType(nodelist) != Rave_ObjectType_SCAN) {
    RAVE_ERROR0("Can not load provided file as a scan");
    return NULL;
  }

  rootattrs = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  scanattrs = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  if (rootattrs == NULL || scanattrs == NULL) {
    RAVE_ERROR0("Failed to create hash table");
    goto done;
  }

  if (!RaveIOInternal_hasNodeByName(nodelist, "/dataset1")) {
    RAVE_ERROR0("Scan file does not contain scan...");
    goto done;
  }

  // Separate scans are a bit different to load since they have what/time, ...
  // that might be overridden by /dataset1/what/time so we need to fetch
  // the / group and then merge it with the /dataset1/what group so that we
  // get proper settings.
  if (!RaveIOInternal_loadAttributesAndDataForObject(nodelist, "", (RaveCoreObject*)rootattrs)) {
    goto done;
  }
  if (!RaveIOInternal_loadAttributesAndDataForObject(nodelist, "/dataset1", (RaveCoreObject*)scanattrs)) {
    goto done;
  }
  scan = RaveIOInternal_loadSpecificScan(nodelist, "/dataset1");
  if (scan == NULL) {
    goto done;
  }
  if (!RaveObjectHashTable_exists(scanattrs, "what/startdate")) {

    RaveUtilities_addStringAttributeToList()
    /*
    /what/date                               is an attribute
    /what/object                             is an attribute
    /what/source                             is an attribute
    /what/time                               is an attribute
    /what/version                            is an attribute
    /where                                   is a group
    /where/height                            is an attribute
    /where/lat                               is an attribute
    /where/lon
    */
  }

  result = RAVE_OBJECT_COPY(scan);
done:
  RAVE_OBJECT_RELEASE(scan);
  RAVE_OBJECT_RELEASE(attrs);
  return result;
}
#endif

/**
 * Loads a polar volume.
 * @param[in] nodelist - the node list
 * @returns a polar volume on success otherwise NULL
 */
static PolarVolume_t* RaveIOInternal_loadVolume(HL_NodeList* nodelist)
{
  int sindex = 1;
  PolarVolume_t* result = NULL;
  PolarVolume_t* volume = NULL;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  if (RaveIOInternal_getObjectType(nodelist) != Rave_ObjectType_PVOL) {
    RAVE_ERROR0("Can not load provided file as a volume");
    return NULL;
  }
  volume = RAVE_OBJECT_NEW(&PolarVolume_TYPE);
  if (volume == NULL) {
    RAVE_ERROR0("Failed to create volume object");
    goto done;
  }
  if (!RaveIOInternal_loadAttributesAndDataForObject(nodelist, "", (RaveCoreObject*)volume)) {
    goto done;
  }

  while (RaveIOInternal_hasNodeByName(nodelist, "/dataset%d", sindex)) {
    PolarScan_t* scan = (PolarScan_t*)RaveIOInternal_loadSpecificScan(nodelist, "/dataset%d", sindex);
    if (scan == NULL) {
      RAVE_ERROR0("Failed to read scan");
      goto done;
    }
    if (!PolarVolume_addScan(volume, scan)) {
      RAVE_ERROR0("Failed to add scan to volume");
      RAVE_OBJECT_RELEASE(scan);
      goto done;
    }
    RAVE_OBJECT_RELEASE(scan);
    sindex++;
  }
  result = RAVE_OBJECT_COPY(volume);
done:
  RAVE_OBJECT_RELEASE(volume);
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
    RAVE_ERROR1("Failed to create parameter name from fmt=%s", fmt);
    goto done;
  }

  if (!RaveIOInternal_hasNodeByName(nodelist, nodeName)) {
    if (!RaveIOInternal_createGroup(nodelist, nodeName)) {
      goto done;
    }
  }

  attributes = PolarScanParam_getAttributeValues(object);

  if (attributes == NULL || !RaveIOInternal_addAttributes(nodelist, attributes, nodeName)) {
    goto done;
  }

  if (!RaveIOInternal_addData(nodelist,
                              PolarScanParam_getData(object),
                              PolarScanParam_getNbins(object),
                              PolarScanParam_getNrays(object),
                              PolarScanParam_getDataType(object),
                              nodeName)) {
    goto done;
  }

  result = 1;
done:
  RaveList_freeAndDestroy(&keys);
  RAVE_OBJECT_RELEASE(attributes);
  return result;
}

/**
 * Adds a scan to a node list.
 * @param[in] object - the parameter to translate into hlhdf nodes
 * @param[in] nodelist - the nodelist the nodes should be added to
 * @param[in] fmt - the variable arguments format string that should be used to define the name of this scan group
 * @returns 1 on success otherwise 0
 */
static int RaveIOInternal_addScanToNodeList(PolarScan_t* object, HL_NodeList* nodelist, const char* fmt, ...)
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
    if (!RaveIOInternal_createGroup(nodelist, nodeName)) {
      goto done;
    }
  }

  attributes = PolarScan_getAttributeValues(object);
  if (attributes != NULL) {
    if (!RaveUtilities_addStringAttributeToList(attributes, "what/product", RaveIO_ProductType_SCAN_STR)) {
      goto done;
    }
    if (!RaveUtilities_addLongAttributeToList(attributes, "where/nbins", PolarScan_getNbins(object)) ||
        !RaveUtilities_addLongAttributeToList(attributes, "where/nrays", PolarScan_getNrays(object))) {
      goto done;
    }
  }

  if (attributes == NULL || !RaveIOInternal_addAttributes(nodelist, attributes, nodeName)) {
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
done:
  RaveList_freeAndDestroy(&keys);
  RAVE_OBJECT_RELEASE(attributes);
  return result;
}

/**
 * Adds a volume to a node list.
 * @param[in] object - the parameter to translate into hlhdf nodes
 * @param[in] nodelist - the nodelist the nodes should be added to
 * @returns 1 on success otherwise 0
 */
static int RaveIOInternal_addVolumeToNodeList(PolarVolume_t* object, HL_NodeList* nodelist)
{
  int result = 0;
  RaveObjectList_t* attributes = NULL;
  int nscans = 0;
  int i = 0;

  RAVE_ASSERT((object != NULL), "object == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  // First verify that no bogus data is entered into the system.
  if (!RaveIOInternal_validateVolume(object)) {
    goto done;
  }

  attributes = PolarVolume_getAttributeValues(object);
  if (attributes != NULL) {
    if (!RaveUtilities_addStringAttributeToList(attributes, "what/object", RaveIO_ObjectType_PVOL_STR) ||
        !RaveUtilities_addStringAttributeToList(attributes, "what/version", RaveIO_ODIM_H5rad_Version_2_0_STR)) {
      RAVE_ERROR0("Failed to add what/object or what/version to attributes");
      goto done;
    }
  }

  if (attributes == NULL || !RaveIOInternal_addAttributes(nodelist, attributes, "")) {
    goto done;
  }

  result = 1; // Set result to 1 now and if adding of scans fails, it will become 0 again

  nscans = PolarVolume_getNumberOfScans(object);
  for (i = 0; result == 1 && i < nscans; i++) {
    PolarScan_t* scan = PolarVolume_getScan(object, i);
    result = RaveIOInternal_addScanToNodeList(scan, nodelist, "/dataset%d", (i+1));
    RAVE_OBJECT_RELEASE(scan);
  }

done:
  RAVE_OBJECT_RELEASE(attributes);
  return result;
}

/**
 * Validates that the cartesian not contains any bogus data before
 * atempting to store it.
 * @param[in] cartesian - the cartesian to validate
 * @returns 1 if all is valid, otherwise 0.
 */
static int RaveIOInternal_validateCartesian(Cartesian_t* cartesian)
{
  int result = 0;
  Projection_t* projection = NULL;

  if (Cartesian_getObjectType(cartesian) == Rave_ObjectType_UNDEFINED) {
    RAVE_INFO0("Storing a cartesian with UNDEFINED ObjectType?");
    goto done;
  }
  if (Cartesian_getDate(cartesian) == NULL) {
    RAVE_INFO0("date == NULL");
    goto done;
  }
  if (Cartesian_getTime(cartesian) == NULL) {
    RAVE_INFO0("time == NULL");
    goto done;
  }
  if (Cartesian_getSource(cartesian) == NULL) {
    RAVE_INFO0("source == NULL");
    goto done;
  }

  projection = Cartesian_getProjection(cartesian);
  if (projection == NULL) {
    RAVE_INFO0("no projection for cartesian product");
    goto done;
  }
  if (Projection_getDefinition(projection) == NULL) {
    RAVE_INFO0("projection does not have a definition?");
    goto done;
  }
  if (Cartesian_getXSize(cartesian) <= 0) {
    RAVE_INFO0("xsize <= 0");
    goto done;
  }
  if (Cartesian_getYSize(cartesian) <= 0) {
    RAVE_INFO0("ysize <= 0");
    goto done;
  }
  if (Cartesian_getXScale(cartesian) <= 0.0) {
    RAVE_INFO0("xscale <= 0");
    goto done;
  }
  if (Cartesian_getYScale(cartesian) <= 0) {
    RAVE_INFO0("yscale <= 0");
    goto done;
  }
  if (Cartesian_getProduct(cartesian) == Rave_ProductType_UNDEFINED) {
    RAVE_INFO0("Undefined ProductType ?");
    goto done;
  }
  if (Cartesian_getQuantity(cartesian) == NULL) {
    RAVE_INFO0("quantity == NULL");
    goto done;
  }
  if (Cartesian_getData(cartesian) == NULL) {
    RAVE_INFO0("data == NULL");
    goto done;
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(projection);
  return result;
}

static int RaveIOInternal_createCartesianExtent(HL_NodeList* nodelist, Cartesian_t* cartesian, Projection_t* projection)
{
  int result = 0;
  double llX = 0.0L, llY = 0.0L, urX = 0.0L, urY = 0.0;
  double LL_lon = 0.0L, LL_lat = 0.0L, UL_lon = 0.0L, UL_lat = 0.0L;
  double UR_lon = 0.0L, UR_lat = 0.0L, LR_lon = 0.0L, LR_lat = 0.0L;
  RAVE_ASSERT((nodelist != NULL),"nodelist == NULL");
  RAVE_ASSERT((projection != NULL), "projection == NULL");
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");

  Cartesian_getAreaExtent(cartesian, &llX, &llY, &urX, &urY);

  if (!Projection_inv(projection, llX, llY, &LL_lon, &LL_lat)) {
    RAVE_ERROR0("Failed to generate LL-coordinate pair");
    goto done;
  }

  if (!Projection_inv(projection, llX, urY, &UL_lon, &UL_lat)) {
    RAVE_ERROR0("Failed to generate UL-coordinate pair");
    goto done;
  }

  if (!Projection_inv(projection, urX, urY, &UR_lon, &UR_lat)) {
    RAVE_ERROR0("Failed to generate UR-coordinate pair");
    goto done;
  }

  if (!Projection_inv(projection, urX, llY, &LR_lon, &LR_lat)) {
    RAVE_ERROR0("Failed to generate LR-coordinate pair");
    goto done;
  }

  if (!RaveIOInternal_createDoubleValue(nodelist, LL_lon * 180.0 / M_PI, "/where/LL_lon")) {
    goto done;
  }
  if (!RaveIOInternal_createDoubleValue(nodelist, LL_lat * 180.0 / M_PI, "/where/LL_lat")) {
    goto done;
  }
  if (!RaveIOInternal_createDoubleValue(nodelist, UL_lon * 180.0 / M_PI, "/where/UL_lon")) {
    goto done;
  }
  if (!RaveIOInternal_createDoubleValue(nodelist, UL_lat * 180.0 / M_PI, "/where/UL_lat")) {
    goto done;
  }
  if (!RaveIOInternal_createDoubleValue(nodelist, UR_lon * 180.0 / M_PI, "/where/UR_lon")) {
    goto done;
  }
  if (!RaveIOInternal_createDoubleValue(nodelist, UR_lat * 180.0 / M_PI, "/where/UR_lat")) {
    goto done;
  }
  if (!RaveIOInternal_createDoubleValue(nodelist, LR_lon * 180.0 / M_PI, "/where/LR_lon")) {
    goto done;
  }
  if (!RaveIOInternal_createDoubleValue(nodelist, LR_lat * 180.0 / M_PI, "/where/LR_lat")) {
    goto done;
  }

  result = 1;
done:
  return result;
}

static int RaveIOInternal_loadCartesianExtent(HL_NodeList* nodelist, Cartesian_t* cartesian, Projection_t* projection)
{
  int result = 0;
  double llX = 0.0L, llY = 0.0L, urX = 0.0L, urY = 0.0;
  double LL_lon = 0.0L, LL_lat = 0.0L, UR_lon = 0.0L, UR_lat = 0.0L;
  RAVE_ASSERT((nodelist != NULL),"nodelist == NULL");
  RAVE_ASSERT((projection != NULL), "projection == NULL");
  RAVE_ASSERT((cartesian != NULL), "cartesian == NULL");

  if (!RaveIOInternal_getDoubleValue(nodelist, &LL_lon, "/where/LL_lon") ||
      !RaveIOInternal_getDoubleValue(nodelist, &LL_lat, "/where/LL_lat") ||
      !RaveIOInternal_getDoubleValue(nodelist, &UR_lon, "/where/UR_lon") ||
      !RaveIOInternal_getDoubleValue(nodelist, &UR_lat, "/where/UR_lat")) {
    RAVE_ERROR0("Could not get cartesian extent coordinates LL & UR");
    goto done;
  }

  if (!Projection_fwd(projection, LL_lon * M_PI/180.0, LL_lat * M_PI/180.0, &llX, &llY)) {
    RAVE_ERROR0("Could not generate XY pair for LL");
    goto done;
  }

  if (!Projection_fwd(projection, UR_lon * M_PI/180.0, UR_lat * M_PI/180.0, &urX, &urY)) {
    RAVE_ERROR0("Could not generate XY pair for UR");
    goto done;
  }

  Cartesian_setAreaExtent(cartesian, llX, llY, urX, urY);

  result = 1;
done:
  return result;
}

/**
 * Loads and returns a cartesian object.
 * @param[in] nodelist - the hlhdf nodelist
 * @returns a cartesian object or NULL on failure
 */
static RaveCoreObject* RaveIOInternal_loadCartesian(HL_NodeList* nodelist)
{
  Cartesian_t* result = NULL;
  Projection_t* projection = NULL;
  char* date = NULL;
  char* time = NULL;
  char* src = NULL;
  char* projdef = NULL;
  long xsize = 0, ysize = 0;
  double xscale = 0.0L, yscale = 0.0L;
  char* quantity = NULL;
  char* startdate = NULL;
  char* starttime = NULL;
  char* enddate = NULL;
  char* endtime = NULL;
  double gain = 0.0L;
  double offset = 0.0L;
  double nodata = 0.0L;
  double undetect = 0.0L;
  HL_Node* node = NULL;
  RaveDataType dataType = RaveDataType_UNDEFINED;
  Rave_ObjectType objectType = Rave_ObjectType_UNDEFINED;
  Rave_ProductType productType = Rave_ProductType_UNDEFINED;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  objectType = RaveIOInternal_getObjectType(nodelist);
  if (objectType != Rave_ObjectType_CVOL && objectType != Rave_ObjectType_IMAGE && objectType != Rave_ObjectType_COMP) {
    RAVE_ERROR0("Can not load provided file as a cartesian product");
    return NULL;
  }
  // What information
  if (!RaveIOInternal_getStringValue(nodelist, &date, "/what/date") ||
      !RaveIOInternal_getStringValue(nodelist, &time, "/what/time") ||
      !RaveIOInternal_getStringValue(nodelist, &src, "/what/source")) {
    RAVE_ERROR0("Could not read /what information");
    return NULL;
  }

  if (!RaveIOInternal_getStringValue(nodelist, &projdef, "/where/projdef") ||
      !RaveIOInternal_getLongValue(nodelist, &xsize, "/where/xsize") ||
      !RaveIOInternal_getLongValue(nodelist, &ysize, "/where/ysize") ||
      !RaveIOInternal_getDoubleValue(nodelist, &xscale, "/where/xscale") ||
      !RaveIOInternal_getDoubleValue(nodelist, &yscale, "/where/yscale")) {
    RAVE_ERROR0("Could not read /where information");
    return NULL;
  }

  projection = RAVE_OBJECT_NEW(&Projection_TYPE);
  if (projection == NULL) {
    RAVE_ERROR0("Could not create projection");
    goto error;
  }
  if (!Projection_init(projection, "raveio-projection", "autoloaded projection", projdef)) {
    RAVE_ERROR0("Could not initialize projection");
    goto error;
  }

  result = RAVE_OBJECT_NEW(&Cartesian_TYPE);
  if (result == NULL) {
    RAVE_CRITICAL0("Could not create cartesian object");
    goto error;
  }

  if (!RaveIOInternal_loadCartesianExtent(nodelist, result, projection)) {
    RAVE_ERROR0("Failed to load cartesian extent");
    goto error;
  }

  if (!Cartesian_setDate(result, date)) {
    RAVE_ERROR0("Illegal date string");
    goto error;
  }
  if (!Cartesian_setTime(result, time)) {
    RAVE_ERROR0("Illegal time string");
    goto error;
  }
  if (!Cartesian_setSource(result, src)) {
    RAVE_ERROR0("Illegal src string");
    goto error;
  }

  if (!Cartesian_setObjectType(result, objectType)) {
    RAVE_ERROR0("Could not set object type");
    goto error;
  }

  Cartesian_setXScale(result, xscale);
  Cartesian_setYScale(result, yscale);
  Cartesian_setProjection(result, projection);

  productType = RaveIOInternal_getProductType(nodelist, "/dataset1/what/product");
  if (!Cartesian_setProduct(result, productType)) {
    RAVE_ERROR0("Could not set product type");
    goto error;
  }

  if (!RaveIOInternal_getStringValue(nodelist, &quantity, "/dataset1/what/quantity") ||
      !RaveIOInternal_getStringValue(nodelist, &startdate, "/dataset1/what/startdate") ||
      !RaveIOInternal_getStringValue(nodelist, &starttime, "/dataset1/what/starttime") ||
      !RaveIOInternal_getStringValue(nodelist, &enddate, "/dataset1/what/enddate") ||
      !RaveIOInternal_getStringValue(nodelist, &endtime, "/dataset1/what/endtime") ||
      !RaveIOInternal_getDoubleValue(nodelist, &gain, "/dataset1/what/gain") ||
      !RaveIOInternal_getDoubleValue(nodelist, &offset, "/dataset1/what/offset") ||
      !RaveIOInternal_getDoubleValue(nodelist, &nodata, "/dataset1/what/nodata") ||
      !RaveIOInternal_getDoubleValue(nodelist, &undetect, "/dataset1/what/undetect")) {
    RAVE_ERROR0("Could not read /dataset1/what information");
    goto error;
  }
  if (!Cartesian_setQuantity(result, quantity)) {
    RAVE_ERROR0("Could not set quantity");
    goto error;
  }
  // Do not bother about setting start and end dates/times
  Cartesian_setGain(result, gain);
  Cartesian_setOffset(result, offset);
  Cartesian_setNodata(result, nodata);
  Cartesian_setUndetect(result, undetect);

  // And finally read the data.
  node = RaveIOInternal_getNode(nodelist, "/dataset1/data1/data");
  if (node == NULL) {
    RAVE_ERROR0("Failed to load data");
    goto error;
  }
  if (HLNode_getRank(node) != 2) {
    RAVE_ERROR0("Data is not 2-dimensional");
    goto error;
  }

  if (HLNode_getDimension(node, 1) != xsize || HLNode_getDimension(node, 0) != ysize) {
    RAVE_ERROR0("xsize/ysize does not correspond to actual data array dimensions");
    goto error;
  }

  dataType = RaveIOInternal_hlhdfToRaveType(HLNode_getFormat(node));
  if (dataType == RaveDataType_UNDEFINED) {
    RAVE_ERROR0("Bad data type");
    goto error;
  }

  if (!Cartesian_setData(result, xsize, ysize, HLNode_getData(node), dataType)) {
    RAVE_ERROR0("Could not set data");
    goto error;
  }

  RAVE_OBJECT_RELEASE(projection);
  return (RaveCoreObject*)result;
error:
  RAVE_OBJECT_RELEASE(projection);
  RAVE_OBJECT_RELEASE(result);
  return NULL;
}

/**
 * Saves a cartesian object as specified according in ODIM HDF5 format specification.
 * @param[in] raveio - self
 * @returns 1 on success, otherwise 0
 */
int RaveIOInternal_saveCartesian(RaveIO_t* raveio)
{
  int result = 0;
  HL_NodeList* nodelist = NULL;
  Projection_t* projection = NULL;
  Rave_ProductType productType = Rave_ProductType_UNDEFINED;
  Cartesian_t* object = NULL; //  DO not release this
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");

  if (raveio->object == NULL || !RAVE_OBJECT_CHECK_TYPE(raveio->object, &Cartesian_TYPE)) {
    RAVE_ERROR0("Atempting to save an object that not is cartesian");
    return 0;
  }

  object = (Cartesian_t*)raveio->object; // So that I dont have to cast the object all the time. DO not release this
  if (!RaveIOInternal_validateCartesian(object)) {
    goto done;
  }

  productType = Cartesian_getProduct(object);

  nodelist = HLNodeList_new();
  if (nodelist == NULL) {
    RAVE_CRITICAL0("Failed to allocate nodelist");
    goto done;
  }

  if (!RaveIOInternal_createStringValue(nodelist, RaveIO_ODIM_Version_2_0_STR, "/Conventions")) {
    goto done;
  }

  // WHAT
  if (!RaveIOInternal_createGroup(nodelist, "/what")) {
    goto done;
  }
  if (!RaveIOInternal_createStringValue(nodelist,
       RaveIOInternal_getObjectTypeString(Cartesian_getObjectType(object)), "/what/object")) {
    goto done;
  }

  if (!RaveIOInternal_createStringValue(nodelist, RaveIO_ODIM_H5rad_Version_2_0_STR, "/what/version")) {
    goto done;
  }
  if (!RaveIOInternal_createStringValue(nodelist, Cartesian_getDate(object), "/what/date")) {
    goto done;
  }
  if (!RaveIOInternal_createStringValue(nodelist, Cartesian_getTime(object), "/what/time")) {
    goto done;
  }
  if (!RaveIOInternal_createStringValue(nodelist, Cartesian_getSource(object), "/what/source")) {
    goto done;
  }

  // WHERE
  if (!RaveIOInternal_createGroup(nodelist, "/where")) {
    goto done;
  }
  projection = Cartesian_getProjection(object);
  if (projection == NULL) {
    RAVE_CRITICAL0("Cartesian product does not have a projection");
    goto done;
  }

  if (!RaveIOInternal_createStringValue(nodelist, Projection_getDefinition(projection), "/where/projdef")) {
    goto done;
  }
  if (!RaveIOInternal_createLongValue(nodelist, Cartesian_getXSize(object), "/where/xsize")) {
    goto done;
  }

  if (!RaveIOInternal_createLongValue(nodelist, Cartesian_getYSize(object), "/where/ysize")) {
    goto done;
  }
  if (!RaveIOInternal_createDoubleValue(nodelist, Cartesian_getXScale(object), "/where/xscale")) {
    goto done;
  }
  if (!RaveIOInternal_createDoubleValue(nodelist, Cartesian_getYScale(object), "/where/yscale")) {
    goto done;
  }
  if (!RaveIOInternal_createCartesianExtent(nodelist, object, projection)) {
    goto done;
  }

  // dataset
  if (!RaveIOInternal_createGroup(nodelist, "/dataset1")) {
    goto done;
  }
  if (!RaveIOInternal_createGroup(nodelist, "/dataset1/what")) {
    goto done;
  }
  if (!RaveIOInternal_createStringValue(nodelist,
        RaveIOInternal_getProductString(productType), "/dataset1/what/product")) {
    goto done;
  }

  if (productType == Rave_ProductType_CAPPI || productType == Rave_ProductType_PPI ||
      productType == Rave_ProductType_ETOP || productType == Rave_ProductType_RHI ||
      productType == Rave_ProductType_VIL) {
    // @todo: FIX prodpar
  }

  if (!RaveIOInternal_createStringValue(nodelist, Cartesian_getQuantity(object), "/dataset1/what/quantity")) {
    goto done;
  }
  if (!RaveIOInternal_createStringValue(nodelist, Cartesian_getDate(object), "/dataset1/what/startdate")) {
    goto done;
  }
  if (!RaveIOInternal_createStringValue(nodelist, Cartesian_getTime(object), "/dataset1/what/starttime")) {
    goto done;
  }
  if (!RaveIOInternal_createStringValue(nodelist, Cartesian_getDate(object), "/dataset1/what/enddate")) {
    goto done;
  }
  if (!RaveIOInternal_createStringValue(nodelist, Cartesian_getTime(object), "/dataset1/what/endtime")) {
    goto done;
  }
  if (!RaveIOInternal_createDoubleValue(nodelist, Cartesian_getGain(object), "/dataset1/what/gain")) {
    goto done;
  }
  if (!RaveIOInternal_createDoubleValue(nodelist, Cartesian_getOffset(object), "/dataset1/what/offset")) {
    goto done;
  }
  if (!RaveIOInternal_createDoubleValue(nodelist, Cartesian_getNodata(object), "/dataset1/what/nodata")) {
    goto done;
  }
  if (!RaveIOInternal_createDoubleValue(nodelist, Cartesian_getUndetect(object), "/dataset1/what/undetect")) {
    goto done;
  }
  if (!RaveIOInternal_createGroup(nodelist, "/dataset1/data1")) {
    goto done;
  }

  if (!RaveIOInternal_createDataset(nodelist, Cartesian_getData(object),
                                    Cartesian_getXSize(object), Cartesian_getYSize(object),
                                    Cartesian_getDataType(object),
                                    "/dataset1/data1/data")) {
    goto done;
  }

  // If data type is 8-bit UCHAR, IMAGE attributes shall be stored.
  if (Cartesian_getDataType(object) == RaveDataType_UCHAR) {
    if (!RaveIOInternal_createStringValue(nodelist, "IMAGE", "/dataset1/data1/data/CLASS")) {
      goto done;
    }
    if (!RaveIOInternal_createStringValue(nodelist, "1.2", "/dataset1/data1/data/IMAGE_VERSION")) {
      goto done;
    }
  }

  if (!HLNodeList_setFileName(nodelist, raveio->filename)) {
    RAVE_CRITICAL0("Could not set filename on nodelist");
    goto done;
  }

  if (!HLNodeList_write(nodelist, NULL, NULL)) {
    RAVE_CRITICAL0("Could not save file");
    goto done;
  }

  result = 1;
done:
  HLNodeList_free(nodelist);
  RAVE_OBJECT_RELEASE(projection);
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
  if (objectType == Rave_ObjectType_CVOL || objectType == Rave_ObjectType_IMAGE || objectType == Rave_ObjectType_COMP) {
    object = RaveIOInternal_loadCartesian(nodelist);
  } else if (objectType == Rave_ObjectType_PVOL) {
    object = (RaveCoreObject*)RaveIOInternal_loadVolume(nodelist);
  }
#ifdef KALLE
  else if (objectType == Rave_ObjectType_SCAN) {
    object = (RaveCoreObject*)RaveIOInternal_loadScan(nodelist);
  }
#endif
  else {
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

int RaveIO_save(RaveIO_t* raveio)
{
  int result = 0;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  if (raveio->filename == NULL) {
    RAVE_ERROR0("Atempting to save an object without a filename");
    return 0;
  }

  if (raveio->object != NULL) {
    if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &Cartesian_TYPE)) {
      result = RaveIOInternal_saveCartesian(raveio);
    } else if (RAVE_OBJECT_CHECK_TYPE(raveio->object, &PolarVolume_TYPE)) {
      HL_NodeList* nodelist = HLNodeList_new();

      if (nodelist != NULL) {
        result = RaveIOInternal_createStringValue(nodelist, RaveIO_ODIM_Version_2_0_STR, "/Conventions");
        if (result == 1) {
          result = RaveIOInternal_addVolumeToNodeList((PolarVolume_t*)raveio->object, nodelist);
        }
        if (result == 1) {
          result = HLNodeList_setFileName(nodelist, raveio->filename);
        }
        if (result == 1) {
          result = HLNodeList_write(nodelist, NULL, NULL);
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

/*@} End of Interface functions */

RaveCoreObjectType RaveIO_TYPE = {
    "RaveIO",
    sizeof(RaveIO_t),
    RaveIO_constructor,
    RaveIO_destructor
};
