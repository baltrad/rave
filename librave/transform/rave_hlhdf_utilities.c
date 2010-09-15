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
 * Utilities for working with H5 files
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-09-10
 */
#include "rave_hlhdf_utilities.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_utilities.h"
#include "string.h"
#include "stdarg.h"

/*@{ Constants */

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

/*@} End of Constants */

/*@{ Interface functions */

int RaveHL_hasNodeByName(HL_NodeList* nodelist, const char* fmt, ...)
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

int RaveHL_createGroup(HL_NodeList* nodelist, const char* fmt, ...)
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

int RaveHL_createStringValue(HL_NodeList* nodelist, const char* value, const char* fmt, ...)
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
 * Puts an attribute in the nodelist as a hlhdf node.
 * @param[in] nodelist - the node list
 * @param[in] attribute - the attribute, the name of the attribute will be used as attr-member
 * @param[in] fmt - the root name, specified as a varargs
 * @param[in] ... - the varargs list
 * @returns 1 on success otherwise 0
 */
int RaveHL_addAttribute(HL_NodeList* nodelist, RaveAttribute_t* attribute, const char* fmt, ...)
{
  const char* attrname = NULL;
  int result = 0;
  char nodeName[1024];
  va_list ap;
  int n = 0;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((attribute != NULL), "attribute == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n < 0 || n >= 1024) {
    RAVE_ERROR0("Failed to generate name for data entry");
    goto done;
  }

  attrname = RaveAttribute_getName(attribute);
  if (attrname != NULL) {
    char attrNodeName[1024];
    sprintf(attrNodeName, "%s/%s", nodeName, attrname);
    if (!HLNodeList_hasNodeByName(nodelist, attrNodeName)) {
      HL_Node* node = HLNode_newAttribute(attrNodeName);
      if (node == NULL) {
        RAVE_CRITICAL1("Failed to create an attribute with name %s", attrNodeName);
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
          RAVE_ERROR1("Could not add node %s", attrNodeName);
        }
      }
    } else {
      /* If attribute already has been added, we just count this as successful */
      result = 1;
    }
  }

done:
  return result;
}


int RaveHL_addAttributes(HL_NodeList* nodelist, RaveObjectList_t* attributes, const char* name)
{
  RaveAttribute_t* attribute = NULL;
  int result = 0;
  int nattrs = 0;
  int i = 0;
  int hashow = 0, haswhat = 0, haswhere = 0;
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((attributes != NULL), "attributes == NULL");

  hashow = RaveHL_hasNodeByName(nodelist, "%s/how", name);
  haswhat = RaveHL_hasNodeByName(nodelist, "%s/what", name);
  haswhere = RaveHL_hasNodeByName(nodelist, "%s/where", name);

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
      haswhat = RaveHL_createGroup(nodelist, "%s/what",name);
      if (haswhat == 0) {
        RAVE_ERROR1("Failed to create group %s/what", name);
        goto done;
      }
    } else if (haswhere==0 && strncasecmp(attrname, "where/", 6)==0) {
      haswhere = RaveHL_createGroup(nodelist, "%s/where",name);
      if (haswhere == 0) {
        RAVE_ERROR1("Failed to create group %s/where", name);
        goto done;
      }
    } else if (hashow==0 && strncasecmp(attrname, "how/", 4)==0) {
      hashow = RaveHL_createGroup(nodelist, "%s/how",name);
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

    if (!RaveHL_addAttribute(nodelist, attribute, name)) {
      RAVE_ERROR2("Failed to add attribute %s/%s to nodelist", name, attrname);
      goto done;
    }
  }
  result = 1;
done:
  RAVE_OBJECT_RELEASE(attribute);
  return result;
}

int RaveHL_createDataset(HL_NodeList* nodelist, void* data, long xsize, long ysize, RaveDataType dataType, const char* fmt, ...)
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
    HL_FormatSpecifier specifier = RaveHL_raveToHlhdfType(dataType);
    const char* hlhdfFormat = HL_getFormatSpecifierString(specifier);
    hsize_t dims[2];
    dims[0] = ysize;
    dims[1] = xsize;
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

int RaveHL_addData(
  HL_NodeList* nodelist,
  void* data,
  long xsize,
  long ysize,
  RaveDataType dataType,
  const char* fmt,
  ...)
{
  int result = 0;
  char nodeName[1024];
  va_list ap;
  int n = 0;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n < 0 || n >= 1024) {
    RAVE_ERROR0("Failed to generate name for data entry");
    goto done;
  }

  if (data == NULL) {
    goto done;
  }

  if (!RaveHL_createDataset(nodelist, data, xsize, ysize, dataType, "%s/data", nodeName)) {
    RAVE_CRITICAL1("Failed to create dataset with name %s/data", nodeName);
    goto done;
  }

  result = 1; // Set result to 1 now, if hdfview specific fails, result will be set back to 0.

  if (dataType == RaveDataType_UCHAR) {
    RaveAttribute_t* imgAttribute = RaveAttributeHelp_createString("CLASS", "IMAGE");
    RaveAttribute_t* verAttribute = RaveAttributeHelp_createString("IMAGE_VERSION", "1.2");
    if (imgAttribute == NULL || verAttribute == NULL) {
      result = 0;
    }
    if (result == 1) {
      result = RaveHL_addAttribute(nodelist, imgAttribute, "%s/data", nodeName);
    }
    if (result == 1) {
      result = RaveHL_addAttribute(nodelist, verAttribute, "%s/data", nodeName);
    }
    RAVE_OBJECT_RELEASE(imgAttribute);
    RAVE_OBJECT_RELEASE(verAttribute);
  }

done:
  return result;
}

HL_FormatSpecifier RaveHL_raveToHlhdfType(RaveDataType format)
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

RaveDataType RaveHL_hlhdfToRaveType(HL_FormatSpecifier format)
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
/*@} End of Interface functions */
