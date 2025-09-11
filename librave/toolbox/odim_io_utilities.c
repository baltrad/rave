/* --------------------------------------------------------------------
Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Utilities when working with ODIM H5 files.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2011-09-30
 */
#include "odim_io_utilities.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_hlhdf_utilities.h"
#include "hlhdf_compound_utils.h"
#include "rave_legend.h"
#include <string.h>
#include <math.h>

/*@{ Private functions */

#define SPEED_OF_LIGHT 299792458  /* m/s */

#define DEFAULT_MINKEYLEN 64
#define DEFAULT_MINVALUELEN 32

/**
 * Called when an attribute belonging to a rave field
 * is found.
 * @param[in] object - the PolarOdimArg pointing to a rave field
 * @param[in] attribute - the attribute found
 * @return 1 on success otherwise 0
 */
static int OdimIoUtilitiesInternal_loadFieldAttribute(void* object, RaveAttribute_t* attribute)
{
  RaveField_t* field = NULL;
  const char* name;
  int result = 0;
  RaveIO_ODIM_Version version;

  RAVE_ASSERT((object != NULL), "object == NULL");
  RAVE_ASSERT((attribute != NULL), "attribute == NULL");

  field = (RaveField_t*)((OdimIoUtilityArg*)object)->object;
  name = RaveAttribute_getName(attribute);
  version = (RaveIO_ODIM_Version)((OdimIoUtilityArg*)object)->version;
  if (name != NULL) {
    result = RaveField_addAttributeVersion(field, attribute, version);
  }

  return result;
}

/**
 * Called when an dataset belonging to a field parameter
 * is found.
 * @param[in] object - the PolarOdimArg pointing to a rave field
 * @param[in] xsize - the x size
 * @param[in] ysize - the y size
 * @param[in] data  - the data
 * @param[in] dtype - the type of the data.
 * @return 1 on success otherwise 0
 */
static int OdimIoUtilitiesInternal_loadFieldDataset(void* object, hsize_t xsize, hsize_t ysize, void* data, RaveDataType dtype, const char* nodeName)
{
  RaveField_t* field = NULL;
  int result = 0;

  field = (RaveField_t*)((OdimIoUtilityArg*)object)->object;
  if (data == NULL && ((OdimIoUtilityArg*)object)->lazyReader != NULL) {
    LazyDataset_t* datasetReader = RAVE_OBJECT_NEW(&LazyDataset_TYPE);
    if (datasetReader != NULL) {
      result = LazyDataset_init(datasetReader, ((OdimIoUtilityArg*)object)->lazyReader, nodeName);
    }
    if (result) {
      result = RaveField_setLazyDataset(field, datasetReader);
    }
    RAVE_OBJECT_RELEASE(datasetReader);
  } else {
    result = RaveField_setData(field, xsize, ysize, data, dtype);
  }
  return result;
}

/*@} End of Private functions */

/*@{ Interface functions */
int OdimIoUtilities_convertGainOffsetFromInternalRave(const char* quantity, RaveIO_ODIM_Version version, double* gain, double* offset)
{
  if (quantity == NULL || gain == NULL || offset == NULL) {
    return 0;
  }

  if (version < RaveIO_ODIM_Version_2_4) {
    return 1;
  }

  if (strcasecmp("HGHT", quantity) == 0) {
    *gain = (*gain)*1000.0;
    *offset = (*offset)*1000.0;
  } else if (strcasecmp("MESH", quantity) == 0) {
    *gain = (*gain)*10.0;
    *offset = (*offset)*10.0;
  }

  return 1;
}

int OdimIoUtilities_convertGainOffsetToInternalRave(const char* quantity, RaveIO_ODIM_Version version, double* gain, double* offset)
{
  if (quantity == NULL || gain == NULL || offset == NULL) {
    return 0;
  }

  if (version < RaveIO_ODIM_Version_2_4) {
    return 1;
  }

  if (strcasecmp("HGHT", quantity) == 0) {
    *gain = (*gain)/1000.0;
    *offset = (*offset)/1000.0;
  } else if (strcasecmp("MESH", quantity) == 0) {
    *gain = (*gain)/10.0;
    *offset = (*offset)/10.0;
  }

  return 1;
}

int OdimIoUtilities_addRaveField(RaveField_t* field, HL_NodeList* nodelist, RaveIO_ODIM_Version version, const char* fmt, ...)
{
  int result = 0;
  va_list ap;
  char name[1024];
  int nName = 0;
  RaveObjectList_t* attributes = NULL;

  RAVE_ASSERT((field != NULL), "field == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((fmt != NULL), "fmt == NULL");

  va_start(ap, fmt);
  nName = vsnprintf(name, 1024, fmt, ap);
  va_end(ap);
  if (nName < 0 || nName >= 1024) {
    RAVE_ERROR0("NodeName would evaluate to more than 1024 characters.");
    goto done;
  }
  if (!RaveHL_hasNodeByName(nodelist, name)) {
    if (!RaveHL_createGroup(nodelist, name)) {
      goto done;
    }
  }

  if (version == RaveIO_ODIM_Version_UNDEFINED) {
    attributes = RaveField_getAttributeValues(field);
  } else {
    attributes = RaveField_getAttributeValuesVersion(field, version);
  }

  if (attributes == NULL || !RaveHL_addAttributes(nodelist, attributes, name)) {
    goto done;
  }

  if (!RaveHL_addData(nodelist,
                      RaveField_getData(field),
                      RaveField_getXsize(field),
                      RaveField_getYsize(field),
                      RaveField_getDataType(field),
                      name)) {
    goto done;
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(attributes);
  return result;
}

/**
 * Adds a list of quality fields (RaveField_t) to a nodelist.
 *
 * @param[in] fields - the list of fields
 * @param[in] nodelist - the hlhdf node list
 * @param[in] fmt - the varargs format string
 * @param[in] ... - the varargs
 * @return 1 on success otherwise 0
 */
int OdimIoUtilities_addQualityFields(RaveObjectList_t* fields, HL_NodeList* nodelist, RaveIO_ODIM_Version version, const char* fmt, ...)
{
  int result = 0;
  va_list ap;
  char name[1024];
  int nName = 0;
  int pindex = 0;
  int nrfields = 0;

  RAVE_ASSERT((fields != NULL), "fields == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((fmt != NULL), "fmt == NULL");

  va_start(ap, fmt);
  nName = vsnprintf(name, 1024, fmt, ap);
  va_end(ap);
  if (nName < 0 || nName >= 1024) {
    RAVE_ERROR0("NodeName would evaluate to more than 1024 characters.");
    goto done;
  }

  result = 1;
  nrfields = RaveObjectList_size(fields);
  for (pindex = 0; result == 1 && pindex < nrfields; pindex++) {
    RaveField_t* field = (RaveField_t*)RaveObjectList_get(fields, pindex);
    if (field != NULL) {
      result = OdimIoUtilities_addRaveField(field, nodelist, version, "%s/quality%d", name, (pindex+1));
    } else {
      result = 0;
    }
    RAVE_OBJECT_RELEASE(field);
  }

done:
  return result;
}

/**
 * Loads a rave field. A rave field can be just about anything with a mapping
 * between attributes and a dataset.
 * @param[in] nodelist - the hlhdf node list
 * @param[in] nodelist - version of the file read
 * @param[in] fmt - the variable argument list string format
 * @param[in] ... - the variable argument list
 * @return a rave field on success otherwise NULL
 */
RaveField_t* OdimIoUtilities_loadField(LazyNodeListReader_t* lazyReader, RaveIO_ODIM_Version version, const char* fmt, ...)
{
  OdimIoUtilityArg arg;
  RaveField_t* field = NULL;
  RaveField_t* result = NULL;
  va_list ap;
  char name[1024];
  int nName = 0;

  RAVE_ASSERT((lazyReader != NULL), "lazyReader == NULL");
  RAVE_ASSERT((fmt != NULL), "fmt == NULL");

  va_start(ap, fmt);
  nName = vsnprintf(name, 1024, fmt, ap);
  va_end(ap);
  if (nName < 0 || nName >= 1024) {
    RAVE_ERROR0("NodeName would evaluate to more than 1024 characters.");
    goto fail;
  }

  field = RAVE_OBJECT_NEW(&RaveField_TYPE);
  if (field == NULL) {
    RAVE_CRITICAL0("Failed to allocate memory for field");
    goto fail;
  }
  arg.lazyReader = lazyReader;
  arg.nodelist = LazyNodeListReader_getHLNodeList(lazyReader);
  arg.object = (RaveCoreObject*)field;
  arg.version = version;

  if (!RaveHL_loadAttributesAndData(arg.nodelist, &arg,
                                    OdimIoUtilitiesInternal_loadFieldAttribute,
                                    OdimIoUtilitiesInternal_loadFieldDataset,
                                    name)) {
    goto fail;
  }

  result = RAVE_OBJECT_COPY(field);
fail:
  RAVE_OBJECT_RELEASE(field);
  return result;
}

RaveLegend_t* OdimIOUtilities_loadLegend(LazyNodeListReader_t* lazyReader, RaveIO_ODIM_Version version, const char* fmt, ...)
{
  RaveLegend_t* legend = NULL;
  RaveLegend_t* result = NULL;
  va_list ap;
  char name[1024];
  int nName = 0;
  RAVE_ASSERT((lazyReader != NULL), "lazyReader == NULL");
  RAVE_ASSERT((fmt != NULL), "fmt == NULL");

  va_start(ap, fmt);
  nName = vsnprintf(name, 1024, fmt, ap);
  va_end(ap);
  if (nName < 0 || nName >= 1024) {
    RAVE_ERROR0("NodeName would evaluate to more than 1024 characters.");
    goto fail;
  }
  
  legend = LazyNodeListReader_getLegend(lazyReader, name);
  if (legend == NULL) {
    goto fail;
  }

  result = RAVE_OBJECT_COPY(legend);
fail:
  RAVE_OBJECT_RELEASE(legend);
  return result;
}


int OdimIoUtilities_getIdFromSource(const char* source, const char* id, char* buf, size_t buflen)
{
  int result = 0;
  if (source != NULL && id != NULL) {
    char* p = strstr(source, id);
    if (p != NULL) {
      int len = 0;
      char* pbrk = NULL;
      p += strlen(id);
      len = strlen(p);
      pbrk = strpbrk((const char*)p, ",");

      if (pbrk != NULL) {
        len = pbrk - p;
      }
      if (len + 1 < buflen) {
        strncpy(buf, p, len);
        buf[len] = '\0';
        result = 1;
      }
    }
  }
  return result;
}

int OdimIoUtilities_getNodOrCmtFromSource(const char* source, char* buf, size_t buflen)
{
  int result = 0;
  result = OdimIoUtilities_getIdFromSource(source, "NOD:", buf, buflen);
  if (!result)
    result = OdimIoUtilities_getIdFromSource(source, "CMT:", buf, buflen);
  return result;
}


int OdimIoUtilities_createLegend(RaveLegend_t* legend, HL_NodeList* nodelist, RaveIO_ODIM_Version version, const char* fmt, ...)
{
  int result = 0;
  va_list ap;
  char name[1024];
  int nName = 0;
  int keylen = 0, valuelen = 0, entrylen = 0, nentries = 0, i = 0;
  hsize_t dims[1] = {0};
  hid_t type_id = -1;
  hid_t strkey_id = -1;
  hid_t strval_id = -1;
  HL_Node* node = NULL;

  char* datatowrite = NULL;
  char* legendentry = NULL;

  RAVE_ASSERT((legend != NULL), "legend == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((fmt != NULL), "fmt == NULL");

  if (RaveLegend_size(legend) <= 0) {
    RAVE_INFO0("Legend does not contain any entries, leaving without doing anything");
    return 1;
  }

  va_start(ap, fmt);
  nName = vsnprintf(name, 1024, fmt, ap);
  va_end(ap);
  if (nName < 0 || nName >= 1024) {
    RAVE_ERROR0("NodeName would evaluate to more than 1024 characters.");
    goto done;
  }

  keylen = RaveLegend_maxKeyLength(legend) + 1;     /* +1 to include newline */
  valuelen = RaveLegend_maxValueLength(legend) + 1;

  if (keylen < DEFAULT_MINKEYLEN) {
    keylen = DEFAULT_MINKEYLEN;
  }

  if (valuelen < DEFAULT_MINVALUELEN) {
    valuelen = DEFAULT_MINVALUELEN;
  }

  entrylen = keylen + valuelen;
  nentries = RaveLegend_size(legend);
  dims[0] = nentries;
  
  strkey_id =   H5Tcopy(H5T_C_S1);
  strval_id =   H5Tcopy(H5T_C_S1);
  if (strkey_id < 0 || strval_id < 0) {
    RAVE_ERROR0("Failed to copy string types to be added to compound");
    goto done;
  }
  H5Tset_size(strkey_id, keylen);
  H5Tset_size(strval_id, valuelen);

  type_id = createCompoundType(sizeof(char)*(entrylen));
  if (type_id < 0) {
    RAVE_ERROR0("Failed to create compound type definition");
    goto done;
  }

  if(addAttributeToCompoundType(type_id, "key", 0, strkey_id) < 0) {
    RAVE_ERROR0("Could not add key to compound type legend");
    goto done;
  }

  if(addAttributeToCompoundType(type_id,"value", keylen, strval_id) < 0) {
    RAVE_ERROR0("Could not add value to compound type legend");
    goto done;
  }

  legendentry = RAVE_MALLOC(sizeof(char) * entrylen);
  datatowrite = RAVE_MALLOC(sizeof(char) * entrylen * nentries);

  for (i = 0; i < nentries; i++) {
    memset(legendentry, 0, sizeof(char) * entrylen);
    snprintf(legendentry, entrylen, "%s", RaveLegend_getNameAt(legend, i));
    snprintf(legendentry+keylen, entrylen-keylen, "%s", RaveLegend_getValueAt(legend, i));
    memcpy(datatowrite+(i*entrylen), legendentry, sizeof(char)*entrylen);
  }

  node = HLNode_newDataset(name);
  if (node == NULL) {
    RAVE_ERROR0("Failed to allocate memory for legend");
    goto done;
  }

  if (!HLNode_setArrayValue(node,(size_t)sizeof(char)*entrylen, 1, dims, (unsigned char*)datatowrite, "compound", type_id)) {
    HLNode_free(node);
    goto done;
  }

  if (!HLNodeList_addNode(nodelist, node)) {
    RAVE_CRITICAL1("Failed to add dataset node with name %s", name);
    HLNode_free(node);
    goto done;
  }

  result = 1;
done:
  if (type_id >= 0) {
    H5Tclose(type_id);
  }
  if (strkey_id >= 0) {
    H5Tclose(strkey_id);
  }
  if (strval_id >= 0) {
    H5Tclose(strval_id);
  }
  RAVE_FREE(legendentry);
  RAVE_FREE(datatowrite);
  return result;
}

int OdimIoUtilities_createSubGroups(const char* attrname, HL_NodeList* nodelist)
{
  RaveList_t* subgroups = NULL;
  int result = 0;

  if (attrname != NULL && nodelist != NULL) {
    subgroups = RaveHL_extractSubGroups(attrname);
    if (subgroups != NULL) {
      int nsubs  = RaveList_size(subgroups);
      int i = 0;

      for (i = 0; i < nsubs; i++) {
        const char* sname = (const char*)RaveList_get(subgroups, i);
        if (sname != NULL && strcmp(sname, "") != 0) {
          char nodename[2048];
          snprintf(nodename, 2048, "/%s", sname);
          if (!HLNodeList_hasNodeByName(nodelist, nodename)) {
            if (!RaveHL_createGroupUnlessExists(nodelist, nodename)) {
              RAVE_ERROR1("Failed to create group: %s", nodename);
              goto fail;
            }
          }
        }
      }
    }
  }

  result = 1;
fail:
  if (subgroups != NULL) {
    RaveList_freeAndDestroy(&subgroups);
  }
  return result;
}

int OdimIoUtilities_addValuesToFile(RaveValue_t* hashtable, HL_NodeList* nodelist)
{
  RaveList_t* keys = NULL;
  int result = 0;

  int nkeys = 0, i = 0;
  if (hashtable == NULL || nodelist == NULL) {
    RAVE_WARNING0("Must provide both rave value hashtable and nodelist");
    return 0;
  }
  if (RaveValue_type(hashtable) != RaveValue_Type_Hashtable) {
    RAVE_WARNING0("Rave value must be of hashtable type");
    return 0;
  }

  keys = RaveValueHash_keys(hashtable);
  if (keys != NULL) {
    nkeys = RaveList_size(keys);
    for (i = 0; i < nkeys; i++) {
      const char* name = (const char*)RaveList_get(keys, i);
      if (name != NULL) {
        RaveValue_t* value = RaveValueHash_get(hashtable, name);
        if (value != NULL) {
          RaveValue_Type valuetype = RaveValue_type(value);
          if (valuetype == RaveValue_Type_String ||
              valuetype == RaveValue_Type_Double ||
              valuetype == RaveValue_Type_Long) {
            if (!OdimIoUtilities_createSubGroups(name, nodelist)) {
              RAVE_ERROR1("Failed to create subgroups for %s", name);
              RAVE_OBJECT_RELEASE(value);
              goto fail;
            }
            if (!RaveHL_addRaveValue(nodelist, value, name)) {
              RAVE_ERROR1("Failed to create attribute %s", name);
              RAVE_OBJECT_RELEASE(value);
              goto fail;
            }
          } else {
            RAVE_WARNING1("Unsupported RaveValueType when adding values to file: %d", valuetype);
          }
          RAVE_OBJECT_RELEASE(value);
        }
      }
    }
  }

  result = 1;
fail:
  if (keys != NULL) {
    RaveList_freeAndDestroy(&keys);
  }
  return result;
}
