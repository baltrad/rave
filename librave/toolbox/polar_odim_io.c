/* --------------------------------------------------------------------
Copyright (C) 2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Polar ODIM IO functions
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-11-12
 */
#include "polar_odim_io.h"
#include "rave_hlhdf_utilities.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>
#include "odim_io_utilities.h"

/**
 * The Polar ODIM IO adaptor
 */
struct _PolarOdimIO_t {
  RAVE_OBJECT_HEAD /** Always on top */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int PolarOdimIO_constructor(RaveCoreObject* obj)
{
  return 1;
}

/**
 * Copy constructor
 */
static int PolarOdimIO_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  return 1;
}

/**
 * Destroys the object
 * @param[in] obj - the instance
 */
static void PolarOdimIO_destructor(RaveCoreObject* obj)
{
}

/**
 * Scan root attributes.
 * @param[in] object - the OdimIoUtilityArg pointing to a polar scan
 * @param[in] attribute - the attribute found
 * @return 1 on success otherwise 0
 */
static int PolarOdimIOInternal_loadRootScanAttribute(void* object, RaveAttribute_t* attribute)
{
  PolarScan_t* scan = (PolarScan_t*)((OdimIoUtilityArg*)object)->object;
  const char* name;
  int result = 0;

  RAVE_ASSERT((attribute != NULL), "attribute == NULL");

  name = RaveAttribute_getName(attribute);

  if (strcasecmp("what/date", name)==0) {
    char* value = NULL;
    if (!RaveAttribute_getString(attribute, &value)) {
      RAVE_ERROR0("Failed to extract what/date as a string");
      goto done;
    }
    result = PolarScan_setDate(scan, value);
  } else if (strcasecmp("what/time", name)==0) {
    char* value = NULL;
    if (!RaveAttribute_getString(attribute, &value)) {
      RAVE_ERROR0("Failed to extract what/time as a string");
      goto done;
    }
    result = PolarScan_setTime(scan, value);
  } else if (strcasecmp("what/source", name)==0) {
    char* value = NULL;
    if (!RaveAttribute_getString(attribute, &value)) {
      RAVE_ERROR0("Failed to extract what/source as a string");
      goto done;
    }
    result = PolarScan_setSource(scan, value);
  } else if (strcasecmp("how/beamwidth", name)==0) {
    double value = 0.0;
    if (!(result = RaveAttribute_getDouble(attribute, &value))) {
      RAVE_ERROR0("Failed to extract how/beamwidth as a double");
      goto done;
    }
    PolarScan_setBeamwidth(scan, value * M_PI/180.0);
  } else if (strcasecmp("where/lon", name)==0) {
    double value = 0.0;
    if (!(result = RaveAttribute_getDouble(attribute, &value))) {
      RAVE_ERROR0("Failed to extract where/lon as a double");
      goto done;
    }
    PolarScan_setLongitude(scan, value * M_PI/180.0);
  } else if (strcasecmp("where/lat", name)==0) {
    double value = 0.0;
    if (!(result = RaveAttribute_getDouble(attribute, &value))) {
      RAVE_ERROR0("Failed to extract where/lat as a double");
      goto done;
    }
    PolarScan_setLatitude(scan, value * M_PI/180.0);
  } else if (strcasecmp("where/height", name)==0) {
    double value = 0.0;
    if (!(result = RaveAttribute_getDouble(attribute, &value))) {
      RAVE_ERROR0("Failed to extract where/height as a double");
      goto done;
    }
    PolarScan_setHeight(scan, value);
  } else if (strcasecmp("what/object", name) == 0) {
    result = 1;
  } else {
    PolarScan_addAttribute(scan, attribute);
    result = 1;
  }

done:
  return result;
}

/**
 * Scan root attributes.
 * @param[in] object - the OdimIoUtilityArg pointing to a polar scan
 * @param[in] attribute - the attribute found
 * @return 1 on success otherwise 0
 */
static int PolarOdimIOInternal_loadRootVolumeAttribute(void* object, RaveAttribute_t* attribute)
{
  PolarVolume_t* volume = (PolarVolume_t*)((OdimIoUtilityArg*)object)->object;
  const char* name;
  int result = 0;

  RAVE_ASSERT((attribute != NULL), "attribute == NULL");

  name = RaveAttribute_getName(attribute);

  if (strcasecmp("what/date", name)==0) {
    char* value = NULL;
    if (!RaveAttribute_getString(attribute, &value)) {
      RAVE_ERROR0("Failed to extract what/date as a string");
      goto done;
    }
    result = PolarVolume_setDate(volume, value);
  } else if (strcasecmp("what/time", name)==0) {
    char* value = NULL;
    if (!RaveAttribute_getString(attribute, &value)) {
      RAVE_ERROR0("Failed to extract what/time as a string");
      goto done;
    }
    result = PolarVolume_setTime(volume, value);
  } else if (strcasecmp("what/source", name)==0) {
    char* value = NULL;
    if (!RaveAttribute_getString(attribute, &value)) {
      RAVE_ERROR0("Failed to extract what/source as a string");
      goto done;
    }
    result = PolarVolume_setSource(volume, value);
  } else if (strcasecmp("how/beamwidth", name)==0) {
    double value = 0.0;
    if (!(result = RaveAttribute_getDouble(attribute, &value))) {
      RAVE_ERROR0("Failed to extract how/beamwidth as a double");
      goto done;
    }
    PolarVolume_setBeamwidth(volume, value * M_PI/180.0);
  } else if (strcasecmp("where/lon", name)==0) {
    double value = 0.0;
    if (!(result = RaveAttribute_getDouble(attribute, &value))) {
      RAVE_ERROR0("Failed to extract where/lon as a double");
      goto done;
    }
    PolarVolume_setLongitude(volume, value * M_PI/180.0);
  } else if (strcasecmp("where/lat", name)==0) {
    double value = 0.0;
    if (!(result = RaveAttribute_getDouble(attribute, &value))) {
      RAVE_ERROR0("Failed to extract where/lat as a double");
      goto done;
    }
    PolarVolume_setLatitude(volume, value * M_PI/180.0);
  } else if (strcasecmp("where/height", name)==0) {
    double value = 0.0;
    if (!(result = RaveAttribute_getDouble(attribute, &value))) {
      RAVE_ERROR0("Failed to extract where/height as a double");
      goto done;
    }
    PolarVolume_setHeight(volume, value);
  } else {
    result = PolarVolume_addAttribute(volume, attribute);
  }

done:
  return result;
}


/**
 * Called when an attribute belonging to a dataset in a scan
 * is found.
 * @param[in] object - the OdimIoUtilityArg pointing to a polar scan
 * @param[in] attribute - the attribute found
 * @return 1 on success otherwise 0
 */
static int PolarOdimIOInternal_loadDsScanAttribute(void* object, RaveAttribute_t* attribute)
{
  PolarScan_t* scan = NULL;
  const char* name;
  int result = 0;

  RAVE_ASSERT((object != NULL), "object == NULL");
  RAVE_ASSERT((attribute != NULL), "attribute == NULL");

  scan = (PolarScan_t*)((OdimIoUtilityArg*)object)->object;
  name = RaveAttribute_getName(attribute);
  if (name != NULL) {
    if (strcasecmp("where/elangle", name)==0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract where/elangle as a double");
      }
      PolarScan_setElangle(scan, value * M_PI/180.0);
    } else if (strcasecmp("where/a1gate", name)==0) {
      long value = 0;
      if (!(result = RaveAttribute_getLong(attribute, &value))) {
        RAVE_ERROR0("Failed to extract where/a1gate as a long");
      }
      PolarScan_setA1gate(scan, value);
    } else if (strcasecmp("where/rscale", name)==0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract where/rscale as a double");
      }
      PolarScan_setRscale(scan, value);
    } else if (strcasecmp("where/rstart", name)==0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract where/rstart as a double");
      }
      PolarScan_setRstart(scan, value);
    } else if (strcasecmp("what/startdate", name)==0) {
      char* value = NULL;
      if (!RaveAttribute_getString(attribute, &value)) {
        RAVE_ERROR0("Failed to extract what/startdate as a string");
        goto done;
      }
      result = PolarScan_setStartDate(scan, value);
    } else if (strcasecmp("what/starttime", name)==0) {
      char* value = NULL;
      if (!RaveAttribute_getString(attribute, &value)) {
        RAVE_ERROR0("Failed to extract what/starttime as a string");
        goto done;
      }
      result = PolarScan_setStartTime(scan, value);
    } else if (strcasecmp("what/enddate", name)==0) {
      char* value = NULL;
      if (!RaveAttribute_getString(attribute, &value)) {
        RAVE_ERROR0("Failed to extract what/enddate as a string");
        goto done;
      }
      result = PolarScan_setEndDate(scan, value);
    } else if (strcasecmp("what/endtime", name)==0) {
      char* value = NULL;
      if (!RaveAttribute_getString(attribute, &value)) {
        RAVE_ERROR0("Failed to extract what/endtime as a string");
        goto done;
      }
      result = PolarScan_setEndTime(scan, value);
    } else if (strcasecmp("what/product", name) == 0) {
      char* value = NULL;
      if (!RaveAttribute_getString(attribute, &value)) {
        RAVE_ERROR0("Failed to extract what/product as a string");
        goto done;
      }
      if (RaveTypes_getObjectTypeFromString(value) != Rave_ObjectType_SCAN) {
        RAVE_WARNING0("what/product did not identify as a SCAN!");
      }
      result = 1;
    } else if (strcasecmp("where/nbins", name) == 0 ||
               strcasecmp("where/nrays", name) == 0) {
      result = 1;
    } else {
      PolarScan_addAttribute(scan, attribute);
      result = 1;
    }
  }
done:
  return result;
}

/**
 * Called when an attribute belonging to a scan parameter
 * is found.
 * @param[in] object - the OdimIoUtilityArg pointing to a polar scan param
 * @param[in] attribute - the attribute found
 * @return 1 on success otherwise 0
 */
static int PolarOdimIOInternal_loadDsScanParamAttribute(void* object, RaveAttribute_t* attribute)
{
  PolarScanParam_t* param = NULL;
  const char* name;
  int result = 0;

  RAVE_ASSERT((object != NULL), "object == NULL");
  RAVE_ASSERT((attribute != NULL), "attribute == NULL");

  param = (PolarScanParam_t*)((OdimIoUtilityArg*)object)->object;
  name = RaveAttribute_getName(attribute);
  if (name != NULL) {
    if (strcasecmp("what/gain", name)==0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract what/gain as a double");
      }
      PolarScanParam_setGain(param, value);
    } else if (strcasecmp("what/offset", name)==0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract what/offset as a double");
      }
      PolarScanParam_setOffset(param, value);
    } else if (strcasecmp("what/nodata", name)==0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract what/nodata as a double");
      }
      PolarScanParam_setNodata(param, value);
    } else if (strcasecmp("what/undetect", name)==0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract what/undetect as a double");
      }
      PolarScanParam_setUndetect(param, value);
    } else if (strcasecmp("what/quantity", name)==0) {
      char* value = NULL;
      if(!RaveAttribute_getString(attribute, &value)) {
        RAVE_ERROR0("Failed to extract what/quantity as a string");
        goto done;
      }
      if (!(result = PolarScanParam_setQuantity(param, value))) {
        RAVE_ERROR1("Failed to add %s attribute", name);
        goto done;
      }
    } else {
      result = PolarScanParam_addAttribute(param, attribute);
    }
  }
done:
  return result;
}

/**
 * Called when an dataset belonging to a scan parameter
 * is found.
 * @param[in] object - the OdimIoUtilityArg pointing to a polar scan param
 * @param[in] nbins - the number of bins
 * @param[in] nrays - the number of rays
 * @param[in] data  - the data
 * @param[in] dtype - the type of the data.
 * @return 1 on success otherwise 0
 */
static int PolarOdimIOInternal_loadDsScanParamDataset(void* object, hsize_t nbins, hsize_t nrays, void* data, RaveDataType dtype)
{
  PolarScanParam_t* param = NULL;

  param = (PolarScanParam_t*)((OdimIoUtilityArg*)object)->object;

  return PolarScanParam_setData(param, nbins, nrays, data, dtype);
}

/**
 * Loads a scan parameter.
 * @param[in] nodelist - the hlhdf node list
 * @param[in] fmt - the variable argument list string format
 * @param[in] ... - the variable argument list
 * @return a scan parameter on success otherwise NULL
 */
static PolarScanParam_t* PolarOdimIOInternal_loadScanParam(HL_NodeList* nodelist, const char* fmt, ...)
{
  OdimIoUtilityArg arg;
  PolarScanParam_t* param = NULL;
  PolarScanParam_t* result = NULL;
  va_list ap;
  char name[1024];
  int nName = 0;
  int pindex = 1;
  int status = 0;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((fmt != NULL), "fmt == NULL");

  va_start(ap, fmt);
  nName = vsnprintf(name, 1024, fmt, ap);
  va_end(ap);
  if (nName < 0 || nName >= 1024) {
    RAVE_ERROR0("NodeName would evaluate to more than 1024 characters.");
    goto fail;
  }

  param = RAVE_OBJECT_NEW(&PolarScanParam_TYPE);
  if (param != NULL) {
    arg.nodelist = nodelist;
    arg.object = (RaveCoreObject*)param;
  }

  if (!RaveHL_loadAttributesAndData(nodelist, &arg,
                                    PolarOdimIOInternal_loadDsScanParamAttribute,
                                    PolarOdimIOInternal_loadDsScanParamDataset,
                                    name)) {
    goto fail;
  }

  pindex = 1;
  status = 1;
  while (status == 1 && RaveHL_hasNodeByName(nodelist, "%s/quality%d", name, pindex)) {
    RaveField_t* field = OdimIoUtilities_loadField(nodelist, "%s/quality%d", name, pindex);
    if (field != NULL) {
      status = PolarScanParam_addQualityField(param, field);
    } else {
      status = 0;
    }
    pindex++;
    RAVE_OBJECT_RELEASE(field);
  }
  if (status == 1) {
    result = RAVE_OBJECT_COPY(param);
  }
fail:
  RAVE_OBJECT_RELEASE(param);
  return result;
}

/**
 * Fills the scan with information from the dataset and below. I.e. root
 * attributes are not read.
 * @param[in] nodelist - the hlhdf node list
 * @param[in] scan - the scan
 * @param[in] fmt - the varargs format string
 * @param[in] ... - the varargs
 * @return 1 on success otherwise 0
 */
static int PolarOdimIOInternal_fillScanDataset(HL_NodeList* nodelist, PolarScan_t* scan, const char* fmt, ...)
{
  int result = 0;
  OdimIoUtilityArg arg;

  va_list ap;
  char name[1024];
  int nName = 0;
  int pindex = 1;

  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  RAVE_ASSERT((fmt != NULL), "fmt == NULL");

  va_start(ap, fmt);
  nName = vsnprintf(name, 1024, fmt, ap);
  va_end(ap);
  if (nName < 0 || nName >= 1024) {
    RAVE_ERROR0("NodeName would evaluate to more than 1024 characters.");
    goto done;
  }

  arg.nodelist = nodelist;
  arg.object = (RaveCoreObject*)scan;

  if (!RaveHL_loadAttributesAndData(nodelist,
                                    &arg,
                                    PolarOdimIOInternal_loadDsScanAttribute,
                                    NULL,
                                    "%s",
                                    name)) {
    RAVE_ERROR1("Failed to load attributes for scan at %s level", name);
    goto done;
  }

  result = 1;
  pindex = 1;
  while (result == 1 && RaveHL_hasNodeByName(nodelist, "%s/data%d", name, pindex)) {
    PolarScanParam_t* param = PolarOdimIOInternal_loadScanParam(nodelist, "%s/data%d", name, pindex);
    if (param != NULL) {
      result = PolarScan_addParameter(scan, param);
    } else {
      result = 0;
    }
    pindex++;
    RAVE_OBJECT_RELEASE(param);
  }

  pindex = 1;
  while (result == 1 && RaveHL_hasNodeByName(nodelist, "%s/quality%d", name, pindex)) {
    RaveField_t* field = OdimIoUtilities_loadField(nodelist, "%s/quality%d", name, pindex);
    if (field != NULL) {
      result = PolarScan_addQualityField(scan, field);
    } else {
      result = 0;
    }
    pindex++;
    RAVE_OBJECT_RELEASE(field);
  }

done:
  return result;
}

/**
 * Adds a scan parameter to the nodelist.
 *
 * @param[in] param - the scan parameter
 * @param[in] nodelist - the hlhdf node list
 * @param[in] fmt - the varargs format string
 * @param[in] ... - the varargs
 * @return 1 on success otherwise 0
 */
static int PolarOdimIOInternal_addParameter(PolarScanParam_t* param, HL_NodeList* nodelist, const char* fmt, ...)
{
  int result = 0;
  RaveObjectList_t* attributes = NULL;
  va_list ap;
  char name[1024];
  int nName = 0;
  RaveObjectList_t* qualityfields = NULL;

  RAVE_ASSERT((param != NULL), "param == NULL");
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
  if ((attributes = PolarScanParam_getAttributeValues(param)) == NULL) {
    goto done;
  }

  if (!RaveUtilities_replaceDoubleAttributeInList(attributes, "what/gain", PolarScanParam_getGain(param)) ||
      !RaveUtilities_replaceDoubleAttributeInList(attributes, "what/offset", PolarScanParam_getOffset(param)) ||
      !RaveUtilities_replaceDoubleAttributeInList(attributes, "what/nodata", PolarScanParam_getNodata(param)) ||
      !RaveUtilities_replaceDoubleAttributeInList(attributes, "what/undetect", PolarScanParam_getUndetect(param)) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/quantity", PolarScanParam_getQuantity(param))) {
    goto done;
  }

  if (!RaveHL_addAttributes(nodelist, attributes, name)) {
    goto done;
  }

  if (!RaveHL_addData(nodelist,
                      PolarScanParam_getData(param),
                      PolarScanParam_getNbins(param),
                      PolarScanParam_getNrays(param),
                      PolarScanParam_getDataType(param),
                      name)) {
    goto done;
  }

  if ((qualityfields = PolarScanParam_getQualityFields(param)) == NULL) {
    goto done;
  }

  result = OdimIoUtilities_addQualityFields(qualityfields, nodelist, name);

done:
  RAVE_OBJECT_RELEASE(attributes);
  RAVE_OBJECT_RELEASE(qualityfields);
  return result;
}

/**
 * Adds scan parameters to the nodelist.
 *
 * @param[in] scan - the scan
 * @param[in] nodelist - the hlhdf node list
 * @param[in] fmt - the varargs format string
 * @param[in] ... - the varargs
 * @return 1 on success otherwise 0
 */
static int PolarOdimIOInternal_addParameters(PolarScan_t* scan, HL_NodeList* nodelist, const char* fmt, ...)
{
  int result = 0;
  RaveObjectList_t* parameters = NULL;
  int nparams = 0;

  va_list ap;
  char name[1024];
  int nName = 0;
  int pindex = 1;

  RAVE_ASSERT((scan != NULL), "scan == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((fmt != NULL), "fmt == NULL");

  va_start(ap, fmt);
  nName = vsnprintf(name, 1024, fmt, ap);
  va_end(ap);
  if (nName < 0 || nName >= 1024) {
    RAVE_ERROR0("NodeName would evaluate to more than 1024 characters.");
    goto done;
  }

  if ((parameters = PolarScan_getParameters(scan)) == NULL) {
    goto done;
  }
  nparams = RaveObjectList_size(parameters);

  result = 1;
  for (pindex = 0; result == 1 && pindex < nparams; pindex++) {
    PolarScanParam_t* param = (PolarScanParam_t*)RaveObjectList_get(parameters, pindex);
    if (param != NULL) {
      result = PolarOdimIOInternal_addParameter(param, nodelist, "%s/data%d", name, (pindex+1));
    } else {
      result = 0;
    }
    RAVE_OBJECT_RELEASE(param);
  }


done:
  RAVE_OBJECT_RELEASE(parameters);
  return result;
}

static void PolarOdimIOInternal_removeVolumeAttributesFromList(RaveObjectList_t* attributes, PolarVolume_t* volume)
{
  int nrattrs = 0;
  int index = 0;

  RAVE_ASSERT((attributes != NULL), "attributes == NULL");
  RAVE_ASSERT((volume != NULL), "volume == NULL");

  nrattrs = RaveObjectList_size(attributes);
  for (index = nrattrs-1; index >= 0; index--) {
    RaveAttribute_t* attr = (RaveAttribute_t*)RaveObjectList_get(attributes, index);
    if (attr != NULL) {
      const char* name = RaveAttribute_getName(attr);
      if (name != NULL && PolarVolume_hasAttribute(volume, name)) {
        RaveAttribute_t* pvolattr = PolarVolume_getAttribute(volume, name);
        if (pvolattr != NULL) {
          RaveAttribute_Format format = RaveAttribute_getFormat(attr);
          if (format == RaveAttribute_getFormat(pvolattr)) {
            if (format == RaveAttribute_Format_Double) {
              double v1 = 0.0;
              double v2 = 0.0;
              RaveAttribute_getDouble(attr, &v1);
              RaveAttribute_getDouble(pvolattr, &v2);
              if (v1 == v2) {
                RaveObjectList_release(attributes, index);
              }
            } else if (format == RaveAttribute_Format_Long) {
              long v1 = 0;
              long v2 = 0;
              RaveAttribute_getLong(attr, &v1);
              RaveAttribute_getLong(pvolattr, &v2);
              if (v1 == v2) {
                RaveObjectList_release(attributes, index);
              }
            } else if (format == RaveAttribute_Format_String) {
              char* v1 = NULL;
              char* v2 = NULL;
              RaveAttribute_getString(attr, &v1);
              RaveAttribute_getString(pvolattr, &v2);
              if ((v1 != NULL && v2 != NULL && strcmp(v1, v2) == 0) || (v1 == NULL && v2 == NULL)) {
                RaveObjectList_release(attributes, index);
              }
            }
          } else {
            RAVE_WARNING1("Conflicting data types between volume and scan attribute %s", name);
            RaveObjectList_release(attributes, index);
          }
        }
        RAVE_OBJECT_RELEASE(pvolattr);
      }
    }
    RAVE_OBJECT_RELEASE(attr);
  }
}

static int PolarOdimIOInternal_addVolumeScan(PolarScan_t* scan, HL_NodeList* nodelist, PolarVolume_t* volume, const char* fmt, ...)
{
  int result = 0;
  RaveObjectList_t* attributes = NULL;
  RaveObjectList_t* qualityfields = NULL;

  va_list ap;
  char name[1024];
  int nName = 0;

  RAVE_ASSERT((scan != NULL), "scan == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((volume != NULL), "volume == NULL");
  RAVE_ASSERT((fmt != NULL), "fmt == NULL");

  va_start(ap, fmt);
  nName = vsnprintf(name, 1024, fmt, ap);
  va_end(ap);
  if (nName < 0 || nName >= 1024) {
    RAVE_ERROR0("NodeName would evaluate to more than 1024 characters.");
    goto done;
  }
  if ((attributes = PolarScan_getAttributeValues(scan)) == NULL) {
    goto done;
  }

  PolarOdimIOInternal_removeVolumeAttributesFromList(attributes, volume);

  if (PolarScan_getBeamwidth(scan) != PolarVolume_getBeamwidth(volume)) {
    if (!RaveUtilities_addDoubleAttributeToList(attributes, "how/beamwidth", PolarScan_getBeamwidth(scan)*180.0/M_PI)) {
      RAVE_WARNING0("Failed to add how/beamwidth to scan");
      goto done;
    }
  }

  if (!RaveUtilities_replaceStringAttributeInList(attributes, "what/product", "SCAN") ||
      !RaveUtilities_replaceLongAttributeInList(attributes, "where/a1gate", PolarScan_getA1gate(scan)) ||
      !RaveUtilities_replaceDoubleAttributeInList(attributes, "where/elangle", PolarScan_getElangle(scan)*180.0/M_PI) ||
      !RaveUtilities_replaceLongAttributeInList(attributes, "where/nbins", PolarScan_getNbins(scan)) ||
      !RaveUtilities_replaceLongAttributeInList(attributes, "where/nrays", PolarScan_getNrays(scan)) ||
      !RaveUtilities_replaceDoubleAttributeInList(attributes, "where/rscale", PolarScan_getRscale(scan)) ||
      !RaveUtilities_replaceDoubleAttributeInList(attributes, "where/rstart", PolarScan_getRstart(scan)) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/startdate", PolarScan_getStartDate(scan)) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/starttime", PolarScan_getStartTime(scan)) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/enddate", PolarScan_getEndDate(scan)) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/endtime", PolarScan_getEndTime(scan))) {
    goto done;
  }

  if (!RaveHL_hasNodeByName(nodelist, name)) {
    if (!RaveHL_createGroup(nodelist, name)) {
      goto done;
    }
  }

  if (!RaveHL_addAttributes(nodelist, attributes, name)) {
    goto done;
  }

  if (!PolarOdimIOInternal_addParameters(scan, nodelist, name)) {
    goto done;
  }

  if ((qualityfields = PolarScan_getQualityFields(scan)) == NULL) {
    goto done;
  }

  result = OdimIoUtilities_addQualityFields(qualityfields, nodelist, name);

done:
  RAVE_OBJECT_RELEASE(attributes);
  RAVE_OBJECT_RELEASE(qualityfields);
  return result;
}

/*@} End of Private functions */

/*@{ Interface functions */

int PolarOdimIO_readScan(PolarOdimIO_t* self, HL_NodeList* nodelist, PolarScan_t* scan)
{
  int result = 0;
  OdimIoUtilityArg arg;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((scan != NULL), "scan == NULL");

  arg.nodelist = nodelist;
  arg.object = (RaveCoreObject*)scan;

  if (!RaveHL_hasNodeByName(nodelist, "/dataset1") ||
      !RaveHL_hasNodeByName(nodelist, "/dataset1/data1")) {
    RAVE_ERROR0("Scan file does not contain scan...");
    goto done;
  }

  if (!RaveHL_loadAttributesAndData(nodelist, &arg,
                                    PolarOdimIOInternal_loadRootScanAttribute,
                                    NULL,
                                    "")) {
    RAVE_ERROR0("Failed to load attributes for scan at root level");
    goto done;
  }

  if (!PolarOdimIOInternal_fillScanDataset(nodelist, scan, "/dataset1")) {
    RAVE_ERROR0("Failed to fill scan");
    goto done;
  }

  result = 1;
done:
  return result;
}

int PolarOdimIO_readVolume(PolarOdimIO_t* self, HL_NodeList* nodelist, PolarVolume_t* volume)
{
  int result = 0;
  int pindex = 1;
  OdimIoUtilityArg arg;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");
  RAVE_ASSERT((volume != NULL), "volume == NULL");

  arg.nodelist = nodelist;
  arg.object = (RaveCoreObject*)volume;

  if (!RaveHL_loadAttributesAndData(nodelist, &arg,
                                    PolarOdimIOInternal_loadRootVolumeAttribute,
                                    NULL,
                                    "")) {
    RAVE_ERROR0("Failed to load attributes for volume at root level");
    goto done;
  }

  result = 1;
  pindex = 1;
  while (result == 1 && RaveHL_hasNodeByName(nodelist, "/dataset%d", pindex)) {
    PolarScan_t* scan = RAVE_OBJECT_NEW(&PolarScan_TYPE);
    if (scan != NULL) {
      result = PolarOdimIOInternal_fillScanDataset(nodelist, scan, "/dataset%d", pindex);
      if (result == 1) {
        result = PolarVolume_addScan(volume, scan);
      }
    } else {
      result = 0;
    }
    pindex++;
    RAVE_OBJECT_RELEASE(scan);
  }

done:
  return result;
}

int PolarOdimIO_fillScan(PolarOdimIO_t* self, PolarScan_t* scan, HL_NodeList* nodelist)
{
  int result = 0;
  RaveObjectList_t* attributes = NULL;
  RaveObjectList_t* qualityfields = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((scan != NULL), "scan == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  if (!RaveHL_hasNodeByName(nodelist, "/Conventions")) {
    if (!RaveHL_createStringValue(nodelist, RAVE_ODIM_VERSION_2_2_STR, "/Conventions")) {
      goto done;
    }
  }

  attributes = PolarScan_getAttributeValues(scan);
  if (attributes != NULL) {
    const char* objectType = RaveTypes_getStringFromObjectType(Rave_ObjectType_SCAN);
    if (!RaveUtilities_addStringAttributeToList(attributes, "what/object", objectType) ||
        !RaveUtilities_replaceStringAttributeInList(attributes, "what/version", RAVE_ODIM_H5RAD_VERSION_2_2_STR)) {
      RAVE_ERROR0("Failed to add what/object or what/version to attributes");
      goto done;
    }
  } else {
    RAVE_ERROR0("Failed to aquire attributes for polar scan");
    goto done;
  }

  if (!RaveUtilities_replaceDoubleAttributeInList(attributes, "how/beamwidth", PolarScan_getBeamwidth(scan)*180.0/M_PI) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/date", PolarScan_getDate(scan)) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/time", PolarScan_getTime(scan)) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/source", PolarScan_getSource(scan)) ||
      !RaveUtilities_replaceDoubleAttributeInList(attributes, "where/height", PolarScan_getHeight(scan)) ||
      !RaveUtilities_replaceDoubleAttributeInList(attributes, "where/lat", PolarScan_getLatitude(scan)*180.0/M_PI) ||
      !RaveUtilities_replaceDoubleAttributeInList(attributes, "where/lon", PolarScan_getLongitude(scan)*180.0/M_PI)) {
    goto done;
  }

  if (attributes == NULL || !RaveHL_addAttributes(nodelist, attributes, "")) {
    goto done;
  }

  if (!RaveHL_createGroup(nodelist, "/dataset1")) {
    goto done;
  }

  RAVE_OBJECT_RELEASE(attributes);
  attributes = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (attributes == NULL) {
    goto done;
  }
  if (!RaveUtilities_replaceStringAttributeInList(attributes, "what/product", "SCAN") ||
      !RaveUtilities_replaceLongAttributeInList(attributes, "where/a1gate", PolarScan_getA1gate(scan)) ||
      !RaveUtilities_replaceDoubleAttributeInList(attributes, "where/elangle", PolarScan_getElangle(scan)*180.0/M_PI) ||
      !RaveUtilities_replaceLongAttributeInList(attributes, "where/nbins", PolarScan_getNbins(scan)) ||
      !RaveUtilities_replaceLongAttributeInList(attributes, "where/nrays", PolarScan_getNrays(scan)) ||
      !RaveUtilities_replaceDoubleAttributeInList(attributes, "where/rscale", PolarScan_getRscale(scan)) ||
      !RaveUtilities_replaceDoubleAttributeInList(attributes, "where/rstart", PolarScan_getRstart(scan)) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/startdate", PolarScan_getStartDate(scan)) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/starttime", PolarScan_getStartTime(scan)) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/enddate", PolarScan_getEndDate(scan)) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/endtime", PolarScan_getEndTime(scan))) {
    goto done;
  }
  if (!RaveHL_addAttributes(nodelist, attributes, "/dataset1")) {
    goto done;
  }

  if (!PolarOdimIOInternal_addParameters(scan, nodelist, "/dataset1")) {
    goto done;
  }

  if ((qualityfields = PolarScan_getQualityFields(scan)) == NULL) {
    goto done;
  }

  result = OdimIoUtilities_addQualityFields(qualityfields, nodelist, "/dataset1");
done:
  RAVE_OBJECT_RELEASE(attributes);
  RAVE_OBJECT_RELEASE(qualityfields);
  return result;
}

int PolarOdimIO_fillVolume(PolarOdimIO_t* self, PolarVolume_t* volume, HL_NodeList* nodelist)
{
  int result = 0;
  RaveObjectList_t* attributes = NULL;

  int nrscans = 0;
  int index = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((volume != NULL), "volume == NULL");
  RAVE_ASSERT((nodelist != NULL), "nodelist == NULL");

  if (!RaveHL_hasNodeByName(nodelist, "/Conventions")) {
    if (!RaveHL_createStringValue(nodelist, RAVE_ODIM_VERSION_2_2_STR, "/Conventions")) {
      goto done;
    }
  }

  attributes = PolarVolume_getAttributeValues(volume);
  if (attributes != NULL) {
    const char* objectType = RaveTypes_getStringFromObjectType(Rave_ObjectType_PVOL);
    if (!RaveUtilities_addStringAttributeToList(attributes, "what/object", objectType) ||
        !RaveUtilities_replaceStringAttributeInList(attributes, "what/version", RAVE_ODIM_H5RAD_VERSION_2_2_STR)) {
      RAVE_ERROR0("Failed to add what/object or what/version to attributes");
      goto done;
    }
  } else {
    RAVE_ERROR0("Failed to aquire attributes for polar volume");
    goto done;
  }

  if (!RaveUtilities_replaceDoubleAttributeInList(attributes, "how/beamwidth", PolarVolume_getBeamwidth(volume)*180.0/M_PI) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/date", PolarVolume_getDate(volume)) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/time", PolarVolume_getTime(volume)) ||
      !RaveUtilities_replaceStringAttributeInList(attributes, "what/source", PolarVolume_getSource(volume)) ||
      !RaveUtilities_replaceDoubleAttributeInList(attributes, "where/height", PolarVolume_getHeight(volume)) ||
      !RaveUtilities_replaceDoubleAttributeInList(attributes, "where/lat", PolarVolume_getLatitude(volume)*180.0/M_PI) ||
      !RaveUtilities_replaceDoubleAttributeInList(attributes, "where/lon", PolarVolume_getLongitude(volume)*180.0/M_PI)) {
    goto done;
  }

  if (attributes == NULL || !RaveHL_addAttributes(nodelist, attributes, "")) {
    goto done;
  }

  result = 1;
  nrscans = PolarVolume_getNumberOfScans(volume);
  for (index = 0; result == 1 && index < nrscans; index++) {
    PolarScan_t* scan = PolarVolume_getScan(volume, index);
    if (scan != NULL) {
      result = PolarOdimIOInternal_addVolumeScan(scan, nodelist, volume, "/dataset%d", (index+1));
    } else {
      result = 0;
    }
    RAVE_OBJECT_RELEASE(scan);
  }

done:
  RAVE_OBJECT_RELEASE(attributes);
  return result;
}
/*@} End of Interface functions */

RaveCoreObjectType PolarOdimIO_TYPE = {
    "PolarOdimIO",
    sizeof(PolarOdimIO_t),
    PolarOdimIO_constructor,
    PolarOdimIO_destructor,
    PolarOdimIO_copyconstructor
};
