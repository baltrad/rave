/* --------------------------------------------------------------------
Copyright (C) 2025 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Feature map used when working with acqva. 
 *
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-11-11
 */
#include "acqvafeaturemap.h"
#include "rave_data2d.h"
#include "rave_debug.h"
#include <math.h>
#include <string.h>
#include "rave_hlhdf_utilities.h"

#define ACQVA_FEATURE_MAP_VERSION "ACQVA Feature Map 1.0"

struct _AcqvaFeatureMap_t {
  RAVE_OBJECT_HEAD          /** Always on top */
  char nod[11];             /**< the nod */
  double latitude;          /**< the latitude in radians */
  double longitude;         /**< the longitude in radians */
  double height;            /**< the height in meters */
  RaveDateTime_t* startdate;    /**< the start of the period */
  RaveDateTime_t* enddate;    /**< the end of the period */
  RaveObjectList_t* elevations; /**< the elevations */   
};

struct _AcqvaFeatureMapElevation_t {
  RAVE_OBJECT_HEAD          /** Always on top */
  double elangle;           /**< the elevation angle */
  RaveObjectList_t* fields; /**< the fields */
};

struct _AcqvaFeatureMapField_t {
  RAVE_OBJECT_HEAD          /** Always on top */
  RaveData2D_t* data;        /**< the data field */
  double elangle;           /**< the elangle */
  RaveDataType type;
  long nbins;
  long nrays;
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int AcqvaFeatureMap_constructor(RaveCoreObject* obj)
{
  AcqvaFeatureMap_t* this = (AcqvaFeatureMap_t*)obj;
  strcpy(this->nod, "");
  this->longitude = 0.0;
  this->latitude = 0.0;
  this->height = 0.0;
  this->elevations = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  this->startdate = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  this->enddate = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  if (this->elevations == NULL || this->startdate == NULL || this->enddate == NULL) {
    goto fail;
  }
  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->elevations);
  RAVE_OBJECT_RELEASE(this->startdate);
  RAVE_OBJECT_RELEASE(this->enddate);
  return 0;
}

/**
 * Copy constructor
 */
static int AcqvaFeatureMap_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  AcqvaFeatureMap_t* this = (AcqvaFeatureMap_t*)obj;
  AcqvaFeatureMap_t* src = (AcqvaFeatureMap_t*)srcobj;

  strcpy(this->nod, src->nod);
  this->longitude = src->longitude;
  this->latitude = src->latitude;
  this->height = src->height;
  this->elevations = NULL;
  this->startdate = NULL;
  this->enddate = NULL;

  this->elevations = RAVE_OBJECT_CLONE(src->elevations);
  if (this->elevations == NULL) {
    goto fail;
  }
  this->startdate = RAVE_OBJECT_CLONE(src->startdate);
  this->enddate = RAVE_OBJECT_CLONE(src->enddate);
  if (this->startdate == NULL || this->enddate == NULL) {
    goto fail;
  }

  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->elevations);
  RAVE_OBJECT_RELEASE(this->startdate);
  RAVE_OBJECT_RELEASE(this->enddate);
  return 0;
}

/**
 * Destuctor
 */
static void AcqvaFeatureMap_destructor(RaveCoreObject* obj)
{
  AcqvaFeatureMap_t* this = (AcqvaFeatureMap_t*)obj;
  RAVE_OBJECT_RELEASE(this->elevations);
  RAVE_OBJECT_RELEASE(this->startdate);
  RAVE_OBJECT_RELEASE(this->enddate);
}

/**
 * Constructor.
 */
static int AcqvaFeatureMapElevation_constructor(RaveCoreObject* obj)
{
  AcqvaFeatureMapElevation_t* this = (AcqvaFeatureMapElevation_t*)obj;
  this->elangle = 0.0;
  this->fields = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (this->fields == NULL) {
    return 0;
  }
  return 1;
}

/**
 * Copy constructor
 */
static int AcqvaFeatureMapElevation_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  AcqvaFeatureMapElevation_t* this = (AcqvaFeatureMapElevation_t*)obj;
  AcqvaFeatureMapElevation_t* src = (AcqvaFeatureMapElevation_t*)srcobj;
  this->elangle = src->elangle;
  this->fields = RAVE_OBJECT_CLONE(src->fields);
  if (this->fields == NULL) {
    return 0;
  }
  return 1;
}

/**
 * Destuctor
 */
static void AcqvaFeatureMapElevation_destructor(RaveCoreObject* obj)
{
  AcqvaFeatureMapElevation_t* this = (AcqvaFeatureMapElevation_t*)obj;
  RAVE_OBJECT_RELEASE(this->fields);
}


/**
 * Constructor.
 */
static int AcqvaFeatureMapField_constructor(RaveCoreObject* obj)
{
  AcqvaFeatureMapField_t* this = (AcqvaFeatureMapField_t*)obj;
  this->elangle = 0.0;
  this->nbins = 0;
  this->nrays = 0;
  this->data = RAVE_OBJECT_NEW(&RaveData2D_TYPE);
  if (this->data == NULL) {
    return 0;
  }
  return 1;
}

/**
 * Copy constructor
 */
static int AcqvaFeatureMapField_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  AcqvaFeatureMapField_t* this = (AcqvaFeatureMapField_t*)obj;
  AcqvaFeatureMapField_t* src = (AcqvaFeatureMapField_t*)srcobj;
  this->elangle = src->elangle;
  this->data = NULL;
  if (src->data != NULL) {
    this->data = RAVE_OBJECT_CLONE(src->data);
    if (this->data == NULL) {
        return 0;
    }
  }
  this->nbins = src->nbins;
  this->nrays = src->nrays;
  return 1;
}

/**
 * Destuctor
 */
static void AcqvaFeatureMapField_destructor(RaveCoreObject* obj)
{
  AcqvaFeatureMapField_t* this = (AcqvaFeatureMapField_t*)obj;
  RAVE_OBJECT_RELEASE(this->data);
}

static int AcqvaFeatureMapInternal_fillNodeListWithField(AcqvaFeatureMapField_t* field, HL_NodeList* nodelist, const char* fmt, ...)
{
  int result = 0;
  va_list ap;
  char name[1024];
  int nName = 0;
  RaveObjectList_t* attributes = NULL;

  if (field == NULL || nodelist == NULL || fmt == NULL) {
    RAVE_ERROR0("Programming error");
    goto done;
  }

  va_start(ap, fmt);
  nName = vsnprintf(name, 1024, fmt, ap);
  va_end(ap);
  if (nName < 0 || nName >= 1024) {
    RAVE_ERROR0("name would evaluate to more than 1024 characters.");
    goto done;
  }

  attributes = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (attributes == NULL) {
    goto done;
  }

  if (!RaveUtilities_addDoubleAttributeToList(attributes, "where/elangle", AcqvaFeatureMapField_getElangle(field)*180.0/M_PI)) {
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

  if (!RaveHL_addData(nodelist,
                      AcqvaFeatureMapField_getData(field),
                      AcqvaFeatureMapField_getNbins(field),
                      AcqvaFeatureMapField_getNrays(field),
                      AcqvaFeatureMapField_getDatatype(field),
                      name)) {
    goto done;
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(attributes);
  return result;
}

static int AcqvaFeatureMapInternal_fillNodeListWithElevation(AcqvaFeatureMapElevation_t* elevation, HL_NodeList* nodelist, const char* fmt, ...)
{
  int result = 0;
  va_list ap;
  char name[1024];
  int nName = 0;
  RaveObjectList_t* attributes = NULL;
  int nfields = 0, fi = 0;

  if (elevation == NULL || nodelist == NULL || fmt == NULL) {
    RAVE_ERROR0("Programming error");
    goto done;
  }

  va_start(ap, fmt);
  nName = vsnprintf(name, 1024, fmt, ap);
  va_end(ap);
  if (nName < 0 || nName >= 1024) {
    RAVE_ERROR0("name would evaluate to more than 1024 characters.");
    goto done;
  }

  attributes = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (attributes == NULL) {
    goto done;
  }

  if (!RaveUtilities_addDoubleAttributeToList(attributes, "where/elangle", AcqvaFeatureMapElevation_getElangle(elevation)*180.0/M_PI)) {
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

  nfields = AcqvaFeatureMapElevation_size(elevation);
  for (fi = 0; fi < nfields; fi++) {
    AcqvaFeatureMapField_t* field = AcqvaFeatureMapElevation_get(elevation, fi);
    if (field == NULL || !AcqvaFeatureMapInternal_fillNodeListWithField(field, nodelist, "%s/data%d", name, fi+1)) {
      RAVE_ERROR0("Could not add field");
      RAVE_OBJECT_RELEASE(field);
      goto done;
    }
    RAVE_OBJECT_RELEASE(field);
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(attributes);
  return result;
}

static int AcqvaFeatureMapInternal_fillNodelist(AcqvaFeatureMap_t* self, HL_NodeList* nodelist)
{
  int result = 0;
  int ngroups = 0, gi = 0;
  RaveObjectList_t* attributes = NULL;

  attributes = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (attributes == NULL) {
    goto fail;
  }

  if (!RaveUtilities_addStringAttributeToList(attributes, "what/nod", AcqvaFeatureMap_getNod(self)) ||
      !RaveUtilities_addStringAttributeToList(attributes, "what/startdate", AcqvaFeatureMap_getStartdate(self)) ||
      !RaveUtilities_addStringAttributeToList(attributes, "what/enddate", AcqvaFeatureMap_getEnddate(self)) ||
      !RaveUtilities_addDoubleAttributeToList(attributes, "where/lon", AcqvaFeatureMap_getLongitude(self)*180.0/M_PI) ||
      !RaveUtilities_addDoubleAttributeToList(attributes, "where/lat", AcqvaFeatureMap_getLatitude(self)*180.0/M_PI) ||
      !RaveUtilities_addDoubleAttributeToList(attributes, "where/height", AcqvaFeatureMap_getHeight(self))) {
    RAVE_ERROR0("Not able to add attributes");
    goto fail;
  }

  if (!RaveHL_addAttributes(nodelist, attributes, "")) {
    goto fail;
  }  

  ngroups = AcqvaFeatureMap_getNumberOfElevations(self);
  for (gi = 0; gi < ngroups; gi++) {
    AcqvaFeatureMapElevation_t* elevation = AcqvaFeatureMap_getElevation(self, gi);
    if (elevation == NULL || !AcqvaFeatureMapInternal_fillNodeListWithElevation(elevation, nodelist, "/dataset%d", gi+1)) {
      RAVE_ERROR0("Could not add elevation group");
      RAVE_OBJECT_RELEASE(elevation);
      goto fail;
    }
    RAVE_OBJECT_RELEASE(elevation);
  }

  result = 1;
fail:
  RAVE_OBJECT_RELEASE(attributes);
  return result;
}

static int AcqvaFeatureMapInternal_loadRootAttributes(void* object, RaveAttribute_t* attribute)
{
  AcqvaFeatureMap_t* featuremap = (AcqvaFeatureMap_t*)object;
  int result = 0;
  if (attribute != NULL) {
    const char* name = RaveAttribute_getName(attribute);
    if (strcasecmp("what/nod", name) == 0) {
      char* value = NULL;
      if (!RaveAttribute_getString(attribute, &value)) {
        RAVE_ERROR0("Failed to extract what/nod as a string");
        goto done;
      }
      if (!(result = AcqvaFeatureMap_setNod(featuremap, value))) {
        RAVE_ERROR1("Failed to set what/nod to %s",value);
      }
    } else if (strcasecmp("what/startdate", name) == 0) {
      char* value = NULL;
      if (!RaveAttribute_getString(attribute, &value)) {
        RAVE_ERROR0("Failed to extract what/startdate as a string");
        goto done;
      }
      if (!(result = AcqvaFeatureMap_setStartdate(featuremap, value))) {
        RAVE_ERROR1("Failed to set what/startdate to %s",value);
      }
    } else if (strcasecmp("what/enddate", name) == 0) {
      char* value = NULL;
      if (!RaveAttribute_getString(attribute, &value)) {
        RAVE_ERROR0("Failed to extract what/enddate as a string");
        goto done;
      }
      if (!(result = AcqvaFeatureMap_setEnddate(featuremap, value))) {
        RAVE_ERROR1("Failed to set what/enddate to %s",value);
      }
    } else if (strcasecmp("where/lon", name) == 0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract what/lon as a double");
        goto done;
      }
      AcqvaFeatureMap_setLongitude(featuremap, value * M_PI / 180.0);
    } else if (strcasecmp("where/lat", name) == 0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract what/lat as a double");
        goto done;
      }
      AcqvaFeatureMap_setLatitude(featuremap, value * M_PI / 180.0);
    } else if (strcasecmp("where/height", name) == 0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract what/height as a double");
        goto done;
      }
      AcqvaFeatureMap_setHeight(featuremap, value);
    } else {
      RAVE_INFO1("Unsupported attribute: %s", name);
      result = 1;
    }
  }
done:
  return result;
}

static int AcqvaFeatureMapInternal_loadFieldAttributeFunc(void* object, RaveAttribute_t* attribute)
{
  int result = 0;
  AcqvaFeatureMapField_t* field = (AcqvaFeatureMapField_t*)object;
  const char* name;
  RAVE_ASSERT((object != NULL), "object == NULL");
  RAVE_ASSERT((attribute != NULL), "attribute == NULL");

  name = RaveAttribute_getName(attribute);
  if (name != NULL) {
    if (strcasecmp("where/elangle", name)==0) {
      double value = 0.0;
      if (!(result = RaveAttribute_getDouble(attribute, &value))) {
        RAVE_ERROR0("Failed to extract where/elangle as a double");
        goto done;
      }
      AcqvaFeatureMapField_setElangle(field, value * M_PI / 180.0);
    } else {
      RAVE_INFO1("Unsupported attribute: %s", name);
      result = 1;
    }
  }
done:
  return result;
}

static int AcqvaFeatureMapInternal_loadFieldDataFunc(void* object, hsize_t nbins, hsize_t nrays, void* data, RaveDataType dtype, const char* nodeName)
{
  AcqvaFeatureMapField_t* field = (AcqvaFeatureMapField_t*)object;
  if (field != NULL) {
    return AcqvaFeatureMapField_setData(field, nbins, nrays, data, dtype);
  }
  return 0;
}

static AcqvaFeatureMapElevation_t* AcqvaFeatureMapInternal_loadElevation(AcqvaFeatureMap_t* self, HL_NodeList* nodelist, const char* fmt, ...)
{
  int status = 0, pindex = 1;
  va_list ap;
  char name[1024];
  int nName = 0;
  RaveAttribute_t* elangle = NULL;
  AcqvaFeatureMapElevation_t *elevation = NULL, *result = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (nodelist == NULL || fmt == NULL) {
    RAVE_ERROR0("Programming error");
    goto done;
  }

  va_start(ap, fmt);
  nName = vsnprintf(name, 1024, fmt, ap);
  va_end(ap);
  if (nName < 0 || nName >= 1024) {
    RAVE_ERROR0("name would evaluate to more than 1024 characters.");
    goto done;
  }

  elevation = RAVE_OBJECT_NEW(&AcqvaFeatureMapElevation_TYPE);
  if (elevation == NULL) {
    RAVE_ERROR0("Could not create elevation");
    goto done;
  }

  elangle = RaveHL_getAttribute(nodelist, "%s/where/elangle", name);
  if (elangle != NULL) {
    double value = 0.0;
    if (!RaveAttribute_getDouble(elangle, &value)) {
      RAVE_ERROR0("Could not get elangle as a double");
      goto done;
    }
    AcqvaFeatureMapElevation_setElangle(elevation, value * M_PI / 180.0);
  } else {
    RAVE_ERROR1("No %s/where/elangle", name);
    goto done;
  }

  status = 1;
  pindex = 1;
  while (status == 1 && RaveHL_hasNodeByName(nodelist, "%s/data%d", name, pindex)) {
    AcqvaFeatureMapField_t* field = RAVE_OBJECT_NEW(&AcqvaFeatureMapField_TYPE);
    if (field != NULL) {
      if (!RaveHL_loadAttributesAndData(nodelist, (void*)field, 
                                        AcqvaFeatureMapInternal_loadFieldAttributeFunc, 
                                        AcqvaFeatureMapInternal_loadFieldDataFunc, 
                                        "%s/data%d", name, pindex)) {
        fprintf(stderr, "Failed to load attrs and data\n");
        status = 0;
      }

      if (status != 0) {
        if (!(status = AcqvaFeatureMapElevation_add(elevation, field))) {
          RAVE_ERROR0("Failed to add field to elevation group");
        }
      }
    }
    pindex++;
    RAVE_OBJECT_RELEASE(field);
  }

  if (status == 0) {
    RAVE_ERROR0("Failed to load field");
    goto done;
  }

  result = RAVE_OBJECT_COPY(elevation);
done:
  RAVE_OBJECT_RELEASE(elangle);
  RAVE_OBJECT_RELEASE(elevation);
  return result;
}
/*@} End of Private functions */

/*@{ AcqvaFeatureMap public methods */
AcqvaFeatureMap_t* AcqvaFeatureMap_load(const char* filename)
{
  int status = 0, pindex = 0;
  HL_NodeList* nodelist = NULL;
  AcqvaFeatureMap_t *featuremap = NULL, *result = NULL;
  char* version = NULL;

  if (!HL_isHDF5File(filename)) {
    RAVE_ERROR0("Not a HDF5 file");
    return NULL;
  }

  nodelist = HLNodeList_read(filename);
  if (nodelist == NULL) {
    RAVE_ERROR1("Failed to read %s", nodelist);
    return NULL;
  }

  HLNodeList_selectAllNodes(nodelist);
  if (!HLNodeList_fetchMarkedNodes(nodelist)) {
    RAVE_ERROR1("Failed to load hdf5 file '%s'", filename);
    goto fail;
  }

  if (!RaveHL_getStringValue(nodelist, &version, "/Conventions")) {
    RAVE_ERROR0("Failed to read attribute /Conventions");
    goto fail;
  }

  if (strcmp(ACQVA_FEATURE_MAP_VERSION, version) != 0) {
    RAVE_ERROR0("Not a valid feature map version");
    goto fail;
  }

  featuremap = RAVE_OBJECT_NEW(&AcqvaFeatureMap_TYPE);
  if (featuremap == NULL) {
    goto fail;
  }

  if (!RaveHL_loadAttributesAndData(nodelist, (void*)featuremap,
                                    AcqvaFeatureMapInternal_loadRootAttributes,
                                    NULL,
                                    "")) {
    RAVE_ERROR0("Failed to load attributes for volume at root level");
    goto fail;
  }

  status = 1;
  pindex = 1;
  while (status == 1 && RaveHL_hasNodeByName(nodelist, "/dataset%d", pindex)) {
    AcqvaFeatureMapElevation_t* group = AcqvaFeatureMapInternal_loadElevation(featuremap, nodelist, "/dataset%d", pindex);
    if (group != NULL) {
      status = AcqvaFeatureMap_addElevation(featuremap, group);
    } else {
      status = 0;
    }
    pindex++;
    RAVE_OBJECT_RELEASE(group);
  }

  if (status == 0) {
    RAVE_ERROR0("Failed to load elevation groups");
    goto fail;
  }

  result = RAVE_OBJECT_COPY(featuremap);
fail:
  RAVE_OBJECT_RELEASE(featuremap);
  if (nodelist != NULL) {
    HLNodeList_free(nodelist);
  }
  return result;
}

int AcqvaFeatureMap_save(AcqvaFeatureMap_t* self, const char* filename)
{
  int result = 0;

  HL_Compression* compression = HLCompression_new(CT_ZLIB);
  HL_FileCreationProperty* property = HLFileCreationProperty_new();
  HL_NodeList* nodelist = HLNodeList_new();

  if (strcmp("", AcqvaFeatureMap_getNod(self)) == 0 ||
      AcqvaFeatureMap_getStartdate(self) == NULL ||
      AcqvaFeatureMap_getEnddate(self) == NULL) {
    RAVE_ERROR0("Feature meta data must contain nod, startdate and enddate");
    goto fail;
  }

  if (nodelist == NULL) {
    RAVE_ERROR0("Failed to create nodelist");
    goto fail;
  }

  if (compression == NULL || property == NULL) {
    RAVE_ERROR0("Failed to create compression or file creation properties");
    goto fail;
  }

  compression->level = (int)6;
  property->userblock = (hsize_t)0;
  property->sizes.sizeof_size = (size_t)4;
  property->sizes.sizeof_addr = (size_t)4;
  property->sym_k.ik = (int)1;
  property->sym_k.lk = (int)1;
  property->istore_k = (long)1;
  property->meta_block_size = (long)0;

  result = RaveHL_createStringValue(nodelist, ACQVA_FEATURE_MAP_VERSION, "/Conventions");
  if (result == 1) {
    result = AcqvaFeatureMapInternal_fillNodelist(self, nodelist);
  }

  if (result == 1) {
    result = HLNodeList_setFileName(nodelist, filename);
  }

  if (result == 1) {
    result = HLNodeList_write(nodelist, property, compression);
  }

fail:
  if (compression != NULL) {
    HLCompression_free(compression);
  }
  if (property != NULL) {
    HLFileCreationProperty_free(property);
  }
  if (nodelist != NULL) {
    HLNodeList_free(nodelist);
  }
  return result;
}


int AcqvaFeatureMap_setNod(AcqvaFeatureMap_t* self, const char* source)
{
  RAVE_ASSERT((self != NULL), "self == NULL");

  if (source != NULL) {
    if (strlen(source) > 10) {
      RAVE_ERROR0("Can not handle nods > 10 characters");
      return 0;
    } else {
      strcpy(self->nod, source);
    }
  } else {
    strcpy(self->nod, "");
  }

  return 1;
}

const char* AcqvaFeatureMap_getNod(AcqvaFeatureMap_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->nod;
}

void AcqvaFeatureMap_setLongitude(AcqvaFeatureMap_t* self, double lon)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->longitude = lon;
}

double AcqvaFeatureMap_getLongitude(AcqvaFeatureMap_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->longitude;
}

void AcqvaFeatureMap_setLatitude(AcqvaFeatureMap_t* self, double lat)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->latitude = lat;
}

double AcqvaFeatureMap_getLatitude(AcqvaFeatureMap_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->latitude;
}

void AcqvaFeatureMap_setHeight(AcqvaFeatureMap_t* self, double height)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->height = height;
}

double AcqvaFeatureMap_getHeight(AcqvaFeatureMap_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->height;
}

int AcqvaFeatureMap_setStartdate(AcqvaFeatureMap_t* self, const char* date)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveDateTime_setDate(self->startdate, date);
}

const char* AcqvaFeatureMap_getStartdate(AcqvaFeatureMap_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveDateTime_getDate(self->startdate);
}

int AcqvaFeatureMap_setEnddate(AcqvaFeatureMap_t* self, const char* date)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveDateTime_setDate(self->enddate, date);
}

const char* AcqvaFeatureMap_getEnddate(AcqvaFeatureMap_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveDateTime_getDate(self->enddate);
}

AcqvaFeatureMapField_t* AcqvaFeatureMap_createField(AcqvaFeatureMap_t* self, long nbins, long nrays, RaveDataType type, double elangle)
{
  AcqvaFeatureMapElevation_t* elevation = NULL;
  AcqvaFeatureMapField_t *field = NULL, *result = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");

  field = AcqvaFeatureMapField_createField(nbins, nrays, type, elangle);
  if (field == NULL) {
    goto fail;
  }
  if (!AcqvaFeatureMap_addField(self, field)) {
    goto fail;
  }

  result = RAVE_OBJECT_COPY(field);
fail:
  RAVE_OBJECT_RELEASE(elevation);
  RAVE_OBJECT_RELEASE(field);
  return result;
}

int AcqvaFeatureMap_addField(AcqvaFeatureMap_t* self, AcqvaFeatureMapField_t* field)
{
  AcqvaFeatureMapElevation_t* elevation = NULL;
  int result = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (field == NULL) {
    RAVE_ERROR0("Can not add NULL field");
    goto fail;
  }

  elevation = AcqvaFeatureMap_createElevation(self, AcqvaFeatureMapField_getElangle(field));
  if (elevation == NULL || !AcqvaFeatureMapElevation_add(elevation, field)) {
    RAVE_ERROR0("Could not create elevation of add field to elevation");
    goto fail;
  }

  result = 1;
fail:
  RAVE_OBJECT_RELEASE(elevation);
  return result;
}

AcqvaFeatureMapElevation_t* AcqvaFeatureMap_createElevation(AcqvaFeatureMap_t* self, double elangle)
{
  AcqvaFeatureMapElevation_t *elevation = NULL, *result = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  elevation = AcqvaFeatureMap_findElevation(self, elangle);
  if (elevation == NULL) {
    elevation = RAVE_OBJECT_NEW(&AcqvaFeatureMapElevation_TYPE);
    if (elevation != NULL) {
      AcqvaFeatureMapElevation_setElangle(elevation, elangle);
      if (!RaveObjectList_add(self->elevations, (RaveCoreObject*)elevation)) {
        RAVE_ERROR0("Could not add elevation group to feature map");
        goto fail;
      }
    } else {
      RAVE_ERROR0("Could not create elevation group");
      goto fail;
    }
  }

  result = RAVE_OBJECT_COPY(elevation);
fail:
  RAVE_OBJECT_RELEASE(elevation);
  return result;
}

int AcqvaFeatureMap_addElevation(AcqvaFeatureMap_t* self, AcqvaFeatureMapElevation_t* elevation)
{
  int result = 0;

  AcqvaFeatureMapElevation_t *found = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  found = AcqvaFeatureMap_findElevation(self, AcqvaFeatureMapElevation_getElangle(elevation));
  if (found == NULL) {
    if (!RaveObjectList_add(self->elevations, (RaveCoreObject*)elevation)) {
      RAVE_ERROR0("Could not add elevation group to feature map");
      goto fail;
    }
  } else {
    RAVE_ERROR0("Elevation group already exists");
    goto fail;
  }
  result = 1;
fail:
  RAVE_OBJECT_RELEASE(found);
  return result;
}

int AcqvaFeatureMap_getNumberOfElevations(AcqvaFeatureMap_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveObjectList_size(self->elevations);
}

AcqvaFeatureMapElevation_t* AcqvaFeatureMap_getElevation(AcqvaFeatureMap_t* self, int index)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (AcqvaFeatureMapElevation_t*)RaveObjectList_get(self->elevations, index);
}

void AcqvaFeatureMap_removeElevation(AcqvaFeatureMap_t* self, int index)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RaveObjectList_release(self->elevations, index);
}


AcqvaFeatureMapField_t* AcqvaFeatureMap_findField(AcqvaFeatureMap_t* self, long nbins, long nrays, double elangle)
{
  AcqvaFeatureMapElevation_t *elevation = NULL;
  AcqvaFeatureMapField_t* result = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");

  elevation = AcqvaFeatureMap_findElevation(self, elangle);
  if (elevation != NULL) {
    result = AcqvaFeatureMapElevation_find(elevation, nbins, nrays);
  }
  RAVE_OBJECT_RELEASE(elevation);
  return result;
}

AcqvaFeatureMapElevation_t* AcqvaFeatureMap_findElevation(AcqvaFeatureMap_t* self, double elangle)
{
  int ngroups = 0, i = 0;
  AcqvaFeatureMapElevation_t *result = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  ngroups = RaveObjectList_size(self->elevations);
  for (i = 0; result == NULL && i < ngroups; i++) {
    AcqvaFeatureMapElevation_t* group = (AcqvaFeatureMapElevation_t*)RaveObjectList_get(self->elevations, i);
    if (fabs(group->elangle - elangle) < 1e-4) {
      result = RAVE_OBJECT_COPY(group);
    }
    RAVE_OBJECT_RELEASE(group);
  }
  return result;
}
/*@} End of AcqvaFeatureMap public methods */

/*@{ AcqvaFeatureMapElevation public methods */
int AcqvaFeatureMapElevation_setElangle(AcqvaFeatureMapElevation_t* self, double elangle)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->elangle = elangle;
  return 1;
}

double AcqvaFeatureMapElevation_getElangle(AcqvaFeatureMapElevation_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->elangle;
}

int AcqvaFeatureMapElevation_add(AcqvaFeatureMapElevation_t* self, AcqvaFeatureMapField_t* field)
{
  RAVE_ASSERT((self != NULL), "self == NULL");

  if (field == NULL || field->data == NULL) {
    RAVE_ERROR0("Must provide a field with data set");
    return 0;
  }

  if (fabs(self->elangle - field->elangle) >= 1e-4) {
    RAVE_ERROR0("Not same elevation angle");
    return 0;
  }

  if (field != NULL) {
    AcqvaFeatureMapField_t* alreadyField = AcqvaFeatureMapElevation_find(self, field->nbins, field->nrays);
    if (alreadyField != NULL) {
      RAVE_ERROR2("Field with dimension %ld x %ld  already exists", field->nbins, field->nrays);
      RAVE_OBJECT_RELEASE(alreadyField);
      return 0;
    }
  }

  if (!RaveObjectList_add(self->fields, (RaveCoreObject*)field)) {
    return 0;
  }

  return 1;
}

int AcqvaFeatureMapElevation_size(AcqvaFeatureMapElevation_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveObjectList_size(self->fields);
}

AcqvaFeatureMapField_t* AcqvaFeatureMapElevation_get(AcqvaFeatureMapElevation_t* self, int index)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (AcqvaFeatureMapField_t*)RaveObjectList_get(self->fields, index);
}

void AcqvaFeatureMapElevation_remove(AcqvaFeatureMapElevation_t* self, int index)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RaveObjectList_release(self->fields, index);
}

AcqvaFeatureMapField_t* AcqvaFeatureMapElevation_find(AcqvaFeatureMapElevation_t* self, long nbins, long nrays)
{
  int nfields = 0, i = 0;
  AcqvaFeatureMapField_t* result = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  nfields = RaveObjectList_size(self->fields);
  for (i = 0; result == NULL && i < nfields; i++) {
    AcqvaFeatureMapField_t* field = (AcqvaFeatureMapField_t*)RaveObjectList_get(self->fields, i);
    if (field->nbins == nbins && field->nrays == nrays) {
      result = RAVE_OBJECT_COPY(field);
    }
    RAVE_OBJECT_RELEASE(field);
  }
  return result;
}

int AcqvaFeatureMapElevation_has(AcqvaFeatureMapElevation_t* self, long nbins, long nrays)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  AcqvaFeatureMapField_t* field = NULL;
  field = AcqvaFeatureMapElevation_find(self, nbins, nrays);
  if (field != NULL) {
    result = 1;
  }
  RAVE_OBJECT_RELEASE(field);
  return result;
}

/*@} End of AcqvaFeatureMapElevation public methods */

/*@{ AcqvaFeatureMapField public methods */
int AcqvaFeatureMapField_setElangle(AcqvaFeatureMapField_t* self, double elangle)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->elangle = elangle;
  return 1;
}

double AcqvaFeatureMapField_getElangle(AcqvaFeatureMapField_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->elangle;
}

long AcqvaFeatureMapField_getNbins(AcqvaFeatureMapField_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->nbins;
}

long AcqvaFeatureMapField_getNrays(AcqvaFeatureMapField_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->nrays;
}

RaveDataType AcqvaFeatureMapField_getDatatype(AcqvaFeatureMapField_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveData2D_getType(self->data);
}

int AcqvaFeatureMapField_createData(AcqvaFeatureMapField_t* self, long nbins, long nrays, RaveDataType type)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "scanparam == NULL");

  result = RaveData2D_createData(self->data, nbins, nrays, type, 0.0);
  if (result) {
    self->nbins = nbins;
    self->nrays = nrays;
  }
  return result;
}

int AcqvaFeatureMapField_setData(AcqvaFeatureMapField_t* self, long nbins, long nrays, void* data, RaveDataType type)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  result = RaveData2D_setData(self->data, nbins, nrays, data, type);
  if (result) {
    self->nbins = nbins;
    self->nrays = nrays;
  }
  return result;
}

void* AcqvaFeatureMapField_getData(AcqvaFeatureMapField_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveData2D_getData(self->data);
}

int AcqvaFeatureMapField_fill(AcqvaFeatureMapField_t* self, double value)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveData2D_fill(self->data, value);
}

int AcqvaFeatureMapField_setValue(AcqvaFeatureMapField_t* self, int bin, int ray, double v)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveData2D_setValue(self->data, bin, ray, v);
}

int AcqvaFeatureMapField_getValue(AcqvaFeatureMapField_t* self, int bin, int ray, double* v)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (v != NULL) {
    double value = 0.0;
    result = RaveData2D_getValue(self->data, bin, ray, &value);
    if (result) {
      *v = value;
    }
  }
  return result;
}

AcqvaFeatureMapField_t* AcqvaFeatureMapField_createField(long nbins, long nrays, RaveDataType type, double elangle)
{
  AcqvaFeatureMapField_t *field = NULL, *result = NULL;

  field = RAVE_OBJECT_NEW(&AcqvaFeatureMapField_TYPE);
  if (field != NULL) {
    if (!AcqvaFeatureMapField_createData(field, nbins, nrays, type)) {
      RAVE_ERROR0("Could not create data field");
      goto fail;
    }
    if (!AcqvaFeatureMapField_setElangle(field, elangle)) {
      RAVE_ERROR0("Could not set elangle in created field");
      goto fail;
    }
  }
  result = RAVE_OBJECT_COPY(field);
fail:
  RAVE_OBJECT_RELEASE(field);
  return result;
}

/*@} End of AcqvaFeatureMapElevation public methods */

RaveCoreObjectType AcqvaFeatureMap_TYPE = {
    "AcqvaFeatureMap",
    sizeof(AcqvaFeatureMap_t),
    AcqvaFeatureMap_constructor,
    AcqvaFeatureMap_destructor,
    AcqvaFeatureMap_copyconstructor
};

RaveCoreObjectType AcqvaFeatureMapElevation_TYPE = {
    "AcqvaFeatureMapElevation",
    sizeof(AcqvaFeatureMapElevation_t),
    AcqvaFeatureMapElevation_constructor,
    AcqvaFeatureMapElevation_destructor,
    AcqvaFeatureMapElevation_copyconstructor
};

RaveCoreObjectType AcqvaFeatureMapField_TYPE = {
    "AcqvaFeatureMapField",
    sizeof(AcqvaFeatureMapField_t),
    AcqvaFeatureMapField_constructor,
    AcqvaFeatureMapField_destructor,
    AcqvaFeatureMapField_copyconstructor
};
