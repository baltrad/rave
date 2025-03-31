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
 * Provides functionality for creating composites according to the acqva method.
 * @file
 * @deprecated
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2024-01-17
 */
#include "acqva.h"
#include "cartesianparam.h"
#include "polarscan.h"
#include "polarvolume.h"
#include "rave_object.h"
#include "rave_types.h"
#include "raveobject_list.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_datetime.h"
#include "projection_pipeline.h"
#include <string.h>
#include "rave_field.h"
#include <float.h>
#include <stdio.h>
#include <math.h>


/**
 * Represents the cartesian product.
 */
struct _Acqva_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveList_t* parameters; /**< the parameters to generate */
  RaveDateTime_t* datetime;  /**< the date and time */
  RaveList_t* objectList;
};

/**
 * Keeps track of what object has what radar index value.
 */
typedef struct AcqvaCompositeRadarItem {
  RaveCoreObject* object;
  int radarIndexValue;
} AcqvaCompositeRadarItem_t;

/**
 * Structure for keeping track on parameters that should be composited.
 */
typedef struct AcqvaCompositingParameter_t {
  char* name;      /**< quantity */
  double gain;     /**< gain to be used in composite data*/
  double offset;   /**< offset to be used in composite data*/
} AcqvaCompositingParameter_t;

/**
 * Structure for holding information regarding a specific position and values 
 * connected with it.
 */
typedef struct AcqvaCompositeValuePosition_t {
  PolarNavigationInfo navinfo;
  double value;       /**< value */
  double qivalue;     /**< quality index value */
  RaveValueType type; /**< value type */
  int valid;          /**< 1 if position valid, otherwise 0 */ 
} AcqvaCompositeValuePosition_t;

/**
 * Structure for keeping track on values / parameter
 */
typedef struct AcqvaCompositeValues_t {
  RaveValueType vtype; /**< value type */
  double value;       /**< value */
  double mindist;     /**< min distance */
  double radardist;   /**< distance to radar */
  int radarindex;     /**< radar index in list of radars */
  const char* name;   /**< name of quantity */
  PolarNavigationInfo navinfo; /**< the navigation info */
  CartesianParam_t* parameter; /**< the cartesian parameter */
} AcqvaCompositeValues_t;

/** The resolution to use for scaling the distance from pixel to used radar. */
/** By multiplying the values in the distance field by 2000, we get the value in unit meters. */
#define DISTANCE_TO_RADAR_RESOLUTION 2000.0

/** Same for height, scaled to 100 m resolution up to 25.5 km */
#define HEIGHT_RESOLUTION 100.0

/** The name of the task for specifying distance to radar */
#define DISTANCE_TO_RADAR_HOW_TASK "se.smhi.composite.distance.radar"

/** The name of the task for specifying height above sea level */
#define HEIGHT_ABOVE_SEA_HOW_TASK "se.smhi.composite.height.radar"

/** The name of the task for indexing the radars used */
#define RADAR_INDEX_HOW_TASK "se.smhi.composite.index.radar"

/*@{ Private functions */
/**
 * Creates a parameter that should be composited
 * @param[in] name - quantity
 * @param[in] gain - gain
 * @param[in] offset - offset
 * @return the parameter or NULL on failure
 */
static AcqvaCompositingParameter_t* AcqvaInternal_createParameter(const char* name, double gain, double offset)
{
  AcqvaCompositingParameter_t* result = NULL;
  if (name != NULL) {
    result = RAVE_MALLOC(sizeof(AcqvaCompositingParameter_t));
    if (result != NULL) {
      result->name = RAVE_STRDUP(name);
      result->gain = gain;
      result->offset = offset;
      if (result->name == NULL) {   
        RAVE_FREE(result);
        result = NULL;
      }
    }
  }
  return result;
}

/**
 * Frees the parameter including its members
 * @param[in] p - the parameter to release
 */
static void AcqvaInternal_freeParameter(AcqvaCompositingParameter_t* p)
{
  if (p != NULL) {
    RAVE_FREE(p->name);
    RAVE_FREE(p);
  }
}

/**
 * Releases the complete parameter list including its items
 * @param[in] p - a pointer to the parameter list
 */
static void AcqvaInternal_freeParameterList(RaveList_t** p)
{
  if (p != NULL && *p != NULL) {
    AcqvaCompositingParameter_t* cp = RaveList_removeLast(*p);
    while (cp != NULL) {
      AcqvaInternal_freeParameter(cp);
      cp = RaveList_removeLast(*p);
    }
    RAVE_OBJECT_RELEASE(*p);
  }
}

/**
 * Frees the radar item
 * @param[in] p - the radar item to release
 */
static void AcqvaInternal_freeRadarItem(AcqvaCompositeRadarItem_t* p)
{
  if (p != NULL) {
    RAVE_OBJECT_RELEASE(p->object);
    RAVE_FREE(p);
  }
}

/**
 * Frees the list of radar items
 * @param[in] p - the list to be released
 */
static void AcqvaInternal_freeObjectList(RaveList_t** p)
{
  if (p != NULL && *p != NULL) {
    AcqvaCompositeRadarItem_t* ri = RaveList_removeLast(*p);
    while (ri != NULL) {
      AcqvaInternal_freeRadarItem(ri);
      ri = RaveList_removeLast(*p);
    }
    RAVE_OBJECT_RELEASE(*p);
  }
}


/**
 * Clones a radar item
 * @param[in] p - item to clone
 * @return the clone or NULL on failure
 */
static AcqvaCompositeRadarItem_t* AcqvaInternal_cloneRadarItem(AcqvaCompositeRadarItem_t* p)
{
  AcqvaCompositeRadarItem_t* result = NULL;
  if (p != NULL) {
    result = RAVE_MALLOC(sizeof(AcqvaCompositeRadarItem_t));
    if (result != NULL) {
      result->object = RAVE_OBJECT_CLONE(p->object);
      result->radarIndexValue = p->radarIndexValue;
      if (result->object == NULL) {
        RAVE_FREE(result);
        result = NULL;
      }
    }
  }
  return result;
}

/**
 * Clones a parameter list
 * @param[in] p - the parameter list to clone
 * @return the clone or NULL on failure
 */
static RaveList_t* AcqvaInternal_cloneRadarItemList(RaveList_t* p)
{
  int len = 0, i = 0;
  RaveList_t *result = NULL, *clone = NULL;;
  if (p != NULL) {
    clone = RAVE_OBJECT_NEW(&RaveList_TYPE);
    if (clone != NULL) {
      len = RaveList_size(p);
      for (i = 0; i < len; i++) {
        AcqvaCompositeRadarItem_t* cp = RaveList_get(p, i);
        AcqvaCompositeRadarItem_t* cpclone = AcqvaInternal_cloneRadarItem(cp);
        if (cpclone == NULL || !RaveList_add(clone, cpclone)) {
          if (cpclone != NULL) {
            RAVE_OBJECT_RELEASE(cpclone->object);
            RAVE_FREE(cpclone);
          }
          goto done;
        }
      }
    }
  }

  result = RAVE_OBJECT_COPY(clone);
done:
  RAVE_OBJECT_RELEASE(clone);
  return result;
}

/**
 * Clones a parameter
 * @param[in] p - parameter to clone
 * @return the clone or NULL on failure
 */
static AcqvaCompositingParameter_t* AcqvaInternal_cloneParameter(AcqvaCompositingParameter_t* p)
{
  AcqvaCompositingParameter_t* result = NULL;
  if (p != NULL) {
    result = RAVE_MALLOC(sizeof(AcqvaCompositingParameter_t));
    if (result != NULL) {
      result->name = RAVE_STRDUP(p->name);
      result->gain = p->gain;
      result->offset = p->offset;
      if (result->name == NULL) {
        RAVE_FREE(result);
        result = NULL;
      }
    }
  }
  return result;
}

/**
 * Clones a parameter list
 * @param[in] p - the parameter list to clone
 * @return the clone or NULL on failure
 */
static RaveList_t* AcqvaInternal_cloneParameterList(RaveList_t* p)
{
  int len = 0, i = 0;
  RaveList_t *result = NULL, *clone = NULL;;
  if (p != NULL) {
    clone = RAVE_OBJECT_NEW(&RaveList_TYPE);
    if (clone != NULL) {
      len = RaveList_size(p);
      for (i = 0; i < len; i++) {
        AcqvaCompositingParameter_t* cp = RaveList_get(p, i);
        AcqvaCompositingParameter_t* cpclone = AcqvaInternal_cloneParameter(cp);
        if (cpclone == NULL || !RaveList_add(clone, cpclone)) {
          AcqvaInternal_freeParameter(cpclone);
          goto done;
        }
      }
    }
  }

  result = RAVE_OBJECT_COPY(clone);
done:
  RAVE_OBJECT_RELEASE(clone);
  return result;
}

/**
 * Verifies that the radar index mapping contains a mapping between string - long rave atttribute
 * @param[in] src - the mapping
 * @return 1 if ok, otherwise 0
 */
static int AcqvaInternal_verifyRadarIndexMapping(RaveObjectHashTable_t* src)
{
  RaveList_t* keys = NULL;
  RaveAttribute_t* attr = NULL;
  int result = 0;

  if (src == NULL) {
    goto done;
  }

  keys = RaveObjectHashTable_keys(src);
  if (keys != NULL) {
    int i;
    int nattrs = RaveList_size(keys);
    for (i = 0; i < nattrs; i++) {
      const char* key = (const char*)RaveList_get(keys, i);
      attr = (RaveAttribute_t*)RaveObjectHashTable_get(src, key);
      if (attr == NULL || !RAVE_OBJECT_CHECK_TYPE(attr, &RaveAttribute_TYPE) || RaveAttribute_getFormat(attr) != RaveAttribute_Format_Long) {
        RAVE_ERROR0("Could not handle radar index mapping, must be mapping between key - long attribute");
        goto done;
      }
      RAVE_OBJECT_RELEASE(attr);
    }
  }

  result = 1;
done:

  RaveList_freeAndDestroy(&keys);
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

/**
 * Gets the quality value at the specified position for the specified quantity and quality field.
 * @param[in] self - self
 * @param[in] obj - the object
 * @param[in] quantity - the quantity
 * @param[in] qualityField - the quality field
 * @param[in] nav - the navigation information
 * @param[out] value - the value
 * @return 1 on success or 0 if value not could be retrieved
 */
static int AcqvaInternal_getQualityValueAtPosition(
  Acqva_t* self,
  RaveCoreObject* obj,
  const char* quantity,
  const char* qualityField,
  PolarNavigationInfo* nav,
  double* value)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((nav != NULL), "nav == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  RAVE_ASSERT((qualityField != NULL), "qualityField == NULL");
  *value = 0.0;

  if (obj != NULL) {
    if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
      if (!PolarScan_getQualityValueAt((PolarScan_t*)obj, quantity, nav->ri, nav->ai, qualityField, 1, value)) {
        *value = 0.0;
      }
    } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
      if (!PolarVolume_getQualityValueAt((PolarVolume_t*)obj, quantity, nav->ei, nav->ri, nav->ai, qualityField, 1, value)) {
        *value = 0.0;
      }
    } else {
      RAVE_WARNING0("Unsupported object type");
      goto done;
    }
  }

  result = 1;
done:
  return result;
}


static int AcqvaInternal_addGainAndOffsetToField(RaveField_t* field, double gain, double offset) {
  RaveAttribute_t* gainattribute = NULL;
  int result = 0;

  RAVE_ASSERT((field != NULL), "field == NULL");

  gainattribute = RaveAttributeHelp_createDouble("what/gain", gain);
  if (gainattribute == NULL ||
      !RaveField_addAttribute(field, gainattribute)) {
    RAVE_ERROR0("Failed to create gain attribute for quality field");
    goto done;
  }
  RAVE_OBJECT_RELEASE(gainattribute);

  gainattribute = RaveAttributeHelp_createDouble("what/offset", offset);
  if (gainattribute == NULL ||
      !RaveField_addAttribute(field, gainattribute)) {
    RAVE_ERROR0("Failed to create offset attribute for quality field");
    goto done;
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(gainattribute);
  return result;

}

/**
 * Uses the navigation information of the value positions and fills all 
 * associated cartesian quality with the composite objects quality fields. If
 * there is more than one value position in the valuePositions-array, an 
 * an interpolation of the quality value will be performed, along the dimensions
 * defined in the interpolationDimensions-array.
 * 
 * @param[in] composite - self
 * @param[in] x - x coordinate
 * @param[in] y - y coordinate
 * @param[in] cvalues - the composite values
 * @param[in] interpolationDimensions - dimensions to perform interpolation in                      
 */
static void AcqvaInternal_fillQualityInformation(
  Acqva_t* self,
  int x,
  int y,
  AcqvaCompositeValues_t* cvalues)
{
  int nfields = 0, i = 0;
  const char* quantity;
  CartesianParam_t* param = NULL;
  double radardist = 0;
  int radarindex = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((cvalues != NULL), "cvalues == NULL");

  param = cvalues->parameter;
  radardist = cvalues->radardist;
  radarindex = cvalues->radarindex;

  nfields = CartesianParam_getNumberOfQualityFields(param);
  quantity = CartesianParam_getQuantity(param);

  for (i = 0; i < nfields; i++) {
    RaveField_t* field = NULL;
    RaveAttribute_t* attribute = NULL;
    char* name = NULL;
    double value = 0.0;

    field = CartesianParam_getQualityField(param, i);
    if (field != NULL) {
      attribute = RaveField_getAttribute(field, "how/task");
    }
    if (attribute != NULL) {
      RaveAttribute_getString(attribute, &name);
    }

    if (name != NULL) {
      RaveCoreObject* obj = Acqva_get(self, radarindex);
      if (obj != NULL) {
        if (strcmp(DISTANCE_TO_RADAR_HOW_TASK, name) == 0) {
          value = radardist/DISTANCE_TO_RADAR_RESOLUTION;
        } else if (strcmp(HEIGHT_ABOVE_SEA_HOW_TASK, name) == 0) {
          value = cvalues->navinfo.actual_height / HEIGHT_RESOLUTION;
        } else if (strcmp(RADAR_INDEX_HOW_TASK, name) == 0) {
          value = (double)Acqva_getRadarIndexValue(self, radarindex);
        } else {
          if (AcqvaInternal_getQualityValueAtPosition(self, obj, quantity, name, &cvalues->navinfo, &value)) {
            value = (value - ACQVA_QUALITY_FIELDS_OFFSET) / ACQVA_QUALITY_FIELDS_GAIN;
          }
        }
        RaveField_setValue(field, x, y, value);
      }
      RAVE_OBJECT_RELEASE(obj);
    }

    RAVE_OBJECT_RELEASE(field);
    RAVE_OBJECT_RELEASE(attribute);
  }
}


static RaveField_t* AcqvaInternal_createQualityField(char* howtaskstr, int xsize, int ysize, double gain, double offset) {
  RaveField_t* qfield = RAVE_OBJECT_NEW(&RaveField_TYPE);
  RaveAttribute_t* howtaskattribute = NULL;

  if (qfield == NULL) {
    RAVE_ERROR0("Failed to create quality field");
    goto error;
  }

  howtaskattribute = RaveAttributeHelp_createString("how/task", howtaskstr);
  if (howtaskattribute == NULL) {
    RAVE_ERROR0("Failed to create quality field (how/task attribute could not be created)");
    goto error;
  }

  if (!RaveField_addAttribute(qfield, howtaskattribute)) {
    RAVE_ERROR0("Failed to create how/task attribute for distance quality field");
    goto error;
  }
  RAVE_OBJECT_RELEASE(howtaskattribute);

  if (!AcqvaInternal_addGainAndOffsetToField(qfield, gain, offset)) {
    RAVE_ERROR0("Failed to add gain and offset attribute to quality field");
    goto error;
  }

  if(!RaveField_createData(qfield, xsize, ysize, RaveDataType_UCHAR)) {
    RAVE_ERROR0("Failed to create quality field");
    goto error;
  }

  return qfield;
error:
  RAVE_OBJECT_RELEASE(qfield);
  RAVE_OBJECT_RELEASE(howtaskattribute);
  return NULL;

}

static char* AcqvaInternal_getIdFromSource(const char* source, const char* id)
{
  char* result = NULL;
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

      result = RAVE_MALLOC(sizeof(char) * (len + 1));
      if (result != NULL) {
        strncpy(result, p, len);
        result[len] = '\0';
      }
    }
  }
  return result;
}

static char* AcqvaInternal_getAnyIdFromSource(const char* source)
{
  char* result = NULL;
  result = AcqvaInternal_getIdFromSource(source, "WMO:");
  if (result == NULL) {
    result = AcqvaInternal_getIdFromSource(source, "RAD:");
  }
  if (result == NULL) {
    result = AcqvaInternal_getIdFromSource(source, "NOD:");
  }
  if (result == NULL) {
    result = AcqvaInternal_getIdFromSource(source, "CMT:");
  }
  return result;
}

static int AcqvaInternal_concateStr(char** ids, int* len, const char* str)
{
  int result = 0;
  char* p = *ids;
  int n = *len;
  int currStrLen = strlen(p);
  if (currStrLen + strlen(str) + 1 > n) {
    int newsize = n + strlen(str) + 1;
    char* newp = RAVE_REALLOC(p, newsize * sizeof(char));
    if (newp != NULL) {
      p = newp;
      n = newsize;
    } else {
      goto done;
    }
  }
  strcat(p, str);

  *ids = p;
  *len = n;
  result = 1;
done:
  return result;
}

static int AcqvaInternal_concateInt(char** ids, int* len, int value)
{
  char buff[16];
  memset(buff, 0, sizeof(char)*16);
  snprintf(buff, 16, "%d", value);
  return AcqvaInternal_concateStr(ids, len, buff);
}

static int AcqvaInternal_addNodeIdsToFieldHowTaskArgs(Acqva_t* self, RaveField_t* field)
{
  int i = 0, n = 0;
  char* ids = NULL;
  int idsLength = 0;
  int result = 0;
  RaveAttribute_t* howTaskArgs = NULL;
  char* srcid = NULL;
  RaveCoreObject* obj = NULL;
  n = Acqva_getNumberOfObjects(self);
  /* We assume that length of ids is <nr radars> * 10 (WMO-number and a ':', followed by a 3-digit number and finally a ',') */
  idsLength = n * 10 + 1;
  ids = RAVE_MALLOC(sizeof(char) * idsLength);
  if (ids == NULL) {
    return 0;
  }
  memset(ids, 0, sizeof(char)*idsLength);

  for (i = 0; i < n; i++) {
    obj = Acqva_get(self, i);
    srcid = NULL;
    if (obj != NULL) {
      if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
        const char* source = PolarScan_getSource((PolarScan_t*)obj);
        srcid = AcqvaInternal_getAnyIdFromSource(source);
      } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
        const char* source = PolarVolume_getSource((PolarVolume_t*)obj);
        srcid = AcqvaInternal_getAnyIdFromSource(source);
      }

      if (srcid != NULL) {
        if (!AcqvaInternal_concateStr(&ids, &idsLength, srcid)) {
          goto done;
        }
      } else {
        if (!AcqvaInternal_concateStr(&ids, &idsLength, "Unknown")) {
          goto done;
        }
      }
      if (!AcqvaInternal_concateStr(&ids, &idsLength, ":")) {
        goto done;
      }
      if (!AcqvaInternal_concateInt(&ids, &idsLength, Acqva_getRadarIndexValue(self, i))) {
        goto done;
      }
      if (i < n-1) {
        if (!AcqvaInternal_concateStr(&ids, &idsLength,",")) {
          goto done;
        }
      }
    }
    RAVE_FREE(srcid);
    RAVE_OBJECT_RELEASE(obj);
  }

  howTaskArgs = RaveAttributeHelp_createString("how/task_args", ids);
  if (howTaskArgs == NULL) {
    goto done;
  }
  if (!RaveField_addAttribute(field, howTaskArgs)) {
    goto done;
  }

  result = 1;
done:
  RAVE_FREE(srcid);
  RAVE_OBJECT_RELEASE(obj);
  RAVE_FREE(ids);
  RAVE_OBJECT_RELEASE(howTaskArgs);
  return result;
}

/**
 * Adds quality flags to the composite.
 * @apram[in] self - self
 * @param[in] image - the image to add quality flags to
 * @param[in] qualityflags - a list of strings identifying the how/task value in the quality fields
 * @return 1 on success otherwise 0
 */
static int AcqvaInternal_addQualityFlags(Acqva_t* self, Cartesian_t* image, RaveList_t* qualityflags)
{
  int result = 0;
  int nqualityflags = 0;
  RaveField_t* field = NULL;
  CartesianParam_t* param = NULL;
  RaveList_t* paramNames = NULL;

  int xsize = 0, ysize = 0;
  int i = 0, j = 0;
  int nparam = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((image != NULL), "image == NULL");

  xsize = Cartesian_getXSize(image);
  ysize = Cartesian_getYSize(image);

  paramNames = Cartesian_getParameterNames(image);
  if (paramNames == NULL) {
    goto done;
  }

  nparam = RaveList_size(paramNames);

  if (qualityflags != NULL) {
    nqualityflags = RaveList_size(qualityflags);
  }

  for (i = 0; i < nqualityflags; i++) {
    char* howtaskvaluestr = (char*)RaveList_get(qualityflags, i);
    double gain = 1.0, offset = 0.0;

    if (strcmp(DISTANCE_TO_RADAR_HOW_TASK, howtaskvaluestr) == 0) {
      gain = DISTANCE_TO_RADAR_RESOLUTION;
      offset = 0.0;
    } else if (strcmp(HEIGHT_ABOVE_SEA_HOW_TASK, howtaskvaluestr) == 0) {
      gain = HEIGHT_RESOLUTION;
      offset = 0.0;
    } else if (strcmp(RADAR_INDEX_HOW_TASK, howtaskvaluestr) == 0) {
      gain = 1.0;
      offset = 0.0;
    } else {
      // set the same, fixed gain and offset that is used for all quality fields (except distance) in the composite
      gain = ACQVA_QUALITY_FIELDS_GAIN;
      offset = ACQVA_QUALITY_FIELDS_OFFSET;
    }

    field = AcqvaInternal_createQualityField(howtaskvaluestr, xsize, ysize, gain, offset);

    if (strcmp(RADAR_INDEX_HOW_TASK, howtaskvaluestr)==0) {
      AcqvaInternal_addNodeIdsToFieldHowTaskArgs(self, field);
    }

    if (field != NULL) {
      for (j = 0; j < nparam; j++) {
        param = Cartesian_getParameter(image, (const char*)RaveList_get(paramNames, j));
        if (param != NULL) {
          RaveField_t* cfield = RAVE_OBJECT_CLONE(field);
          if (cfield == NULL ||
              !CartesianParam_addQualityField(param, cfield)) {
            RAVE_OBJECT_RELEASE(cfield);
            RAVE_ERROR0("Failed to add quality field");
            goto done;
          }
          RAVE_OBJECT_RELEASE(cfield);
        }
        RAVE_OBJECT_RELEASE(param);
      }
    } else {
      RAVE_WARNING1("Could not create quality field for: %s", howtaskvaluestr);
    }

    RAVE_OBJECT_RELEASE(field);
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(field);
  RAVE_OBJECT_RELEASE(param);
  RaveList_freeAndDestroy(&paramNames);
  return result;
}

/**
 * Returns a pointer to the internall stored parameter in the composite.
 * @param[in] composite - composite
 * @param[in] quantity - the parameter
 * @return the found parameter or NULL if not found
 */
static AcqvaCompositingParameter_t* AcqvaInternal_getParameterByName(Acqva_t* self, const char* quantity)
{
  int len = 0, i = 0;
  if (quantity != NULL) {
    len = RaveList_size(self->parameters);
    for (i = 0; i < len; i++) {
      AcqvaCompositingParameter_t* cp = RaveList_get(self->parameters, i);
      if (strcmp(cp->name, quantity) == 0) {
        return cp;
      }
    }
  }
  return NULL;
}
/**
 * Constructor.
 * @param[in] obj - the created object
 */
static int Acqva_constructor(RaveCoreObject* obj)
{
  Acqva_t* this = (Acqva_t*)obj;
  this->parameters = NULL;
  this->objectList = RAVE_OBJECT_NEW(&RaveList_TYPE);
  this->datetime = RAVE_OBJECT_NEW(&RaveDateTime_TYPE);
  this->parameters = RAVE_OBJECT_NEW(&RaveList_TYPE);

  if (this->objectList == NULL || this->parameters == NULL || this->datetime == NULL) {
    goto error;
  }
  return 1;
error:
   AcqvaInternal_freeParameterList(&this->parameters);
   AcqvaInternal_freeObjectList(&this->objectList);
  RAVE_OBJECT_RELEASE(this->datetime);
  return 0;
}

/**
 * Copy constructor.
 * @param[in] obj - the created object
 * @param[in] srcobj - the source (that is copied)
 */
static int Acqva_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  Acqva_t* this = (Acqva_t*)obj;
  Acqva_t* src = (Acqva_t*)srcobj;
  this->parameters = AcqvaInternal_cloneParameterList(src->parameters);
  this->objectList = AcqvaInternal_cloneRadarItemList(src->objectList);
  this->datetime = RAVE_OBJECT_CLONE(src->datetime);

  if (this->objectList == NULL || this->datetime == NULL || this->parameters == NULL) {
    goto error;
  }

  return 1;
error:
  AcqvaInternal_freeParameterList(&this->parameters);
  AcqvaInternal_freeObjectList(&this->objectList);
  RAVE_OBJECT_RELEASE(this->datetime);
  return 0;
}

/**
 * Destructor
 * @param[in] obj - the object to destroy
 */
static void Acqva_destructor(RaveCoreObject* obj)
{
  Acqva_t* this = (Acqva_t*)obj;
  AcqvaInternal_freeObjectList(&this->objectList);
  RAVE_OBJECT_RELEASE(this->datetime);
  AcqvaInternal_freeParameterList(&this->parameters);
}

/**
 * Tries to find the next available integer that is not filtered by indexes.
 * @param[in] indexes - filter of already used integers
 * @param[in] n_objs - number of indexes
 * @param[in] available - the integer that should be tested for availability
 * @return 1 if already exists, otherwise 0
 */
static int AcqvaInternal_containsRadarIndex(int* indexes, int n_objs, int available)
{
  int i = 0;
  for (i = 0; i < n_objs; i++) {
    if (indexes[i] == available) {
      return 1;
    }
  }
  return 0;
}

static char* AcqvaInternal_getTypeAndIdFromSource(const char* source, const char* id)
{
  char* result = NULL;
  if (source != NULL && id != NULL) {
    char* p = strstr(source, id);
    if (p != NULL) {
      int len = 0;
      char* pbrk = NULL;
      len = strlen(p);
      pbrk = strpbrk((const char*)p, ",");

      if (pbrk != NULL) {
        len = pbrk - p;
      }
      result = RAVE_MALLOC(sizeof(char) * (len + 1));
      if (result != NULL) {
        strncpy(result, p, len);
        result[len] = '\0';
      }
    }
  }
  return result;
}

/**
 * Returns the next available integer with a filter of already aquired indexes.
 * We assume that we always want to index radars from 1-gt;N. Which means that the first time
 * you call this function you should specify lastIndex = 0, then the subsequent calls the
 * index returned from this function should be passed in the next iteration.
 *
 * I.e.
 * lastIndex = 0
 * lastIndex = AcqvaInternal_nextAvailableRadarIndexValue(indexes, n_objs, lastIndex);
 * lastIndex = AcqvaInternal_nextAvailableRadarIndexValue(indexes, n_objs, lastIndex);
 * and so on.
 *
 */
static int AcqvaInternal_nextAvailableRadarIndexValue(int* indexes, int n_objs, int lastIndex)
{
  int ctr = lastIndex + 1;
  while(AcqvaInternal_containsRadarIndex(indexes, n_objs, ctr)) {
    ctr++;
  }
  return ctr;
}

/**
 * Makes sure that all objects used in the composite gets a unique value.
 */
static int AcqvaInternal_updateRadarIndexes(Acqva_t* self, RaveObjectHashTable_t* mapping)
{
  int n_objs = 0, i = 0;
  int result = 0;
  int* indexes = NULL;
  int lastIndex = 0;

  if (!AcqvaInternal_verifyRadarIndexMapping(mapping)) {
    goto done;
  }

  /* First reset indexes */
  n_objs = RaveList_size(self->objectList);
  indexes = RAVE_MALLOC(sizeof(int) * n_objs);
  if (indexes == NULL) {
    goto done;
  }
  memset(indexes, 0, sizeof(int)*n_objs);

  for (i = 0; i < n_objs; i++) {
    AcqvaCompositeRadarItem_t* ri = (AcqvaCompositeRadarItem_t*)RaveList_get(self->objectList, i);
    char* src = NULL;
    ri->radarIndexValue = 0;

    if (RAVE_OBJECT_CHECK_TYPE(ri->object, &PolarVolume_TYPE)) {
      src = (char*)PolarVolume_getSource((PolarVolume_t*)ri->object);
    } else if (RAVE_OBJECT_CHECK_TYPE(ri->object, &PolarScan_TYPE)) {
      src = (char*)PolarScan_getSource((PolarScan_t*)ri->object);
    }

    if (src != NULL) {
      char *str = NULL;
      str = AcqvaInternal_getTypeAndIdFromSource(src, "WMO:");
      if (str == NULL || !RaveObjectHashTable_exists(mapping, str)) {
        RAVE_FREE(str);
        str = AcqvaInternal_getTypeAndIdFromSource(src, "RAD:");
      }
      if (str == NULL || !RaveObjectHashTable_exists(mapping, str)) {
        RAVE_FREE(str);
        str = AcqvaInternal_getTypeAndIdFromSource(src, "NOD:");
      }

      if (str != NULL && RaveObjectHashTable_exists(mapping, str)) {
        RaveAttribute_t* attr = (RaveAttribute_t*)RaveObjectHashTable_get(mapping, str);
        long v = 0;
        if (RaveAttribute_getLong(attr, &v)) {
          ri->radarIndexValue = (int)v;
          indexes[i] = (int)v;
        }
        RAVE_OBJECT_RELEASE(attr);
      } else if (RaveObjectHashTable_exists(mapping, src)) {
        RaveAttribute_t* attr = (RaveAttribute_t*)RaveObjectHashTable_get(mapping, src);
        long v = 0;
        if (RaveAttribute_getLong(attr, &v)) {
          ri->radarIndexValue = (int)v;
          indexes[i] = (int)v;
        }
        RAVE_OBJECT_RELEASE(attr);
      }

      RAVE_FREE(str);
    }
  }

  /* Any radarIndexValue = 0, needs to get a suitable value, take first available one */
  for (i = 0; i < n_objs; i++) {
    AcqvaCompositeRadarItem_t* ri = (AcqvaCompositeRadarItem_t*)RaveList_get(self->objectList, i);
    if (ri->radarIndexValue == 0) {
      ri->radarIndexValue = lastIndex = AcqvaInternal_nextAvailableRadarIndexValue(indexes, n_objs, lastIndex);
    }
  }

  result = 1;
done:
  RAVE_FREE(indexes);
  return result;
}


/**
 * Creates the resulting composite image.
 * @param[in] self - self
 * @param[in] area - the area the composite image(s) should have
 * @returns the cartesian on success otherwise NULL
 */
static Cartesian_t* AcqvaInternal_createCompositeImage(Acqva_t* self, Area_t* area)
{
  Cartesian_t *result = NULL, *cartesian = NULL;
  int nparam = 0, i = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");

  cartesian = RAVE_OBJECT_NEW(&Cartesian_TYPE);
  if (cartesian == NULL) {
    goto done;
  }
  Cartesian_init(cartesian, area);

  nparam = Acqva_getParameterCount(self);
  if (nparam <= 0) {
    RAVE_ERROR0("You can not generate a composite without specifying at least one parameter");
    goto done;
  }

  Cartesian_setObjectType(cartesian, Rave_ObjectType_COMP);
  //Cartesian_setProduct(cartesian, "ACQVA"??);
  if (Acqva_getTime(self) != NULL) {
    if (!Cartesian_setTime(cartesian, Acqva_getTime(self))) {
      goto done;
    }
  }
  if (Acqva_getDate(self) != NULL) {
    if (!Cartesian_setDate(cartesian, Acqva_getDate(self))) {
      goto done;
    }
  }
  if (!Cartesian_setSource(cartesian, Area_getID(area))) {
    goto done;
  }

  for (i = 0; i < nparam; i++) {
    double gain = 0.0, offset = 0.0;
    const char* name = Acqva_getParameter(self, i, &gain, &offset);
    CartesianParam_t* cp = Cartesian_createParameter(cartesian, name, RaveDataType_UCHAR, 0);
    if (cp == NULL) {
      goto done;
    }
    CartesianParam_setNodata(cp, 255.0);
    CartesianParam_setUndetect(cp, 0.0);
    CartesianParam_setGain(cp, gain);
    CartesianParam_setOffset(cp, offset);
    RAVE_OBJECT_RELEASE(cp);
  }

  result = RAVE_OBJECT_COPY(cartesian);
done:
  RAVE_OBJECT_RELEASE(cartesian);
  return result;
}
/**
 * Returns the projection object that belongs to this obj.
 * @param[in] obj - the rave core object instance
 * @return the projection or NULL if there is no projection instance
 */
static Projection_t* AcqvaInternal_getProjection(RaveCoreObject* obj)
{
  Projection_t* result = NULL;
  if (obj != NULL) {
    if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
      result = PolarVolume_getProjection((PolarVolume_t*)obj);
    } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
      result = PolarScan_getProjection((PolarScan_t*)obj);
    }
  }
  return result;
}

/**
 * Creates an array of CompositeValues_t with length nparam.
 * @param[in] nparam - the number of items in the array
 * @returns the array on success or NULL on failure
 */
static AcqvaCompositeValues_t* AcqvaInternal_createCompositeValues(int nparam)
{
  AcqvaCompositeValues_t* result = NULL;
  if (nparam > 0) {
    result = RAVE_MALLOC(sizeof(AcqvaCompositeValues_t) * nparam);
    if (result == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for composite values");
    } else {
      memset(result, 0, sizeof(AcqvaCompositeValues_t) * nparam);
    }
  }
  return result;
}

/**
 * Resets the array of composite values except the CartesianParam parameter
 * and the dToRadar field.
 * @param[in] nparam - number of parameters
 * @param[in] p - pointer at the array
 */
static void AcqvaInternal_resetCompositeValues(Acqva_t* composite, int nparam, AcqvaCompositeValues_t* p)
{
  int i = 0;
  for (i = 0; i < nparam; i++) {
    p[i].mindist = 1e10;
    p[i].radarindex = -1;
    p[i].vtype = RaveValueType_NODATA;
    p[i].name = (const char*)((AcqvaCompositingParameter_t*)RaveList_get(composite->parameters, i))->name;
  }
}

/*@{ Interface functions */
int Acqva_add(Acqva_t* self, RaveCoreObject* object)
{
  AcqvaCompositeRadarItem_t* item = NULL;
  int result = 0;
  RAVE_ASSERT((self != NULL), "composite == NULL");
  RAVE_ASSERT((object != NULL), "object == NULL");

  if (!RAVE_OBJECT_CHECK_TYPE(object, &PolarVolume_TYPE)) {
    RAVE_ERROR0("Providing an object that not is a PolarVolume nor a PolarScan during acqva composite generation");
    return 0;
  }
  // if (!RAVE_OBJECT_CHECK_TYPE(object, &PolarVolume_TYPE) &&
  //     !RAVE_OBJECT_CHECK_TYPE(object, &PolarScan_TYPE)) {
  //   RAVE_ERROR0("Providing an object that not is a PolarVolume nor a PolarScan during acqva composite generation");
  //   return 0;
  // }
  item = RAVE_MALLOC(sizeof(AcqvaCompositeRadarItem_t));
  if (item != NULL) {
    item->object = RAVE_OBJECT_COPY(object);
    result = RaveList_add(self->objectList, item);
    if (result == 0) {
      RAVE_OBJECT_RELEASE(item->object);
      RAVE_FREE(item);
    }
    item->radarIndexValue = RaveList_size(self->objectList);
  }

  return result;
}

int Acqva_getNumberOfObjects(Acqva_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveList_size(self->objectList);
}

RaveCoreObject* Acqva_get(Acqva_t* self, int index)
{
  AcqvaCompositeRadarItem_t* item = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  item = (AcqvaCompositeRadarItem_t*)RaveList_get(self->objectList, index);
  if (item != NULL) {
    return RAVE_OBJECT_COPY(item->object);
  }
  return NULL;
}

int Acqva_getRadarIndexValue(Acqva_t* self, int index)
{
  AcqvaCompositeRadarItem_t* item = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  item = (AcqvaCompositeRadarItem_t*)RaveList_get(self->objectList, index);
  if (item != NULL) {
    return item->radarIndexValue;
  }
  return 0;
}

int Acqva_addParameter(Acqva_t* self, const char* quantity, double gain, double offset)
{
  int result = 0;
  AcqvaCompositingParameter_t* param = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");

  param = AcqvaInternal_getParameterByName(self, quantity);
  if (param != NULL) {
    param->gain = gain;
    param->offset = offset;
    result = 1;
  } else {
    param = AcqvaInternal_createParameter(quantity, gain, offset);
    if (param != NULL) {
      result = RaveList_add(self->parameters, param);
      if (!result) {
        AcqvaInternal_freeParameter(param);
      }
    }
  }
  return result;
}

int Acqva_hasParameter(Acqva_t* self, const char* quantity)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (quantity != NULL) {
    int i = 0;
    int len = RaveList_size(self->parameters);
    for (i = 0; result == 0 && i < len ; i++) {
      AcqvaCompositingParameter_t* s = RaveList_get(self->parameters, i);
      if (s != NULL && s->name != NULL && strcmp(quantity, s->name) == 0) {
        result = 1;
      }
    }
  }
  return result;
}

int Acqva_getParameterCount(Acqva_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveList_size(self->parameters);
}

const char* Acqva_getParameter(Acqva_t* self, int index, double* gain, double* offset)
{
  AcqvaCompositingParameter_t* param = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  param = RaveList_get(self->parameters, index);
  if (param != NULL) {
    if (gain != NULL) {
      *gain = param->gain;
    }
    if (offset != NULL) {
      *offset = param->offset;
    }
    return (const char*)param->name;
  }
  return NULL;
}

int Acqva_setTime(Acqva_t* self, const char* value)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveDateTime_setTime(self->datetime, value);
}

const char* Acqva_getTime(Acqva_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveDateTime_getTime(self->datetime);
}

int Acqva_setDate(Acqva_t* self, const char* value)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveDateTime_setDate(self->datetime, value);
}

const char* Acqva_getDate(Acqva_t* self)
{
  RAVE_ASSERT((self != NULL), "composite == NULL");
  return RaveDateTime_getDate(self->datetime);
}

int Acqva_applyRadarIndexMapping(Acqva_t* self, RaveObjectHashTable_t* mapping)
{
  RAVE_ASSERT((self != NULL), "self == NULL");

  return AcqvaInternal_updateRadarIndexes(self, mapping);
}

int Acqva_findLowestUsableValue(Acqva_t* self, PolarVolume_t* pvol, 
  double lon, double lat, const char* qfieldname, double* height, 
  double* elangle, int* ray, int* bin, int* eindex, PolarNavigationInfo* outnavinfo)
{
  int nrelevs = 0, i = 0, found = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (pvol == NULL) {
    RAVE_ERROR0("Providing pvol == NULL");
    return 0;
  }
  nrelevs = PolarVolume_getNumberOfScans(pvol);
  for (i = 0; !found && i < nrelevs; i++) {
    PolarNavigationInfo navinfo;
    PolarScan_t* scan = PolarVolume_getScan(pvol, i);
    if (PolarScan_getNearestNavigationInfo(scan, lon, lat, &navinfo)) {
      double v = 0.0;
      if (PolarScan_getQualityValueAt(scan, NULL, navinfo.ri, navinfo.ai, qfieldname, 1, &v)) {
        if (v != 0.0) {
          *height = navinfo.actual_height;
          *elangle = navinfo.elevation;
          *ray = navinfo.ai;
          *bin = navinfo.ri;
          *eindex = i;
          *outnavinfo = navinfo;
          found = 1;
        }
      }
    }
    RAVE_OBJECT_RELEASE(scan);
  }
  return found;
}

Cartesian_t* Acqva_generate(Acqva_t* self, Area_t* area, RaveList_t* qualityflags)
{
  Cartesian_t* result = NULL;
  Projection_t* projection = NULL;
  AcqvaCompositeValues_t* cvalues = NULL;
  RaveObjectList_t* pipelines = NULL;
  int x = 0, y = 0, i = 0, xsize = 0, ysize = 0, nradars = 0;
  int nqualityflags = 0;
  int nparam = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  
  if (area == NULL) {
    RAVE_ERROR0("Trying to generate composite with NULL area");
    goto fail;
  }

  nparam = Acqva_getParameterCount(self);
  if (nparam <= 0) {
    RAVE_ERROR0("You can not generate a composite without specifying at least one parameter");
    goto fail;
  }

  result =AcqvaInternal_createCompositeImage(self, area);
  if (result == NULL) {
    goto fail;
  }

  if ((cvalues = AcqvaInternal_createCompositeValues(nparam)) == NULL) {
    goto fail;
  }

  for (i = 0; i < nparam; i++) {
    const char* name = Acqva_getParameter(self, i, NULL, NULL);
    cvalues[i].parameter = Cartesian_getParameter(result, name); // Keep track on parameters
    if (cvalues[i].parameter == NULL) {
      RAVE_ERROR0("Failure in parameter handling\n");
      goto fail;
    }
  }

  xsize = Cartesian_getXSize(result);
  ysize = Cartesian_getYSize(result);
  projection = Cartesian_getProjection(result);
  nradars = Acqva_getNumberOfObjects(self);

  if (qualityflags != NULL) {
    nqualityflags = RaveList_size(qualityflags);
    if (!AcqvaInternal_addQualityFlags(self, result, qualityflags)) {
      goto fail;
    }
  }

  pipelines = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (pipelines == NULL) {
    goto fail;
  }
  for (i = 0; i < nradars; i++) {
    RaveCoreObject* obj = Acqva_get(self, i);
    if (obj != NULL) {
      Projection_t* objproj = AcqvaInternal_getProjection(obj);
      ProjectionPipeline_t* pipeline = NULL;
      if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) { 
        PolarVolume_sortByElevations((PolarVolume_t*)obj, 1);
      }
      if (objproj == NULL) {
        RAVE_OBJECT_RELEASE(obj);
        RAVE_ERROR0("No projection for object");
        goto fail;
      }
      pipeline = ProjectionPipeline_createPipeline(projection, objproj);
      RAVE_OBJECT_RELEASE(objproj);
      RAVE_OBJECT_RELEASE(obj);
      if (pipeline == NULL || !RaveObjectList_add(pipelines, (RaveCoreObject*)pipeline)) {
        RAVE_ERROR0("Failed to create pipeline");
        RAVE_OBJECT_RELEASE(pipeline);
        goto fail;
      }
      RAVE_OBJECT_RELEASE(pipeline);
    }
  }

  for (y = 0; y < ysize; y++) {
    double herey = Cartesian_getLocationY(result, y);
    for (x = 0; x < xsize; x++) {
      int cindex = 0;
      double herex = Cartesian_getLocationX(result, x);
      double olon = 0.0, olat = 0.0;
      int debug = (x>210 && x<240 && y>310 && y<340)?0:0;

      AcqvaInternal_resetCompositeValues(self, nparam, cvalues);

      for (i = 0; i < nradars; i++) {
        RaveCoreObject* obj = NULL;
        ProjectionPipeline_t* pipeline = NULL;
        obj = Acqva_get(self, i);
        if (obj != NULL) {
          pipeline = (ProjectionPipeline_t*)RaveObjectList_get(pipelines, i);
        }
        if (pipeline != NULL) {
          /* We will go from surface coords into the lonlat projection assuming that a polar volume uses a lonlat projection*/
          if (!ProjectionPipeline_fwd(pipeline, herex, herey, &olon, &olat)) {
            RAVE_WARNING0("Failed to transform from composite into polar coordinates");
          } else {
            double dist = 0.0;
            double maxdist = 0.0;

            if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
              dist = PolarVolume_getDistance((PolarVolume_t*)obj, olon, olat);
              maxdist = PolarVolume_getMaxDistance((PolarVolume_t*)obj);
            } else {
              RAVE_ERROR0("ACQVA currently only handles polar volumes");
              goto fail;
            }
            /* else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
              dist = PolarScan_getDistance((PolarScan_t*)obj, olon, olat);
              maxdist = PolarScan_getMaxDistance((PolarScan_t*)obj);
            }*/
            if (dist <= maxdist) {
              double height=0.0, elangle=0.0;
              int ray=0, bin=0, eindex=0;
              PolarNavigationInfo navinfo;
              if (Acqva_findLowestUsableValue(self, (PolarVolume_t*)obj, 
                olon, olat, "se.smhi.acqva", &height, &elangle, &ray, &bin, &eindex, &navinfo)) {
                for (cindex = 0; cindex < nparam; cindex++) {
                  RaveValueType otype = RaveValueType_NODATA;
                  double v = 0.0;
                  otype = PolarVolume_getConvertedParameterValueAt((PolarVolume_t*)obj, cvalues[cindex].name, eindex, bin, ray, &v);
                  if (debug) {
                     fprintf(stderr, "height=%f for radar=%d, v=%f, bin=%d, ray=%d, eindex=%d, value=%f\n", height, i, v, bin, ray, eindex, v);
                  }
                  if (otype != RaveValueType_NODATA) {
                    if (cvalues[cindex].mindist > height) {
                      cvalues[cindex].mindist = height;
                      cvalues[cindex].value = v;
                      cvalues[cindex].vtype = otype;
                      cvalues[cindex].navinfo = navinfo;
                      cvalues[cindex].radarindex = i;
                      cvalues[cindex].radardist = cvalues[cindex].navinfo.actual_range;
                    }
                  }
                }
              }
            }
          }
        }
        RAVE_OBJECT_RELEASE(pipeline);
        RAVE_OBJECT_RELEASE(obj);
      }

      for (cindex = 0; cindex < nparam; cindex++) {
        double vvalue = cvalues[cindex].value;
        int vtype = cvalues[cindex].vtype;
        CartesianParam_setConvertedValue(cvalues[cindex].parameter, x, y, vvalue, vtype);
        if ((vtype == RaveValueType_DATA || vtype == RaveValueType_UNDETECT) &&
            cvalues[cindex].radarindex >= 0 && nqualityflags > 0) {
          AcqvaInternal_fillQualityInformation(self, x, y, &cvalues[cindex]);
        }        
      }
    }
  }

  for (i = 0; cvalues != NULL && i < nparam; i++) {
     RAVE_OBJECT_RELEASE(cvalues[i].parameter);
  }
  RAVE_FREE(cvalues);
  RAVE_OBJECT_RELEASE(projection);
  RAVE_OBJECT_RELEASE(pipelines);
  return result;
fail:
  for (i = 0; cvalues != NULL && i < nparam; i++) {
    RAVE_OBJECT_RELEASE(cvalues[i].parameter);
  }
  RAVE_FREE(cvalues);
  RAVE_OBJECT_RELEASE(projection);
  RAVE_OBJECT_RELEASE(pipelines);
  RAVE_OBJECT_RELEASE(result);
  return result;
}

/*@} End of Interface functions */

RaveCoreObjectType Acqva_TYPE = {
    "Acqva",
    sizeof(Acqva_t),
    Acqva_constructor,
    Acqva_destructor,
    Acqva_copyconstructor
};