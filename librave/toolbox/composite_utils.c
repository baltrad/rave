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
 * Contains various utility functions when creating composites
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-0!-17
 */
#include "composite_utils.h"
#include "cartesian.h"
#include "cartesianvolume.h"
#include "compositearguments.h"
#include "projection_pipeline.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_types.h"
#include "polarvolume.h"
#include "polarscan.h"
#include "raveobject_hashtable.h"
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <limits.h>

/*@{ CompositeQualityFlagDefinition definition */
static int CompositeQualityFlagDefinition_constructor(RaveCoreObject* obj)
{
  CompositeQualityFlagDefinition_t* this = (CompositeQualityFlagDefinition_t*)obj;
  this->qualityFieldName = NULL;
  this->datatype = RaveDataType_UNDEFINED;
  this->gain = 1.0;
  this->offset = 0.0;
  return 1;
}
static int CompositeQualityFlagDefinition_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  CompositeQualityFlagDefinition_t* this = (CompositeQualityFlagDefinition_t*)obj;
  CompositeQualityFlagDefinition_t* src = (CompositeQualityFlagDefinition_t*)srcobj;
  this->qualityFieldName = NULL;
  this->datatype = src->datatype;
  this->gain = src->gain;
  this->offset = src->offset;
  if (src->qualityFieldName != NULL) {
    this->qualityFieldName = RAVE_STRDUP(src->qualityFieldName);
    if (this->qualityFieldName == NULL) {
      RAVE_ERROR0("Failed to duplicate quality field name");
      goto fail;
    }
  }
  return 1;
fail:
  RAVE_FREE(this->qualityFieldName);
  return 0;
}

static void CompositeQualityFlagDefinition_destructor(RaveCoreObject* obj)
{
  CompositeQualityFlagDefinition_t* this = (CompositeQualityFlagDefinition_t*)obj;
  RAVE_FREE(this->qualityFieldName);
}

RaveCoreObjectType CompositeQualityFlagDefinition_TYPE = {
  "CompositeQualityFlagDefinition",
  sizeof(CompositeQualityFlagDefinition_t),
  CompositeQualityFlagDefinition_constructor,
  CompositeQualityFlagDefinition_destructor,
  CompositeQualityFlagDefinition_copyconstructor
};

/**
 * Utility function to create and register a quality flag definition in the hash table of quality flag definitions.
 */
int CompositeUtils_registerQualityFlagDefinition(RaveObjectHashTable_t* qualityFlags, const char* qualityFieldName, RaveDataType datatype, double offset, double gain)
{
  CompositeQualityFlagDefinition_t* definition = NULL;
  int result = 0;

  if (qualityFieldName == NULL) {
    RAVE_ERROR0("Must specify quality field name to use this function");
    return 0;
  }
  if (qualityFlags == NULL) {
    RAVE_ERROR0("Must provide a qualityFlags instance to register the quality flag definition in");
    return 0;
  }

  definition = RAVE_OBJECT_NEW(&CompositeQualityFlagDefinition_TYPE);
  if (definition != NULL) {
    definition->datatype = datatype;
    definition->offset = offset;
    definition->gain = gain;
    definition->qualityFieldName = RAVE_STRDUP(qualityFieldName);
    if (definition->qualityFieldName == NULL) {
      RAVE_ERROR0("Failed to duplicate quality field name");
      goto fail;
    }
  }

  result = RaveObjectHashTable_put(qualityFlags, qualityFieldName, (RaveCoreObject*)definition);
fail:
  RAVE_OBJECT_RELEASE(definition);
  return result;
}

int CompositeUtils_registerQualityFlagDefinitionFromSettings(RaveObjectHashTable_t* qualityFlags, CompositeQualityFlagSettings_t* settings)
{
  int ctr = 0;
  if (qualityFlags == NULL) {
    RAVE_ERROR0("Must provide quality flags");
    return 0;
  }

  if (settings != NULL) {
    while (settings[ctr].qualityFieldName != NULL) {
      if (!CompositeUtils_registerQualityFlagDefinition(qualityFlags, settings[ctr].qualityFieldName, settings[ctr].datatype, settings[ctr].offset, settings[ctr].gain)) {
        RAVE_ERROR0("Could not register quality flag definitions");
        goto fail;
      }
      ctr++;
    }
  }
  return 1;
fail:
  return 0;
}
/*@} End of CompositeQualityFlagSettings definition */
 

int CompositeUtils_isValidCartesianArguments(CompositeArguments_t* arguments)
{
  Area_t* area = NULL;
  int result = 0;
  if (arguments == NULL) {
    RAVE_ERROR0("No arguments provided");
    return 0;
  }
  if (CompositeArguments_getParameterCount(arguments) <= 0) {
    RAVE_ERROR0("Must at least provide one parameter to create a cartesian product");
    return 0;
  }
  area = CompositeArguments_getArea(arguments);
  if (area != NULL) {
    result = 1;
  }
  RAVE_OBJECT_RELEASE(area);
  return result;
}

Cartesian_t* CompositeUtils_createCartesianFromArguments(CompositeArguments_t* arguments)
{
  Area_t* area = NULL;
  Cartesian_t *cartesian = NULL, *result = NULL;
  int nparam = 0, i = 0;

  if (arguments == NULL) {
    RAVE_ERROR0("Must provide arguments when creating a cartesian");
    return NULL;
  }

  if (!CompositeUtils_isValidCartesianArguments(arguments)) {
    RAVE_ERROR0("Validate arguments before atempting to create a cartesian");
    return NULL;
  }

  area = CompositeArguments_getArea(arguments);

  cartesian = RAVE_OBJECT_NEW(&Cartesian_TYPE);
  if (cartesian == NULL) {
    goto done;
  }
  Cartesian_init(cartesian, area);
  Cartesian_setObjectType(cartesian, Rave_ObjectType_COMP);
  Cartesian_setProduct(cartesian, CompositeArguments_getProductType(arguments));

  if (CompositeArguments_getTime(arguments) != NULL) {
    if (!Cartesian_setTime(cartesian, CompositeArguments_getTime(arguments))) {
      goto done;
    }
  }
  if (CompositeArguments_getDate(arguments) != NULL) {
    if (!Cartesian_setDate(cartesian, CompositeArguments_getDate(arguments))) {
      goto done;
    }
  }
  if (!Cartesian_setSource(cartesian, Area_getID(area))) {
    goto done;
  }

  nparam = CompositeArguments_getParameterCount(arguments);
  for (i = 0; i < nparam; i++) {
    double gain = 0.0, offset = 0.0, nodata = 0.0, undetect = 0.0;
    RaveDataType datatype = RaveDataType_UNDEFINED;
    const char* name = CompositeArguments_getParameterAtIndex(arguments, i, &gain, &offset, &datatype, &nodata, &undetect);
    CartesianParam_t* cp = Cartesian_createParameter(cartesian, name, datatype, nodata);
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
  RAVE_OBJECT_RELEASE(area);
  return result;
}

CompositeUtilValue_t* CompositeUtils_createCompositeValues(CompositeArguments_t* arguments, Cartesian_t* cartesian, int* nentries)
{
  CompositeUtilValue_t* result = NULL;
  int nparam = 0;
  if (arguments == NULL) {
    RAVE_ERROR0("Must provide arguments");
    return NULL;
  }
  nparam = CompositeArguments_getParameterCount(arguments);
  if (nparam > 0) {
    int i = 0;

    result = RAVE_MALLOC(sizeof(CompositeUtilValue_t) * nparam);
    if (result == NULL) {
      RAVE_CRITICAL0("Failed to allocate memory for composite values");
    } else {
      memset(result, 0, sizeof(CompositeUtilValue_t) * nparam);
    }

    for (i = 0; i < nparam; i++) {
      const char* quantity = CompositeArguments_getParameterName(arguments, i);
      CartesianParam_t* parameter = Cartesian_getParameter(cartesian, quantity);
      if (parameter == NULL) {
        RAVE_ERROR0("Failed to get parameter from cartesian.");
        CompositeUtils_freeCompositeValueParameters(&result, nparam);
        return NULL;
      }
      result[i].parameter = parameter;
      result[i].name = (const char*)CompositeArguments_getParameterName(arguments, i);
    }
    *nentries = nparam;
  }
  return result;  
}

void CompositeUtils_resetCompositeValues(CompositeArguments_t* arguments, CompositeUtilValue_t* cvalues, int nentries)
{
  if (cvalues != NULL) {
    int i = 0;
    for (i = 0; i < nentries; i++) {
      cvalues[i].mindist = 1e10;
      cvalues[i].radarindex = -1;
      cvalues[i].vtype = RaveValueType_NODATA;
    }
  }
}


void CompositeUtils_freeCompositeValueParameters(CompositeUtilValue_t** cvalues, int nparam)
{
  if (cvalues != NULL && *cvalues != NULL) {
    int i = 0;
    for (i = 0; i < nparam; i++) {
      RAVE_OBJECT_RELEASE((*cvalues)[i].parameter);
    }
    RAVE_FREE(*cvalues);
    *cvalues = NULL;
  }
}

Projection_t* CompositeUtils_getProjection(RaveCoreObject* obj)
{
  Projection_t* result = NULL;
  if (obj != NULL) {
    if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
      result = PolarVolume_getProjection((PolarVolume_t*)obj);
    } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
      result = PolarScan_getProjection((PolarScan_t*)obj);
    } else if (RAVE_OBJECT_CHECK_TYPE(obj, &Cartesian_TYPE)) {
      result = Cartesian_getProjection((Cartesian_t*)obj);
    } else if (RAVE_OBJECT_CHECK_TYPE(obj, &CartesianVolume_TYPE)) {
      result = CartesianVolume_getProjection((CartesianVolume_t*)obj);
    }
  }
  return result;
}

int CompositeUtils_getObjectSource(RaveCoreObject* obj, char* source, int nlen)
{
  int result = 0;
  if (obj != NULL) {
    memset(source, 0, sizeof(char)*nlen);
    if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
      const char* objsrc = PolarVolume_getSource((PolarVolume_t*)obj);
      if (objsrc != NULL && strlen(objsrc) < nlen - 1) {
        strncpy(source, objsrc, nlen);
        result = strlen(source);
      }
     } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
      const char* objsrc = PolarScan_getSource((PolarScan_t*)obj);
      if (objsrc != NULL && strlen(objsrc) < nlen - 1) {
        strncpy(source, objsrc, nlen);
        result = strlen(source);
      }
    } else if (RAVE_OBJECT_CHECK_TYPE(obj, &Cartesian_TYPE)) {
      const char* objsrc = Cartesian_getSource((Cartesian_t*)obj);
      if (objsrc != NULL && strlen(objsrc) < nlen - 1) {
        strncpy(source, objsrc, nlen);
        result = strlen(source);
      }
    } else if (RAVE_OBJECT_CHECK_TYPE(obj, &CartesianVolume_TYPE)) {
      const char* objsrc = CartesianVolume_getSource((CartesianVolume_t*)obj);
      if (objsrc != NULL && strlen(objsrc) < nlen - 1) {
        strncpy(source, objsrc, nlen);
        result = strlen(source);
      }
    }
  }
  return result;
}

CompositeRaveObjectBinding_t* CompositeUtils_createRaveObjectBinding(CompositeArguments_t* arguments, Cartesian_t* cartesian, int* nobjects, OdimSources_t* sources)
{
  CompositeRaveObjectBinding_t* result = NULL;
  Projection_t* projection = NULL;

  int i = 0, nlen = 0;

  if (arguments == NULL || cartesian == NULL || nobjects == NULL) {
    RAVE_ERROR0("Must provide arguments, cartesian and nobjects must be provided");
    return NULL;
  }
  projection = Cartesian_getProjection(cartesian);
  if (projection == NULL) {
    RAVE_ERROR0("Cartesian must have a projection");
    return NULL;
  }

  nlen = CompositeArguments_getNumberOfObjects(arguments);
  result = RAVE_MALLOC(sizeof(CompositeRaveObjectBinding_t)* nlen);
  if (result == NULL) {
    RAVE_ERROR0("Failed to allocate array for object bindings");
    goto fail;
  }
  memset(result, 0, sizeof(CompositeRaveObjectBinding_t)*nlen);
  for (i = 0; i < nlen; i++) {
    RaveCoreObject* object = CompositeArguments_getObject(arguments, i);
    if (object != NULL) {
      Projection_t* objproj = CompositeUtils_getProjection(object);
      ProjectionPipeline_t* pipeline = NULL;
      OdimSource_t* source = NULL;
      char strsrc[512];

      if (objproj == NULL) {
        RAVE_ERROR0("Object does not have a projection");
        RAVE_OBJECT_RELEASE(object);
        goto fail;
      }
      pipeline = ProjectionPipeline_createPipeline(projection, objproj);
      if (pipeline == NULL) {
        RAVE_ERROR0("Failed to create pipeline");
        RAVE_OBJECT_RELEASE(object);
        RAVE_OBJECT_RELEASE(objproj);
        goto fail;
      }
      if (sources != NULL && CompositeUtils_getObjectSource(object, strsrc, 512)) {
        source = OdimSources_identify(sources, (const char*)strsrc);
        if (source != NULL) {
          RAVE_SPEWDEBUG1("NOD: %s", OdimSource_getNod(source));
        }
      }

      result[i].object = RAVE_OBJECT_COPY(object);
      result[i].pipeline = RAVE_OBJECT_COPY(pipeline);
      result[i].source = RAVE_OBJECT_COPY(source);
      RAVE_OBJECT_RELEASE(pipeline);
      RAVE_OBJECT_RELEASE(objproj);
      RAVE_OBJECT_RELEASE(source);
    }
    RAVE_OBJECT_RELEASE(object);
  }
  *nobjects = nlen;
  RAVE_OBJECT_RELEASE(projection);
  return result;
fail:
  RAVE_OBJECT_RELEASE(projection);
  CompositeUtils_releaseRaveObjectBinding(&result, nlen);
  return NULL;
}

/**
 * Releases the objects and then deallocates the array
 * @param[in,out] arr - the array to release
 */
void CompositeUtils_releaseRaveObjectBinding(CompositeRaveObjectBinding_t** arr, int nobjects)
{
  int i = 0;
  if (arr == NULL) {
    RAVE_ERROR0("Nothing to release");
    return;
  }
  if (*arr != NULL) {
    for (i = 0; i < nobjects; i++) {
      RAVE_OBJECT_RELEASE((*arr)[i].object);
      RAVE_OBJECT_RELEASE((*arr)[i].pipeline);
      RAVE_OBJECT_RELEASE((*arr)[i].source)
    }
    RAVE_FREE(*arr);
    *arr = NULL;
  }
}

void CompositeUtils_getQualityFlagSettings(CompositeQualityFlagSettings_t* settings, const char* flagname, double* offset, double* gain, RaveDataType* datatype)
{
  int ctr = 0;
  double voffset = 0.0, vgain = 1.0;
  RaveDataType vdatatype = RaveDataType_UCHAR;

  if (settings != NULL && flagname != NULL) {
    while (settings[ctr].qualityFieldName != NULL) {
      if (strcmp(flagname, settings[ctr].qualityFieldName) == 0) {
        voffset = settings[ctr].offset;
        vgain = settings[ctr].gain;
        vdatatype = settings[ctr].datatype;
        break;
      }
      ctr++;
    }
  }
  if (offset != NULL) {
    *offset = voffset;
  }
  if (gain != NULL) {
    *gain = vgain;
  }
  if (datatype != NULL) {
    *datatype = vdatatype;
  }
}

int CompositeUtils_addQualityFlagsToCartesian(CompositeArguments_t* arguments, Cartesian_t* cartesian, RaveObjectHashTable_t* qualityFlagDefinitions)
{
  int xsize = 0, ysize = 0, i = 0, j = 0, nparam = 0, nqfields = 0;
  RaveField_t* field = NULL;
  CartesianParam_t* param = NULL;
  int result = 0;

  if (arguments == NULL || cartesian == NULL || qualityFlagDefinitions == NULL) {
    RAVE_ERROR0("Must provide arguments, cartesian and quality flag definitions");
    return 0;
  }

  xsize = Cartesian_getXSize(cartesian);
  ysize = Cartesian_getYSize(cartesian);

  nqfields = CompositeArguments_getNumberOfQualityFlags(arguments);
  nparam = CompositeArguments_getParameterCount(arguments);

  for (i = 0; i < nqfields; i++) {
    const char* howtaskvaluestr = (const char*)CompositeArguments_getQualityFlagAt(arguments, i);
    double gain = 1.0/UCHAR_MAX, offset = 0.0;
    RaveDataType datatype = RaveDataType_UCHAR;
    CompositeQualityFlagDefinition_t* definition = (CompositeQualityFlagDefinition_t*)RaveObjectHashTable_get(qualityFlagDefinitions, howtaskvaluestr);
    if (definition != NULL) {
      offset = definition->offset;
      gain = definition->gain;
      datatype = definition->datatype;
    }
    field = CompositeUtils_createQualityField(howtaskvaluestr, xsize, ysize, datatype, gain, offset);
    if (field != NULL) {
      for (j = 0; j < nparam; j++) {
        param = Cartesian_getParameter(cartesian, CompositeArguments_getParameterName(arguments, j));
        if (param != NULL) {
          RaveField_t* cfield = RAVE_OBJECT_CLONE(field);
          if (cfield == NULL ||
              !CartesianParam_addQualityField(param, cfield)) {
            RAVE_OBJECT_RELEASE(cfield);
            RAVE_OBJECT_RELEASE(definition);
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
    RAVE_OBJECT_RELEASE(definition);
    RAVE_OBJECT_RELEASE(field);
  }
  result = 1;
done:
  RAVE_OBJECT_RELEASE(field);
  RAVE_OBJECT_RELEASE(param);
  return result;  
}

int CompositeUtils_addQualityFlagsToCartesianFromSettings(CompositeArguments_t* arguments, Cartesian_t* cartesian, CompositeQualityFlagSettings_t* settings)
{
  int result = 0;
  RaveObjectHashTable_t* definitions = NULL;
  if (arguments == NULL || cartesian == NULL) {
    RAVE_ERROR0("Must provide arguments and cartesian");
    return 0;
  }
  definitions = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  if (definitions == NULL) {
    RAVE_ERROR0("Failed to allocate memory");
    return 0;
  }
  if (!CompositeUtils_registerQualityFlagDefinitionFromSettings(definitions, settings)) {
    RAVE_ERROR0("Failed to create definitions from settings");
    goto fail;
  }

  result = 1;
fail:
  RAVE_OBJECT_RELEASE(definitions);
  return result;
}

int CompositeUtils_addGainAndOffsetToField(RaveField_t* field, double gain, double offset) {
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

RaveField_t* CompositeUtils_createQualityField(const char* howtaskvaluestr, int xsize, int ysize, RaveDataType datatype, double gain, double offset)
{
  RaveField_t* qfield = RAVE_OBJECT_NEW(&RaveField_TYPE);
  RaveAttribute_t* howtaskattribute = NULL;

  if (qfield == NULL) {
    RAVE_ERROR0("Failed to create quality field");
    goto error;
  }

  howtaskattribute = RaveAttributeHelp_createString("how/task", howtaskvaluestr);
  if (howtaskattribute == NULL) {
    RAVE_ERROR0("Failed to create quality field (how/task attribute could not be created)");
    goto error;
  }

  if (!RaveField_addAttribute(qfield, howtaskattribute)) {
    RAVE_ERROR0("Failed to create how/task attribute for distance quality field");
    goto error;
  }
  RAVE_OBJECT_RELEASE(howtaskattribute);

  if (!CompositeUtils_addGainAndOffsetToField(qfield, gain, offset)) {
    RAVE_ERROR0("Failed to add gain and offset attribute to quality field");
    goto error;
  }

  if(!RaveField_createData(qfield, xsize, ysize, datatype)) {
    RAVE_ERROR0("Failed to create quality field");
    goto error;
  }

  return qfield;
error:
  RAVE_OBJECT_RELEASE(qfield);
  RAVE_OBJECT_RELEASE(howtaskattribute);
  return NULL;

}

int CompositeUtils_getPolarValueAtPosition(RaveCoreObject* obj, const char* quantity, PolarNavigationInfo* nav, const char* qiFieldName, RaveValueType* type, double* value, double* qualityValue)
{
  int result = 0;
  RAVE_ASSERT((nav != NULL), "nav == NULL");
  RAVE_ASSERT((type != NULL), "type == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  if (qualityValue != NULL) {
    *qualityValue = 0.0;
  }

  if (obj != NULL) {
    if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
      *type = PolarScan_getConvertedParameterValue((PolarScan_t*)obj, quantity, nav->ri, nav->ai, value);
      if (qiFieldName != NULL && qualityValue != NULL) {
        if (!PolarScan_getQualityValueAt((PolarScan_t*)obj, quantity, nav->ri, nav->ai, qiFieldName, 0, qualityValue)) {
          *qualityValue = 0.0;
        }
      }
    } else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
      *type = PolarVolume_getConvertedParameterValueAt((PolarVolume_t*)obj, quantity, nav->ei, nav->ri, nav->ai, value);
      if (qiFieldName != NULL && qualityValue != NULL) {
        if (!PolarVolume_getQualityValueAt((PolarVolume_t*)obj, quantity, nav->ei, nav->ri, nav->ai, qiFieldName, 0, qualityValue)) {
          *qualityValue = 0.0;
        }
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

int CompositeUtils_getPolarQualityValueAtPosition(RaveCoreObject* obj, const char* quantity, const char* qualityField, PolarNavigationInfo* nav, double* value)
{
  int result = 0;
  if (nav == NULL || value == NULL || qualityField == NULL) {
    RAVE_ERROR0("nav, value and qualityField must all be provided");
    return 0;
  }
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

int CompositeUtils_getVerticalMaxValue(
  RaveCoreObject* object,
  const char* quantity,
  const char* qiFieldName,
  double lon,
  double lat,
  RaveValueType* vtype,
  double* vvalue,
  PolarNavigationInfo* navinfo,
  double* qiv)
{
  RaveCoreObject* obj = NULL;
  PolarNavigationInfo info;

  RAVE_ASSERT((object != NULL) , "object == NULL");
  RAVE_ASSERT((vtype != NULL), "vtype == NULL");
  RAVE_ASSERT((vvalue != NULL), "vvalue == NULL");

  if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
    *vtype = PolarScan_getNearestConvertedParameterValue((PolarScan_t*)obj, quantity, lon, lat, vvalue, &info);
    if (qiFieldName != NULL && qiv != NULL) {
      if (!PolarScan_getQualityValueAt((PolarScan_t*)obj, quantity, info.ri, info.ai, qiFieldName, 0, qiv)) {
        *qiv = 0.0;
      }
    }
  } else {
    *vtype = PolarVolume_getConvertedVerticalMaxValue((PolarVolume_t*)obj, quantity, lon, lat, vvalue, &info);
    if (qiFieldName != NULL && qiv != NULL) {
      if (!PolarVolume_getQualityValueAt((PolarVolume_t*)obj, quantity, info.ei, info.ri, info.ai, qiFieldName, 0, qiv)) {
        *qiv = 0.0;
      }
    }
  }

  if (navinfo != NULL) {
    *navinfo = info;
  }

  RAVE_OBJECT_RELEASE(obj);
  return 1;
}

RaveList_t* CompositeUtils_cloneRaveListStrings(RaveList_t* inlist)
{
  RaveList_t *result = NULL, *tmplist = NULL;

  if (inlist != NULL) {
    tmplist = RAVE_OBJECT_NEW(&RaveList_TYPE);
    if (tmplist != NULL) {
      int nlen = 0, i = 0;
      nlen = RaveList_size(inlist);
      for (i = 0; i < nlen; i++) {
        char* str = RAVE_STRDUP((const char*)RaveList_get(inlist, i));
        if (str == NULL) {
          RAVE_ERROR0("Could not duplicate string");
          goto fail;
        }
        if (!RaveList_add(tmplist, str)) {
          goto fail;
        }
      }
    }
  }

  result = tmplist;
  tmplist = NULL;
fail:
  if (tmplist != NULL) {
    RaveList_freeAndDestroy(&tmplist);
  }
  return result;
}