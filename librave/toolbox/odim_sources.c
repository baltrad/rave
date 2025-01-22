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
 * Provides support for reading odim sources from an xml-file and managing these in a structure.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-01-15
 */
#include "odim_sources.h"
#include "odim_source.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_object.h"
#include "raveobject_list.h"
#include "raveobject_hashtable.h"
#include "rave_simplexml.h"
#include "rave_utilities.h"
#include "expat.h"
#include <string.h>

/**
 * Represents the odim sources registry
 */
struct _OdimSources_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveObjectHashTable_t* nod; /**< the mapping of sources */
  RaveObjectHashTable_t* wigos; /**< the mapping of sources */
  RaveObjectHashTable_t* wmo; /**< the mapping of sources */
  RaveObjectHashTable_t* rad; /**< the mapping of sources */
  RaveObjectHashTable_t* plc; /**< the mapping of sources */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int OdimSources_constructor(RaveCoreObject* obj)
{
  OdimSources_t* this = (OdimSources_t*)obj;
  this->nod = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  this->wigos = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  this->wmo = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  this->rad = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);
  this->plc = RAVE_OBJECT_NEW(&RaveObjectHashTable_TYPE);

  if (this->nod == NULL || this->wigos == NULL || this->wmo == NULL || this->rad == NULL || this->plc == NULL) {
    goto error;
  }
  return 1;
error:
  RAVE_OBJECT_RELEASE(this->nod);
  RAVE_OBJECT_RELEASE(this->wigos);
  RAVE_OBJECT_RELEASE(this->wmo);
  RAVE_OBJECT_RELEASE(this->rad);
  RAVE_OBJECT_RELEASE(this->plc);
  return 0;
}

/**
 * Copy constructor
 */
static int OdimSources_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  OdimSources_t* this = (OdimSources_t*)obj;
  OdimSources_t* src = (OdimSources_t*)srcobj;

  this->nod = RAVE_OBJECT_CLONE(src->nod);
  this->wigos = RAVE_OBJECT_CLONE(src->wigos);
  this->wmo = RAVE_OBJECT_CLONE(src->wmo);
  this->rad = RAVE_OBJECT_CLONE(src->rad);
  this->plc = RAVE_OBJECT_CLONE(src->plc);
  if (this->nod == NULL || this->wigos == NULL || this->wmo == NULL || this->rad == NULL || this->plc == NULL) {
    goto error;
  }
  return 1;
error:
  RAVE_OBJECT_RELEASE(this->nod);
  RAVE_OBJECT_RELEASE(this->wigos);
  RAVE_OBJECT_RELEASE(this->wmo);
  RAVE_OBJECT_RELEASE(this->rad);
  RAVE_OBJECT_RELEASE(this->plc);  
  return 0;
}

/**
 * Destroys the registry
 * @param[in] obj - the the AreaRegistry_t instance
 */
static void OdimSources_destructor(RaveCoreObject* obj)
{
  OdimSources_t* this = (OdimSources_t*)obj;
  RAVE_OBJECT_RELEASE(this->nod);
  RAVE_OBJECT_RELEASE(this->wigos);
  RAVE_OBJECT_RELEASE(this->wmo);
  RAVE_OBJECT_RELEASE(this->rad);
  RAVE_OBJECT_RELEASE(this->plc);
}

static int OdimSourcesInternal_loadSources(OdimSources_t* sources, const char* filename)
{
  SimpleXmlNode_t* node = NULL;
  int result = 0, nchildren = 0, i = 0;

  node = SimpleXmlNode_parseFile(filename);
  if (node == NULL) {
    goto done;
  }

  nchildren = SimpleXmlNode_getNumberOfChildren(node);
  for (i = 0; i < nchildren; i++) {
    SimpleXmlNode_t* country = SimpleXmlNode_getChild(node, i);
    if (country != NULL) {
      int j = 0;
      int nrsites = SimpleXmlNode_getNumberOfChildren(country);
      const char* cccc = SimpleXmlNode_getAttribute(country, "CCCC"); /* NOTE! SimpleXmlNode should be corrected to allow for any case attribute and always return lower case names */
      const char* org = SimpleXmlNode_getAttribute(country, "org");
      for (j = 0; j < nrsites; j++) {
        SimpleXmlNode_t* site = SimpleXmlNode_getChild(country, j);
        if (site != NULL) {
          const char* nod = SimpleXmlNode_getName(site);
          const char* plc = SimpleXmlNode_getAttribute(site, "plc");
          const char* rad = SimpleXmlNode_getAttribute(site, "rad");
          const char* wmo = SimpleXmlNode_getAttribute(site, "wmo");
          const char* wigos = SimpleXmlNode_getAttribute(site, "wigos");
          OdimSource_t* source = OdimSource_create(nod, wmo, wigos, plc, rad, cccc, org);
          if (source == NULL || !OdimSources_add(sources, source)) {
            RAVE_ERROR1("Failed to add source (%s) to sources", nod);
            RAVE_OBJECT_RELEASE(source);
            RAVE_OBJECT_RELEASE(site);
            RAVE_OBJECT_RELEASE(country);
            goto done;
          }
          RAVE_OBJECT_RELEASE(source);
        }
        RAVE_OBJECT_RELEASE(site);
      }
    }
    RAVE_OBJECT_RELEASE(country);
  }
  result = 1;
done:
  RAVE_OBJECT_RELEASE(node);
  return result;
}

/*@} End of Private functions */

/*@{ Interface functions */

OdimSources_t* OdimSources_load(const char* filename)
{
  OdimSources_t* result = NULL;
  if (filename != NULL) {
    result = RAVE_OBJECT_NEW(&OdimSources_TYPE);
    if (result != NULL) {
      if (!OdimSourcesInternal_loadSources(result, filename)) {
        RAVE_OBJECT_RELEASE(result);
      }
    }
  }
  return result;
}

int OdimSources_add(OdimSources_t* self, OdimSource_t* source)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (source != NULL) {
    if (OdimSource_getNod(source) != NULL) {
      result = RaveObjectHashTable_put(self->nod, OdimSource_getNod(source), (RaveCoreObject*)source); 

      if (OdimSource_getWigos(source) != NULL) {
        if (RaveObjectHashTable_exists(self->wigos, OdimSource_getWigos(source)) || !RaveObjectHashTable_put(self->wigos, OdimSource_getWigos(source), (RaveCoreObject*)source)) {
          RAVE_WARNING1("Failed to add wigos to odim sources WIGOS: %s", OdimSource_getWigos(source));
        }
      }

      if (OdimSource_getWmo(source) != NULL && strcmp("00000", OdimSource_getWmo(source)) != 0) {
        if (RaveObjectHashTable_exists(self->wmo,OdimSource_getWmo(source)) || !RaveObjectHashTable_put(self->wmo, OdimSource_getWmo(source), (RaveCoreObject*)source)) {
          RAVE_WARNING1("Failed to add wmo to odim sources WMO: %s", OdimSource_getWmo(source));
        }
      }

      if (OdimSource_getRad(source) != NULL) {
        if (RaveObjectHashTable_exists(self->rad,OdimSource_getRad(source)) || !RaveObjectHashTable_put(self->rad, OdimSource_getRad(source), (RaveCoreObject*)source)) {
          RAVE_WARNING1("Failed to add rad to odim sources RAD: %s", OdimSource_getRad(source));
        }
      }

      if (OdimSource_getPlc(source) != NULL) {
        if (RaveObjectHashTable_exists(self->plc, OdimSource_getPlc(source)) || !RaveObjectHashTable_put(self->plc, OdimSource_getPlc(source), (RaveCoreObject*)source)) {
          RAVE_WARNING1("Failed to add plc to odim sources PLC: %s", OdimSource_getPlc(source));
        }
      }

    } else {
      RAVE_ERROR0("Must specify nod in source to be able to add to registry");
    }
  }
  return result;
}

int OdimSources_size(OdimSources_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveObjectHashTable_size(self->nod);
}

OdimSource_t* OdimSources_get(OdimSources_t* self, const char* nod)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (OdimSource_t*)RaveObjectHashTable_get(self->nod, nod);
}

OdimSource_t* OdimSources_get_wmo(OdimSources_t* self, const char* wmo)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (OdimSource_t*)RaveObjectHashTable_get(self->wmo, wmo);
}

OdimSource_t* OdimSources_get_wigos(OdimSources_t* self, const char* wigos)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (OdimSource_t*)RaveObjectHashTable_get(self->wigos, wigos);
}

OdimSource_t* OdimSources_get_rad(OdimSources_t* self, const char* rad)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (OdimSource_t*)RaveObjectHashTable_get(self->rad, rad);
}

OdimSource_t* OdimSources_get_plc(OdimSources_t* self, const char* plc)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (OdimSource_t*)RaveObjectHashTable_get(self->plc, plc);
}

OdimSource_t* OdimSources_identify(OdimSources_t* self, const char* sourcestr)
{
  char *nod = NULL, *wmo = NULL, *wigos = NULL, *rad = NULL, *plc = NULL;
  OdimSource_t* result = NULL;
  // RaveObjectList_t* sources = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  nod = OdimSource_getIdFromOdimSource(sourcestr, "NOD:");
  if (nod != NULL) {
    result = OdimSources_get(self, nod);
  } else {
    /* If no NOD, then we will have to identify using the other information.
     * Do it in order, WIGOS, WMO, RAD and finally PLC. If WMO is 00000, then we will 
     * ignore it completely.
     */
    wigos = OdimSource_getIdFromOdimSource(sourcestr, "WIGOS:");
    wmo = OdimSource_getIdFromOdimSource(sourcestr, "WMO:");
    rad = OdimSource_getIdFromOdimSource(sourcestr, "RAD:");
    plc = OdimSource_getIdFromOdimSource(sourcestr, "PLC:");

    if (wigos != NULL) {
      result = OdimSources_get_wigos(self, wigos);
    }
    if (result == NULL && wmo != NULL && strcmp(wmo, "00000") != 0) {
      result = OdimSources_get_wmo(self, wmo);
    }
    if (result == NULL && rad != NULL) {
      result = OdimSources_get_rad(self, rad);
    }
    if (result == NULL && plc != NULL) {
      result = OdimSources_get_plc(self, plc);
    }
    // sources = RaveObjectHashTable_values(self->nod);


    // if (sources != NULL) {
    //   int i = 0, nlen = RaveObjectList_size(sources);
    //   int foundSourceCtr = 0, foundDuplicateSourceCtr = 0;
    //   OdimSource_t* foundSource = NULL;

    //   for (i = 0; i < nlen; i++) {
    //     OdimSource_t* source = (OdimSource_t*)RaveObjectList_get(sources, i);
    //     int sourcectr = 0;
    //     if (source != NULL) { 
    //        sourcectr += (wigos != NULL && OdimSource_getWigos(source) != NULL && strcmp(wigos,OdimSource_getWigos(source)) == 0)?1:0;
    //        sourcectr += (wmo != NULL && OdimSource_getWmo(source) != NULL && strcmp(wmo,OdimSource_getWmo(source)) == 0)?1:0;
    //        sourcectr += (rad != NULL && OdimSource_getRad(source) != NULL && strcmp(rad,OdimSource_getRad(source)) == 0)?1:0;
    //        sourcectr += (plc != NULL && OdimSource_getPlc(source) != NULL && strcmp(plc,OdimSource_getPlc(source)) == 0)?1:0;
    //        if (sourcectr > foundSourceCtr) {
    //         foundSourceCtr = sourcectr;
    //         foundDuplicateSourceCtr = 0;
    //         foundSource = RAVE_OBJECT_COPY(source);
    //         if (returnFirstMatch) {
    //           RAVE_OBJECT_RELEASE(source);
    //           break;
    //         }
    //        } else if (sourcectr == foundSourceCtr) {
    //         foundDuplicateSourceCtr = 1;
    //        }
    //     }
    //     RAVE_OBJECT_RELEASE(source);
    //   }
    //   if (foundDuplicateSourceCtr) {
    //     RAVE_ERROR0("Found at least two sources that matches source string. Don't know what to return.");
    //     RAVE_OBJECT_RELEASE(foundSource);
    //   }
    //   result = RAVE_OBJECT_COPY(foundSource);
    //   RAVE_OBJECT_RELEASE(foundSource);
    // }
    // RAVE_OBJECT_RELEASE(sources);
  }
  RAVE_FREE(nod);
  RAVE_FREE(wigos);
  RAVE_FREE(wmo);
  RAVE_FREE(rad);
  RAVE_FREE(plc);
  return result;
}

RaveList_t* OdimSources_nods(OdimSources_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveObjectHashTable_keys(self->nod);
}

/*@} End of Interface functions */

RaveCoreObjectType OdimSources_TYPE = {
    "OdimSources",
    sizeof(OdimSources_t),
    OdimSources_constructor,
    OdimSources_destructor,
    OdimSources_copyconstructor
};
