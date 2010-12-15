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
 * Provides support for reading and writing areas to and from
 * an xml-file.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-12-08
 */
#include "arearegistry.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "raveobject_list.h"
#include "rave_simplexml.h"
#include "rave_utilities.h"
#include "expat.h"
#include <string.h>

/**
 * Represents the registry
 */
struct _AreaRegistry_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveObjectList_t* areas; /**< the list of areas */
  ProjectionRegistry_t* projRegistry; /**< the projection registry */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int AreaRegistry_constructor(RaveCoreObject* obj)
{
  AreaRegistry_t* this = (AreaRegistry_t*)obj;
  this->areas = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  this->projRegistry = RAVE_OBJECT_NEW(&ProjectionRegistry_TYPE);
  if (this->areas == NULL || this->projRegistry == NULL) {
    goto error;
  }

  return 1;
error:
  RAVE_OBJECT_RELEASE(this->areas);
  RAVE_OBJECT_RELEASE(this->projRegistry);
  return 0;
}

/**
 * Copy constructor
 */
static int AreaRegistry_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  AreaRegistry_t* this = (AreaRegistry_t*)obj;
  AreaRegistry_t* src = (AreaRegistry_t*)srcobj;

  this->areas = RAVE_OBJECT_CLONE(src->areas);
  this->projRegistry = RAVE_OBJECT_CLONE(src->projRegistry);

  if (this->areas == NULL || this->projRegistry == NULL) {
    goto error;
  }
  return 1;
error:
  RAVE_OBJECT_RELEASE(this->areas);
  RAVE_OBJECT_RELEASE(this->projRegistry);
  return 0;
}

/**
 * Destroys the registry
 * @param[in] obj - the the AreaRegistry_t instance
 */
static void AreaRegistry_destructor(RaveCoreObject* obj)
{
  AreaRegistry_t* this = (AreaRegistry_t*)obj;
  RAVE_OBJECT_RELEASE(this->areas);
  RAVE_OBJECT_RELEASE(this->projRegistry);
}

/**
 * Parses a string containing an extent into the lower left and upper right coordinates.
 * @param[in] extent - the extent in format llx, lly, urx, ury
 * @param[in,out] llx - the lower left x coordinate
 * @param[in,out] lly - the lower left y coordinate
 * @param[in,out] urx - the upper right x coordinate
 * @param[in,out] ury - the upper right y coordinate
 * @return 1 on success otherwise 0
 */
static int AreaRegistryInternal_parseExtent(const char* extent, double* llx, double* lly, double* urx, double* ury)
{
  int result = 0;
  RaveList_t* tokens = NULL;
  RAVE_ASSERT((llx != NULL), "llx == NULL");
  RAVE_ASSERT((lly != NULL), "lly == NULL");
  RAVE_ASSERT((urx != NULL), "urx == NULL");
  RAVE_ASSERT((ury != NULL), "ury == NULL");

  tokens = RaveUtilities_getTrimmedTokens(extent, ',');
  if (tokens == NULL || RaveList_size(tokens) != 4) {
    RAVE_ERROR0("Illegal extent definition");
    goto done;
  }
  if (sscanf((const char*)RaveList_get(tokens, 0), "%lf", llx) != 1) {
    RAVE_ERROR0("Could not extract llx");
    goto done;
  }
  if (sscanf((const char*)RaveList_get(tokens, 1), "%lf", lly) != 1) {
    RAVE_ERROR0("Could not extract lly");
    goto done;
  }
  if (sscanf((const char*)RaveList_get(tokens, 2), "%lf", urx) != 1) {
    RAVE_ERROR0("Could not extract urx");
    goto done;
  }
  if (sscanf((const char*)RaveList_get(tokens, 3), "%lf", ury) != 1) {
    RAVE_ERROR0("Could not extract ury");
    goto done;
  }

  result = 1;
done:
  return result;
}

/**
 * Creates an area from a xml-node
 * @param[in] node the xml node
 * @returns an area on success or NULL on failure
 */
static Area_t* AreaRegistryInternal_createAreaFromNode(AreaRegistry_t* self, SimpleXmlNode_t* node)
{
  Area_t* area = NULL;
  Area_t* result = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");

  if (node != NULL && SimpleXmlNode_getName(node) != NULL && strcasecmp("area", SimpleXmlNode_getName(node)) == 0) {
    area = RAVE_OBJECT_NEW(&Area_TYPE);
    if (area != NULL) {
      const char* id = SimpleXmlNode_getAttribute(node, "id");
      const char* description = NULL;
      const char* pcs = NULL;
      const char* xsizestr = NULL;
      const char* ysizestr = NULL;
      const char* xscalestr = NULL;
      const char* yscalestr = NULL;
      const char* extentstr = NULL;

      SimpleXmlNode_t* dNode = SimpleXmlNode_getChildByName(node, "description");
      SimpleXmlNode_t* areadef = SimpleXmlNode_getChildByName(node, "areadef");

      if (dNode != NULL) {
        description = SimpleXmlNode_getText(dNode);
        RAVE_OBJECT_RELEASE(dNode);
      }

      if (areadef != NULL) {
        int nchild = SimpleXmlNode_getNumberOfChildren(areadef);
        int i = 0;
        for (i = 0; i < nchild; i++) {
          SimpleXmlNode_t* child = SimpleXmlNode_getChild(areadef, i);
          if (child != NULL && strcasecmp("arg", SimpleXmlNode_getName(child)) == 0) {
            const char* cid = SimpleXmlNode_getAttribute(child, "id");
            if (cid != NULL) {
              if (strcasecmp("pcs", cid)==0) {
                pcs = SimpleXmlNode_getText(child);
              } else if (strcasecmp("xsize", cid)==0) {
                xsizestr = SimpleXmlNode_getText(child);
              } else if (strcasecmp("ysize", cid)==0) {
                ysizestr = SimpleXmlNode_getText(child);
              } else if (strcasecmp("scale", cid)==0) {
                xscalestr = yscalestr = SimpleXmlNode_getText(child);
              } else if (strcasecmp("xscale", cid)==0) {
                xscalestr = SimpleXmlNode_getText(child);
              } else if (strcasecmp("yscale", cid)==0) {
                yscalestr = SimpleXmlNode_getText(child);
              } else if (strcasecmp("extent", cid)==0) {
                extentstr = SimpleXmlNode_getText(child);
              }
            }
          }
          RAVE_OBJECT_RELEASE(child);
        }
        RAVE_OBJECT_RELEASE(areadef);
      }

      if (id != NULL) {
        if (!Area_setID(area, id)) {
          RAVE_ERROR1("Failed to allocate memory for id = %s", id);
          goto done;
        }
      } else {
        RAVE_ERROR0("id missing for area");
        goto done;
      }

      if (pcs != NULL) {
        if (self->projRegistry != NULL) {
          Projection_t* proj = ProjectionRegistry_getByName(self->projRegistry, pcs);
          if (proj != NULL) {
            Area_setProjection(area, proj);
          } else {
            if (!Area_setPcsid(area, pcs)) {
              RAVE_ERROR0("Failed to set pcs id");
              goto done;
            }
          }
        } else {
          if (!Area_setPcsid(area, pcs)) {
            RAVE_ERROR0("Failed to set pcs id");
            goto done;
          }
        }
      } else {
        RAVE_ERROR0("No pcs id for area");
        goto done;
      }

      if (description != NULL) {
        if (!Area_setDescription(area, description)) {
          RAVE_ERROR1("Failed to allocate memory for description = %s", description);
          goto done;
        }
      }

      if (xsizestr != NULL) {
        long xsize = 0;
        if (sscanf(xsizestr, "%ld", &xsize) != 1) {
          RAVE_ERROR1("xsize missing for %s", id);
          goto done;
        }
        Area_setXSize(area, xsize);
      } else {
        RAVE_ERROR1("xsize missing for area %s", id);
        goto done;
      }

      if (ysizestr != NULL) {
        long ysize = 0;
        if (sscanf(ysizestr, "%ld", &ysize) != 1) {
          RAVE_ERROR1("ysize missing for %s", id);
          goto done;
        }
        Area_setYSize(area, ysize);
      } else {
        RAVE_ERROR1("ysize missing for area %s", id);
        goto done;
      }

      if (xscalestr != NULL) {
        double xscale = 0;
        if (sscanf(xscalestr, "%lf", &xscale) != 1) {
          RAVE_ERROR1("xscale missing for %s", id);
          goto done;
        }
        Area_setXScale(area, xscale);
      } else {
        RAVE_ERROR1("xscale missing for area %s", id);
        goto done;
      }

      if (yscalestr != NULL) {
        double yscale = 0;
        if (sscanf(yscalestr, "%lf", &yscale) != 1) {
          RAVE_ERROR1("yscale missing for %s", id);
          goto done;
        }
        Area_setYScale(area, yscale);
      } else {
        RAVE_ERROR1("yscale missing for area %s", id);
        goto done;
      }

      if (extentstr != NULL) {
        double llx=0.0,lly=0.0,urx=0.0,ury=0.0;
        if (!AreaRegistryInternal_parseExtent(extentstr, &llx,&lly,&urx,&ury)) {
          RAVE_ERROR1("Failed to parse extent for area %s", id);
          goto done;
        }
        Area_setExtent(area, llx, lly, urx, ury);
      } else {
        RAVE_ERROR1("extent missing for area %s", id);
        goto done;
      }
    }
  }

  result = RAVE_OBJECT_COPY(area);

done:
  RAVE_OBJECT_RELEASE(area);
  return result;
}

/*@} End of Private functions */

/*@{ Interface functions */
int AreaRegistry_loadRegistry(AreaRegistry_t* self, const char* filename)
{
  SimpleXmlNode_t* node = NULL;
  int result = 0;
  int nrchildren = 0;
  int i = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((filename != NULL), "filename == NULL");

  node = SimpleXmlNode_parseFile(filename);
  if (node == NULL) {
    goto done;
  }

  nrchildren = SimpleXmlNode_getNumberOfChildren(node);
  for (i = 0; i < nrchildren; i++) {
    SimpleXmlNode_t* child = SimpleXmlNode_getChild(node, i);
    Area_t* area = AreaRegistryInternal_createAreaFromNode(self, child);
    if (area != NULL) {
      RaveObjectList_add(self->areas, (RaveCoreObject*)area);
    }
    RAVE_OBJECT_RELEASE(child);
    RAVE_OBJECT_RELEASE(area);
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(node);
  return result;
}

AreaRegistry_t* AreaRegistry_load(const char* filename, ProjectionRegistry_t* pRegistry)
{
  AreaRegistry_t* result = NULL;
  if (filename != NULL) {
    result = RAVE_OBJECT_NEW(&AreaRegistry_TYPE);
    if (result != NULL) {
      AreaRegistry_setProjectionRegistry(result, pRegistry);
      if (!AreaRegistry_loadRegistry(result, filename)) {
        RAVE_OBJECT_RELEASE(result);
      }
    }
  }
  return result;
}

int AreaRegistry_size(AreaRegistry_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveObjectList_size(self->areas);
}

Area_t* AreaRegistry_get(AreaRegistry_t* self, int index)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (Area_t*)RaveObjectList_get(self->areas, index);
}

void AreaRegistry_setProjectionRegistry(AreaRegistry_t* self, ProjectionRegistry_t* registry)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_OBJECT_RELEASE(self->projRegistry);
  self->projRegistry = RAVE_OBJECT_COPY(registry);
}

ProjectionRegistry_t* AreaRegistry_getProjectionRegistry(AreaRegistry_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RAVE_OBJECT_COPY(self->projRegistry);
}

/*@} End of Interface functions */

RaveCoreObjectType AreaRegistry_TYPE = {
    "AreaRegistry",
    sizeof(AreaRegistry_t),
    AreaRegistry_constructor,
    AreaRegistry_destructor,
    AreaRegistry_copyconstructor
};
