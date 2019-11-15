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
  RaveList_freeAndDestroy(&tokens);
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

  if (node != NULL && SimpleXmlNode_getName(node) != NULL &&
      strcasecmp("area", SimpleXmlNode_getName(node)) == 0) {
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
            RAVE_OBJECT_RELEASE(proj);
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

/**
 * Loads an area registry from a xml file
 * @param[in] self - self
 * @param[in] filename - the xml file to load
 * @returns 1 on success 0 otherwise
 */
int AreaRegistryInternal_loadRegistry(AreaRegistry_t* self, const char* filename)
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

/**
 * Adds all arguments to the area def node
 * @param[in] area - the area
 * @param[in] areadefNode - the xml node to get arguments added
 * @returns 1 on success or 0 on failure
 */
static int AreaRegistryInternal_addArgsToNode(Area_t* area, SimpleXmlNode_t* areadefNode)
{
  SimpleXmlNode_t* argNode = NULL;
  int result = 0;

  RAVE_ASSERT((area != NULL), "area == NULL");
  RAVE_ASSERT((areadefNode != NULL), "areadefNode == NULL");

  // pcs
  argNode = SimpleXmlNode_create(areadefNode, "arg");
  if (argNode != NULL) {
    const char* pcsid = Area_getPcsid(area);
    if (pcsid == NULL ||
        !SimpleXmlNode_addAttribute(argNode, "id", "pcs") ||
        !SimpleXmlNode_setText(argNode, pcsid, strlen(pcsid))) {
      RAVE_ERROR0("Failed to add pcs id to area");
      goto done;
    }
  } else {
    goto done;
  }
  RAVE_OBJECT_RELEASE(argNode);

  // xsize
  argNode = SimpleXmlNode_create(areadefNode, "arg");
  if (argNode != NULL) {
    char xsize[32];
    sprintf(xsize, "%ld", Area_getXSize(area));
    if (!SimpleXmlNode_addAttribute(argNode, "id", "xsize") ||
        !SimpleXmlNode_addAttribute(argNode, "type", "int") ||
        !SimpleXmlNode_setText(argNode, xsize, strlen(xsize))) {
      RAVE_ERROR0("Failed to add xsize to area");
      goto done;
    }
  } else {
    goto done;
  }
  RAVE_OBJECT_RELEASE(argNode);

  // ysize
  argNode = SimpleXmlNode_create(areadefNode, "arg");
  if (argNode != NULL) {
    char ysize[32];
    sprintf(ysize, "%ld", Area_getYSize(area));
    if (!SimpleXmlNode_addAttribute(argNode, "id", "ysize") ||
        !SimpleXmlNode_addAttribute(argNode, "type", "int") ||
        !SimpleXmlNode_setText(argNode, ysize, strlen(ysize))) {
      RAVE_ERROR0("Failed to add ysize to area");
      goto done;
    }
  } else {
    goto done;
  }
  RAVE_OBJECT_RELEASE(argNode);

  // xscale
  argNode = SimpleXmlNode_create(areadefNode, "arg");
  if (argNode != NULL) {
    char xscale[32];
    sprintf(xscale, "%lf", Area_getXScale(area));
    if (!SimpleXmlNode_addAttribute(argNode, "id", "xscale") ||
        !SimpleXmlNode_addAttribute(argNode, "type", "float") ||
        !SimpleXmlNode_setText(argNode, xscale, strlen(xscale))) {
      RAVE_ERROR0("Failed to add xscale to area");
      goto done;
    }
  } else {
    goto done;
  }
  RAVE_OBJECT_RELEASE(argNode);

  // yscale
  argNode = SimpleXmlNode_create(areadefNode, "arg");
  if (argNode != NULL) {
    char yscale[32];
    sprintf(yscale, "%lf", Area_getYScale(area));
    if (!SimpleXmlNode_addAttribute(argNode, "type", "float") ||
        !SimpleXmlNode_addAttribute(argNode, "id", "yscale") ||
        !SimpleXmlNode_setText(argNode, yscale, strlen(yscale))) {
      RAVE_ERROR0("Failed to add yscale to area");
      goto done;
    }
  } else {
    goto done;
  }
  RAVE_OBJECT_RELEASE(argNode);

  // extent
  argNode = SimpleXmlNode_create(areadefNode, "arg");
  if (argNode != NULL) {
    double llX = 0.0, llY = 0.0, urX = 0.0, urY = 0.0;
    char extent[512];
    Area_getExtent(area, &llX, &llY, &urX, &urY);
    if (sprintf(extent, "%lf, %lf, %lf, %lf", llX, llY, urX, urY) >= 511) {
      RAVE_ERROR0("Extent became too large, can not complete writing");
      goto done;
    }
    if (!SimpleXmlNode_addAttribute(argNode, "id", "extent") ||
        !SimpleXmlNode_addAttribute(argNode, "type", "sequence") ||
        !SimpleXmlNode_setText(argNode, extent, strlen(extent))) {
      RAVE_ERROR0("Failed to add extent to area");
      goto done;
    }
  } else {
    goto done;
  }
  RAVE_OBJECT_RELEASE(argNode);

  result = 1;
done:
  RAVE_OBJECT_RELEASE(argNode);
  return result;
}

/**
 * Creates a xml node from a area instance.
 * @param[in] proj - the area instance
 * @returns a xml node on success or NULL on failure
 */
static SimpleXmlNode_t* AreanRegistryInternal_createNode(Area_t* area)
{
  SimpleXmlNode_t* node = NULL;
  SimpleXmlNode_t* result = NULL;
  SimpleXmlNode_t* descrNode = NULL;
  SimpleXmlNode_t* areadefNode = NULL;
  const char* description = NULL;

  RAVE_ASSERT((area != NULL), "area == NULL");

  node = RAVE_OBJECT_NEW(&SimpleXmlNode_TYPE);
  descrNode = RAVE_OBJECT_NEW(&SimpleXmlNode_TYPE);
  areadefNode = RAVE_OBJECT_NEW(&SimpleXmlNode_TYPE);

  if (node == NULL || descrNode == NULL || areadefNode == NULL) {
    goto done;
  }
  if(!SimpleXmlNode_setName(node, "area") ||
     !SimpleXmlNode_setName(descrNode, "description") ||
     !SimpleXmlNode_setName(areadefNode, "areadef")) {
    goto done;
  }
  if (!SimpleXmlNode_addAttribute(node, "id", Area_getID(area))) {
    goto done;
  }
  description = Area_getDescription(area);
  if (description != NULL &&
      !SimpleXmlNode_setText(descrNode,description, strlen(description))) {
    goto done;
  }
  if (!AreaRegistryInternal_addArgsToNode(area, areadefNode)) {
    goto done;
  }

  if (!SimpleXmlNode_addChild(node, descrNode) ||
      !SimpleXmlNode_addChild(node, areadefNode)) {
    goto done;
  }

  result = RAVE_OBJECT_COPY(node);
done:
  RAVE_OBJECT_RELEASE(node);
  RAVE_OBJECT_RELEASE(descrNode);
  RAVE_OBJECT_RELEASE(areadefNode);
  return result;
}

/*@} End of Private functions */

/*@{ Interface functions */

AreaRegistry_t* AreaRegistry_load(const char* filename, ProjectionRegistry_t* pRegistry)
{
  AreaRegistry_t* result = NULL;
  if (filename != NULL) {
    result = RAVE_OBJECT_NEW(&AreaRegistry_TYPE);
    if (result != NULL) {
      AreaRegistry_setProjectionRegistry(result, pRegistry);
      if (!AreaRegistryInternal_loadRegistry(result, filename)) {
        RAVE_OBJECT_RELEASE(result);
      }
    }
  }
  return result;
}

int AreaRegistry_add(AreaRegistry_t* self, Area_t* area)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (area != NULL) {
    result = RaveObjectList_add(self->areas, (RaveCoreObject*)area);
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

Area_t* AreaRegistry_getByName(AreaRegistry_t* self, const char* id)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (id != NULL) {
    int n = 0;
    int i = 0;
    n = RaveObjectList_size(self->areas);
    for (i = 0; i < n; i++) {
      Area_t* area = (Area_t*)RaveObjectList_get(self->areas, i);
      if (area != NULL &&
          Area_getID(area) != NULL &&
          strcmp(id, Area_getID(area))==0) {
        return area;
      }
      RAVE_OBJECT_RELEASE(area);
    }
  }
  return NULL;
}

void AreaRegistry_remove(AreaRegistry_t* self, int index)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RaveObjectList_release(self->areas, index);
}

void AreaRegistry_removeByName(AreaRegistry_t* self, const char* id)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (id != NULL) {
    int nlen = 0;
    int i = 0;
    int found = 0;
    nlen = RaveObjectList_size(self->areas);
    for (i = 0; found == 0 && i < nlen; i++) {
      Area_t* area = (Area_t*)RaveObjectList_get(self->areas, i);
      if (area != NULL && strcmp(id, Area_getID(area)) == 0) {
        found = 1;
        RaveObjectList_release(self->areas, i);
      }
      RAVE_OBJECT_RELEASE(area);
    }
  }
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

int AreaRegistry_write(AreaRegistry_t* self, const char* filename)
{
  SimpleXmlNode_t* root = NULL;
  Area_t* childarea = NULL;
  SimpleXmlNode_t* childnode = NULL;
  FILE* fp = NULL;
  int result = 0;
  int nproj = 0;
  int i = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");

  if (filename == NULL) {
    RAVE_ERROR0("Trying to save registry without filename");
    goto done;
  }

  root = RAVE_OBJECT_NEW(&SimpleXmlNode_TYPE);
  if (root == NULL) {
    RAVE_ERROR0("Failed to allocate memory for root node");
    goto done;
  }
  if (!SimpleXmlNode_setName(root, "areas")) {
    goto done;
  }

  nproj = RaveObjectList_size(self->areas);
  for (i = 0;  i < nproj; i++) {
    childarea = (Area_t*)RaveObjectList_get(self->areas, i);
    if (childarea == NULL) {
      goto done;
    }

    childnode = AreanRegistryInternal_createNode(childarea);
    if (childnode == NULL || !SimpleXmlNode_addChild(root, childnode)) {
      goto done;
    }
    RAVE_OBJECT_RELEASE(childarea);
    RAVE_OBJECT_RELEASE(childnode);
  }

  fp = fopen(filename, "w");
  if (fp == NULL) {
    RAVE_ERROR1("Failed to open %s for writing", filename);
    goto done;
  }

  if (!SimpleXmlNode_write(root, fp)) {
    RAVE_ERROR0("Failed to write xml file");
    goto done;
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(childarea);
  RAVE_OBJECT_RELEASE(childnode);
  RAVE_OBJECT_RELEASE(root);
  if (fp != NULL) {
    fclose(fp);
  }
  return result;
}
/*@} End of Interface functions */

RaveCoreObjectType AreaRegistry_TYPE = {
    "AreaRegistry",
    sizeof(AreaRegistry_t),
    AreaRegistry_constructor,
    AreaRegistry_destructor,
    AreaRegistry_copyconstructor
};
