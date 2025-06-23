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
#include "tileregistry.h"
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
struct _TileRegistry_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveObjectList_t* tiledefs; /**< the list of tiledefs */
};


/*@{ Private functions */
/**
 * Constructor.
 */
static int TileRegistry_constructor(RaveCoreObject* obj)
{
  TileRegistry_t* this = (TileRegistry_t*)obj;
  this->tiledefs = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (this->tiledefs == NULL) {
    goto error;
  }

  return 1;
error:
  RAVE_OBJECT_RELEASE(this->tiledefs);
  return 0;
}

/**
 * Copy constructor
 */
static int TileRegistry_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  TileRegistry_t* this = (TileRegistry_t*)obj;
  TileRegistry_t* src = (TileRegistry_t*)srcobj;

  this->tiledefs = RAVE_OBJECT_CLONE(src->tiledefs);

  if (this->tiledefs == NULL) {
    goto error;
  }
  return 1;
error:
  RAVE_OBJECT_RELEASE(this->tiledefs);
  return 0;
}

/**
 * Destroys the registry
 * @param[in] obj - the the TileRegistry_t instance
 */
static void TileRegistry_destructor(RaveCoreObject* obj)
{
  TileRegistry_t* this = (TileRegistry_t*)obj;
  RAVE_OBJECT_RELEASE(this->tiledefs);
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
static int TileRegistryInternal_parseExtent(const char* extent, double* llx, double* lly, double* urx, double* ury)
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
 * Creates an tiledef from a xml-node
 * @param[in] node the xml node
 * @returns an tiledef on success or NULL on failure
 */
static TileDef_t* TileRegistryInternal_createTileFromNode(TileRegistry_t* self, SimpleXmlNode_t* node)
{
  TileDef_t* tiledef = NULL;
  TileDef_t* result = NULL;
  RAVE_ASSERT((self != NULL), "self == NULL");
  
  /*
   * 
   * <area id="odc_area">
   < tile* id="nw" extent="-1950002.529151, -100001.025971, -50002.529151, 2099998.974029" />
   <tile id="w"  extent="-1950002.529151, -900001.025971, -50002.529151, -100001.025971" />
   <tile id="sw" extent="-1950002.529151, -2300001.025971, -50002.529151, -900001.025971" />
   <tile id="ne" extent="-50002.529151, 499998.974029, 1849997.470849, 2099998.974029" />
   <tile id="e"  extent="-50002.529151, -300001.025971, 1849997.470849, 499998.97402899992" />
   <tile id="se" extent="-50002.529151, -2300001.025971, 1849997.470849, -300001.025971" />
   </area>
   */

  if (node != NULL && SimpleXmlNode_getName(node) != NULL && strcasecmp("tile", SimpleXmlNode_getName(node)) == 0) {
    tiledef = RAVE_OBJECT_NEW(&TileDef_TYPE);
    if (tiledef != NULL) {
      
      const char* idstr = SimpleXmlNode_getAttribute(node, "id");
      if (idstr != NULL) {
        if (!TileDef_setID(tiledef, idstr)) {
          RAVE_ERROR1("Failed to allocate memory for tileid = %s", idstr);
          goto done;
        }
      } else {
        RAVE_ERROR0("id missing for tiledef");
        goto done;
      }
      
      const char* extentstr = SimpleXmlNode_getAttribute(node, "extent");
      if (extentstr != NULL) {
        //RAVE_DEBUG1("exstent: %s", extentstr);
        double llx=0.0,lly=0.0,urx=0.0,ury=0.0;
        if (!TileRegistryInternal_parseExtent(extentstr, &llx,&lly,&urx,&ury)) {
          RAVE_ERROR1("Failed to parse extent for tiledef %s", idstr);
          goto done;
        }
        TileDef_setExtent(tiledef, llx, lly, urx, ury);
      } else {
        RAVE_ERROR1("extent missing for tiledef %s",idstr);
        goto done;
      }
    result = RAVE_OBJECT_COPY(tiledef);
    }
  }
done:
    RAVE_OBJECT_RELEASE(tiledef);
    return result;
}

/**
 * Loads an tiledef registry from a xml file
 * @param[in] self - self
 * @param[in] filename - the xml file to load
 * @returns 1 on success 0 otherwise
 */
static int TileRegistryInternal_loadRegistry(TileRegistry_t* self, const char* filename)
{
  SimpleXmlNode_t* node = NULL;
  int result = 0;
  int nrchildren = 0;
  int ntiledefs = 0;
  int i = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((filename != NULL), "filename == NULL");

  node = SimpleXmlNode_parseFile(filename);
  if (node == NULL) {
    goto done;
  }

  nrchildren = SimpleXmlNode_getNumberOfChildren(node);
  //RAVE_DEBUG1("nrchildren: %d", nrchildren);
  for (i = 0; i < nrchildren; i++) {
    SimpleXmlNode_t* child = SimpleXmlNode_getChild(node, i);
    if (child != NULL && SimpleXmlNode_getName(child) != NULL && strcasecmp("area", SimpleXmlNode_getName(child)) == 0) {
      ntiledefs = SimpleXmlNode_getNumberOfChildren(child);
      const char* areaid = SimpleXmlNode_getAttribute(child, "id");
      //RAVE_DEBUG1("areid: %s", areaid);
      for (int j = 0; j < ntiledefs; j++) {
        SimpleXmlNode_t* tile = SimpleXmlNode_getChild(child, j);
        if (tile != NULL && SimpleXmlNode_getName(tile) != NULL && strcasecmp("tile", SimpleXmlNode_getName(tile)) == 0) {
          TileDef_t* tiledef = TileRegistryInternal_createTileFromNode(self, tile);
          if (tiledef != NULL) {
            if (areaid != NULL) {
              if (!TileDef_setAreaID(tiledef, areaid)) {
                RAVE_ERROR1("Failed to allocate memory for areaid = %s", areaid);
                goto done;
              }
            } else {
              RAVE_ERROR0("areaid missing for tiledef");
              goto done;
            }
            RaveObjectList_add(self->tiledefs, (RaveCoreObject*)tiledef);
            RAVE_OBJECT_RELEASE(tiledef);
          }
          RAVE_OBJECT_RELEASE(tile);
        }
      }
      RAVE_OBJECT_RELEASE(child);
    }
    
    
  }
  result = 1;
done:
  RAVE_OBJECT_RELEASE(node);
  return result;
}

/**
 * Adds all arguments to the tiledef def node
 * @param[in] tiledef - the tiledef
 * @param[in] tiledefdefNode - the xml node to get arguments added
 * @returns 1 on success or 0 on failure
 */
static int TileRegistryInternal_addArgsToNode(TileDef_t* tiledef, SimpleXmlNode_t* tiledefNode)
{
  SimpleXmlNode_t* argNode = NULL;
  int result = 0;

  RAVE_ASSERT((tiledef != NULL), "tiledef == NULL");
  RAVE_ASSERT((tiledefNode != NULL), "tiledefNode == NULL");

  // id
  argNode = SimpleXmlNode_create(tiledefNode, "arg");
  if (argNode != NULL) {
    const char* id = TileDef_getID(tiledef);
    if (id == NULL ||
        !SimpleXmlNode_addAttribute(argNode, "id", "id") ||
        !SimpleXmlNode_setText(argNode, id, strlen(id))) {
      RAVE_ERROR0("Failed to add id to tiledef");
      goto done;
    }
  } else {
    goto done;
  }
  RAVE_OBJECT_RELEASE(argNode);

  // extent
  argNode = SimpleXmlNode_create(tiledefNode, "arg");
  if (argNode != NULL) {
    double llX = 0.0, llY = 0.0, urX = 0.0, urY = 0.0;
    char extent[512];
    TileDef_getExtent(tiledef, &llX, &llY, &urX, &urY);
    if (snprintf(extent, 512, "%lf, %lf, %lf, %lf", llX, llY, urX, urY) >= 511) {
      RAVE_ERROR0("Extent became too large, can not complete writing");
      goto done;
    }
    if (!SimpleXmlNode_addAttribute(argNode, "id", "extent") ||
        !SimpleXmlNode_addAttribute(argNode, "type", "sequence") ||
        !SimpleXmlNode_setText(argNode, extent, strlen(extent))) {
      RAVE_ERROR0("Failed to add extent to tiledef");
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
 * Creates a xml node from a tiledef instance.
 * @param[in] proj - the tiledef instance
 * @returns a xml node on success or NULL on failure
 */
static SimpleXmlNode_t* TilenRegistryInternal_createNode(TileDef_t* tiledef)
{
  SimpleXmlNode_t* node = NULL;
  SimpleXmlNode_t* result = NULL;
  SimpleXmlNode_t* tiledefNode = NULL;

  RAVE_ASSERT((tiledef != NULL), "tiledef == NULL");

  node = RAVE_OBJECT_NEW(&SimpleXmlNode_TYPE);
  tiledefNode = RAVE_OBJECT_NEW(&SimpleXmlNode_TYPE);

  if (node == NULL || tiledefNode == NULL) {
    goto done;
  }
  if(!SimpleXmlNode_setName(node, "area") ||
     !SimpleXmlNode_setName(tiledefNode, "tile")) {
    goto done;
  }
  if (!SimpleXmlNode_addAttribute(node, "id", TileDef_getAreaID(tiledef))) {
    goto done;
  }
  if (!SimpleXmlNode_addAttribute(tiledefNode, "id", TileDef_getID(tiledef))) {
    goto done;
  }
  // extent
  
  double llX = 0.0, llY = 0.0, urX = 0.0, urY = 0.0;
  char extent[512];
  TileDef_getExtent(tiledef, &llX, &llY, &urX, &urY);
  if (snprintf(extent, 512, "%lf, %lf, %lf, %lf", llX, llY, urX, urY) >= 511) {
    RAVE_ERROR0("Extent became too large, can not complete writing");
    goto done;
  }
  if (!SimpleXmlNode_addAttribute(tiledefNode, "id", "extent") ||
    !SimpleXmlNode_addAttribute(tiledefNode, "type", "sequence") ||
    !SimpleXmlNode_setText(tiledefNode, extent, strlen(extent))) {
    RAVE_ERROR0("Failed to add extent to tiledef");
    goto done;
  }

  if (!TileRegistryInternal_addArgsToNode(tiledef, tiledefNode)) {
    goto done;
  }

  if (!SimpleXmlNode_addChild(node, tiledefNode)) {
    goto done;
  }

  result = RAVE_OBJECT_COPY(node);
done:
  RAVE_OBJECT_RELEASE(node);
  RAVE_OBJECT_RELEASE(tiledefNode);
  return result;
}

/*@} End of Private functions */

/*@{ Interface functions */

TileRegistry_t* TileRegistry_load(const char* filename)
{
  TileRegistry_t* result = NULL;
  if (filename != NULL) {
    result = RAVE_OBJECT_NEW(&TileRegistry_TYPE);
    if (result != NULL) {
      if (!TileRegistryInternal_loadRegistry(result, filename)) {
        RAVE_OBJECT_RELEASE(result);
      }
    }
  }
  return result;
}

int TileRegistry_add(TileRegistry_t* self, TileDef_t* tiledef)
{
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (tiledef != NULL) {
    result = RaveObjectList_add(self->tiledefs, (RaveCoreObject*)tiledef);
  }
  return result;
}

int TileRegistry_size(TileRegistry_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveObjectList_size(self->tiledefs);
}

TileDef_t* TileRegistry_get(TileRegistry_t* self, int index)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (TileDef_t*)RaveObjectList_get(self->tiledefs, index);
}

TileDef_t* TileRegistry_getByName(TileRegistry_t* self, const char* id)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (id != NULL) {
    int n = 0;
    int i = 0;
    n = RaveObjectList_size(self->tiledefs);
    for (i = 0; i < n; i++) {
      TileDef_t* tiledef = (TileDef_t*)RaveObjectList_get(self->tiledefs, i);
      if (tiledef != NULL &&
          TileDef_getID(tiledef) != NULL &&
          strcmp(id, TileDef_getID(tiledef))==0) {
        return tiledef;
      }
      RAVE_OBJECT_RELEASE(tiledef);
    }
  }
  return NULL;
}

RaveObjectList_t * TileRegistry_getByArea(TileRegistry_t* self, const char* areaid)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (areaid != NULL) {
    RaveObjectList_t* result = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
    int n = 0;
    int i = 0;
    n = RaveObjectList_size(self->tiledefs);
    for (i = 0; i < n; i++) {
      TileDef_t* tiledef = (TileDef_t*)RaveObjectList_get(self->tiledefs, i);
      if (tiledef != NULL &&
        TileDef_getAreaID(tiledef) != NULL &&
        strcmp(areaid, TileDef_getAreaID(tiledef))==0) {
          RaveObjectList_add(result, (RaveCoreObject*)tiledef);
        }
        RAVE_OBJECT_RELEASE(tiledef);
    }
    return result;
  }
  return NULL;
}

void TileRegistry_remove(TileRegistry_t* self, int index)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RaveObjectList_release(self->tiledefs, index);
}

void TileRegistry_removeByName(TileRegistry_t* self, const char* id)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (id != NULL) {
    int nlen = 0;
    int i = 0;
    int found = 0;
    nlen = RaveObjectList_size(self->tiledefs);
    for (i = 0; found == 0 && i < nlen; i++) {
      TileDef_t* tiledef = (TileDef_t*)RaveObjectList_get(self->tiledefs, i);
      if (tiledef != NULL && strcmp(id, TileDef_getID(tiledef)) == 0) {
        found = 1;
        RaveObjectList_release(self->tiledefs, i);
      }
      RAVE_OBJECT_RELEASE(tiledef);
    }
  }
}

int TileRegistry_write(TileRegistry_t* self, const char* filename)
{
  SimpleXmlNode_t* root = NULL;
  TileDef_t* childtiledef = NULL;
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
  if (!SimpleXmlNode_setName(root, "rave-tile-registry")) {
    goto done;
  }

  nproj = RaveObjectList_size(self->tiledefs);
  for (i = 0;  i < nproj; i++) {
    childtiledef = (TileDef_t*)RaveObjectList_get(self->tiledefs, i);
    if (childtiledef == NULL) {
      goto done;
    }

    childnode = TilenRegistryInternal_createNode(childtiledef);
    if (childnode == NULL || !SimpleXmlNode_addChild(root, childnode)) {
      goto done;
    }
    RAVE_OBJECT_RELEASE(childtiledef);
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
  RAVE_OBJECT_RELEASE(childtiledef);
  RAVE_OBJECT_RELEASE(childnode);
  RAVE_OBJECT_RELEASE(root);
  if (fp != NULL) {
    fclose(fp);
  }
  return result;
}
/*@} End of Interface functions */

RaveCoreObjectType TileRegistry_TYPE = {
    "TileRegistry",
    sizeof(TileRegistry_t),
    TileRegistry_constructor,
    TileRegistry_destructor,
    TileRegistry_copyconstructor
};
