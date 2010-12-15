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
 * Provides support for reading and writing projections to and from
 * an xml-file.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-12-09
 */
#include "projectionregistry.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "raveobject_list.h"
#include "rave_simplexml.h"
#include <string.h>

/**
 * Represents the registry
 */
struct _ProjectionRegistry_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveObjectList_t* projections; /**< the list of areas */
};


/*@{ Private functions */
/**
 * Constructor.
 */
static int ProjectionRegistry_constructor(RaveCoreObject* obj)
{
  ProjectionRegistry_t* this = (ProjectionRegistry_t*)obj;
  this->projections = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  if (this->projections == NULL) {
    goto error;
  }

  return 1;
error:
  RAVE_OBJECT_RELEASE(this->projections);
  return 0;
}

/**
 * Copy constructor
 */
static int ProjectionRegistry_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  ProjectionRegistry_t* this = (ProjectionRegistry_t*)obj;
  ProjectionRegistry_t* src = (ProjectionRegistry_t*)srcobj;

  this->projections = RAVE_OBJECT_CLONE(src->projections);

  if (this->projections == NULL) {
    goto error;
  }
  return 1;
error:
  RAVE_OBJECT_RELEASE(this->projections);
  return 0;
}

/**
 * Destroys the registry
 * @param[in] obj - the the AreaRegistry_t instance
 */
static void ProjectionRegistry_destructor(RaveCoreObject* obj)
{
  ProjectionRegistry_t* this = (ProjectionRegistry_t*)obj;
  RAVE_OBJECT_RELEASE(this->projections);
}

/**
 * Creates an projection from a xml-node
 * @param[in] node the xml node
 * @returns an projection on success or NULL on failure
 */
static Projection_t* ProjectionRegistryInternal_createProjFromNode(SimpleXmlNode_t* node)
{
  Projection_t* proj = NULL;
  Projection_t* result = NULL;
  if (node != NULL && SimpleXmlNode_getName(node) != NULL &&
      strcasecmp("projection", SimpleXmlNode_getName(node)) == 0) {
    proj = RAVE_OBJECT_NEW(&Projection_TYPE);
    if (proj != NULL) {
      const char* id = SimpleXmlNode_getAttribute(node, "id");
      const char* descr = NULL;
      const char* projdef = NULL;

      SimpleXmlNode_t* descrNode = SimpleXmlNode_getChildByName(node, "description");
      SimpleXmlNode_t* projdefNode = SimpleXmlNode_getChildByName(node, "projdef");

      if (descrNode != NULL) {
        descr = SimpleXmlNode_getText(descrNode);
        RAVE_OBJECT_RELEASE(descrNode);
      }
      if (projdefNode != NULL) {
        projdef = SimpleXmlNode_getText(projdefNode);
        RAVE_OBJECT_RELEASE(projdefNode);
      }

      if (id == NULL || descr == NULL || projdef == NULL) {
        RAVE_ERROR0("Id, description or projection definition missing");
        goto done;
      }

      fprintf(stderr, "id='%s', descr='%s', projdef='%s'\n", id, descr, projdef);

      if (!Projection_init(proj, id, descr, projdef)) {
        RAVE_ERROR0("Failed to initialize projection");
        goto done;
      }
    }
  }

  result = RAVE_OBJECT_COPY(proj);
done:
  RAVE_OBJECT_RELEASE(proj);
  return result;
}
/*@} End of Private functions */

/*@{ Interface functions */
ProjectionRegistry_t* ProjectionRegistry_loadRegistry(const char* filename)
{
  SimpleXmlNode_t* node = NULL;
  ProjectionRegistry_t* registry = NULL;
  ProjectionRegistry_t* result = NULL;

  int nrchildren = 0;
  int i = 0;
  RAVE_ASSERT((filename != NULL), "filename == NULL");

  node = SimpleXmlNode_parseFile(filename);
  if (node == NULL) {
    goto done;
  }
  registry = RAVE_OBJECT_NEW(&ProjectionRegistry_TYPE);
  if (registry == NULL) {
    goto done;
  }

  nrchildren = SimpleXmlNode_getNumberOfChildren(node);
  for (i = 0; i < nrchildren; i++) {
    SimpleXmlNode_t* child = SimpleXmlNode_getChild(node, i);
    Projection_t* projection = ProjectionRegistryInternal_createProjFromNode(child);
    if (projection != NULL) {
      RaveObjectList_add(registry->projections, (RaveCoreObject*)projection);
    }
    RAVE_OBJECT_RELEASE(child);
    RAVE_OBJECT_RELEASE(projection);
  }

  result = RAVE_OBJECT_COPY(registry);
done:
  RAVE_OBJECT_RELEASE(node);
  RAVE_OBJECT_RELEASE(registry);
  return result;

}

int ProjectionRegistry_size(ProjectionRegistry_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveObjectList_size(self->projections);
}

Projection_t* ProjectionRegistry_get(ProjectionRegistry_t* self, int index)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (Projection_t*)RaveObjectList_get(self->projections, index);
}

Projection_t* ProjectionRegistry_getByName(ProjectionRegistry_t* self, const char* pcsid)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (pcsid != NULL) {
    int n = 0;
    int i = 0;
    n = RaveObjectList_size(self->projections);
    for (i = 0; i < n; i++) {
      Projection_t* proj = (Projection_t*)RaveObjectList_get(self->projections, i);
      if (proj != NULL &&
          Projection_getID(proj) != NULL &&
          strcmp(pcsid, Projection_getID(proj))==0) {
        return proj;
      }
      RAVE_OBJECT_RELEASE(proj);
    }
  }
  return NULL;
}

/*@} End of Interface functions */

RaveCoreObjectType ProjectionRegistry_TYPE = {
    "ProjectionRegistry",
    sizeof(ProjectionRegistry_t),
    ProjectionRegistry_constructor,
    ProjectionRegistry_destructor,
    ProjectionRegistry_copyconstructor
};
