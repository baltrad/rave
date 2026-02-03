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
 * File Object ODIM IO functions
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-11-12
 */
#include "file_object_odim_io.h"
#include "rave_hlhdf_utilities.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>
#include "odim_io_utilities.h"
#include "lazy_dataset.h"
#include <math.h>

/**
 * The Polar ODIM IO adaptor
 */
struct _FileObjectOdimIO_t {
  RAVE_OBJECT_HEAD /** Always on top */
  RaveIO_ODIM_Version version;
  int strict; /**< if strict writing should be enforced or not */
  char error_message[1024];                /**< if an error occurs during writing an error message might give you the reason */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int FileObjectOdimIO_constructor(RaveCoreObject* obj)
{
  ((FileObjectOdimIO_t*)obj)->version = RaveIO_ODIM_Version_2_4;
  ((FileObjectOdimIO_t*)obj)->strict = 0;
  strcpy(((FileObjectOdimIO_t*)obj)->error_message, "");
  return 1;
}

/**
 * Copy constructor
 */
static int FileObjectOdimIO_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  FileObjectOdimIO_t* self = (FileObjectOdimIO_t*)obj;
  FileObjectOdimIO_t* src = (FileObjectOdimIO_t*)srcobj;

  self->version = src->version;
  self->strict = src->strict;
  strcpy(self->error_message, src->error_message);

  return 1;
}

/**
 * Destroys the object
 * @param[in] obj - the instance
 */
static void FileObjectOdimIO_destructor(RaveCoreObject* obj)
{
}

/*@} End of Private functions */

/*@{ Interface functions */
void FileObjectOdimIO_setVersion(FileObjectOdimIO_t* self, RaveIO_ODIM_Version version)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->version = version;
}

RaveIO_ODIM_Version FileObjectOdimIO_getVersion(FileObjectOdimIO_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->version;
}

void FileObjectOdimIO_setStrict(FileObjectOdimIO_t* self, int strict)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->strict = strict;
}

int FileObjectOdimIO_isStrict(FileObjectOdimIO_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->strict;
}

const char* FileObjectOdimIO_getErrorMessage(FileObjectOdimIO_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->error_message;
}

static int FileObjectOdimIOInternal_getGroupChildName(const char* name, char* groupname, size_t glen, char* childname, size_t clen)
{
  char* lastptr = NULL;
  if (name != NULL) {
    lastptr = strrchr(name, '/');
    if (lastptr != NULL) {
      size_t plen = (lastptr - name);
      size_t olen = strlen(name) - plen - 1;
      if (glen > plen + 1) {
        if (plen == 0) { /* Must be first character and it is /, allow it. */
          if (glen > 2) {
            groupname[0] = '/';
            groupname[1] = '\0';
          } else {
            goto fail;
          }
        } else {
          memcpy(groupname, name, sizeof(char)*plen);
          groupname[plen] = '\0';
        }
      } else {
        goto fail;
      }

      if (clen > olen + 1) {
        memcpy(childname, lastptr + 1, sizeof(char)*olen);
        childname[olen] = '\0';
      } else {
        goto fail;
      }
    }
  }
  return 1;
fail:
  return 0;
}

FileObject_t* FileObjectOdimIO_read(FileObjectOdimIO_t* self, LazyNodeListReader_t* lazyReader)
{
  int n = 0, i = 0;
  OdimIoUtilityArg arg;
  FileObject_t *result = NULL, *fileobject = NULL;
  RaveValue_t* firststr = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((lazyReader != NULL), "lazyReader == NULL");

  fileobject = FileObject_createRoot();
  if (fileobject == NULL) {
    RAVE_ERROR0("Failed to allocate memory");
    goto fail;
  }

  arg.lazyReader = lazyReader;
  arg.nodelist = LazyNodeListReader_getHLNodeList(lazyReader);
  arg.object = (RaveCoreObject*)fileobject;
  arg.version = self->version;

  n = HLNodeList_getNumberOfNodes(arg.nodelist);
  for (i = 0; i < n; i++) {
    HL_Node* node = HLNodeList_getNodeByIndex(arg.nodelist, i);
    const char* nodeName = HLNode_getName(node);
    FileObject_t* fot = NULL;

    HL_Type hltype = HLNode_getType(node);
    if (hltype == GROUP_ID) {
      fot = FileObject_create(fileobject, nodeName);
    } else if (hltype == DATASET_ID) {
      fot = FileObject_create(fileobject, nodeName);  /* Dataset is a type of group so just create it directly without extracting group/dataset */
      if (fot != NULL) {
        hsize_t d0 = HLNode_getDimension(node, 0);
        hsize_t d1 = HLNode_getDimension(node, 1);
        RaveDataType dataType = RaveHL_hlhdfToRaveType(HLNode_getFormat(node));
        if (dataType != RaveDataType_UNDEFINED) {
          void* data = HLNode_getData(node);
          if (data == NULL) {
            LazyDataset_t* datasetReader = RAVE_OBJECT_NEW(&LazyDataset_TYPE);
            if (datasetReader == NULL || !LazyDataset_init(datasetReader, lazyReader, nodeName) || !FileObject_setLazyDataset(fot, datasetReader)) {
              RAVE_ERROR0("Failed to initialize lazy reader");
              RAVE_OBJECT_RELEASE(datasetReader);
              RAVE_OBJECT_RELEASE(fot);
              goto fail;
            }
            RAVE_OBJECT_RELEASE(datasetReader);
          } else {
            RaveData2D_t* data2d = RAVE_OBJECT_NEW(&RaveData2D_TYPE);
            if (data2d == NULL || !RaveData2D_setData(data2d, d1, d0, data, dataType) || !FileObject_setData(fot, data2d)) {
              RAVE_ERROR0("Failed to set data2d data");
              RAVE_OBJECT_RELEASE(data2d);
              RAVE_OBJECT_RELEASE(fot);
              goto fail;
            }
            RAVE_OBJECT_RELEASE(data2d);
          }
        } else {
          RAVE_ERROR0("Undefined datatype for dataset");
          result = 0;
        }
      }
    } else if (hltype == ATTRIBUTE_ID) {
      char groupname[2048];
      char attrname[2048];
      if (FileObjectOdimIOInternal_getGroupChildName(nodeName, groupname, sizeof(groupname), attrname, sizeof(attrname))) {
        fot = FileObject_create(fileobject, groupname);
        if (fot != NULL) {
          RaveValue_t* value = RaveHL_createValue(node);
          if (value == NULL || !FileObject_addAttribute(fot, attrname, value)) {
            RAVE_ERROR1("Failed to read attribute: %s", attrname);
            RAVE_OBJECT_RELEASE(value);
            RAVE_OBJECT_RELEASE(fot);
            goto fail;
          }
          RAVE_OBJECT_RELEASE(value);
        }
      }
    }
    RAVE_OBJECT_RELEASE(fot);
  }

  result = RAVE_OBJECT_COPY(fileobject);
fail:
  RAVE_OBJECT_RELEASE(fileobject);
  RAVE_OBJECT_RELEASE(firststr);
  return result;
}

static int FileObjectOdimIOInternal_fill(FileObjectOdimIO_t* self, FileObject_t* fobj, HL_NodeList* nodelist, const char* basename)
{
  int gi = 0, ngroups = 0;
  char nodename[1024];
  RaveList_t* keys = NULL;
  RaveValue_t* attributes = NULL;

  strcpy(nodename, "");
  if (strcmp(FileObject_getName(fobj), "")!=0) {
    sprintf(nodename, "%s/%s", basename, FileObject_getName(fobj));
  }

  if (!FileObject_isDataset(fobj) && strcmp("", FileObject_getName(fobj)) != 0) {
    /* The root is not created as a group since that is implicit */
    if (!RaveHL_createGroupUnlessExists(nodelist, nodename)) {
      RAVE_ERROR0("Failed to create group in nodelist");
      goto fail;
    }
  } else if (FileObject_isDataset(fobj)) {
    RaveData2D_t* d2d = FileObject_getData(fobj);
    void* data = NULL;
    long xsize = 0, ysize = 0;
    RaveDataType datatype = RaveDataType_UNDEFINED;
    if (d2d == NULL) {
      RAVE_ERROR0("Failed to get data from file object");
      goto fail;
    }
    data = RaveData2D_getData(d2d);
    xsize = RaveData2D_getXsize(d2d);
    ysize = RaveData2D_getYsize(d2d);
    datatype = RaveData2D_getType(d2d);

    RAVE_OBJECT_RELEASE(d2d);

    if (!RaveHL_createDataset(nodelist, data, xsize, ysize, datatype, nodename)) {
      RAVE_ERROR0("Failed to create dataset in nodelist");
      goto fail;
    }
  }

  ngroups = FileObject_numberOfGroups(fobj);
  for (gi = 0; gi < ngroups; gi++) {
    FileObject_t* group = FileObject_getByIndex(fobj, gi);
    if (group == NULL || !FileObjectOdimIOInternal_fill(self, group, nodelist, nodename)) {
      RAVE_OBJECT_RELEASE(group);
      goto fail;
    }
    RAVE_OBJECT_RELEASE(group);
  }

  attributes = FileObject_attributes(fobj);
  if (attributes == NULL) {
    goto fail;
  }

  keys = RaveValueHash_keys(attributes);
  if (keys != NULL) {
    int i = 0, alen = RaveList_size(keys);
    for (i = 0; i < alen; i++) {
      const char* key = (const char*)RaveList_get(keys, i);
      RaveValue_t* value = RaveValueHash_get(attributes, key);
      if (!RaveHL_addRaveValue(nodelist, value, "%s/%s", nodename, key)) {
        RAVE_ERROR0("Failed to add rave value to attributes");
        RAVE_OBJECT_RELEASE(value);
        goto fail;
      }
      RAVE_OBJECT_RELEASE(value);
    }
  } else {
    RAVE_ERROR0("Could not get keys");
    goto fail;
  }

  if (keys != NULL) {
      RaveList_freeAndDestroy(&keys);
  }
  RAVE_OBJECT_RELEASE(attributes);

  return 1;
fail:
  if (keys != NULL) {
      RaveList_freeAndDestroy(&keys);
  }
  RAVE_OBJECT_RELEASE(attributes);

  return 0;
}

int FileObjectOdimIO_fill(FileObjectOdimIO_t* self, FileObject_t* fobj, HL_NodeList* nodelist)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (fobj == NULL || nodelist == NULL) {
    RAVE_ERROR0("Programming error, must provide file object and nodelist");
    return 0;
  }

  if (!FileObject_areNamesSet(fobj)) {
    RAVE_ERROR0("Must set all names in file object structure");
    return 0;
  }

  return FileObjectOdimIOInternal_fill(self, fobj, nodelist, "");
}

/*@} End of Interface functions */

RaveCoreObjectType FileObjectOdimIO_TYPE = {
    "FileObjectOdimIO",
    sizeof(FileObjectOdimIO_t),
    FileObjectOdimIO_constructor,
    FileObjectOdimIO_destructor,
    FileObjectOdimIO_copyconstructor
};
