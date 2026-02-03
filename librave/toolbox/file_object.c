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
 * Defines a general file object
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-12-17
 */
#include "file_object.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>
#include <stdio.h>


/**
 * Represents the file object
 */
struct _FileObject_t {
  RAVE_OBJECT_HEAD /** Always on top */
  FileObjectRestrictionMode mode; /**< the restrictions that should be applied on object */
  char name[512];
  RaveObjectList_t* groups;
  RaveValue_t* attributes; /**< the attributes (as a hash table) */
  RaveData2D_t* data;
  LazyDataset_t* lazyDataset;
};

/*@{ Private functions */
/**
 * Ensures that we have got data in the data-table set
 */
static RaveData2D_t* FileObjectInternal_ensureData2D(FileObject_t* self)
{
  if (self->lazyDataset != NULL) {
    RaveData2D_t* loaded = LazyDataset_get(self->lazyDataset);
    if (loaded != NULL) {
      RAVE_DEBUG0("FileObject_ensureData2D: LazyDataset fetched");
      RAVE_OBJECT_RELEASE(self->data);
      self->data = RAVE_OBJECT_COPY(loaded);
      RAVE_OBJECT_RELEASE(self->lazyDataset);
    } else {
      RAVE_ERROR0("Failed to load dataset");
    }
    RAVE_OBJECT_RELEASE(loaded);
  }
  return self->data;
}

/**
 * Constructor.
 */
static int FileObject_constructor(RaveCoreObject* obj)
{
  FileObject_t* this = (FileObject_t*)obj;
  this->mode = FileObjectRestrictionMode_ODIM;
  this->groups = RAVE_OBJECT_NEW(&RaveObjectList_TYPE);
  this->attributes = RaveValue_createHashTable(NULL);
  this->data = NULL;
  this->lazyDataset = NULL;
  strcpy(this->name, "");
  if (this->groups == NULL || this->attributes == NULL) {
    RAVE_OBJECT_RELEASE(this->groups);
    RAVE_OBJECT_RELEASE(this->attributes);
    return 0;
  }
  return 1;
}

/**
 * Copy constructor
 */
static int FileObject_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  FileObject_t* this = (FileObject_t*)obj;
  FileObject_t* src = (FileObject_t*)srcobj;
  this->mode = src->mode;
  this->groups = RAVE_OBJECT_CLONE(src->groups);
  this->attributes = RAVE_OBJECT_CLONE(src->attributes);
  this->data = NULL;
  this->lazyDataset = NULL;
  strcpy(this->name, src->name);
  if (this->groups == NULL || this->attributes == NULL) {
    goto fail;
  }
  if (src->data != NULL || src->lazyDataset != NULL) {
    this->data = RAVE_OBJECT_CLONE(FileObjectInternal_ensureData2D(src));
    if (this->data == NULL) {
      goto fail;
    }
  }
  return 1;
fail:
  RAVE_OBJECT_RELEASE(this->groups);
  RAVE_OBJECT_RELEASE(this->attributes);
  RAVE_OBJECT_RELEASE(this->data);
  return 0;
}

/**
 * Destroys the odim source
 * @param[in] obj - the the OdimSource_t instance
 */
static void FileObject_destructor(RaveCoreObject* obj)
{
  FileObject_t* this = (FileObject_t*)obj;
  RAVE_OBJECT_RELEASE(this->groups);
  RAVE_OBJECT_RELEASE(this->attributes);
  RAVE_OBJECT_RELEASE(this->data);
  RAVE_OBJECT_RELEASE(this->lazyDataset);
}

static FileObject_t* FileObjectInternal_find(FileObject_t* self, const char* name)
{
  int nlen = 0, i = 0;
  nlen = RaveObjectList_size(self->groups);
  for (i = 0; i < nlen; i++) {
    FileObject_t* found = (FileObject_t*)RaveObjectList_get(self->groups, i);
    if (found != NULL) {
      if (strcmp(found->name, name) == 0) {
        return found;
      }
    }
    RAVE_OBJECT_RELEASE(found);
  }
  return NULL;
}

static FileObject_t* FileObjectInternal_createGroup(FileObject_t* self, const char* name)
{
  FileObject_t* result =  RAVE_OBJECT_NEW(&FileObject_TYPE);
  if (result == NULL || !RaveObjectList_add(self->groups, (RaveCoreObject*)result)) {
    RAVE_ERROR0("Could not create file object or add to groups");
    goto fail;
  }
  if (!FileObject_setName(result, name)) {
    RAVE_ERROR0("Could not set name\n");
    goto fail;
  }
  return result;
fail:
  RAVE_OBJECT_RELEASE(result);
  return NULL;
}

static char* FileObjectInternal_getPathPart(char* firstptr, char* buff, int* bufflen)
{
  int ctr = 0;
  if (firstptr != NULL && buff != NULL) {
    char* iptr = firstptr;
    while (*iptr != '\0' && *iptr != '/') {
      *buff = *iptr;
      ctr += 1;
      buff++; 
      iptr++;
    }
    *buff='\0';
    *bufflen = ctr;
    if (*iptr == '/') {
      iptr++;
    }
    return iptr;
  }
  *bufflen = 0;
  return NULL;
}

static FileObject_t* FileObjectInternal_get(FileObject_t* self, const char* name, size_t namelen, int createOnMissing)
{
  FileObject_t* found = NULL;
  FileObject_t* result = NULL;
  char *firstptr = NULL;

  char buff[1024], tmpbuff[1024];
  int tblen = 0;
  int bytestocopy = namelen < 1023 ? namelen : 1022;
  strncpy(buff, name, bytestocopy);

  buff[bytestocopy] = '\0';
  if (bytestocopy < 1) {
    /* Always return self if providing empty name */
    return RAVE_OBJECT_COPY(self);
  }

  firstptr = (char*)buff;

  if (*firstptr == '/') { /* Remove leading / since we are already here. */
    firstptr ++;
  }

  if (*firstptr == '\0') {
    return RAVE_OBJECT_COPY(self);
  } else {
    firstptr = FileObjectInternal_getPathPart(firstptr, (char*)tmpbuff, &tblen);
    if (tblen > 0 && strcmp("", tmpbuff) != 0) {
      found = FileObjectInternal_find(self, tmpbuff);
      if (found == NULL && createOnMissing) {
        found = FileObjectInternal_createGroup(self, tmpbuff);
        if (found == NULL) {
          goto fail;
        }
      }  RAVE_ASSERT((self != NULL), "self == NULL");

      if (found && firstptr != NULL && strlen(firstptr) > 0) {
        FileObject_t* child = FileObjectInternal_get(found, firstptr, strlen(firstptr), createOnMissing);
        RAVE_OBJECT_RELEASE(found);
        found = RAVE_OBJECT_COPY(child);
        RAVE_OBJECT_RELEASE(child);
      }
    }
  }

  result = RAVE_OBJECT_COPY(found);
fail:
  RAVE_OBJECT_RELEASE(found);
  return result;
}

static int FileObjectInternal_addStringToBuffer(char** ppstr, size_t* pnalloc, size_t* ppos, const char* nstr)
{
  char* pstr = *ppstr;
  size_t pos = *ppos;
  size_t nalloc = *pnalloc;
  int len = strlen(nstr);
  if (nalloc < pos + len + 1) {
    char* newbuf = RAVE_MALLOC(sizeof(char) * (nalloc * 2));
    if (newbuf != NULL) {
      memcpy(newbuf, pstr, sizeof(char)*nalloc);
      nalloc *= 2;
      RAVE_FREE(pstr);
      pstr = newbuf;
      *ppstr = pstr;
      *pnalloc = nalloc;
    } else {
      RAVE_ERROR0("Failure when reallocating memory for string buffer");
      return 0;
    }
  }
  memcpy(pstr + pos, nstr, len);
  pos += len;
  *(pstr + pos) = '\0';
  *ppos = pos;
  return 1;
}

static int FileObjectInternal_toString(FileObject_t* self, char** ppstr, size_t* pnalloc, size_t* ppos, int indent)
{
  RaveList_t* keys = NULL;
  char tmpbuf[1024];
  int i = 0, ngroups = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (ppstr == NULL || pnalloc == NULL || ppos == NULL) {
    return 0;
  }
  /* Group or dataset */
  if (!FileObject_isDataset(self)) {
    if (strcmp("", self->name) == 0) {
      snprintf(tmpbuf, 1024, "%*sGROUP '/' {\n", indent, "");
    } else {
      snprintf(tmpbuf, 1024, "%*sGROUP '%s' {\n", indent, "", self->name);
    }
    if (!FileObjectInternal_addStringToBuffer(ppstr, pnalloc, ppos, tmpbuf)) {
      RAVE_ERROR0("Failed to add string to buffer\n");
      goto fail;
    }
  } else {
    long xsize = 0, ysize = 0;
    RaveDataType datatype = RaveDataType_UNDEFINED;

    if (strcmp("", self->name) == 0) {
      snprintf(tmpbuf, 1024, "%*sDATASET '/' {\n", indent, "");
    } else {
      snprintf(tmpbuf, 1024, "%*sDATASET '%s' {\n", indent, "", self->name);
    }
    if (!FileObjectInternal_addStringToBuffer(ppstr, pnalloc, ppos, tmpbuf)) {
      RAVE_ERROR0("Failed to add string to buffer\n");
      goto fail;
    }

    xsize = FileObject_getDatasetX(self);
    ysize = FileObject_getDatasetY(self);
    datatype = FileObject_getDatasetType(self);


    snprintf(tmpbuf, 1024, "%*sDIMENSIONS = %ld x %ld\n", indent+2, "", xsize, ysize);
    if (!FileObjectInternal_addStringToBuffer(ppstr, pnalloc, ppos, tmpbuf)) {
      RAVE_ERROR0("Failed to add string to buffer\n");
      goto fail;
    }
    snprintf(tmpbuf, 1024, "%*sDATATYPE = %s\n", indent+2, "", RaveTypes_getDataTypeName(datatype));
    if (!FileObjectInternal_addStringToBuffer(ppstr, pnalloc, ppos, tmpbuf)) {
      RAVE_ERROR0("Failed to add string to buffer\n");
      goto fail;
    }

    if (FileObject_isDatasetLoaded(self)) {
      snprintf(tmpbuf, 1024, "%*sLOADED\n", indent+2, "");
    } else {
      snprintf(tmpbuf, 1024, "%*sNOT LOADED\n", indent+2, "");
    }
    if (!FileObjectInternal_addStringToBuffer(ppstr, pnalloc, ppos, tmpbuf)) {
      RAVE_ERROR0("Failed to add string to buffer\n");
      goto fail;
    }
  }

  /* Attributes */
  keys = RaveValueHash_keys(self->attributes);
  if (keys != NULL) {
    int i = 0, alen = RaveList_size(keys);
    for (i = 0; i < alen; i++) {
      const char* key = (const char*)RaveList_get(keys, i);
      RaveValue_t* val = RaveValueHash_get(self->attributes, key);
      if (RaveValue_type(val) == RaveValue_Type_Boolean) {
        int x = RaveValue_toBoolean(val);
        if (x == 0) {
          snprintf(tmpbuf, 1024, "%*s = false\n", indent+2, key);
        } else {
          snprintf(tmpbuf, 1024, "%*s = true\n", indent+2, key);
        }
      } else if (RaveValue_type(val) == RaveValue_Type_Long) {
        long x = RaveValue_toLong(val);
        snprintf(tmpbuf, 1024, "%*s%s = %ld\n", indent+2, "", key, x);
      } else if (RaveValue_type(val) == RaveValue_Type_Double) {
        double x = RaveValue_toDouble(val);
        snprintf(tmpbuf, 1024, "%*s%s = %g\n", indent+2, "", key, x);
      } else if (RaveValue_type(val) == RaveValue_Type_String) {
        const char* x = RaveValue_toString(val);
        snprintf(tmpbuf, 1024, "%*s%s = %s\n", indent+2, "", key, x);
      } else if (RaveValue_type(val) == RaveValue_Type_List) {
        snprintf(tmpbuf, 1024, "%*s%s = LIST\n", indent+2, "", key);
      } else if (RaveValue_type(val) == RaveValue_Type_Hashtable) {
        snprintf(tmpbuf, 1024, "%*s%s = HASH\n", indent+2, "", key);
      } else {
        RAVE_OBJECT_RELEASE(val);
        continue;
      }
      RAVE_OBJECT_RELEASE(val);
      if (!FileObjectInternal_addStringToBuffer(ppstr, pnalloc, ppos, tmpbuf)) {
        RAVE_ERROR0("Failed to add string to buffer\n");
        goto fail;
      }
    }
    RaveList_freeAndDestroy(&keys);
  }

  /* Child groups */
  ngroups = RaveObjectList_size(self->groups);
  for (i = 0; i < ngroups; i++) {
    FileObject_t* group = (FileObject_t*)RaveObjectList_get(self->groups, i);
    if (group != NULL) {
      if (!FileObjectInternal_toString(group, ppstr, pnalloc, ppos, indent + 2)) {
        RAVE_ERROR0("Failure when adding child group information");
        RAVE_OBJECT_RELEASE(group);
        goto fail;
      }
      RAVE_OBJECT_RELEASE(group);
    } else {
      RAVE_ERROR0("Programming error");
      goto fail;
    }
  }

  snprintf(tmpbuf, 1024, "%*s}\n", indent, "");
  if (!FileObjectInternal_addStringToBuffer(ppstr, pnalloc, ppos, tmpbuf)) {
    RAVE_ERROR0("Failed to add string to buffer\n");
    goto fail;
  }

  return 1;
fail:
  return 0;
}

/*@} End of Private functions */

/*@{ Interface functions */
FileObject_t* FileObject_createRoot(void)
{
  FileObject_t* result = RAVE_OBJECT_NEW(&FileObject_TYPE);
  return result;
}

int FileObjectInternal_areNamesSet(FileObject_t* self)
{
  int ngroups = 0, gi = 0;
  int result = 1;

  if (self == NULL) {
    return 0;
  }

  if (strcmp(self->name, "") == 0) {
    return 0;
  }

  ngroups = FileObject_numberOfGroups(self);
  for (gi = 0; result == 1 && gi < ngroups; gi++) {
    FileObject_t* group = FileObject_getByIndex(self, gi);
    if (group == NULL || !FileObjectInternal_areNamesSet(group)) {
      result = 0;
    }
    RAVE_OBJECT_RELEASE(group);
  }
  return result;

}

int FileObject_areNamesSet(FileObject_t* self)
{
  int ngroups = 0, gi = 0;
  int result = 1;

  if (self == NULL) {
    return 0;
  }
  /* We assume this node is root and hence empty string is allowed so just check sub groups for != "" */

  ngroups = FileObject_numberOfGroups(self);
  for (gi = 0; result == 1 && gi < ngroups; gi++) {
    FileObject_t* group = FileObject_getByIndex(self, gi);
    if (group == NULL || !FileObjectInternal_areNamesSet(group)) {
      result = 0;
    }
    RAVE_OBJECT_RELEASE(group);
  }
  return result;
}

char* FileObject_toString(FileObject_t* self)
{
  char* pstr = NULL;
  size_t nalloc = 1024;
  size_t pos = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  pstr = RAVE_MALLOC(sizeof(char) * nalloc);
  if (pstr == NULL) {
    RAVE_ERROR0("Failed to allocate memory for string representation");
    return NULL;
  }
  if (!FileObjectInternal_toString(self, &pstr, &nalloc, &pos, 0)) {
    return NULL;
  }
  return pstr;
}

void FileObject_setRestrictionMode(FileObject_t* self, FileObjectRestrictionMode mode)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  self->mode = mode;
}

FileObjectRestrictionMode FileObject_getRestrictionMode(FileObject_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->mode;
}


FileObject_t* FileObject_get(FileObject_t* self, const char* name)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return FileObjectInternal_get(self, name, strlen(name), 0);
}

FileObject_t* FileObject_getByIndex(FileObject_t* self, int index)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (FileObject_t*)RaveObjectList_get(self->groups, index);
}

FileObject_t* FileObject_create(FileObject_t* self, const char* name)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return FileObjectInternal_get(self, name, strlen(name), 1);
}

RaveData2D_t* FileObject_getData(FileObject_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  FileObjectInternal_ensureData2D(self);
  return RAVE_OBJECT_COPY(self->data);
}

int FileObject_setData(FileObject_t* self, RaveData2D_t* data)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_OBJECT_RELEASE(self->data);
  RAVE_OBJECT_RELEASE(self->lazyDataset);
  self->data = RAVE_OBJECT_COPY(data);
  return 1;
}

int FileObject_isDataset(FileObject_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->data != NULL || self->lazyDataset != NULL) {
    return 1;
  }
  return 0;
}

int FileObject_isDatasetLoaded(FileObject_t* self)
{
  if (self->data != NULL) {
    return 1;
  }
  return 0;
}

long FileObject_getDatasetX(FileObject_t* self)
{
  if (self->lazyDataset != NULL) {
    return LazyDataset_getXsize(self->lazyDataset);
  }
  return RaveData2D_getXsize(self->data);
}

long FileObject_getDatasetY(FileObject_t* self)
{
  if (self->lazyDataset != NULL) {
    return LazyDataset_getYsize(self->lazyDataset);
  }
  return RaveData2D_getYsize(self->data);
}

RaveDataType FileObject_getDatasetType(FileObject_t* self)
{
  if (self->lazyDataset != NULL) {
    return LazyDataset_getDataType(self->lazyDataset);
  }
  return RaveData2D_getType(self->data);
}

int FileObject_setName(FileObject_t* self, const char* name)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (strpbrk(name, "/") != NULL) {
    RAVE_ERROR0("Names must not contain '/'");
    return 0;
  }
  strcpy(self->name, name);
  return 1;  
}

const char* FileObject_getName(FileObject_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->name;
}

size_t FileObject_numberOfGroups(FileObject_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveObjectList_size(self->groups);
}

int FileObject_exists(FileObject_t* self, const char* name)
{
  FileObject_t* found = NULL;
  int result = 0;
  RAVE_ASSERT((self != NULL), "self == NULL");
  found = FileObjectInternal_get(self, name, strlen(name), 0);
  result = (found != NULL);
  RAVE_OBJECT_RELEASE(found);
  return result;
}

RaveObjectList_t* FileObject_groups(FileObject_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RAVE_OBJECT_COPY(self->groups);
}

RaveValue_t* FileObject_attributes(FileObject_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RAVE_OBJECT_COPY(self->attributes);
}

int FileObject_addAttribute(FileObject_t* self, const char* name, RaveValue_t* value)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return RaveValueHash_put(self->attributes, name, value);
}

int FileObject_setLazyDataset(FileObject_t* self, LazyDataset_t* lazyDataset)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (self->data == NULL) {
    self->lazyDataset = RAVE_OBJECT_COPY(lazyDataset);
    return 1;
  } else {
    RAVE_ERROR0("Trying to set lazy dataset loader when data exists");
    return 0;
  }
}
/*@} End of Interface functions */

RaveCoreObjectType FileObject_TYPE = {
    "FileObject",
    sizeof(FileObject_t),
    FileObject_constructor,
    FileObject_destructor,
    FileObject_copyconstructor
};
