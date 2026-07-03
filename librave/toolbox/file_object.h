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
 * This is a general file object.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2025-12-17
 */
#ifndef FILEOBJECT_H
#define FILEOBJECT_H
#include "rave_object.h"
#include "rave_value.h"
#include "rave_types.h"
#include "rave_list.h"
#include "raveobject_list.h"
#include "lazy_dataset.h"
#include "rave_data2d.h"

typedef enum FileObjectRestrictionMode {
    FileObjectRestrictionMode_NONE=0,    
    FileObjectRestrictionMode_ODIM=1
} FileObjectRestrictionMode;

/**
 * Defines a general file object
 */
typedef struct _FileObject_t FileObject_t;

/**
 * Type definition to use when creating a rave object.
 */
extern RaveCoreObjectType FileObject_TYPE;

/**
 * Creates the root group to get started.
 * @return the root group with name set to ""
 */
FileObject_t* FileObject_createRoot(void);

/**
 * Sets the name of this file object.
 * @param[in] self - self
 * @param[in] name - name of the file object. Must not contain '/'.
 * @return 1 on success otherwise 0
 */
int FileObject_setName(FileObject_t* self, const char* name);
    
/**
 * @param[in] self - self
 * @return the name of this file object
 */
const char* FileObject_getName(FileObject_t* self);

/**
 * Verifies that all names are set in self and all child groups.
 * The root group is allowed to have an empty string as name. All childs must
 * have a name != "".
 * @param[in] self - self
 * @return 1 if groups names are ok, otherwise 0
 */
int FileObject_areNamesSet(FileObject_t* self);

/**
 * Creates a string representation of the file object including subgroups and attributes.
 * Arrays in attributes and dataset values are not displayed.
 * @param[in] self - self
 * @return a string representation of self
 */
char* FileObject_toString(FileObject_t* self);

/**
 * Sets a restriction mode on the file object structure. For example if specifying
 * ODIM, then datasets are not allowed to contain subgroups.
 * @param[in] self - self
 * @param[in] mode - the mode to use
 */
void FileObject_setRestrictionMode(FileObject_t* self, FileObjectRestrictionMode mode);

/**
 * @param[in] self - self
 * @return the restriction mode
 */
FileObjectRestrictionMode FileObject_getRestrictionMode(FileObject_t* self);

/**
 * Gets the subgroup within self.
 * Can specify more than one level of subgroup, for example it is allowed to call
 * FileObject_get(self, "/this/that/group1"). Note, it is only group/dataset names
 * that can be handled.
 * @param[in] self - self
 * @param[in] name - name of group wanted
 * @return the file object instance if found otherwise NULL
 */
FileObject_t* FileObject_get(FileObject_t* self, const char* name);

/**
 * Gets (or creates) the subgroup within self.
 * Can specify more than one level of subgroup, for example it is allowed to call
 * FileObject_get(self, "/this/that/group1"). Note, it is only group/dataset names
 * that can be handled.
 * @param[in] self - self
 * @param[in] name - name of group wanted
 * @return the file object instance if found/created otherwise NULL
 */
FileObject_t* FileObject_create(FileObject_t* self, const char* name);

/**
 * @param[in] self - self
 * @return number of sub groups this file object has
 */
size_t FileObject_numberOfGroups(FileObject_t* self);

/**
 * Returns the sub group at specified index
 * @param[in] self - self
 * @param[in] index - index of group
 */
FileObject_t* FileObject_getByIndex(FileObject_t* self, int index);

/**
 * @param[in] self - self
 * @param[in] name - name of the node that should be checked for
 */
int FileObject_exists(FileObject_t* self, const char* name);

/**
 * Returns the sub groups
 * @param[in] self
 * @return the sub groups
 */
RaveObjectList_t* FileObject_groups(FileObject_t* self);

/**
 * Returns the attributes as a RaveValueHash.
 * @param[in] self - self
 * @return the attributes belonging to this group
 */
RaveValue_t* FileObject_attributes(FileObject_t* self);

/**
 * Adds an attribute to this group.
 * @param[in] self - self
 * @param[in] name - name of the attribute
 * @param[in] value - the value that should be added
 * @return 1 on success otherwise 0
 */
int FileObject_addAttribute(FileObject_t* self, const char* name, RaveValue_t* value);

/**
 * @param[in] self - self
 * @return the data 2d field if this group is a dataset
 */
RaveData2D_t* FileObject_getData(FileObject_t* self);

/**
 * @param[in] self - self
 * @param[in] data - the 2d data field
 * @return 1 on success otherwise 0
 */
int FileObject_setData(FileObject_t* self, RaveData2D_t* data);

/**
 * Returns if this group is a dataset node. Can either be that the data 2d field is set or that
 * a lazy dataset instance has been set in this object
 * @param[in] self - self
 * @return if this object is a dataset or not
 */
int FileObject_isDataset(FileObject_t* self);

/**
 * Checks if the data 2d field has been set. I.e. if lazy dataset has been set but
 * not loaded, then this method will return 0
 * @param[in] self - self
 * @return if dataset has been loaded or not
 */
int FileObject_isDatasetLoaded(FileObject_t* self);

/**
 * If this object is a dataset then this will return the xsize of the dataset. Can be useful
 * if lazy loading is used and data not loaded.
 * @param[in] self - self
 * @return the xsize
 */
long FileObject_getDatasetX(FileObject_t* self);

/**
 * If this object is a dataset then this will return the ysize of the dataset. Can be useful
 * if lazy loading is used and data not loaded.
 * @param[in] self - self
 * @return the ysize
 */
long FileObject_getDatasetY(FileObject_t* self);

/**
 * If this object is a dataset then this will return the data type of the dataset. Can be useful
 * if lazy loading is used and data not loaded.
 * @param[in] self - self
 * @return the datatype
 */
RaveDataType FileObject_getDatasetType(FileObject_t* self);

/**
 * Sets a lazy dataset in the file object
 * @param[in] self - self
 * @param[in] lazyDataset - the lazy dataset
 * @return 1 on success otherwise 0
 */
int FileObject_setLazyDataset(FileObject_t* self, LazyDataset_t* lazyDataset);
#endif
