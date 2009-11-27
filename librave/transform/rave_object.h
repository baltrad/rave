/* --------------------------------------------------------------------
Copyright (C) 2009 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Generic implementation of an object that is used within rave. All
 * objects should use this as template for their structure.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-11-25
 */
#ifndef RAVE_OBJECT_H
#define RAVE_OBJECT_H
#include <stdlib.h>

/**
 * Always should be at top of a struct that implements a RaveObject.
 */
#define RAVE_OBJECT_HEAD \
  int roh_refCnt; \
  struct _raveobjecttype* roh_type; \
  void* roh_bindingData;

/**
 * The basic raveobject that contains the header information for all
 * rave objects.
 */
typedef struct _raveobject {
  RAVE_OBJECT_HEAD /** Always on top */
} RaveCoreObject;

/**
 * The rave object type definition.
 */
typedef struct _raveobjecttype {
  const char* name; /**< the name, for printout */
  size_t type_size; /**< the size of the object, sizeof(type) */
  int (*constructor)(RaveCoreObject* obj); /** function to be called for initialization of the object */
  void (*destructor)(RaveCoreObject* obj); /** function to be called for release of members in the object */
} RaveCoreObjectType;

/**
 * Creates a new instance of a specified object type, should be implemented within
 * the object itself until we have a generic way to create instances by type.
 * Typically:
 * SomeObject_t* obj = RAVE_OBJECT_NEW(&SomeObject_Type);
 */
#define RAVE_OBJECT_NEW(type) \
  (void*)RaveCoreObject_new(type, __FILE__, __LINE__);

/**
 * Releases the provided object once (i.e. decrements the reference counter and
 * sets the invalue to NULL. Ie. Do not do something funny like:
 * RAVE_OBJECT_RELEASE(x[index++]) since that will mess up the memory, instead write
 * RAVE_OBJECT_RELEASE(x[index]);
 * index++
 */
#define RAVE_OBJECT_RELEASE(obj) \
  RaveCoreObject_release((RaveCoreObject*)(obj), __FILE__, __LINE__); \
  (obj) = NULL;

/**
 * Increments the reference counter once for the specified object and returns
 * the pointer. Typically used as:
 * x = RAVE_OBJECT_COPY(obj);
 * ....
 * RAVE_OBJECT_RELEASE(x);
 */
#define RAVE_OBJECT_COPY(src) \
  (void*)RaveCoreObject_copy((RaveCoreObject*)(src), __FILE__, __LINE__)

/**
 * Returns the provided objects reference count.
 */
#define RAVE_OBJECT_REFCNT(src) \
  RaveCoreObject_getRefCount((RaveCoreObject*)src)

/**
 * Binds the rave object to some sort of pointer value. E.g. a python object pointer.
 */
#define RAVE_OBJECT_BIND(this, bound) \
  RaveCoreObject_bind((RaveCoreObject*)this, bound)

/**
 * Unbinds the rave object from a pointer value. E.g. a python object pointer.
 */
#define RAVE_OBJECT_UNBIND(this, bound) \
  RaveCoreObject_unbind((RaveCoreObject*)this, bound)

/**
 * Returns if this object currently is bound to a pointer or not.
 */
#define RAVE_OBJECT_ISBOUND(this) \
  (RaveCoreObject_getBindingData((RaveCoreObject*)this) != NULL)

/**
 * Returns the current binding or NULL if there is none.
 */
#define RAVE_OBJECT_GETBINDING(this) \
  RaveCoreObject_getBindingData((RaveCoreObject*)this)

/**
 * Creates a new instance of the provided type.
 * @param[in] type - the object type
 * @param[in] filename - the filename that this allocation was performed in
 * @param[in] lineno - the linenumber that this allocation was performed in
 * @returns a new instance
 */
RaveCoreObject* RaveCoreObject_new(RaveCoreObjectType* type, const char* filename, int lineno);

/**
 * Decrements the reference counter and if reference counts gets to 0, the destructor that
 * was provided when creating the instance will be called. Note, regardless on if and how the
 * destructor has been implemented, the object will be freed.
 * @param[in] obj - the object to release (including the object itself)
 */
void RaveCoreObject_release(RaveCoreObject* obj, const char* filename, int lineno);

/**
 * Initializes the object. Will set the reference counter to 1 and assign the object name and destructor.
 * @param[in] obj - the object to initialize
 * @param[in] rohname - the object name
 * @param[in] destructor - the destructor to be called
 */
void RaveCoreObject_initialize(RaveCoreObject* obj, const char* rohname, void (*destructor)(struct _raveobject* obj));

/**
 * Increments the reference counter and returns a pointer to the provided object.
 * @param[in] src - the object to be copied
 * @returns a pointer to the copied object
 */
RaveCoreObject* RaveCoreObject_copy(RaveCoreObject* src, const char* filename, int lineno);

/**
 * Returns the current reference count.
 * @param[in] src - the the object that should be queried for reference count
 * @returns the reference count.
 */
int RaveCoreObject_getRefCount(RaveCoreObject* src);

/**
 * Sets the binding data field, useful when for example associating this
 * object with another object in order to manage objects from python
 * or other languages. Do not use this function directly but use
 * the macros for binding instead.
 * @param[in] src - this object
 * @param[in] bindingData - the associated object (or similar)
 */
void RaveCoreObject_bind(RaveCoreObject* src, void* bindingData);

/**
 * Removes the binding from the rave object. Do not use this function directly
 * but use the macros for binding instead. If bindingData != the stored binding data nothing will happen.
 * @param[in] src - this object
 * @param[in] bindingData - the binding to remove
 */
void RaveCoreObject_unbind(RaveCoreObject* src, void* bindingData);

/**
 * Returns the extra data. Do not use this function directly but use
 * the macros for binding objects instead.
 * @param[in] src - this object
 * @returns the binding data or NULL if none.
 */
void* RaveCoreObject_getBindingData(RaveCoreObject* src);


#ifdef KALLE
/**
 * Sets a callback function that should be called after the objects destructor
 * has been called but before the object itself is claimed. The destructor callback
 * will be invoked with the provided cbdata.
 * @param[in] src - the rave core object
 * @param[in] cbdata - the data that should be provided to the destructor callback
 * @param[in] destructorcb - the destructor callback
 */
void RaveCoreObject_setDestructorCB(RaveCoreObject* src, void* cbdata, void (*destructorcb)(void*));

/**
 * Returns the data that should be sent in the destructor callback, this function
 * can be handy if for example the destructor callback should be resetted but
 * something has to be done with the data.
 * @param[in] src - the rave core object
 * @returns the destructor cb data
 */
void* RaveCoreObject_getDestructorCBData(RaveCoreObject* src);
#endif

/**
 * Prints the rave object statistics.
 */
void RaveCoreObject_printStatistics(void);

#endif /* RAVE_OBJECT_H */
