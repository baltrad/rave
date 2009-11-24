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
 * Implementation of a simple list.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-11-20
 */
#include "rave_list.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include <string.h>

#define LIST_EXPAND_NR_ENTRIES 20  /**< Expand the list array with this number of entries / reallocation */
#define DEFAULT_NR_RAVE_LIST_ENTRIES 20 /**< Default number of list entries */

/**
 * Represents a list
 */
struct _RaveList_t {
  long ps_refCount;
  void** list;
  int nrEntries;
  int nrAlloc;
};

/*@{ Private functions */

/**
 * Should be called before adding an entry to a list so that the size of the list
 * never is to small when inserting a new entry.
 * @param[in] list - the list
 * @returns 1 on success or 0 on any kind of failure.
 */
static int RaveListInternal_ensureCapacity(RaveList_t* list)
{
  int result = 0;
  RAVE_ASSERT((list != NULL), "list == NULL");
  if (list->nrEntries >= list->nrAlloc - 1) {
    int nsz = list->nrAlloc + LIST_EXPAND_NR_ENTRIES;
    void** narr = RAVE_REALLOC(list->list, nsz * sizeof(void*));
    int i;
    if (narr == NULL) {
      RAVE_CRITICAL0("Failed to reallocate memory for list");
      goto done;
    }
    list->list = narr;
    for (i = list->nrEntries; i < nsz; i++) {
      list->list[i] = NULL;
    }
    list->nrAlloc = nsz;
  }
  result = 1;
done:
  return result;
}

/**
 * Destroys the list
 * @param[in] list - the list to destroy
 */
static void RaveList_destroy(RaveList_t* list)
{
  if (list != NULL) {
    RAVE_FREE(list->list);
    RAVE_FREE(list);
  }
}
/*@} End of Private functions */

/*@{ Interface functions */
RaveList_t* RaveList_new(void)
{
  RaveList_t* result = NULL;
  result = RAVE_MALLOC(sizeof(RaveList_t));
  if (result != NULL) {
    result->ps_refCount = 1;
    result->list = RAVE_MALLOC(sizeof(void*)*DEFAULT_NR_RAVE_LIST_ENTRIES);
    result->nrAlloc = DEFAULT_NR_RAVE_LIST_ENTRIES;
    result->nrEntries = 0;
    if (result->list == NULL) {
      RaveList_destroy(result);
      result = NULL;
      goto done;
    }
  }
done:
  return result;
}

void RaveList_release(RaveList_t* list)
{
  if (list != NULL) {
    list->ps_refCount--;
    if (list->ps_refCount <= 0) {
      RaveList_destroy(list);
    }
  }
}

RaveList_t* RaveList_copy(RaveList_t* list)
{
  if (list != NULL) {
    list->ps_refCount++;
  }
  return list;
}

int RaveList_add(RaveList_t* list, void* ob)
{
  int result = 0;
  RAVE_ASSERT((list != NULL), "list == NULL");

  if (!RaveListInternal_ensureCapacity(list)) {
    RAVE_CRITICAL0("Can not add entry to list since size does not allow it");
    goto done;
  }

  list->list[list->nrEntries++] = ob;

  result = 1;
done:
  return result;
}

int RaveList_size(RaveList_t* list)
{
  RAVE_ASSERT((list != NULL), "list == NULL");
  return list->nrEntries;
}

void* RaveList_get(RaveList_t* list, int index)
{
  RAVE_ASSERT((list != NULL), "list == NULL");
  if (index >= 0 && index < list->nrEntries) {
    return list->list[index];
  }
  return NULL;
}

void* RaveList_remove(RaveList_t* list, int index)
{
  void* result = NULL;
  RAVE_ASSERT((list != NULL), "list == NULL");
  if (index >= 0 && index < list->nrEntries) {
    int i = 0;
    int ne = list->nrEntries-1;
    result = list->list[index];
    for (i = index; i < ne; i++) {
      list->list[i] = list->list[i+1];
    }
    list->nrEntries--;
  }
  return result;
}

void RaveList_sort(RaveList_t* list, int (*sortfun)(const void*, const void*))
{
  RAVE_ASSERT((list != NULL), "list == NULL");
  qsort(list->list, list->nrEntries, sizeof(void*), sortfun);
}
