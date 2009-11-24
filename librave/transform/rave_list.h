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
#ifndef RAVE_LIST_H
#define RAVE_LIST_H

/**
 * Defines a list
 */
typedef struct _RaveList_t RaveList_t;

/**
 * Creates a new list instance
 * @return a new instance or NULL on failure
 */
RaveList_t* RaveList_new(void);

/**
 * Releases the responsibility for the list, it is not certain that
 * it will be deleted though if there still are references existing
 * to this list. It should also be observed that currently, the
 * list will not release the actual data that has been added to
 * the list, only the internally used memory
 * @param[in] list - the list
 */
void RaveList_release(RaveList_t* list);

/**
 * Copies the reference to this instance by increasing a
 * reference counter.
 * @param[in] list - the list
 * @return a pointer to the scan
 */
RaveList_t* RaveList_copy(RaveList_t* list);

/**
 * Add one instance to the list.
 * @param[in] list - the list
 * @param[in] obj - the object
 * @returns 1 on success, otherwise 0
 */
int RaveList_add(RaveList_t* list, void* ob);

/**
 * Returns the number of items in this list.
 * @param[in] list - the list
 * @returns the number of items in this list.
 */
int RaveList_size(RaveList_t* list);

/**
 * Returns the item at the specified position.
 * @param[in] list - the list
 * @param[in] index - the index of the requested item
 * @returns the object
 */
void* RaveList_get(RaveList_t* list, int index);

/**
 * Removes the item at the specified position and returns it.
 * @param[in] list - the list
 * @param[in] index - the index of the requested item
 * @returns the object
 */
void* RaveList_remove(RaveList_t* list, int index);

/**
 * Sorts the list according to the provided sort function.
 * The sort function should return an integer less than,
 * equal to or greater than zero depending on how the first
 * argument is in relation to the second argument.
 *
 * @param[in] list - the list
 * @param[in] sortfun - the sorting function.
 */
void RaveList_sort(RaveList_t* list, int (*sortfun)(const void*, const void*));

#endif /* RAVE_LIST_H */
