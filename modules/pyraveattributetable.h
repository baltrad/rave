/*
 * pyraveattributetable.h
 *
 *  Created on: Mar 30, 2022
 *      Author: anders
 */

#ifndef PYRAVEATTRIBUTETABLE_H
#define PYRAVEATTRIBUTETABLE_H


#include "rave_attribute_table.h"

/**
 * A rave attribute table
 */
typedef struct {
  PyObject_HEAD /*Always has to be on top*/
  RaveAttributeTable_t* table; /**< the attributes */
} PyRaveAttributeTable;

#define PyRaveAttributeTable_Type_NUM 0                                 /**< index of type */

#define PyRaveAttributeTable_GetNative_NUM 1                            /**< index of GetNative */
#define PyRaveAttributeTable_GetNative_RETURN RaveAttributeTable_t*     /**< return type for GetNative */
#define PyRaveAttributeTable_GetNative_PROTO (PyRaveAttributeTable*)    /**< arguments for GetNative */

#define PyRaveAttributeTable_New_NUM 2                                  /**< index of New */
#define PyRaveAttributeTable_New_RETURN PyRaveAttributeTable*           /**< return type for New */
#define PyRaveAttributeTable_New_PROTO (RaveAttributeTable_t*)          /**< arguments for New */

#define PyRaveAttributeTable_API_pointers 3                             /**< number of API pointers */

#define PyRaveAttributeTable_CAPSULE_NAME "_raveattributetable._C_API"

#ifdef PY_RAVE_ATTRIBUTE_TABLE_MODULE
/** Forward declaration of type */
extern PyTypeObject PyRaveAttributeTable_Type;

/** Checks if the object is a PyArea or not */
#define PyRaveAttributeTable_Check(op) ((op)->ob_type == &PyRaveAttributeTable_Type)

/** Forward declaration of PyArea_GetNative */
static PyRaveAttributeTable_GetNative_RETURN PyRaveAttributeTable_GetNative PyRaveAttributeTable_GetNative_PROTO;

/** Forward declaration of PyArea_New */
static PyRaveAttributeTable_New_RETURN PyRaveAttributeTable_New PyRaveAttributeTable_New_PROTO;

#else
/** Pointers to types and functions */
static void **PyRaveAttributeTable_API;

/**
 * Returns a pointer to the internal rave attribute table, remember to release the reference
 * when done with the object. (RAVE_OBJECT_RELEASE).
 */
#define PyRaveAttributeTable_GetNative \
  (*(PyRaveAttributeTable_GetNative_RETURN (*)PyRaveAttributeTable_GetNative_PROTO) PyRaveAttributeTable_API[PyRaveAttributeTable_GetNative_NUM])

/**
 * Creates a rave attribute table instance. Release this object with Py_DECREF.  If a RaveAttributeTable_t area is
 * provided and this object already is bound to a python instance, this instance will be increfed and
 * returned.
 * @param[in] table - the RaveAttributeTable_t instance.
 * @returns the PyRaveAttributeTable instance.
 */
#define PyRaveAttributeTable_New \
  (*(PyRaveAttributeTable_New_RETURN (*)PyRaveAttributeTable_New_PROTO) PyRaveAttributeTable_API[PyRaveAttributeTable_New_NUM])

/**
 * Checks if the object is a python attribute table.
 */
#define PyRaveAttributeTable_Check(op) \
   (Py_TYPE(op) == &PyRaveAttributeTable_Type)

#define PyRaveAttributeTable_Type (*(PyTypeObject*)PyRaveAttributeTable_API[PyRaveAttributeTable_Type_NUM])

/**
 * Imports the PyRaveAttributeTable module (like import _attributetable in python).
 */
#define import_pyraveattributetabley() \
    PyRaveAttributeTable_API = (void **)PyCapsule_Import(PyRaveAttributeTable_CAPSULE_NAME, 1);

#endif


#endif /* PYRAVEATTRIBUTETABLE_H */
