/**
 * Helper code for accessing python objects.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 1998-
 */
#ifndef GET_PY_H
#define GET_PY_H

#include <Python.h>
#include <arrayobject.h>

#include <projects.h>

/**
 * Creates a PROJ.4 projection object from a python array of strings.
 * @param[in] pcs a python sequence
 * @return a PROJ.4 projection or NULL on failure
 */
PJ* initProjection(PyObject* pcs);

/**
 * Get a double from a python dictionary.
 * @param[in] name the name of the key
 * @param[in,out] val the value
 * @param[in] dictionary the python dictionary
 * @return 1 on success, otherwise 0
 */
int getDoubleFromDictionary(char* name,double* val,PyObject* dictionary);

/**
 * Get a int from a python dictionary.
 * @param[in] name the name of the key
 * @param[in,out] val the value
 * @param[in] dictionary the python dictionary
 * @return 1 on success, otherwise 0
 */
int getIntFromDictionary(char* name, int* val, PyObject* dictionary);

/**
 * Get a double from a object instance. Equivalent of float(instance.name).
 * @param[in] name the name of the attribute
 * @param[in,out] val the value
 * @param[in] instance the python instance
 * @return 1 on success, otherwise 0
 */
int getDoubleFromInstance(char* name,double* val,PyObject* instance);

/**
 * Get a int from a object instance. Equivalent of int(instance.name).
 * @param[in] name the name of the attribute
 * @param[in,out] val the value
 * @param[in] instance the python instance
 * @return 1 on success, otherwise 0
 */
int getIntFromInstance(char* name, int* val,PyObject* instance);

/**
 * Get a double at index idx from the provided tuple.
 * @param[in] idx the index of the value
 * @param[in,out] val the value
 * @param[in] tuple the python tuple
 * @return 1 on success, otherwise 0
 */
int getIdxDoubleFromTuple(int idx, double* val,PyObject* tuple);

/**
 * Get a int at index idx from the provided tuple.
 * @param[in] idx the index of the value
 * @param[in,out] val the value
 * @param[in] tuple the python tuple
 * @return 1 on success, otherwise 0
 */
int getIdxIntFromTuple(int idx, int* val, PyObject* tuple);

/**
 * Appends a double to a list.
 * @param[in] list the python list
 * @param[in] val the double value to be appended
 * @return 1 on success, otherwise 0
 */
int AppendFloatToList(PyObject* list, double val);

/**
 * Inserts double val into a list at position i.
 * @param[in] list the python list
 * @param[in] i the index where to insert the value
 * @param[in] val the value
 * @return 1 on success, otherwise 0
 */
int InsertFloatInList(PyObject* list, int i, double val);

/**
 * Extracts PyFloat from a list at position i and returns it as a double.
 * @param[in] list the python list
 * @param[in] i the index to get the value from
 * @return the double
 */
double getDoubleFromList(PyObject* list, int i);


#endif
