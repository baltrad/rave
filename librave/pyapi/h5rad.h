/**
 * Accesses various contents of a RAVE INFO object.
 * @file
 * @author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2005-
 */
#ifndef H5RAD_H
#define H5RAD_H

#include <Python.h>
#include <arrayobject.h>


/**
 * Get int variable from an object using key. Same as int(inobj.get(key)).
 * @param[in] inobj the object
 * @param[in] key the name of the variable
 * @param[in,out] val the value
 * @return 1 on success, otherwise 0
 */
int GetIntFromINFO(PyObject* inobj, char* key, int* val);

/**
 * Get double variable from an object using key. Same as inobj.get(key).
 * @param[in] inobj the object
 * @param[in] key the name of the variable
 * @param[in,out] val the value
 * @return 1 on success, otherwise 0
 */
int GetDoubleFromINFO(PyObject* inobj, char* key, double* val);

/**
 * Get C string from inobj using 'key'.
 * @param[in] inobj the object.
 * @param[in] key the name of the variable
 * @param[in,out] val the allocated string
 * @return 1 on success, otherwise 0
*/
int GetStringFromINFO(PyObject* inobj, char* key, char** val);

/**
 * Return sequence object from inobj using 'key'.
 * @param[in] inobj the object
 * @param[in] key the name of the variable
 * @return the sequence object if found, otherwise NULL
 */
PyObject* GetSequenceFromINFO(PyObject* inobj, char* key);

/**
 * Return a python string object from object.
 * @param[in] inobj the object
 * @param[in] key the name of the variable
 * @return the python string object if found, otherwise NULL
 */
PyObject* GetPyStringFromINFO(PyObject* inobj, char* key);

/**
 * Return a double from a sequence at index i.
 * @param[in] inobj the object
 * @param[in] i the index in the sequence
 * @param[in,out] val the found value
 * @return 1 on success, otherwise 0
 */
int GetDoubleFromSequence(PyObject* inobj, int i, double* val);

/**
 * Return a int from a sequence at index i.
 * @param[in] inobj the object
 * @param[in] i the index in the sequence
 * @param[in,out] val the found value
 * @return 1 on success, otherwise 0
 */
int GetIntFromSequence(PyObject* inobj, int i, int* val);

/**
 * Return a unicode string from inobj using 'key'.
 * Do not read directly using this function! It will leak!
 * @param[in] inobj the object
 * @param[in] key the key
 * @return the unicode string if found, otherwise NULL
 */
PyObject* getUnicodeFromINFO(PyObject* inobj, char* key);

#endif
