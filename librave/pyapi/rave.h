/**
 * rave object wrapper functionality, accesses the python rave object through
 * the c-api.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-08-07
 */
#ifndef RAVE_H
#define RAVE_H

/*Standard includes*/
#include <Python.h>
#include <arrayobject.h>
#include "rave_transform.h"
#include "h5rad.h"
#include "getpy.h"

/**
 * Defines the extents and resolution of an area.
 */
typedef struct {
   double xscale; /**< X-scale */
   double yscale; /**< Y-scale */
   int xsize;     /**< X-size */
   int ysize;     /**< Y-size */
   double nodata; /**< nodata value */
   double noecho; /**< noecho value */
   UV lowleft;    /**< lower-left corner */
   UV uppright;   /**< upper-right corner */
} RaveImageStruct;

/**
 * A python(ified) version of the RavePolarVolume to keep track
 * on all python arrays.
 */
typedef struct PyRavePolarVolume {
  RavePolarVolume volume;           /**< The RavePolarVolume */
  PyArrayObject** fields;            /**< The python arrays defining the volume */
  int fieldsn;                       /**< The number of fields */
} PyRavePolarVolume;

/**
 * The actual RaveObject which maps agains an two-dimensional image.
 */
typedef struct {
   PyArrayObject* data;    /**< The data array */
   PyArrayObject* topo;    /**< @deprecated Used by compositing algorithms */
   PyArrayObject* bitmap;  /**< @deprecated Used by compositing algorithms */
   PyObject* info;         /**< Information about this image */
} RaveObject;

/**
 * Initializes a RaveObject as a failsafe against screwy compilers.
 * @param[in] v the object to initialize.
 */
void initialize_RaveObject(RaveObject* v);

/**
 * Returns the type of the array.
 * @param[in] arr the python array
 * @return the type
 */
char array_type_2d(PyArrayObject* arr);

/**
 * Returns the type_num of the array
 * for example if PyArray_FLOAT/PyArray_UINT/PyArray_DOUBLE etc.
 * @param[in] arr the python array
 * @return the type
 */
int array_type_num_2d(PyArrayObject* arr);

/**
 * Returns how many indices one row has.
 * @param[in] arr the python array
 * @return the stride
 */
int array_stride_xsize_2d(PyArrayObject* arr);

/**
 * Returns an pointer to the data in the array
 * object as an array of unsigned char.
 * @param[in] arr the python array
 * @return a pointer to the data structure of the array (DO NOT FREE)
 */
unsigned char* array_data_2d(PyArrayObject* arr);

/**
 * Translates a array typecode into a rave data type.
 * @param[in] type the array typecode
 * @return the rave data type
 */
RaveDataType translate_pytype_to_ravetype(char type);

/**
 * Translates a numpy array type into a rave data type.
 * @param[in] type the numpy array type
 * @return the rave data type
 */
RaveDataType translate_pyarraytype_to_ravetype(int type);

/**
 * Returns the corresponding PyArray_Type from the typecode,
 * e.g. input 'f' will return PyArray_FLOAT.
 * @param[in] type the array type code
 * @return the numpy array identifier
 */
int pyarraytype_from_type(char type);

/**
 * Returns the result of the expression:
 * (R*R-r*r)/(R*R+r*r).
 * @param[in] R the max radius (May NOT be 0.0)
 * @param[in] r the actual radius
 * @return (R² - r²)/(R² + r²)
 */
double calculate_cressman(double r, double R);

/**
 * Returns the result of the expression:
 * 1.0-r/R
 * @param[in] R the max radius (May NOT be 0.0)
 * @param[in] r the actual radius
 * @return 1.0 - r / R
 */
double calculate_inverse(double r, double R);

/**
 * Imports the pcs module, and gets the projection
 * defininition by using the dictionary attribute pcs
 * returns NULL if it could not create the projection.
 * @param[in] dict the python dictionary
 * @return the projection or NULL on failure
 */
PJ* get_rave_projection(PyObject* dict);

/**
 * Fills the rave image struct with all mandatory
 * information from the Info structure.
 * @param[in] inobj the rave info dictionary
 * @param[in,out] p the structure to be filled with information
 * @param[in] set the /image index
 * Returns 0 on failure and 1 on success.
 */
int fill_rave_image_info(PyObject* inobj, RaveImageStruct *p, int set);

/**
 * Fills the rave image struct with the area extent.
 * @param[in] dict the dictionary
 * @param[in,out] p the structure to be filled
 * @return 0 on failure and sets the py error, otherwise 1 is returned.
 */
int fill_rave_area_extent(PyObject* dict, RaveImageStruct* p);

/**
 * Fills the RaveObject with the parts as defined in the python object.
 * @param[in] robj The rave object in python
 * @param[in,out] obj The RaveObject
 * @param[in] set the set index
 * @param[in] dsetname the set identifier name
 * @return 1 on success, otherwise 0
 */
int fill_rave_object(PyObject *robj, RaveObject *obj, int set,
		     const char *dsetname);

/**
 * Just decrefs the RaveObject structs, it does not
 * free the actual rave object.
 * @param[in] obj the rave object
 */
void free_rave_object(RaveObject* obj);

/**
 * Releases the internals of a rave pypolar volume by decrefing
 * the fields and freeing the field array.
 * @param[in] vol the pypolar volume
 */
void free_pypolar_volume(PyRavePolarVolume* vol);

/**
 * Builds all relevant information for a python polar volume.
 * @param[in] vol the volume to build
 * @param[in] pyobj the python object that should define the volume
 * @return 1 on success, otherwise failure.
 */
int fill_pypolar_volume(PyRavePolarVolume* vol, PyObject* pyobj);

#endif
