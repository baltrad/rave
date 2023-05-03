/* --------------------------------------------------------------------
Copyright (C) 2017 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Cartesian CF convention handler
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2017-11-24
 */
#include "cartesian_cf_io.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "raveobject_hashtable.h"
#include "odim_io_utilities.h"
#include "cartesiancomposite.h"
#include "proj_wkt_helper.h"
#define __USE_XOPEN
#define _XOPEN_SOURCE
#include <time.h>
#include <string.h>

#define UNITS "units"
#define LONG_NAME "long_name"
#define STANDARD_NAME "standard_name"
#define POSITIVE "positive"
#define DESCRIPTION "description"

#define UNIT_METERS "m"
#define UNIT_DEGREE_EAST "degree_east"
#define UNIT_DEGREE_NORTH "degree_north"
#define UNIT_SECONDS_SINCE_19700101 "seconds since 1970-01-01 00:00:00 +00:00"
#define UNITS_DBZ "dBZ"

#define HEIGHT_DESCRIPTION "height above sea level"

#define TEXT_UP "up"
#define TEXT_HEIGHT "height"
#define TEXT_LONGITUDE "longitude"
#define TEXT_LATITUDE "latitude"
#define TEXT_TIME "time"

typedef struct QuantityNodataUndetectMapping {
  char* quantity;
  float nodata;
  float undetect;
  char* units;
  char* long_name;
} QuantityNodataUndetectMapping;

QuantityNodataUndetectMapping qmapping[] = {
    {"DBZH", -34.0, -32.0, UNITS_DBZ, "equivalent_reflectivity_factor_h"},
    {NULL, 0.0, 0.0, NULL, NULL}
};


/**
 * Represents the adaptor
 */
struct _CartesianCfIO_t {
  RAVE_OBJECT_HEAD /** Always on top */
  int deflate_level; /**< Compression level 0=no compression, 1-9 means level of compression */
};

/*@{ Private functions */
/**
 * Constructor.
 */
static int CartesianCfIO_constructor(RaveCoreObject* obj)
{
  ((CartesianCfIO_t*)obj)->deflate_level = 0;
  return 1;
}

/**
 * Copy constructor
 */
static int CartesianCfIO_copyconstructor(RaveCoreObject* obj, RaveCoreObject* srcobj)
{
  ((CartesianCfIO_t*)obj)->deflate_level = ((CartesianCfIO_t*)srcobj)->deflate_level;
  return 1;
}

/**
 * Destroys the object
 * @param[in] obj - the instance
 */
static void CartesianCfIO_destructor(RaveCoreObject* obj)
{
}

/**
 * Searches for a mapping with specified name.
 * @param[in] name - the quantity that is wanted
 * @returns the mapping if found otherwise NULL
 */
static const QuantityNodataUndetectMapping* CartesianCfIO_getQuantityMapping(const char* name)
{
  int i=0;
  while (qmapping[i].quantity != NULL) {
    if (strcmp(qmapping[i].quantity, name)==0) {
      return &qmapping[i];
    }
  }
  return NULL;
}

/**
 * Searches for the nodata value for the specified name
 * @param[in] name - the quantity that is wanted
 * @returns the nodata value if found, otherwise 0.0
 */
static float CartesianCfIO_getNodata(const char* name)
{
  const QuantityNodataUndetectMapping* mapping = CartesianCfIO_getQuantityMapping(name);
  if (mapping != NULL) {
    return mapping->nodata;
  }
  return 0.0;
}


/**
 * Adds a string attribute to a netcdf file.
 * @param[in] ncid - the netcdf file id
 * @param[in] varid - the variable id (-1 for global)
 * @param[in] name - the name of the variable
 * @param[in] value - the value to write
 * @returns 1 on success otherwise 0
 */
static int CartesianCfIOInternal_addStringAttribute(int ncid, int varid, const char* name, const char* fmt, ...)
{
  char value[1024];
  va_list ap;
  int n = 0;

  RAVE_ASSERT((fmt != NULL), "fmt == NULL");

  va_start(ap, fmt);
  n = vsnprintf(value, 1024, fmt, ap);
  va_end(ap);
  if (n < 0 || n >= 1024) {
    RAVE_ERROR0("Failed to generate value");
    return 0;
  }

  if (nc_put_att_text(ncid, varid, name, strlen(value), value))
  {
    RAVE_ERROR0("Failed to write Conventions file");
    return 0;
  }
  return 1;
}

/**
 * Adds a string attribute to a netcdf file. Would be the same as calling \ref CartesianCfIOInternal_addStringAttribute with varid=NC_GLOBAL.
 * @param[in] ncid - the netcdf file id
 * @param[in] name - the name of the variable
 * @param[in] value - the value to write
 * @returns 1 on success otherwise 0
 */
static int CartesianCfIOInternal_addGlobalStringAttribute(int ncid, const char* name, const char* fmt, ...)
{
  char value[1024];
  va_list ap;
  int n = 0;

  RAVE_ASSERT((fmt != NULL), "fmt == NULL");

  va_start(ap, fmt);
  n = vsnprintf(value, 1024, fmt, ap);
  va_end(ap);
  if (n < 0 || n >= 1024) {
    RAVE_ERROR0("Failed to generate value");
    return 0;
  }
  return CartesianCfIOInternal_addStringAttribute(ncid, NC_GLOBAL, name, value);
}

/**
 * Adds a double attribute to a netcdf file
 */
static int CartesianCfIOInternal_addDoubleAttribute(int ncid, int varid, const char* name, double value)
{
  if (nc_put_att_double(ncid, varid, name, NC_DOUBLE, 1, &value))
  {
    RAVE_ERROR1("Failed to put double attribute %s", name);
    return 0;
  }
  return 1;
}

/**
 * Adds a double array attribute to a netcdf file
 */
static int CartesianCfIOInternal_addDoubleArrayAttribute(int ncid, int varid, const char* name, size_t vlen, double* value)
{
  if (nc_put_att_double(ncid, varid, name, NC_DOUBLE, vlen, value))
  {
    RAVE_ERROR1("Failed to put double array attribute %s", name);
    return 0;
  }
  return 1;
}

/**
 * Adds a float attribute to a netcdf file
 */
static int CartesianCfIOInternal_addFloatAttribute(int ncid, int varid, const char* name, float value)
{
  if (nc_put_att_float(ncid, varid, name, NC_FLOAT, 1, &value))
  {
    RAVE_ERROR1("Failed to put float attribute %s", name);
    return 0;
  }
  return 1;
}

/**
 * Adds the global attributes that are essential to have in the CF-convention file.
 * @param[in] self - self
 * @param[in] ncid - the netcdf file id
 * @param[in] cartesian - the cartesian product
 * @return 1 on success otherwise 0
 */
static int CartesianCfIOInternal_addGlobalAttributes(CartesianCfIO_t* self, int ncid, Cartesian_t* cartesian)
{
  int result = 0;
  char areaid[256];
  if (!OdimIoUtilities_getNodOrCmtFromSource(Cartesian_getSource(cartesian), areaid, 256)) {
    char tmpid[256];
    RAVE_INFO0("Could not find NOD or CMT in source\n");
    if (!OdimIoUtilities_getIdFromSource(Cartesian_getSource(cartesian), "PLC:", tmpid, 256)) {
      if (OdimIoUtilities_getIdFromSource(Cartesian_getSource(cartesian), "WMO:", tmpid, 256)) {
        snprintf(areaid, 256, "WMO:%s", tmpid);
      } else {
        goto done;
      }
    } else {
      snprintf(areaid, 256, "PLC:%s", tmpid);
    }
  }

  if (!CartesianCfIOInternal_addGlobalStringAttribute(ncid, "Conventions", "CF-1.7"))
    goto done;

  if (!CartesianCfIOInternal_addGlobalStringAttribute(ncid, "comment", "cartesian %s", RaveTypes_getStringFromProductType(Cartesian_getProduct(cartesian))))
    goto done;

  if (!CartesianCfIOInternal_addGlobalStringAttribute(ncid, "institution", "rave"))
    goto done;

  if (!CartesianCfIOInternal_addGlobalStringAttribute(ncid, "references", "rave"))
    goto done;

  if (!CartesianCfIOInternal_addGlobalStringAttribute(ncid, "source", "rave"))
    goto done;

  if (!CartesianCfIOInternal_addGlobalStringAttribute(ncid, "history", "original"))
    goto done;

  if (!CartesianCfIOInternal_addGlobalStringAttribute(ncid, "title", "cartesian %s in %s", RaveTypes_getStringFromProductType(Cartesian_getProduct(cartesian)), areaid))
    goto done;

  result = 1;
done:
  return result;
}

/**
 * Verify the relevant values and parameters so that we don't have any surprises when we generate the data. For example that all x/y sizes are the same.
 * @param[in] cartesian - the cartesian product
 * @param[in,out] xsize - the identified x-size
 * @param[in,out] ysize - the identified y-size
 * @return 1 on success otherwise 0
 */
static int CartesianCfIOInternal_verifyParameters(CartesianCfIO_t* self, Cartesian_t* cartesian, long* xsize, long* ysize)
{
  RaveList_t* names = Cartesian_getParameterNames(cartesian);
  int i, nlen;
  int result = 0;
  nlen = RaveList_size(names);
  if (nlen > 0) {
    const char* name = RaveList_get(names, 0);
    CartesianParam_t* param = Cartesian_getParameter(cartesian, name);
    *xsize = CartesianParam_getXSize(param);
    *ysize = CartesianParam_getYSize(param);
    RAVE_OBJECT_RELEASE(param);
    result = 1;
    for (i = 1; i < nlen && result; i++) {
      name = RaveList_get(names, i);
      param = Cartesian_getParameter(cartesian, name);
      if (CartesianParam_getXSize(param) != *xsize || CartesianParam_getYSize(param) != *ysize) {
        RAVE_ERROR1("Parameter %s has different xsize/ysize than rest", name);
        result = 0;
      }
      RAVE_OBJECT_RELEASE(param);
    }
  }
  RaveList_freeAndDestroy(&names);
  return result;
}

static int CartesianCfIOInternal_compareNameLists(RaveList_t* l1, RaveList_t* l2)
{
  int i = 0, j = 0, ni = 0, nj = 0;
  if (l1 == NULL || l2 == NULL)
    return 0;
  ni = RaveList_size(l1);
  nj = RaveList_size(l2);
  if (ni != nj) {
    RAVE_ERROR0("Different sizes between lists");
    return 0;
  }
  for (i = 0; i < ni; i++) {
    const char* name = RaveList_get(l1, i);
    int found = 0;
    for (j = 0; found == 0 && j < nj; j++) {
      const char* name2 = RaveList_get(l2, j);
      if (strcmp(name, name2) == 0)
        found = 1;
    }
    if (!found) {
      RAVE_ERROR0("Could not find %s name in other list");
      return 0;;
    }
  }
  return 1;
}

static int CartesianCfIOInternal_compareCartesian(CartesianCfIO_t* self, Cartesian_t* c1, Cartesian_t* c2)
{
  RaveList_t* names1 = NULL;
  RaveList_t* names2 = NULL;
  long xsize1, ysize1, xsize2, ysize2;
  int result = 0;

  if (!CartesianCfIOInternal_verifyParameters(self, c1, &xsize1, &ysize1))
    goto done;
  if (!CartesianCfIOInternal_verifyParameters(self, c2, &xsize2, &ysize2))
    goto done;

  if (xsize1 != xsize2 || ysize1 != ysize2) {
    RAVE_ERROR0("Inconsistant sizes between two cartesian products");
    goto done;
  }

  names1 = Cartesian_getParameterNames(c1);
  names2 = Cartesian_getParameterNames(c2);
  if (!CartesianCfIOInternal_compareNameLists(names1, names2)) {
    RAVE_ERROR0("Cartesian products contains different sets of quantities");
    goto done;
  }
  result = 1;
done:
  RaveList_freeAndDestroy(&names1);
  RaveList_freeAndDestroy(&names2);
  return result;
}

static int CartesianCfIOInternal_verifyVolume(CartesianCfIO_t* self, CartesianVolume_t* volume, long* xsize, long* ysize, long* nheights)
{
  int nvolumes = 0, i = 0;
  int result = 0;
  Cartesian_t* cartesian = NULL;
  Cartesian_t* othercartesian = NULL;

  nvolumes = CartesianVolume_getNumberOfImages(volume);
  if (nvolumes > 0) {
    cartesian = CartesianVolume_getImage(volume, 0);
    if (!CartesianCfIOInternal_verifyParameters(self, cartesian, xsize, ysize))
      goto done;
    *nheights = nvolumes;
    for (i = 1; i < nvolumes; i++) {
      othercartesian = CartesianVolume_getImage(volume, i);
      if (!CartesianCfIOInternal_compareCartesian(self, cartesian, othercartesian)) {
        RAVE_ERROR0("Volume doesn't contain same type of cartesian products");
        goto done;
      }
      RAVE_OBJECT_RELEASE(othercartesian);
    }
  }
  result = 1;
done:
  RAVE_OBJECT_RELEASE(cartesian);
  RAVE_OBJECT_RELEASE(othercartesian);
  return result;
}

/**
 * Defines a variable that is of one dimension.
 * @param[in] ncid - the net cdf id
 * @param[in] name - the name of the variable
 * @param[in] typid - the type of the variable, e.g. NC_FLOAT....
 * @param[in] dimid - the reference to the defined one-dimensional type
 * @param[out] varid - a reference to the created variable
 * @returns 1 on success otherwise 0
 */
static int CartesianCfIOInternal_defineOneDimVariable(int ncid, const char* name, int typid, int dimid, int* varid)
{
  if (nc_def_var(ncid, name, typid, 1, &dimid, varid)) {
    RAVE_ERROR1("Failed to define variable for %s", name);
    return 0;
  }
  return 1;
}

/**
 * Defines a two dimensional variable.
 * @param[in] ncid - the net cdf id
 * @param[in] name - the name of the variable
 * @param[in] typid - the type of the data
 * @param[in] dimids - the two dimensional ids defining this variable
 * @param[out] varid - the created variable id
 * @returns 1 on success otherwise 0
 */
static int CartesianCfIOInternal_defineTwoDimVariable(int ncid, const char* name, int typid, int* dimids, int* varid)
{
  if (nc_def_var(ncid, name, typid, 2, dimids, varid)) {
    RAVE_ERROR1("Failed to define variable for %s", name);
    return 0;
  }
  return 1;
}

/**
 * Adds some text into an attribute
 * @param[in] ncid - the netcdf file id
 * @param[in] varid - the variable id to add the attribute to
 * @param[in] name - the name of the attribute
 * @param[in] vlen - the length of the string to add
 * @param[in] v - the string
 * @returns 1 on success otherwise 0
 */
static int CartesianCfIOInternal_addText(int ncid, int varid, const char* name, size_t vlen, const char* v)
{
  if (nc_put_att_text(ncid, varid, name, vlen, v)) {
    RAVE_ERROR1("Failed to add text for %s", name);
    return 0;
  }
  return 1;
}

/**
 * Defines the x array variable
 * @param[in] ncid - the netcdf file id
 * @param[in] typid - the data type, e.g. NC_DOUBLE
 * @param[in] xsize - the x-size
 * @param[out] xsizevarid - the actual xsize dimensional varid
 * @param[out] varid - the variables varid
 * @return 1 on success otherwise 0
 */
static int CartesianCfIOInternal_defineX(int ncid, int typid, size_t xsize, int* xsizevarid, int* varid)
{
  int result = 0;
  if (nc_def_dim(ncid, "x", xsize, xsizevarid))
     goto done;

  if (!CartesianCfIOInternal_defineOneDimVariable(ncid, "x", typid, *xsizevarid, varid))
    goto done;

  if (!CartesianCfIOInternal_addText(ncid, *varid, UNITS, strlen(UNIT_METERS), UNIT_METERS))
    goto done;

  if (!CartesianCfIOInternal_addText(ncid, *varid, LONG_NAME, strlen("x-coordinate in Cartesian system"), "x-coordinate in Cartesian system"))
    goto done;

  if (!CartesianCfIOInternal_addText(ncid, *varid, STANDARD_NAME, strlen("projection_x_coordinate"), "projection_x_coordinate"))
    goto done;

  result = 1;
done:
  return result;
}

/**
 * Defines the y array variable
 * @param[in] ncid - the netcdf file id
 * @param[in] typid - the data type, e.g. NC_DOUBLE
 * @param[in] ysize - the y-size
 * @param[out] ysizevarid - the actual ysize dimensional varid
 * @param[out] varid - the variables varid
 * @return 1 on success otherwise 0
 */
static int CartesianCfIOInternal_defineY(int ncid, int typid, size_t ysize, int* ysizevarid, int* varid)
{
  int result = 0;
  if (nc_def_dim(ncid, "y", ysize, ysizevarid))
     goto done;

  if (!CartesianCfIOInternal_defineOneDimVariable(ncid, "y", typid, *ysizevarid, varid))
    goto done;

  if (!CartesianCfIOInternal_addText(ncid, *varid, UNITS, strlen(UNIT_METERS), UNIT_METERS))
    goto done;

  if (!CartesianCfIOInternal_addText(ncid, *varid, LONG_NAME, strlen("y-coordinate in Cartesian system"), "y-coordinate in Cartesian system"))
    goto done;

  if (!CartesianCfIOInternal_addText(ncid, *varid, STANDARD_NAME, strlen("projection_y_coordinate"), "projection_y_coordinate"))
    goto done;

  result = 1;
done:
  return result;
}

/**
 * Defines the longitude array variable
 * @param[in] ncid - the netcdf file id
 * @param[in] yxdimids - the two dimensions, first the y-dim varid and then x-dim varid
 * @param[out] lonvarid - the varid for this longitude array
 * @return 1 on success otherwise 0
 */
static int CartesianCfIOInternal_defineLongitude(int ncid, int* yxdimids, int* lonvarid)
{
  int result = 0;
  if (!CartesianCfIOInternal_defineTwoDimVariable(ncid, "longitude", NC_DOUBLE, yxdimids, lonvarid))
    goto done;
  if (!CartesianCfIOInternal_addText(ncid, *lonvarid, UNITS, strlen(UNIT_DEGREE_EAST), UNIT_DEGREE_EAST))
    goto done;
  if (!CartesianCfIOInternal_addText(ncid, *lonvarid, LONG_NAME, strlen(TEXT_LONGITUDE), TEXT_LONGITUDE))
    goto done;
  if (!CartesianCfIOInternal_addText(ncid, *lonvarid, STANDARD_NAME, strlen(TEXT_LONGITUDE), TEXT_LONGITUDE))
    goto done;
  result = 1;
done:
  return result;
}

/**
 * Defines the latitude array variable
 * @param[in] ncid - the netcdf file id
 * @param[in] yxdimids - the two dimensions, first the y-dim varid and then x-dim varid
 * @param[out] latvarid - the varid for this latitude array
 * @return 1 on success otherwise 0
 */
static int CartesianCfIOInternal_defineLatitude(int ncid, int* yxdimids, int* latvarid)
{
  int result = 0;
  if (!CartesianCfIOInternal_defineTwoDimVariable(ncid, "latitude", NC_DOUBLE, yxdimids, latvarid))
    goto done;
  if (!CartesianCfIOInternal_addText(ncid, *latvarid, UNITS, strlen(UNIT_DEGREE_NORTH), UNIT_DEGREE_NORTH))
    goto done;
  if (!CartesianCfIOInternal_addText(ncid, *latvarid, LONG_NAME, strlen(TEXT_LATITUDE), TEXT_LATITUDE))
    goto done;
  if (!CartesianCfIOInternal_addText(ncid, *latvarid, STANDARD_NAME, strlen(TEXT_LATITUDE), TEXT_LATITUDE))
    goto done;
  result = 1;
done:
  return result;
}

/**
 * Defines the time variable
 * @param[in] ncid - the netcdf file id
 * @param[out] ntimevarid - the varid for the unlimited time dimension
 * @param[out] varid - the varid for the time variable
 * @return 1 on success otherwise 0
 */
static int CartesianCfIOInternal_defineTime(int ncid, int* ntimevarid, int* varid)
{
  int result = 0;
  if (nc_def_dim(ncid, "time", NC_UNLIMITED, ntimevarid))
     goto done;

  if (!CartesianCfIOInternal_defineOneDimVariable(ncid, "time", NC_DOUBLE, *ntimevarid, varid))
    goto done;
  if (!CartesianCfIOInternal_addText(ncid, *varid, UNITS, strlen(UNIT_SECONDS_SINCE_19700101), UNIT_SECONDS_SINCE_19700101))
    goto done;
  if (!CartesianCfIOInternal_addText(ncid, *varid, LONG_NAME, strlen(TEXT_TIME), TEXT_TIME))
    goto done;
  if (!CartesianCfIOInternal_addText(ncid, *varid, STANDARD_NAME, strlen(TEXT_TIME), TEXT_TIME))
    goto done;
  result = 1;
done:
  return result;
}


/**
 * Extracts the time from the cartesian object into seconds since 1970-01-01 00:00:00 UTC
 * @param[in] cartesian - cartesian
 * @param[out] t - the seconds
 * @returns 1 on success otherwise 0
 */
static int CartesianCfIOInternal_getTime(Cartesian_t* cartesian, double* t)
{
  char datestr[32];
  struct tm dt;
  int result = 0;

  time_t utct = 0;
  strcpy(datestr, Cartesian_getDate(cartesian));
  strcat(datestr, Cartesian_getTime(cartesian));
  struct tm* x = gmtime(&utct);
  x->tm_isdst = 0; // UTC don't use DST
  if (strptime(datestr, "%Y%m%d%H%M%S", &dt)==NULL) {
    RAVE_ERROR1("Failed to create tm struct from %s", datestr);
    goto done;
  }
  dt.tm_isdst = 0; // UTC don't use DST
  *t = mktime(&dt) - mktime(x);
  result = 1;
done:
  return result;
}

/**
 * Defines the varids for the height0 dimension and the heigh0 variable
 * @param[in] ncid - the netcdf file id
 * @param[in] nheights - number of heights
 * @param[out] nheightvarid - the height dimension varid
 * @param[out] varid - the variable id
 * @returns 1 on success otherwise 0
 */
static int CartesianCfIOInternal_defineHeight0(int ncid, int nheights, int* nheightvarid, int* varid)
{
  int result = 0;
  if (nc_def_dim(ncid, "height0", nheights, nheightvarid))
     goto done;

  if (!CartesianCfIOInternal_defineOneDimVariable(ncid, "height0", NC_DOUBLE, *nheightvarid, varid))
    goto done;
  if (!CartesianCfIOInternal_addText(ncid, *varid, UNITS, strlen(UNIT_METERS), UNIT_METERS))
    goto done;
  if (!CartesianCfIOInternal_addText(ncid, *varid, DESCRIPTION, strlen(HEIGHT_DESCRIPTION), HEIGHT_DESCRIPTION))
    goto done;
  if (!CartesianCfIOInternal_addText(ncid, *varid, POSITIVE, strlen(TEXT_UP), TEXT_UP))
    goto done;
  if (!CartesianCfIOInternal_addText(ncid, *varid, LONG_NAME, strlen(TEXT_HEIGHT), TEXT_HEIGHT))
    goto done;

  result = 1;
done:
  return result;
}

/**
 * Writes a double array
 * @param[in] ncid - the netcdf file id
 * @param[in] varid - the variable id for the double array type
 * @param[in] data - the data to write
 * @param[in] description - description for log output
 */
static int CartesianCfIOInternal_writeDoubleArray(int ncid, int varid, double* data, const char* description)
{
  if (nc_put_var_double(ncid, varid, data)) {
    RAVE_ERROR1("Failed to write %s", description);
    return 0;
  }
  return 1;
}

/**
 * Adds the time value to the time varid
 * @param[in] ncid - the netcdf file id
 * @param[in] timevarid - the time variable id
 * @param[in] time0 the time in seconds since 1970-01-01
 * @returns 1 on success otherwise 0
 */
static int CartesianCfIOInternal_addTimeValue(int ncid, int timevarid, double time0)
{
  size_t sp = 0;
  size_t cp = 1;
  if (nc_put_vara_double(ncid, timevarid, &sp, &cp, &time0)) {
    RAVE_ERROR0("Failed to add time value to time variable");
    return 0;
  }
  return 1;
}

/**
 * Fills the surface information relevant for creating the CF file. Like lon/lat arrays and x/y-arrays
 * @param[in] cartesian - the cartesian product
 * @param[in] xsize - the xsize
 * @param[in] ysize - the ysize
 * @param[in,out] xarr - the x array in meters
 * @param[in,out] yarr - the y array in meters
 * @param[in,out] lonarr - the longitude array that matches each pixel in the data fields
 * @param[in,out] latarr - the latitude array that matches each pixel in the data fields
 * @returns 1 on success otherwise 0
 */
static int CartesianCfIOInternal_createSurfaceInformation(Cartesian_t* cartesian, long xsize, long ysize, double* xarr, double* yarr, double* lonarr, double* latarr)
{
  long x = 0, y = 0;
  int result = 0;
  Projection_t* lonlatPJ = NULL;
  Projection_t* cartesianPJ = NULL;
  lonlatPJ = RAVE_OBJECT_NEW(&Projection_TYPE);
  if (lonlatPJ == NULL || !Projection_init(lonlatPJ, "lonlat", "lonlat", "+proj=latlong +ellps=WGS84 +datum=WGS84")) {
    RAVE_ERROR0("Failed to allocate memory or initialize lonlat projection");
    goto done;
  }

  cartesianPJ = Cartesian_getProjection(cartesian);
  if (cartesianPJ == NULL) {
    RAVE_ERROR0("Cartesian product does not have a projection set");
    goto done;
  }

  for (y = ysize - 1; y >= 0; y--) {
    double herey = Cartesian_getLocationY(cartesian, y);
    *yarr = herey;
    yarr++;
    for (x = 0; x < xsize; x++) {
      double herex = Cartesian_getLocationX(cartesian, x);
      double olon,olat;
      xarr[x] = herex;
      if (!Projection_transformx(cartesianPJ, lonlatPJ, herex, herey, 0.0, &olon, &olat, NULL)) {
        RAVE_ERROR0("Transform failed");
        goto done;
      }
      *lonarr = olon * 180.0 / M_PI;
      *latarr = olat * 180.0 / M_PI;
      lonarr++;
      latarr++;
    }
  }
  result = 1;
done:
  RAVE_OBJECT_RELEASE(cartesianPJ);
  RAVE_OBJECT_RELEASE(lonlatPJ);
  return result;
}

static int CartesianCfIOInternal_getProdparFromCartesian(Cartesian_t* cartesian, double* v)
{
  int result = 0;
  *v = 0.0;
  RaveAttribute_t* attr = Cartesian_getAttribute(cartesian, "what/prodpar");
  if (attr != NULL) {
    if (RaveAttribute_getFormat(attr) == RaveAttribute_Format_Double) {
      double value = 0.0;
      RaveAttribute_getDouble(attr, &value);
      *v = value;
      result = 1;
    } else if (RaveAttribute_getFormat(attr) == RaveAttribute_Format_Long) {
      long value = 0;
      RaveAttribute_getLong(attr, &value);
      *v = (double)value;
      result = 1;
    } else if (RaveAttribute_getFormat(attr) == RaveAttribute_Format_String) {
      char* sv = NULL;
      double tv = 0.0;
      RaveAttribute_getString(attr, &sv);
      if (sscanf(sv, "%lf", &tv))
        *v = tv;
      result = 1;
    }
  }
  RAVE_OBJECT_RELEASE(attr);
  return result;
}

static int CartesianCfIOInternal_createHeight0Array(CartesianVolume_t* volume, double** height0)
{
  int i = 0, nlen = 0;
  double* heights = NULL;
  nlen = CartesianVolume_getNumberOfImages(volume);
  if (nlen <= 0)
    return 0;
  heights = RAVE_MALLOC(sizeof(double) * nlen);
  memset(heights, 0, sizeof(double)*nlen);
  for (i = 0; i < nlen; i++) {
    Cartesian_t* cartesian = CartesianVolume_getImage(volume, i);
    if (Cartesian_getParameterCount(cartesian) > 0) {
      double v = 0.0;
      if (CartesianCfIOInternal_getProdparFromCartesian(cartesian, &v)) {
        heights[i] = v;
      }
    }
    RAVE_OBJECT_RELEASE(cartesian);
  }
  *height0 = heights;
  return 1;
}

/**
 * Adds the projection definition to the file.
 * @param[in] ncid - the netcdf file id
 * @param[in] cartesian - the cartesian instance containing the projection
 * @param[out] projvarid - the projection definitions variable id
 * @param[out] grid_mapping - the name of the this projection
 * @returns 1 on success otherwise 0
 */
static int CartesianCfIOInternal_addProjectionDefinition(int ncid, Cartesian_t* cartesian, int* projvarid, char** grid_mapping)
{
  int result = 0;

  Projection_t* proj = Cartesian_getProjection(cartesian);
  RaveObjectList_t* wktlist = NULL;
  RaveAttribute_t* attr = NULL;
  if (proj != NULL) {
    char areaid[256];
    size_t n = 0;
    size_t i = 0;
    if (!OdimIoUtilities_getNodOrCmtFromSource(Cartesian_getSource(cartesian), areaid, 256)) {
      snprintf(areaid, 256, "projection_def");
    }
    if (nc_def_var(ncid, areaid, NC_INT, 0, NULL, projvarid)) {
      RAVE_ERROR0("Failed to create projection definition %d");
      goto done;
    }
    wktlist = RaveWkt_translate_from_projection(proj);
    n = RaveObjectList_size(wktlist);
    for (i = 0; i < n; i++) {
      attr = (RaveAttribute_t*)RaveObjectList_get(wktlist, i);
      if (RaveAttribute_getFormat(attr) == RaveAttribute_Format_String) {
        char* v = NULL;
        RaveAttribute_getString(attr, &v);
        if (!CartesianCfIOInternal_addStringAttribute(ncid, *projvarid, RaveAttribute_getName(attr), v)) {
          RAVE_ERROR1("Failed to add string attribute %s to projvarid", RaveAttribute_getName(attr));
          goto done;
        }
      } else if (RaveAttribute_getFormat(attr) == RaveAttribute_Format_Double) {
        double v;
        RaveAttribute_getDouble(attr, &v);
        if (!CartesianCfIOInternal_addDoubleAttribute(ncid, *projvarid, RaveAttribute_getName(attr), v)) {
          RAVE_ERROR1("Failed to add double attribute %s to projvarid", RaveAttribute_getName(attr));
          goto done;
        }
      } else if (RaveAttribute_getFormat(attr) == RaveAttribute_Format_DoubleArray) {
        int n = 0;
        double* v = NULL;
        RaveAttribute_getDoubleArray(attr, &v, &n);
        if (!CartesianCfIOInternal_addDoubleArrayAttribute(ncid, *projvarid, RaveAttribute_getName(attr), (size_t)n, v)) {
          RAVE_ERROR1("Failed to write double array for %s", RaveAttribute_getName(attr));
          goto done;
        }
      }
      RAVE_OBJECT_RELEASE(attr);
    }
    if (!CartesianCfIOInternal_addStringAttribute(ncid, *projvarid, "proj4", Projection_getDefinition(proj))) {
      RAVE_ERROR0("Failed to add proj4 definition");
      goto done;
    }
    *grid_mapping = RAVE_STRDUP(areaid);
    if (*grid_mapping==NULL)
      goto done;
    result = 1;
  }

done:
  RAVE_OBJECT_RELEASE(attr);
  RAVE_OBJECT_RELEASE(proj);
  RAVE_OBJECT_RELEASE(wktlist);
  return result;
}

/**
 * Adds the necessary variables ot the parameter
 * @param[in] ncid - the netcdf file id
 * @param[in] varid - the varid for the parameter beeing written
 * @param[in] name - the name of the quantity of this parameter (to get nodata/units/...)
 * @returns 1 on success otherwise 0
 */
static int CartesianCfIOInternal_addVariablesToParameter(int ncid, int varid, const char* name)
{
  int result = 0;
  const QuantityNodataUndetectMapping* mapping = CartesianCfIO_getQuantityMapping(name);
  if (mapping != NULL) {
    if (!CartesianCfIOInternal_addFloatAttribute(ncid, varid, _FillValue, mapping->nodata))
      goto done;
    if (!CartesianCfIOInternal_addStringAttribute(ncid, varid, UNITS, mapping->units))
      goto done;
    if (!CartesianCfIOInternal_addStringAttribute(ncid, varid, LONG_NAME, mapping->long_name))
      goto done;
    if (!CartesianCfIOInternal_addStringAttribute(ncid, varid, "coordinates", "longitude latitude"))
      goto done;
  }
  result = 1;
done:
  return result;
}

/**
 * Writes the cartesian product
 * @param[in] self - self
 * @param[in] ncid - the netcdf file id
 * @param[in] cartesian - the cartesian product beeing written
 * @returns 1 on success otherwise 0
 */
int CartesianCfIOInternal_writeCartesian(CartesianCfIO_t* self, int ncid, Cartesian_t* cartesian)
{
  long xsize, ysize;
  int result = 0;
  int yxdimids[2];
  int fieldids[4];
  size_t startp[4], countp[4];
  char* grid_mapping=NULL;

  int* varids = NULL;
  RaveList_t* names = NULL;
  int nlen, i;
  float* data = NULL;
  double *xarr = NULL, *yarr = NULL, *lonarr = NULL, *latarr = NULL;
  double height0=500.0, time0=0.0; // Just for testing
  int xvarid, yvarid, lonvarid, latvarid, xsizevarid, ysizevarid, timevarid, ntimevarid, heightvarid, nheightvarid, projvarid;

  if (!CartesianCfIOInternal_verifyParameters(self, cartesian, &xsize, &ysize))
    goto done;

  if (!CartesianCfIOInternal_getTime(cartesian, &time0))
    goto done;

  if (!CartesianCfIOInternal_addGlobalAttributes(self, ncid, cartesian))
    goto done;

  if (!CartesianCfIOInternal_defineX(ncid, NC_DOUBLE, xsize, &xsizevarid, &xvarid))
    goto done;

  if (!CartesianCfIOInternal_defineY(ncid, NC_DOUBLE, ysize, &ysizevarid, &yvarid))
    goto done;

  if (!CartesianCfIOInternal_defineTime(ncid, &ntimevarid, &timevarid))
    goto done;

  if (!CartesianCfIOInternal_defineHeight0(ncid, 1, &nheightvarid, &heightvarid))
    goto done;

  fieldids[0] = ntimevarid;
  fieldids[1] = nheightvarid;
  yxdimids[0] = fieldids[2] = ysizevarid;
  yxdimids[1] = fieldids[3] = xsizevarid;

  if (!CartesianCfIOInternal_defineLongitude(ncid, yxdimids, &lonvarid))
    goto done;
  if (!CartesianCfIOInternal_defineLatitude(ncid, yxdimids, &latvarid))
    goto done;

  // Now we need to create variables for all quantities
  names = Cartesian_getParameterNames(cartesian);
  if (names == NULL || RaveList_size(names) <= 0) {
    RAVE_ERROR0("No parameters in cartesian product");
    goto done;
  }
  nlen = RaveList_size(names);
  varids = RAVE_MALLOC(sizeof(int) * nlen);
  if (varids == NULL) {
    RAVE_ERROR0("Failed to allocate memory for varids");
    goto done;
  }

  if (!CartesianCfIOInternal_addProjectionDefinition(ncid, cartesian, &projvarid, &grid_mapping))
    goto done;

  for (i = 0; i < nlen; i++) {
    const char* name = RaveList_get(names, i);
    int varid = 0;
    if (nc_def_var(ncid, name, NC_FLOAT, 4, fieldids, &varid)) {
      RAVE_ERROR1("Failed to create netcdf var definition for %s", name);
      goto done;
    }
    varids[i] = varid;

    if (self->deflate_level > 0) {
      int shuffle = 1; /* ?? */
      int deflate = 1; /* 1 turns compression on */
      if (nc_def_var_deflate(ncid, varid, shuffle, deflate, self->deflate_level)) {
        RAVE_ERROR1("Failed to set %s name for compression", name);
        goto done;
      }
    }

    if (!CartesianCfIOInternal_addVariablesToParameter(ncid, varid, name))
      goto done;

    if (!CartesianCfIOInternal_addStringAttribute(ncid, varid, "grid_mapping", grid_mapping))
      goto done;
  }

  data = RAVE_MALLOC(sizeof(float) * xsize * ysize);
  xarr = RAVE_MALLOC(sizeof(double) * xsize);
  yarr = RAVE_MALLOC(sizeof(double) * ysize);
  lonarr = RAVE_MALLOC(sizeof(double) * xsize * ysize);
  latarr = RAVE_MALLOC(sizeof(double) * xsize * ysize);

  if (data == NULL || xarr == NULL || yarr == NULL || lonarr == NULL || latarr == NULL) {
    goto done;
  }

  if (nc_enddef(ncid))
  {
    RAVE_ERROR0("Failed to finish metadata writing");
    goto done;
  }

  if (!CartesianCfIOInternal_createSurfaceInformation(cartesian, xsize, ysize, xarr, yarr, lonarr, latarr)) {
    goto done;
  }

  if (!CartesianCfIOInternal_getProdparFromCartesian(cartesian, &height0)) {
    RAVE_ERROR0("Failed to get prodpar from cartesian");
    goto done;
  }

  if (!CartesianCfIOInternal_writeDoubleArray(ncid, xvarid, xarr, "x array") ||
      !CartesianCfIOInternal_writeDoubleArray(ncid, yvarid, yarr, "y array") ||
      !CartesianCfIOInternal_writeDoubleArray(ncid, lonvarid, lonarr, "lon array") ||
      !CartesianCfIOInternal_writeDoubleArray(ncid, latvarid, latarr, "lat array") ||
      !CartesianCfIOInternal_writeDoubleArray(ncid, heightvarid, &height0, "height0 array") ||
      !CartesianCfIOInternal_addTimeValue(ncid, timevarid, time0))
    goto done;

  startp[0] = startp[1] = startp[2] = startp[3] = 0;
  countp[0] = 1;
  countp[1] = 1;
  countp[2] = ysize;
  countp[3] = xsize;

  // And as a final feat we try to write all the actual data fields....
  for (i = 0; i < nlen; i++) {
    long y = 0, x = 0;
    const char* name = RaveList_get(names, i);
    float* datap = data;
    float nodata = CartesianCfIO_getNodata(name);
    memset(data, nodata, sizeof(float)*xsize*ysize);

    CartesianParam_t* param = Cartesian_getParameter(cartesian, name);
    for (y = ysize - 1; y >= 0; y--) {
      for (x = 0; x < xsize; x++) {
        double v = 0;
        float sv = nodata;
        if (CartesianParam_getConvertedValue(param, x, y, &v) == RaveValueType_DATA) {
          sv = (float)v;
        }
        *datap = sv;
        datap++;
      }
    }


    if (nc_put_vara_float(ncid, varids[i], startp, countp, data)) {
      RAVE_ERROR0("Failed to write data array");
      RAVE_OBJECT_RELEASE(param);
      goto done;
    }
    RAVE_OBJECT_RELEASE(param);
  }

  result = 1;
done:
  RAVE_FREE(varids);
  RAVE_FREE(xarr);
  RAVE_FREE(yarr);
  RAVE_FREE(lonarr);
  RAVE_FREE(latarr);
  RAVE_FREE(grid_mapping);
  RaveList_freeAndDestroy(&names);
  return result;
}

/**
 * Writes the cartesian volume product
 * @param[in] self - self
 * @param[in] ncid - the netcdf file id
 * @param[in] cartesian - the cartesian product beeing written
 * @returns 1 on success otherwise 0
 */
static int CartesianCfIOInternal_writeCartesianVolume(CartesianCfIO_t* self, int ncid, CartesianVolume_t* volume)
{
  long xsize = 0, ysize = 0, nheights = 0;
  Cartesian_t* cartesian = NULL;
  RaveList_t* names = NULL;
  int yxdimids[2];
  int fieldids[4];
  int result = 0;
  int nlen, i, j;
  char* grid_mapping=NULL;
  int* varids = NULL;
  float* data = NULL;
  size_t startp[4], countp[4];
  double *xarr = NULL, *yarr = NULL, *lonarr = NULL, *latarr = NULL;
  double* height0 = NULL;
  double time0=0.0;
  int xvarid, yvarid, lonvarid, latvarid, xsizevarid, ysizevarid, timevarid, ntimevarid, heightvarid, nheightvarid, projvarid;

  if (!CartesianCfIOInternal_verifyVolume(self, volume, &xsize, &ysize, &nheights))
    goto done;

  if (nheights <= 0) {
    RAVE_ERROR0("Nothing to write");
    goto done;
  }

  cartesian = CartesianVolume_getImage(volume, 0);

  if (!CartesianCfIOInternal_getTime(cartesian, &time0))
    goto done;

  if (!CartesianCfIOInternal_addGlobalAttributes(self, ncid, cartesian))
    goto done;

  if (!CartesianCfIOInternal_defineX(ncid, NC_DOUBLE, xsize, &xsizevarid, &xvarid))
    goto done;

  if (!CartesianCfIOInternal_defineY(ncid, NC_DOUBLE, ysize, &ysizevarid, &yvarid))
    goto done;

  if (!CartesianCfIOInternal_defineTime(ncid, &ntimevarid, &timevarid))
    goto done;

  if (!CartesianCfIOInternal_defineHeight0(ncid, nheights, &nheightvarid, &heightvarid))
    goto done;

  fieldids[0] = ntimevarid;
  fieldids[1] = nheightvarid;
  yxdimids[0] = fieldids[2] = ysizevarid;
  yxdimids[1] = fieldids[3] = xsizevarid;

  if (!CartesianCfIOInternal_defineLongitude(ncid, yxdimids, &lonvarid))
    goto done;
  if (!CartesianCfIOInternal_defineLatitude(ncid, yxdimids, &latvarid))
    goto done;

  // Now we need to create variables for all quantities
  names = Cartesian_getParameterNames(cartesian);
  if (names == NULL || RaveList_size(names) <= 0) {
    RAVE_ERROR0("No parameters in cartesian product");
    goto done;
  }
  nlen = RaveList_size(names);
  varids = RAVE_MALLOC(sizeof(int) * nlen);
  if (varids == NULL) {
    RAVE_ERROR0("Failed to allocate memory for varids");
    goto done;
  }

  if (!CartesianCfIOInternal_addProjectionDefinition(ncid, cartesian, &projvarid, &grid_mapping))
    goto done;

  for (i = 0; i < nlen; i++) {
    const char* name = RaveList_get(names, i);
    int varid = 0;
    if (nc_def_var(ncid, name, NC_FLOAT, 4, fieldids, &varid)) {
      RAVE_ERROR1("Failed to create netcdf var definition for %s", name);
      goto done;
    }
    varids[i] = varid;

    if (self->deflate_level > 0) {
      int shuffle = 1; /* ?? */
      int deflate = 1; /* 1 turns compression on */
      int ccode = nc_def_var_deflate(ncid, varid, shuffle, deflate, self->deflate_level);
      if (ccode) {
        RAVE_ERROR2("Failed to set %s name for compression code=%d", name, ccode);
        goto done;
      }
    }

    if (!CartesianCfIOInternal_addVariablesToParameter(ncid, varid, name))
      goto done;

    if (!CartesianCfIOInternal_addStringAttribute(ncid, varid, "grid_mapping", grid_mapping))
      goto done;
  }

  int datasize = sizeof(float) * xsize * ysize * nheights;
  data = RAVE_MALLOC(datasize);
  xarr = RAVE_MALLOC(sizeof(double) * xsize);
  yarr = RAVE_MALLOC(sizeof(double) * ysize);
  lonarr = RAVE_MALLOC(sizeof(double) * xsize * ysize);
  latarr = RAVE_MALLOC(sizeof(double) * xsize * ysize);

  if (data == NULL || xarr == NULL || yarr == NULL || lonarr == NULL || latarr == NULL) {
    goto done;
  }

  if (nc_enddef(ncid))
  {
    RAVE_ERROR0("Failed to finish metadata writing");
    goto done;
  }

  if (!CartesianCfIOInternal_createSurfaceInformation(cartesian, xsize, ysize, xarr, yarr, lonarr, latarr)) {
    goto done;
  }

  if (!CartesianCfIOInternal_createHeight0Array(volume, &height0))
    goto done;

  if (!CartesianCfIOInternal_writeDoubleArray(ncid, xvarid, xarr, "x array") ||
      !CartesianCfIOInternal_writeDoubleArray(ncid, yvarid, yarr, "y array") ||
      !CartesianCfIOInternal_writeDoubleArray(ncid, lonvarid, lonarr, "lon array") ||
      !CartesianCfIOInternal_writeDoubleArray(ncid, latvarid, latarr, "lat array") ||
      !CartesianCfIOInternal_writeDoubleArray(ncid, heightvarid, height0, "height0 array") ||
      !CartesianCfIOInternal_addTimeValue(ncid, timevarid, time0))
    goto done;

  startp[0] = startp[1] = startp[2] = startp[3] = 0;
  countp[0] = 1;
  countp[1] = nheights;
  countp[2] = ysize;
  countp[3] = xsize;


  // And as a final feat we try to write all the actual data fields....
  for (i = 0; i < nlen; i++) {
    long y = 0, x = 0;
    const char* name = RaveList_get(names, i);
    float* datap = data;
    float nodata = CartesianCfIO_getNodata(name);
    memset(data, nodata, datasize);
    for (j = 0; j < nheights; j++) {
      Cartesian_t* vcartesian = CartesianVolume_getImage(volume, j);
      if (vcartesian != NULL) {
        CartesianParam_t* param = Cartesian_getParameter(vcartesian, name);
        for (y = ysize - 1; y >= 0; y--) {
          for (x = 0; x < xsize; x++) {
            double v = 0;
            float sv = nodata;
            if (CartesianParam_getConvertedValue(param, x, y, &v) == RaveValueType_DATA) {
              sv = (float)v;
            }
            *datap = sv;
            datap++;
          }
        }
        RAVE_OBJECT_RELEASE(param);
      }
      RAVE_OBJECT_RELEASE(vcartesian);
    }

    if (nc_put_vara_float(ncid, varids[i], startp, countp, data)) {
      RAVE_ERROR0("Failed to write data array");
      goto done;
    }
  }

  result = 1;
done:
  RAVE_FREE(varids);
  RAVE_FREE(data);
  RAVE_FREE(xarr);
  RAVE_FREE(yarr);
  RAVE_FREE(lonarr);
  RAVE_FREE(latarr);
  RAVE_FREE(grid_mapping);
  RaveList_freeAndDestroy(&names);
  RAVE_OBJECT_RELEASE(cartesian);

  return result;
}

/*@} End of Private functions */

/*@{ Interface functions */


int CartesianCfIO_setDeflateLevel(CartesianCfIO_t* self, int level)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  if (level >= 0 && level <= 9) {
    self->deflate_level = level;
    return 1;
  }
  return 0;
}

int CartesianCfIO_getDeflateLevel(CartesianCfIO_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return self->deflate_level;
}

RaveCoreObject* CartesianOdimIO_read(CartesianCfIO_t* self, const char* filename)
{
	return NULL;
}

int CartesianCfIO_write(CartesianCfIO_t* self, const char* filename, RaveCoreObject* obj)
{
  int result = 0;
  int ncid = 0;
	RAVE_ASSERT((self != NULL), "self == NULL");
	int cmode = NC_CLOBBER;

	if (obj == NULL)
	{
		RAVE_ERROR0("Trying to write a file without specifying the object to write");
		return 0;
	}

	if (!RAVE_OBJECT_CHECK_TYPE(obj, &Cartesian_TYPE) &&
	    !RAVE_OBJECT_CHECK_TYPE(obj, &CartesianVolume_TYPE)) {
		RAVE_ERROR0("Trying to write an object that is not cartesian as CF convention");
		return 0;
	}

	if (filename == NULL)
	{
		RAVE_ERROR0("Trying to write without specifying filename");
		return 0;
	}

	if (strlen(filename) <= 3 || strncmp(filename + strlen(filename) - 3, ".nc", 3) != 0)
	{
		RAVE_ERROR0("File needs to end with .nc when writing CF");
		return 0;
	}

	if (self->deflate_level > 0) {
	  cmode |= NC_NETCDF4|NC_CLASSIC_MODEL;
	}

	if(nc_create(filename, cmode, &ncid)) {
	  RAVE_ERROR0("Failed to create file");
	  goto done;
	}

	if (RAVE_OBJECT_CHECK_TYPE(obj, &Cartesian_TYPE)) {
	  if (!CartesianCfIOInternal_writeCartesian(self, ncid, (Cartesian_t*)obj)) {
	    goto done;
	  }
	} else {
    if (!CartesianCfIOInternal_writeCartesianVolume(self, ncid, (CartesianVolume_t*)obj)) {
      goto done;
    }
	}

	if (nc_close(ncid)) {
	  RAVE_ERROR0("Failed to write cartesian file\n");
	  goto done;
	}

	result = 1;
done:
  return result;
}

/*@} End of Interface functions */

RaveCoreObjectType CartesianCfIO_TYPE = {
    "CartesianCfIO",
    sizeof(CartesianCfIO_t),
    CartesianCfIO_constructor,
    CartesianCfIO_destructor,
    CartesianCfIO_copyconstructor
};
