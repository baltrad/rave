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
 * Functions for performing rave related IO operations, mostly ODIM-formatted HDF5 files.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-11-12
 */
#include "rave_io.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "hlhdf.h"
#include "string.h"
#include "stdarg.h"

/**
 * Defines the structure for the RaveIO in a volume.
 */
struct _RaveIO_t {
  long ps_refCount;

  HL_NodeList* nodelist;
};

/*@{ Constants */
static const char RaveIO_ObjectType_UNDEFINED_STR[]= "UNDEFINED";
static const char RaveIO_ObjectType_PVOL_STR[]= "PVOL";
static const char RaveIO_ObjectType_CVOL_STR[]= "CVOL";
static const char RaveIO_ObjectType_SCAN_STR[]= "SCAN";
static const char RaveIO_ObjectType_RAY_STR[]= "RAY";
static const char RaveIO_ObjectType_AZIM_STR[]= "AZIM";
static const char RaveIO_ObjectType_IMAGE_STR[]= "IMAGE";
static const char RaveIO_ObjectType_COMP_STR[]= "COMP";
static const char RaveIO_ObjectType_XSEC_STR[]= "XSEC";
static const char RaveIO_ObjectType_VP_STR[]= "VP";
static const char RaveIO_ObjectType_PIC_STR[]= "PIC";

static const char* VALID_RaveIO_ObjectTypes[] = {
  RaveIO_ObjectType_PVOL_STR,
  RaveIO_ObjectType_CVOL_STR,
  RaveIO_ObjectType_SCAN_STR,
  RaveIO_ObjectType_RAY_STR,
  RaveIO_ObjectType_AZIM_STR,
  RaveIO_ObjectType_IMAGE_STR,
  RaveIO_ObjectType_COMP_STR,
  RaveIO_ObjectType_XSEC_STR,
  RaveIO_ObjectType_VP_STR,
  RaveIO_ObjectType_PIC_STR,
  NULL
};

static const char RaveIO_ODIM_Version_2_0_STR[] = "ODIM_H5/V2_0";

struct RaveToHlhdfTypeMap {
  HL_FormatSpecifier hlhdfFormat;
  RaveDataType raveType;
};

static const struct RaveToHlhdfTypeMap RAVE_TO_HLHDF_MAP[] = {
  {HLHDF_UNDEFINED, RaveDataType_UNDEFINED},
  {HLHDF_CHAR, RaveDataType_CHAR},
  {HLHDF_SCHAR, RaveDataType_CHAR},
  {HLHDF_UCHAR, RaveDataType_UCHAR},
  {HLHDF_SHORT, RaveDataType_SHORT},
  {HLHDF_USHORT, RaveDataType_SHORT},
  {HLHDF_INT, RaveDataType_INT},
  {HLHDF_UINT, RaveDataType_INT},
  {HLHDF_LONG, RaveDataType_LONG},
  {HLHDF_ULONG, RaveDataType_LONG},
  {HLHDF_LLONG, RaveDataType_UNDEFINED},
  {HLHDF_ULLONG, RaveDataType_UNDEFINED},
  {HLHDF_FLOAT, RaveDataType_FLOAT},
  {HLHDF_DOUBLE, RaveDataType_DOUBLE},
  {HLHDF_LDOUBLE, RaveDataType_UNDEFINED},
  {HLHDF_HSIZE, RaveDataType_UNDEFINED},
  {HLHDF_HSSIZE, RaveDataType_UNDEFINED},
  {HLHDF_HERR, RaveDataType_UNDEFINED},
  {HLHDF_HBOOL, RaveDataType_UNDEFINED},
  {HLHDF_STRING, RaveDataType_UNDEFINED},
  {HLHDF_COMPOUND, RaveDataType_UNDEFINED},
  {HLHDF_ARRAY, RaveDataType_UNDEFINED},
  {HLHDF_END_OF_SPECIFIERS, RaveDataType_UNDEFINED}
};

/*@} End of Constants */

/*@{ Macros */
/**
 * Quick access function for reading one atomic value from a
 * HLHDF node.
 *
 * @param[in] vt - the type for the read data
 * @param[in] nn - the node
 * @param[in] ss - the size of the data type
 * @param[in] ov - the output value
 * @param[in] ot - the type of the output value where assignment will be done.
 */
#define RAVEIO_GET_ATOMIC_NODEVALUE(vt, nn, ss, ot, ov) \
{ \
  vt v; \
  memcpy(&v, HLNode_getData(nn), ss); \
  ov = (ot)v; \
}

/*@} End of Macros */

/*@{ Private functions */
/**
 * Destroys the RaveIO instance
 * @param[in] scan - the cartesian product to destroy
 */
static void RaveIO_destroy(RaveIO_t* raveio)
{
  if (raveio != NULL) {
    RaveIO_close(raveio);
    RAVE_FREE(raveio);
  }
}

/**
 * Fetches the node with the provided name and returns the
 * value. Note, the value will be the internal pointer so
 * do not free it.
 * @param[in] raveio - the Rave IO instance
 * @param[in] name - the name
 * @param[out] value - the string value
 * @param[out] sz - the string length
 * @returns 0 on failure, otherwise 1
 */
static int RaveIOInternal_getStringValue(RaveIO_t* raveio, const char* name, char** value, size_t* sz)
{
  int result = 0;
  HL_Node* node = NULL;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  RAVE_ASSERT((sz != NULL), "sz == NULL");
  *value = NULL;
  *sz = 0;
  if (raveio->nodelist == NULL) {
    RAVE_ERROR0("Trying to read a file that not has been opened");
    goto done;
  }

  node = HLNodeList_getNodeByName(raveio->nodelist, name);
  if (node == NULL) {
    RAVE_ERROR1("Could not read %s", name);
    goto done;
  }

  if (HLNode_getFormat(node) != HLHDF_STRING) {
    RAVE_ERROR1("%s is not of type HLHDF_STRING", name);
    goto done;
  }

  *value = (char*)HLNode_getData(node);
  *sz = HLNode_getDataSize(node);

  result = 1;
done:
  return result;
}

/**
 * Same as @ref RaveIOInternal_getStringValue but uses variable argument lists for defining name.
 * @param[in] raveio - the Rave IO instance
 * @param[out] value - the string
 * @param[out] sz    - the length of the string
 * @param[in] fmt    - the format string
 * @param[in] ...    - the variable argument list
 * @returns 0 on failure, otherwise 1
 */
static int RaveIOInternal_getStringValueFmt(RaveIO_t* raveio, char** value, size_t* sz, const char* fmt, ...)
{
  va_list ap;
  char nodeName[1024];
  int n = 0;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  RAVE_ASSERT((sz != NULL), "sz == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n >= 0 && n < 1024) {
    return RaveIOInternal_getStringValue(raveio, nodeName, value, sz);
  }
  return 0;
}

/**
 * Fetches the node with the provided name and returns the
 * value.
 * @param[in] raveio - the Rave IO instance
 * @param[in] name - the name
 * @param[out] value - the double value
 * @returns 0 on failure, otherwise 1
 */
static int RaveIOInternal_getDoubleValue(RaveIO_t* raveio, const char* name, double* value)
{
  int result = 0;
  HL_Node* node = NULL;
  HL_FormatSpecifier format = HLHDF_UNDEFINED;
  size_t sz = 0;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  *value = 0.0L;
  if (raveio->nodelist == NULL) {
    RAVE_ERROR0("Trying to read a file that not has been opened");
    goto done;
  }

  node = HLNodeList_getNodeByName(raveio->nodelist, name);
  if (node == NULL) {
    RAVE_ERROR1("Could not read %s", name);
    goto done;
  }

  format = HLNode_getFormat(node);
  if (format < HLHDF_FLOAT || format > HLHDF_LDOUBLE) {
    RAVE_ERROR1("%s is not a float or double", name);
    goto done;
  }
  sz = HLNode_getDataSize(node);

  if (sz == sizeof(float)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(float, node, sz, double, *value);
  } else if (sz == sizeof(double)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(double, node, sz, double, *value);
  } else if (sz == sizeof(long double)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(long double, node, sz, double, *value);
    RAVE_WARNING1("Node %s is of type long double, downcasting",name);
  }

  result = 1;
done:
  return result;
}

/**
 * Same as @ref RaveIOInternal_getDoubleValue but takes a variable argument list instead.
 * @param[in] raveio - the Rave IO instance
 * @param[out] value - the double value
 * @param[in] fmt    - the formatter string
 * @param[in] ...    - the varargs
 * @return 0 on failure, otherwise 1
 */
static int RaveIOInternal_getDoubleValueFmt(RaveIO_t* raveio, double* value, const char* fmt, ...)
{
  va_list ap;
  char nodeName[1024];
  int n = 0;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n >= 0 && n < 1024) {
    return RaveIOInternal_getDoubleValue(raveio, nodeName, value);
  }
  return 0;
}

/**
 * Fetches the node with the provided name and returns the
 * value.
 * @param[in] raveio - the Rave IO instance
 * @param[in] name - the name
 * @param[out] value - the long value
 * @returns 0 on failure, otherwise 1
 */
static int RaveIOInternal_getLongValue(RaveIO_t* raveio, const char* name, long* value)
{
  int result = 0;
  HL_Node* node = NULL;
  HL_FormatSpecifier format = HLHDF_UNDEFINED;
  size_t sz = 0;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");
  *value = 0.0L;
  if (raveio->nodelist == NULL) {
    RAVE_ERROR0("Trying to read a file that not has been opened");
    goto done;
  }

  node = HLNodeList_getNodeByName(raveio->nodelist, name);
  if (node == NULL) {
    RAVE_ERROR1("Could not read %s", name);
    goto done;
  }

  format = HLNode_getFormat(node);
  if (format < HLHDF_SCHAR || format > HLHDF_ULLONG) {
    RAVE_ERROR1("%s is not a integer", name);
    goto done;
  }
  sz = HLNode_getDataSize(node);
  if (sz == sizeof(char)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(char, node, sz, long, *value);
  } else if (sz == sizeof(short)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(short, node, sz, long, *value);
  } else if (sz == sizeof(int)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(int, node, sz, long, *value);
  } else if (sz == sizeof(long)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(long, node, sz, long, *value);
  } else if (sz == sizeof(long long)) {
    RAVEIO_GET_ATOMIC_NODEVALUE(long long, node, sz, long, *value);
    RAVE_WARNING1("Node %s is of type long long, downcasting",name);
  }

  result = 1;
done:
  return result;
}

/**
 * Same as @ref RaveIOInternal_getLongValue but takes a variable argument list instead.
 * @param[in] raveio - the Rave IO instance
 * @param[out] value - the double value
 * @param[in] fmt    - the formatter string
 * @param[in] ...    - the varargs
 * @return 0 on failure, otherwise 1
 */
static int RaveIOInternal_getLongValueFmt(RaveIO_t* raveio, long* value, const char* fmt, ...)
{
  va_list ap;
  char nodeName[1024];
  int n = 0;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  RAVE_ASSERT((value != NULL), "value == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n >= 0 && n < 1024) {
    return RaveIOInternal_getLongValue(raveio, nodeName, value);
  }
  return 0;
}

/**
 * Returns the node with the name as specified by the variable argument
 * list.
 * @param[in] raveio - the Rave IO instance
 * @param[in] fmt    - the variable argument format
 * @param[in] ...    - the argument list
 * @returns the found node or NULL if not found
 */
static HL_Node* RaveIOInternal_getNodeFmt(RaveIO_t* raveio, const char* fmt, ...)
{
  va_list ap;
  char nodeName[1024];
  int n = 0;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n >= 0 && n < 1024) {
    return HLNodeList_getNodeByName(raveio->nodelist, nodeName);
  }
  return NULL;
}

/**
 * Translates a hlhdf format specified into a rave data type.
 * @param[in] format - the hlhdf format specified
 * @returns the RaveDataType
 */
static RaveDataType RaveIOInternal_hlhdfToRaveType(HL_FormatSpecifier format)
{
  int index = 0;
  RaveDataType result = RaveDataType_UNDEFINED;
  while (RAVE_TO_HLHDF_MAP[index].hlhdfFormat != HLHDF_END_OF_SPECIFIERS) {
    if (RAVE_TO_HLHDF_MAP[index].hlhdfFormat == format) {
      result = RAVE_TO_HLHDF_MAP[index].raveType;
      break;
    }
    index++;
  }
  return result;
}

/**
 * Verifies if the file contains a node with the name as specified by the variable
 * argument list.
 * @param[in] raveio - the Rave IO instance
 * @param[in] fmt    - the variable argument format specifier
 * @param[in] ...    - the variable argument list
 * @returns 1 if the node could be found, otherwise 0
 */
int RaveIOInternal_hasNodeByNameFmt(RaveIO_t* raveio, const char* fmt, ...)
{
  va_list ap;
  char nodeName[1024];
  int n = 0;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");

  va_start(ap, fmt);
  n = vsnprintf(nodeName, 1024, fmt, ap);
  va_end(ap);
  if (n >= 0 && n < 1024) {
    return HLNodeList_hasNodeByName(raveio->nodelist, nodeName);
  }
  return 0;
}
/*@} End of Private functions */

RaveIO_t* RaveIO_new(void)
{
  RaveIO_t* result = NULL;
  result = RAVE_MALLOC(sizeof(RaveIO_t));
  if (result != NULL) {
    result->ps_refCount = 1;
    result->nodelist = NULL;
  }
  return result;
}

void RaveIO_release(RaveIO_t* raveio)
{
  if (raveio != NULL) {
    raveio->ps_refCount--;
    if (raveio->ps_refCount <= 0) {
      RaveIO_destroy(raveio);
    }
  }
}

RaveIO_t* RaveIO_copy(RaveIO_t* raveio)
{
  if (raveio != NULL) {
    raveio->ps_refCount++;
  }
  return raveio;
}

void RaveIO_close(RaveIO_t* raveio)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  HLNodeList_free(raveio->nodelist);
  raveio->nodelist = NULL;
}

RaveIO_t* RaveIO_open(const char* filename)
{
  RaveIO_t* result = NULL;
  result = RaveIO_new();
  if (result != NULL) {
    if (!RaveIO_openFile(result, filename)) {
      RaveIO_release(result);
      result = NULL;
    }
  }
  return result;
}

int RaveIO_openFile(RaveIO_t* raveio, const char* filename)
{
  int result = 0;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  RaveIO_close(raveio);
  raveio->nodelist = HLNodeList_read(filename);
  if (raveio->nodelist == NULL) {
    RAVE_ERROR1("Failed to load hdf5 file '%s'", filename);
    goto done;
  }
  HLNodeList_selectAllNodes(raveio->nodelist);
  if (!HLNodeList_fetchMarkedNodes(raveio->nodelist)) {
    RAVE_ERROR1("Failed to load hdf5 file '%s'", filename);
    goto done;
  }
  result = 1;
done:
  return result;
}

int RaveIO_isOpen(RaveIO_t* raveio)
{
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  return (raveio->nodelist != NULL)?1:0;
}

PolarScan_t* RaveIO_loadScanIndex(RaveIO_t* raveio, const int dsindex, const int dindex)
{
  char entryName[30];
  PolarScan_t* result = NULL;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");

  sprintf(entryName, "/dataset%d/data%d", dsindex, dindex);
  if (HLNodeList_hasNodeByName(raveio->nodelist, entryName)) {
    double gain = 0.0L, offset = 0.0L, nodata = 0.0L, undetect = 0.0L;
    double elangle = 0.0L, rscale = 0.0L, rstart = 0.0L;
    long nbins = 0, nrays = 0, a1gate = 0;
    char* quantity = NULL;
    size_t szQuantity = 0;
    HL_Node* dataNode = NULL;
    RaveDataType dataType = RaveDataType_UNDEFINED;
    if (!RaveIOInternal_getDoubleValueFmt(raveio, &gain, "%s/what/gain", entryName)) {
      goto done;
    }
    if (!RaveIOInternal_getDoubleValueFmt(raveio, &offset, "%s/what/offset", entryName)) {
      goto done;
    }
    if (!RaveIOInternal_getDoubleValueFmt(raveio, &nodata, "%s/what/nodata", entryName)) {
      goto done;
    }
    if (!RaveIOInternal_getDoubleValueFmt(raveio, &undetect, "%s/what/undetect", entryName)) {
      goto done;
    }
    if (!RaveIOInternal_getStringValueFmt(raveio, &quantity, &szQuantity, "%s/what/quantity", entryName)) {
      goto done;
    }
    if (!RaveIOInternal_getDoubleValueFmt(raveio, &elangle, "/dataset%d/where/elangle", dsindex)) {
      goto done;
    }
    if (!RaveIOInternal_getLongValueFmt(raveio, &a1gate, "/dataset%d/where/a1gate", dsindex)) {
      goto done;
    }
    if (!RaveIOInternal_getLongValueFmt(raveio, &nbins, "/dataset%d/where/nbins", dsindex)) {
      goto done;
    }
    if (!RaveIOInternal_getLongValueFmt(raveio, &nrays, "/dataset%d/where/nrays", dsindex)) {
      goto done;
    }
    if (!RaveIOInternal_getDoubleValueFmt(raveio, &rscale, "/dataset%d/where/rscale", dsindex)) {
      goto done;
    }
    if (!RaveIOInternal_getDoubleValueFmt(raveio, &rstart, "/dataset%d/where/rstart", dsindex)) {
      goto done;
    }
    dataNode = RaveIOInternal_getNodeFmt(raveio, "%s/data", entryName);
    if (dataNode == NULL) {
      goto done;
    }
    dataType = RaveIOInternal_hlhdfToRaveType(HLNode_getFormat(dataNode));
    if (dataType == RaveDataType_UNDEFINED) {
      goto done;
    }
    result = PolarScan_new();
    if (result == NULL) {
      goto done;
    }
    PolarScan_setGain(result, gain);
    PolarScan_setOffset(result, offset);
    PolarScan_setNodata(result, nodata);
    PolarScan_setUndetect(result, undetect);
    PolarScan_setQuantity(result, quantity);
    PolarScan_setElangle(result, elangle * M_PI/180.0);
    PolarScan_setA1gate(result, a1gate);
    PolarScan_setRscale(result, rscale);
    PolarScan_setRstart(result, rstart);
    PolarScan_setData(result, nbins,nrays,HLNode_getData(dataNode), dataType);
  }

done:
  return result;
}

PolarVolume_t* RaveIO_loadVolume(RaveIO_t* raveio)
{
  PolarVolume_t* result = NULL;
  double lon = 0.0L, lat = 0.0L, height = 0.0L;
  int dsindex = 1;

  RAVE_ASSERT((raveio != NULL), "raveio == NULL");

  if (!RaveIO_isOpen(raveio)) {
    RAVE_ERROR0("Trying to load a volume when no file is open");
    return NULL;
  }

  if (!RaveIO_isSupported(raveio)) {
    RAVE_ERROR0("Loading is not supported for provided file");
    return NULL;
  }

  // What information

  // Where information
  if (!RaveIOInternal_getDoubleValue(raveio, "/where/lon", &lon) ||
      !RaveIOInternal_getDoubleValue(raveio, "/where/lat", &lat) ||
      !RaveIOInternal_getDoubleValue(raveio, "/where/height", &height)) {
    RAVE_ERROR0("Could not read location information");
    return NULL;
  }

  result = PolarVolume_new();
  if (result != NULL) {
    PolarVolume_setLongitude(result, lon*M_PI/180.0);
    PolarVolume_setLatitude(result, lat*M_PI/180.0);
    PolarVolume_setHeight(result, height);
  }

  // Read scans
  while (RaveIOInternal_hasNodeByNameFmt(raveio, "/dataset%d", dsindex)) {
    int dindex = 1;
    while (RaveIOInternal_hasNodeByNameFmt(raveio, "/dataset%d/data%d", dsindex, dindex)) {
      PolarScan_t* scan = RaveIO_loadScanIndex(raveio, dsindex, dindex);
      if (scan != NULL) {
        PolarVolume_addScan(result, scan);
      }
      PolarScan_release(scan);
      dindex++;
    }
    dsindex++;
  }

  return result;
}

int RaveIO_isSupported(RaveIO_t* raveio)
{
  int result = 0;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  if (raveio->nodelist != NULL &&
      RaveIO_getObjectType(raveio) == RaveIO_ObjectType_PVOL &&
      RaveIO_getOdimVersion(raveio) != RaveIO_ODIM_Version_UNDEFINED) {
    result = 1;
  }
  return result;
}

RaveIO_ObjectType RaveIO_getObjectType(RaveIO_t* raveio)
{
  RaveIO_ObjectType result = RaveIO_ObjectType_UNDEFINED;
  char* objectType = NULL;
  size_t sz = 0;
  int index = 0;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");
  if (!RaveIOInternal_getStringValue(raveio, "/what/object", &objectType, &sz)) {
    RAVE_ERROR0("Failed to read attribute /what/object");
    goto done;
  }

  while (VALID_RaveIO_ObjectTypes[index] != NULL) {
    if (strcmp(VALID_RaveIO_ObjectTypes[index], objectType) == 0) {
      result = (RaveIO_ObjectType)index;
      break;
    }
    index++;
  }

done:
  return result;
}

RaveIO_ODIM_Version RaveIO_getOdimVersion(RaveIO_t* raveio)
{
  RaveIO_ODIM_Version result = RaveIO_ODIM_Version_UNDEFINED;
  char* odimVersion = NULL;
  size_t sz = 0;
  RAVE_ASSERT((raveio != NULL), "raveio == NULL");

  if (!RaveIOInternal_getStringValue(raveio, "/Conventions", &odimVersion, &sz)) {
    RAVE_ERROR0("Failed to read attribute /Conventions");
    goto done;
  }

  if (strcmp(RaveIO_ODIM_Version_2_0_STR, odimVersion) == 0) {
    result = RaveIO_ODIM_Version_2_0;
  }

done:
  return result;
}

