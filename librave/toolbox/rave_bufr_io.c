/* --------------------------------------------------------------------
Copyright (C) 2011 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Adaptor for polar BUFR files.
 * This object supports \ref #RAVE_OBJECT_CLONE.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2011-11-08
 */
#include "rave_bufr_io.h"
#include "rave_debug.h"
#include "rave_alloc.h"
#include "rave_utilities.h"
#include "rave_data2d.h"
#include "hlhdf.h"
#include "hlhdf_alloc.h"
#include "hlhdf_debug.h"
#include "string.h"
#include "stdarg.h"
#include "raveobject_hashtable.h"
#include "raveobject_list.h"
#include "polarvolume.h"
#include "cartesianvolume.h"
#include "rave_field.h"
#include "rave_hlhdf_utilities.h"
#include "cartesian_odim_io.h"
#include "polar_odim_io.h"
#include <time.h>
#include <zlib.h>
#include <float.h>
#include <ctype.h>
#include "rave_config.h"

/* In order to allow concurrent use we can use pthread if we are able to.
 */
#ifdef  PTHREAD_SUPPORTED
#include <pthread.h>
#endif

#include "desc.h"
#include "bufr.h"

/**
 * Global rave object that will be filled
 */
static RaveCoreObject* raveObject = NULL;

#ifdef  PTHREAD_SUPPORTED
static int bufrio_mutex_initialized = 0;
static pthread_mutex_t bufrio_mutex;
#endif


/**
 * Defines the structure for the RaveIO in a volume.
 */
struct _RaveBufrIO_t {
  RAVE_OBJECT_HEAD /** Always on top */
  char* tabledir;  /**< the directory for the bufr descriptor tables */
};

/*@{ Private functions */
static int RaveBufrIO_constructor(RaveCoreObject* obj)
{
  RaveBufrIO_t* this = (RaveBufrIO_t*)obj;
  this->tabledir = NULL;

  if (strcmp("", RAVE_BUFR_TABLES_DIR) != 0) {
    if (!RaveBufrIO_setTableDir(this, RAVE_BUFR_TABLES_DIR)) {
      goto done;
    }
  }

  return 1;
done:
  RAVE_FREE(this->tabledir);
  return 0;
}

/**
 * Destroys the list
 * @param[in] list - the list to destroy
 */
static void RaveBufrIO_destructor(RaveCoreObject* obj)
{
  RaveBufrIO_t* this = (RaveBufrIO_t*)obj;
  RAVE_FREE(this->tabledir);
}

/**
 * Checks if the descriptor with specified ids exists in the array
 * @param[in] dds - the data descriptors
 * @param[in] ndescs - the number of data descriptors
 * @param[in] f - f id
 * @param[in] x - x id
 * @param[in] y - y id
 * @®eturn 1 if it exists, otherwise 0
 */
static int RaveBufrInternal_hasDescriptor(dd* dds, int ndescs, int f, int x, int y)
{
  int result = 0;
  int i = 0;


  if (dds == NULL || ndescs <= 0) {
    goto done;
  }
  for (i = 0; result == 0 && i < ndescs; i++) {
    if (bufr_check_fxy(dds+i, f, x, y)) {
      result = 1;
    }
  }

done:
  return result;
}
/**
 * Creates a rave core object by interpreeting the data descriptors.
 * @param[in] dds - the data descriptors on first read
 * @param[in] ndescs - the number of data descriptors in array
 * @returns the rave object, either a cartesian or a polar volume
 */
static RaveCoreObject* RaveBufrIOInternal_createRaveObject(dd* dds, int ndescs)
{
  RaveCoreObject *obj = NULL, *result = NULL;

  if (dds == NULL || ndescs <= 0) {
    goto done;
  }

  if (RaveBufrInternal_hasDescriptor(dds, ndescs, 3,21,204) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 3,1,31) &&
      (RaveBufrInternal_hasDescriptor(dds, ndescs, 3,21,203) ||
       RaveBufrInternal_hasDescriptor(dds, ndescs, 3,21,207))) {
    obj = RAVE_OBJECT_NEW(&PolarVolume_TYPE);
  } else {
    RAVE_ERROR0("Does not reckognize BUFR descriptor combination");
/*
    if (
      RaveBufrInternal_hasDescriptor(dds, ndescs, 3,1,11) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 3,1,13) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 3,21,204) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 0,29,205) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 0,5,33) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 0,6,33) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 2,1,129) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 0,30,21) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 0,30,22) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 2,1,0) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 3,1,21) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 3,21,8) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 3,21,204) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 1,4,0) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 0,31,1) &&
      RaveBufrInternal_hasDescriptor(dds, ndescs, 3,21,205)) {
    obj = RAVE_OBJECT_NEW(&Cartesian_TYPE);
*/
  }

  result = RAVE_OBJECT_COPY(obj);
done:
  RAVE_OBJECT_RELEASE(obj);
  return result;
}

/**
 * Decompresses the indata with the zlib uncompression algorithm.
 * @param[in] indata - the data to be decompressed
 * @param[in] nindata - the length of the indata
 * @param[in] npixels - the number of pixels in the array
 * @param[out] the number of pixels in the result
 */
static varfl* RaveBufrIOInternal_decompress(unsigned char* indata, unsigned long nindata, unsigned long npixels, unsigned long* nresult)
{
  unsigned char *buf = NULL, *result = NULL;
  int c = 0;

  RAVE_ASSERT((indata != NULL), "indata == NULL");
  RAVE_ASSERT((nresult != NULL), "nresult == NULL");

  buf = RAVE_MALLOC(sizeof(varfl) * npixels);
  if (buf == NULL) {
    goto done;
  }
  *nresult = npixels * sizeof(varfl);
  c = uncompress(buf, nresult, indata, nindata);
  if (c != Z_OK) {
    RAVE_ERROR1("Failed to uncompress bufr data array: ERRCODE %d", c);
    goto done;
  }

  result = buf;
  buf = NULL; /* Release responsibility for memory */
done:
  RAVE_FREE(buf);
  return (varfl*)result;
}

static int RaveBufrIOInternal_addHowToObject(varfl* vv, int* ii, void* kvalue,
  int (*RaveBufrIOInternal_addAttributeCB)(void*, RaveAttribute_t*)) {
  int k, n, m, i = *ii;
  int result = 0;
  RaveAttribute_t* attr = NULL;
  n = vv[i++];
  for (m = 0; m < n && result == 0; ++m) {
    char* s1 = calloc(20, sizeof(char));
    char* s2 = calloc(16, sizeof(char));
    char* ts1 = NULL;
    strcpy(s1,"how/");
    for (k = 4; k < 20; ++k) {
      s1[k] = vv[i++];
    }
    ts1 = &s1[19];
    while (isspace(*ts1)) {
      *ts1 = '\0';
      ts1--;
    }
    for (k = 0; k < 16; ++k) {
      s2[k] = vv[i++];
    }
    attr = RaveAttributeHelp_createString(s1, s2);
    if (attr == NULL || !RaveBufrIOInternal_addAttributeCB(kvalue, attr)) {
      result = 1;
    }
    RAVE_OBJECT_RELEASE(attr);
    free(s1);
    free(s2);
  }
  n = vv[i++];
  for (m = 0; m < n && result == 0; ++m) {
    char * s1 = calloc(20, sizeof(char));
    char * s2 = calloc(16, sizeof(char));
    char* ts1 = NULL;
    strcpy(s1,"how/");
    for (k = 4; k < 20; ++k) {
      s1[k] = vv[i++];
    }
    ts1 = &s1[19];
    while (isspace(*ts1)) {
      *ts1 = '\0';
      ts1--;
    }

    for (k = 0; k < 8; ++k) {
      s2[k] = vv[i++];
    }
    attr = RaveAttributeHelp_createDoubleFromString(s1,s2);
    if (attr == NULL || !RaveBufrIOInternal_addAttributeCB(kvalue, attr)) {
      result = 1;
    }
    RAVE_OBJECT_RELEASE(attr);
    free(s1);
    free(s2);
  }
  *ii = i;
  return result;
}

static int RaveBufrIOInternal_addAttributeToPolarVolume(void* kvalue, RaveAttribute_t* raveattr)
{
  return PolarVolume_addAttribute((PolarVolume_t*)kvalue, raveattr);
}

static int RaveBufrIOInternal_addAttributeToPolarScan(void* kvalue, RaveAttribute_t* raveattr)
{
  return PolarScan_addAttribute((PolarScan_t*)kvalue, raveattr);
}

static int RaveBufrIOInternal_addAttributeToPolarScanParam(void* kvalue, RaveAttribute_t* raveattr)
{
  return PolarScanParam_addAttribute((PolarScanParam_t*)kvalue, raveattr);
}

static void RaveBufrIOInternal_getDateTimeStrings(varfl* vv, int* ii, char* datestr, char* timestr) {
  int year, month, day, hour, minute, second;
  int i = *ii;
  year = (int)vv[i++]; month = (int)vv[i++]; day = (int)vv[i++];
  hour = (int)vv[i++]; minute = (int)vv[i++]; second = (int)vv[i++];
  sprintf(datestr, "%04d%02d%02d", year, month, day);
  sprintf(timestr, "%02d%02d%02d", hour, minute, second);
  *ii = i;
}

static void RaveBufrIOInternal_getString(varfl* vv, int* ii, char* buff, int bufflen, int trimRight)
{
  int k = 0, i = *ii;
  for (k = 0; k < bufflen; k++) {
    buff[k] = vv[i++];
  }
  if (trimRight) {
    k = bufflen - 1;
    while (k >= 0 && isspace(buff[k])) {
      buff[k] = '\0';
      k--;
    }
  }
  *ii = i;
}

static PolarScanParam_t* RaveBufrIOInternal_getOdim22Parameter(varfl* vv, int* ii, long nbins, long nrays)
{
  PolarScanParam_t* param = NULL;
  PolarScanParam_t* result = NULL;
  int i = *ii;
  char quantity[6];
  int compression = 0;

  param = RAVE_OBJECT_NEW(&PolarScanParam_TYPE);
  if (param == NULL) {
    goto done;
  }
  RaveBufrIOInternal_addHowToObject(vv, &i, param, RaveBufrIOInternal_addAttributeToPolarScanParam);
  RaveBufrIOInternal_getString(vv, &i, quantity, 6, 1);
  if (!PolarScanParam_setQuantity(param, quantity)) {
    goto done;
  }
  RAVE_DEBUG1("Quantity is: %s\n",quantity);
  PolarScanParam_setGain(param, 1.0);
  PolarScanParam_setOffset(param, 0.0);
  PolarScanParam_setNodata(param, DBL_MAX);
  PolarScanParam_setUndetect(param, -DBL_MAX);
  compression = (int)vv[i++];
  if (compression != 0) {
    RAVE_ERROR0("BUFR file has not been stored with zlib compression");
    goto done;
  } else {
    int k = 0, n = 0, j = 0, niter = 0;
    unsigned char* tempo = NULL;
    unsigned long ndata = 0;
    varfl* data = NULL;
    niter = vv[i++];
    tempo = RAVE_MALLOC(niter * 65534 * sizeof(unsigned char));
    if (tempo == NULL) {
      goto done;
    }
    memset(tempo, 0, niter*65534*sizeof(unsigned char));
    for (n = 0; n < niter; ++n) {
      int ncomp = vv[i++];
      for (k = 0; k < ncomp; ++k) {
        tempo[j++] = (vv[i+k] == MISSVAL ? 255 : vv[i+k]);
      }
      i += ncomp;
    }
    data = RaveBufrIOInternal_decompress(tempo, j, nbins * nrays, &ndata);
    RAVE_FREE(tempo);

    if (data == NULL) {
      goto done;
    }

    if (!PolarScanParam_setData(param, nbins, nrays, data, RaveDataType_DOUBLE)) {
      RAVE_FREE(data);
      goto done;
    }

    result = RAVE_OBJECT_COPY(param);
    RAVE_FREE(data);
  }
done:
  RAVE_OBJECT_RELEASE(param);
  *ii = i;
  return result;
}

/**
 * Callback function that handles polar volumes.
 * @param[in] val - val
 * @param[in] ind - int
 * @return return code
 */
static int RaveBufrIOInternal_polarVolumeCallback(varfl val, int ind)
{
  int result = 0;  /* We start assuming that everything is going to be fine */
  PolarScan_t* scan = NULL;
  PolarScanParam_t* param = NULL;

  RAVE_ASSERT((raveObject != NULL), "raveObject == NULL");
  RAVE_ASSERT((RAVE_OBJECT_CHECK_TYPE(raveObject, &PolarVolume_TYPE)), "raveObject != PolarVolume_TYPE");

  if (ind == _desc_special) {
    if (des[ind]->el->d.f != 2) {
      RAVE_INFO3("fxy: %d %d %d", des[ind]->el->d.f, des[ind]->el->d.x, des[ind]->el->d.y);
    }
    return 1;
  }

  if (des[ind]->id == SEQDESC) {
    dd* d = &(des[ind]->seq->d);
    varfl* vv;
    bufrval_t* v = bufr_open_val_array();
    if (v == NULL) {
      RAVE_ERROR0("Failed to open val array\n");
      goto done;
    }
    if (!bufr_parse_out(des[ind]->seq->del, 0, des[ind]->seq->nel - 1, bufr_val_to_global, 0)) {
      RAVE_ERROR0("Failed to run bufr_parse_out\n");
      goto done;
    }
    vv = v->vals;

    if (bufr_check_fxy(d, 3, 1, 31)) {
      /* WMO block and station number */
      int i = 0;

      varfl tempo = vv[i++];
      if (tempo != MISSVAL) {
        int wmoblock = (int)tempo;
        int wmostat = (int)vv[i++];
        char source[16];
        if (wmostat == MISSVAL) {
          RAVE_ERROR0("Failed to get wmostat\n");
          goto done;
        }
        sprintf(source, "WMO:%02d%03d",wmoblock, wmostat);
        if (!PolarVolume_setSource((PolarVolume_t*)raveObject, source)) {
          goto done;
        }
      } else {
        RAVE_WARNING0("No station number");
      }
      i++;
      i++; /* Year vv[i++]*/
      i++; /* Month  vv[i++]*/
      i++; /* Day  vv[i++]*/
      i++; /* Hour vv[i++] */
      i++; /* Minute vv[i++] */
      PolarVolume_setLatitude((PolarVolume_t*)raveObject, (double)vv[i++] * M_PI / 180.0);
      PolarVolume_setLongitude((PolarVolume_t*)raveObject, (double)vv[i++] * M_PI / 180.0);
      PolarVolume_setHeight((PolarVolume_t*)raveObject, (double)vv[i++]);
    } else if (bufr_check_fxy (d, 3, 21, 203)) {
      int i = 0;
      int si = 0; /* scan index */
      int nscans = vv[i++];
      for (si = 0; si < nscans; si++) {
        char datestr[16];
        char timestr[16];
        long nbins = 0, nrays = 0;
        int pi = 0; /* parameter index */
        int nparams = 0;
        int year, month, day, hour, minute, second;

        scan = RAVE_OBJECT_NEW(&PolarScan_TYPE);
        if (scan == NULL) {
          goto done;
        }
        year = (int)vv[i++]; month = (int)vv[i++]; day = (int)vv[i++];
        hour = (int)vv[i++]; minute = (int)vv[i++]; second = (int)vv[i++];
        sprintf(datestr, "%04d%02d%02d", year, month, day);
        sprintf(timestr, "%02d%02d%02d", hour, minute, second);

        if (!PolarScan_setStartDate(scan, datestr) ||
            !PolarScan_setStartTime(scan, timestr)) {
          goto done;
        }

        year = (int)vv[i++]; month = (int)vv[i++]; day = (int)vv[i++];
        hour = (int)vv[i++]; minute = (int)vv[i++]; second = (int)vv[i++];
        sprintf(datestr, "%04d%02d%02d", year, month, day);
        sprintf(timestr, "%02d%02d%02d", hour, minute, second);

        if (!PolarScan_setEndDate(scan, datestr) ||
            !PolarScan_setEndTime(scan, timestr)) {
          goto done;
        }
        if ((int)vv[i++] != 90) {
          RAVE_WARNING1("vv[%d] is not 90", i);
          goto done;
        }
        PolarScan_setElangle(scan, (double)vv[i++] * M_PI / 180.0);
        nbins = (long)vv[i++];
        PolarScan_setRscale(scan, (double)vv[i++]);
        PolarScan_setRstart(scan, (double)vv[i++]);
        nrays = (long)vv[i++];
        PolarScan_setA1gate(scan, (long)vv[i++]);

        nparams = (int)vv[i++];
        for (pi = 0; pi < nparams; pi++) {
          int paramid = vv[i++];
          char quantity[16];
          int compression = 0;

          param = RAVE_OBJECT_NEW(&PolarScanParam_TYPE);
          if (param == NULL) {
            goto done;
          }
          if (paramid == 0) {
            strcpy(quantity, "DBZH");
          } else if (paramid == 40) {
            strcpy(quantity, "VRAD");
          } else if (paramid == 91) {
            strcpy(quantity, "TH");
          } else if (paramid == 92) {
            strcpy(quantity, "WRAD");
          } else {
            strcpy(quantity, "UNKNOWN");
          }
          if (!PolarScanParam_setQuantity(param, quantity)) {
            goto done;
          }
          PolarScanParam_setGain(param, 1.0);
          PolarScanParam_setOffset(param, 0.0);
          PolarScanParam_setNodata(param, DBL_MAX);
          PolarScanParam_setUndetect(param, -DBL_MAX);
          compression = (int)vv[i++];
          if (compression != 0) {
            RAVE_ERROR0("BUFR file has not been stored with zlib compression");
            goto done;
          } else {
            int k = 0, n = 0, j = 0, niter = 0;
            unsigned char* tempo = NULL;
            unsigned long ndata = 0;
            varfl* data = NULL;
            niter = vv[i++];
            tempo = RAVE_MALLOC(niter * 65534 * sizeof(unsigned char));
            if (tempo == NULL) {
              goto done;
            }
            memset(tempo, 0, niter*65534*sizeof(unsigned char));
            for (n = 0; n < niter; ++n) {
              int ncomp = vv[i++];
              for (k = 0; k < ncomp; ++k) {
                tempo[j++] = (vv[i+k] == MISSVAL ? 255 : vv[i+k]);
              }
              i += ncomp;
            }
            /* my_decompress(tempo, j, &ndecomp);*/
            /* data[i + nbins*j] is value for bin i and ray j*/
            data = RaveBufrIOInternal_decompress(tempo, j, nbins * nrays, &ndata);
            RAVE_FREE(tempo);

            if (data == NULL) {
              goto done;
            }

            if (!PolarScanParam_setData(param, nbins, nrays, data, RaveDataType_DOUBLE)) {
              RAVE_FREE(data);
              goto done;
            }

            if (!PolarScan_addParameter(scan, param)) {
              RAVE_FREE(data);
              goto done;
            }
            RAVE_FREE(data);
          }
          RAVE_OBJECT_RELEASE(param);
        }
        if (!PolarVolume_addScan((PolarVolume_t*)raveObject, scan)) {
          goto done;
        }
        RAVE_OBJECT_RELEASE(scan);
      }
    } else if (bufr_check_fxy (d, 3, 21, 207)) {
      int i = 0, nscans = 0;
      int si = 0; /*scan index */
      RaveBufrIOInternal_addHowToObject(vv, &i, raveObject, RaveBufrIOInternal_addAttributeToPolarVolume);

      nscans = (int)vv[i++];
      for (si = 0; si < nscans; si++) {
        char datestr[16];
        char timestr[16];
        char product[6];
        long nbins = 0, nrays = 0;
        int nparams = 0;
        int pi = 0; /* parameter index */

        scan = RAVE_OBJECT_NEW(&PolarScan_TYPE);
        if (scan == NULL) {
          goto done;
        }
        RaveBufrIOInternal_addHowToObject(vv, &i, scan, RaveBufrIOInternal_addAttributeToPolarScan);
        RaveBufrIOInternal_getDateTimeStrings(vv, &i, datestr, timestr);
        if (!PolarScan_setStartDate(scan, datestr) ||
            !PolarScan_setStartTime(scan, timestr)) {
          goto done;
        }
        RaveBufrIOInternal_getDateTimeStrings(vv, &i, datestr, timestr);
        if (!PolarScan_setEndDate(scan, datestr) ||
            !PolarScan_setEndTime(scan, timestr)) {
          goto done;
        }
        RaveBufrIOInternal_getString(vv, &i, product, 6, 1);
        RAVE_DEBUG1("Product is: %s\n",product);
        PolarScan_setElangle(scan, (double)vv[i++] * M_PI / 180.0);
        nbins = (long)vv[i++];
        PolarScan_setRscale(scan, (double)vv[i++]);
        PolarScan_setRstart(scan, (double)vv[i++]);
        nrays = (long)vv[i++];
        PolarScan_setA1gate(scan, (long)vv[i++]);
        nparams = (int)vv[i++];
        for (pi = 0; pi < nparams; pi++) {
          param = RaveBufrIOInternal_getOdim22Parameter(vv, &i, nbins, nrays);
          if (param != NULL) {
            if (!PolarScan_addParameter(scan, param)) {
              goto done;
            }
          }
          RAVE_OBJECT_RELEASE(param);
        }
        if (!PolarVolume_addScan((PolarVolume_t*)raveObject, scan)) {
          goto done;
        }
        RAVE_OBJECT_RELEASE(scan);
      }
    } else if (bufr_check_fxy (d, 3, 21, 204)) {
      int i = 0, j = 0;
      int nstations = (int)vv[i++] + 1;
      char* sources = NULL;

      if (nstations > 0) {
        sources = RAVE_MALLOC(sizeof(char) * nstations * 25);
        if (sources == NULL) {
          goto done;
        }
        memset(sources, 0, sizeof(char) * nstations * 25);
        for (j = 0; j < nstations-1; ++j) {
          char station[25];
          int k = 0;

          if (strlen(sources) > 0) {
            strcat(sources, ",");
          }

          for (k = 0; k < 3; k++) {
            station[k] = (char)vv[i++];
          }
          station[3] = ':';
          for (k = 4; k < 20; k++) {
            station[k] = (char)vv[i++];
          }
          station[k] = '\0';
          strcat(sources, station);
        }
        if (!PolarVolume_setSource((PolarVolume_t*)raveObject, sources)) {
          RAVE_FREE(sources);
          goto done;
        }
        RAVE_FREE(sources);
      }
    }
  }

  result = 1;
done:
  RAVE_OBJECT_RELEASE(scan);
  RAVE_OBJECT_RELEASE(param);
  bufr_close_val_array();
  return result;
}

/**
 * Callback function used by bufr:s parse out function. Depending on
 * type of global raveObject different types of data will be handled.
 * @param[in] val - val
 * @param[in] ind - int
 * @return return code
 */
static int RaveBufrIOInternal_parseOutCallback(varfl val, int ind)
{
  int result = 0;
  if (raveObject == NULL) {
    RAVE_ERROR0("raveObject not initialized when comming to callback");
  } else if (RAVE_OBJECT_CHECK_TYPE(raveObject, &PolarVolume_TYPE)) {
    result = RaveBufrIOInternal_polarVolumeCallback(val, ind);
  }
  return result;
}

static RaveCoreObject* RaveBufrIOInternal_read(RaveBufrIO_t* self, bufr_t* msg)
{
  sect_1_t s1;
  /*time_t nominalTime;*/
  int desch = -1, ndescs = 0;
  dd* dds = NULL;
  RaveCoreObject* result = NULL;

  RAVE_ASSERT((self != NULL), "self == NULL");
  RAVE_ASSERT((msg != NULL), "msg == NULL");

  if (!bufr_decode_sections01(&s1, msg)) {
    RAVE_ERROR0("Failed to decode section 1!\n");
    goto done;
  }

  /* read descriptor tables, yes, I know, we shouldn't cast a const pointer */
  if (read_tables((char*)RaveBufrIO_getTableDir(self), s1.vmtab, s1.vltab, s1.subcent, s1.gencent)) {
    RAVE_ERROR0("Failed to read desriptor tables");
    goto done;
  }

  /* open bitstreams for section 3 and 4 */
  desch = bufr_open_descsec_r(msg, NULL);
  if (desch < 0) {
    RAVE_ERROR0("Failed to open bitstream");
    goto done;
  }

  if (bufr_open_datasect_r(msg) < 0) {
    RAVE_ERROR0("Failed to open data section");
    goto done;
  }

  /* calculate number of data descriptors  */
  ndescs = bufr_get_ndescs(msg);

  /* allocate memory and read data descriptors from bitstream */
  if (!bufr_in_descsec(&dds, ndescs, desch)) {
    RAVE_ERROR0("Failed to allocate and read descriptors");
    goto done;
  }

  RAVE_OBJECT_RELEASE(raveObject);
  raveObject = RaveBufrIOInternal_createRaveObject(dds, ndescs);
  if (raveObject != NULL) {
    /* output data to our global data structure */
    if (!bufr_parse_out(dds, 0, ndescs - 1, RaveBufrIOInternal_parseOutCallback, 1)) {
      RAVE_ERROR0("Failed to parse data object");
      goto done;
    }
    if (RAVE_OBJECT_CHECK_TYPE(raveObject, &PolarVolume_TYPE)) {
      char ndate[10];
      char ntime[10];
      sprintf(ndate, "%04d%02d%02d", (s1.year < 50 ? s1.year+2000 : s1.year), s1.mon, s1.day);
      sprintf(ntime, "%02d%02d%02d", s1.hour, s1.min, s1.sec);
      if (!PolarVolume_setDate((PolarVolume_t*)raveObject, ndate) ||
          !PolarVolume_setTime((PolarVolume_t*)raveObject, ntime)) {
        RAVE_ERROR0("Failed to set date/time");
        goto done;
      }
    }
  } else {
    RAVE_ERROR0("Could not match BUFR descriptors to a RAVE object");
  }

  result = RAVE_OBJECT_COPY(raveObject);
done:
  RAVE_OBJECT_RELEASE(raveObject);

  /* close bitstreams and free descriptor array */
  if (dds != NULL) {
    free(dds);
  }
  bufr_close_descsec_r(desch);
  bufr_close_datasect_r();
  free_descs();

  return result;
}

/*@} End of Private functions */

/*@{ Interface functions */
int RaveBufrIO_setTableDir(RaveBufrIO_t* self, const char* dirname)
{
  char* tmp = NULL;
  int result = 0;

  RAVE_ASSERT((self != NULL), "self == NULL");
  if (dirname != NULL) {
    tmp = RAVE_STRDUP(dirname);
    if (tmp == NULL) {
      RAVE_ERROR0("Failed to allocate memory");
      goto done;
    }
  }
  RAVE_FREE(self->tabledir);
  self->tabledir = tmp;
  tmp = NULL;
  result = 1;
done:
  RAVE_FREE(tmp);
  return result;
}

const char* RaveBufrIO_getTableDir(RaveBufrIO_t* self)
{
  RAVE_ASSERT((self != NULL), "self == NULL");
  return (const char*)self->tabledir;
}

RaveCoreObject* RaveBufrIO_read(RaveBufrIO_t* self, const char* filename)
{
  bufr_t msg;
  RaveCoreObject* result = NULL;

  /* BUFR library lacks initializer so we must be very careful */
  memset (&msg, 0, sizeof(bufr_t));

#ifdef  PTHREAD_SUPPORTED
  if (bufrio_mutex_initialized == 0) {
    pthread_mutex_init(&bufrio_mutex, NULL);
    bufrio_mutex_initialized = 1;
  }
  pthread_mutex_lock(&bufrio_mutex);
#endif

  RAVE_ASSERT((self != NULL), "self == NULL");

  if (!bufr_read_file(&msg, filename)) {
    RAVE_ERROR1("Failed to read BUFR file %s!\n", filename);
    goto done;
  }

  result = RaveBufrIOInternal_read(self, &msg);

done:
#ifdef  PTHREAD_SUPPORTED
  pthread_mutex_unlock(&bufrio_mutex);
#endif
  bufr_free_data(&msg);
  return result;
}

int RaveBufrIO_isBufr(const char* filename)
{
  FILE* fp = NULL;
  char head[5];
  int result = 0;

  memset(head, 0, 5);

  fp = fopen(filename, "r");
  if (fp != NULL) {
    if (fread(head, sizeof(char), 4, fp) == 4) {
      if (strcmp("BUFR", head) == 0) {
        result = 1;
      }
    }
    fclose(fp);
  }
  return result;
}

/*@} End of Interface functions */
RaveCoreObjectType RaveBufrIO_TYPE = {
    "RaveBufrIO",
    sizeof(RaveBufrIO_t),
    RaveBufrIO_constructor,
    RaveBufrIO_destructor,
    NULL
};
