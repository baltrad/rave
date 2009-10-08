/**
 * polar-to-polar transformations, administered through rave_so.py.
 * Input polar volumes of radial wind velocity and reflectivity
 * factor are interpolated to output polar volumes. Along the way
 * superob variables are calculated and written to a temporary
 * ASCII file, which is then managed by rave_so.py.
 *
 * This module's first incarnation worked with one input and one
 * output volume. Successive versions have been faced with adding
 * variables whose names are not entirely consistent, which should
 * be kept in mind.
 * @file
 * @author Daniel Michelson (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2006-
 */
#include <Python.h>
#include <arrayobject.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "raveutil.h"
#include "rave.h"

static PyObject *ErrorObject;
static FILE* staticFD;

/**
 * Sets a python exception, then performs return NULL.
 */
#define raiseException(type,msg) {PyErr_SetString(type,msg);return NULL;}

/* DEFINES */

/**
 * PPI
 */
#define PPI 1

/**
 * CAPPI
 */
#define CAPPI 2

/**
 * Pseudo-CAPPI
 */
#define PCAPPI 3

/**
 * Use all weights
 */
#define ALL_WEIGHTS 1

/**
 * Only use weights that is not 0
 */
#define NO_ZERO_WEIGHTS 2

/**
 * Round azimuth
 */
#define ROUND_AZIMUTH 0

/**
 * Set the azimuth to the nearest lower integer value
 */
#define FLOOR_AZIMUTH 1

/**
 * Set the azimuth to the nearest higher integer value
 */
#define CEIL_AZIMUTH 2

/**
 * Only use one elevation
 */
#define USE_SINGLE_ELEV 1

/**
 * Use many elevations
 */
#define USE_MANY_ELEV 2

/**
 * Max number of different elevations
 */
#define MAXELEV 32

/**
 * Max number of bins
 */
#define MAXRANGEBINS 1000	     /* This is 250m res and max 250km range */

/**
 * Radius at the equator
 */
#define R_EQU 6378160.0

/**
 * Radius to the poles
 */
#define R_POL 6356780.0

/**
 * No quality control
 */
#define NO_QC 0

/**
 * QC with mean as quality indicator
 */
#define STD_DEVIATION_MEANVALUE 1

/**
 * QC with median as quality indicator
 */
#define STD_DEVIATION_MEDIAN 2

/**
 * Max number of coords
 */
#define MAXCOORD 128

/**
 * Debug printf formatter
 */
#define HFORMAT "# LON: %6.3f  LAT: %5.3f  ALT: %d  WMO: %d  dR: %5.1f  BEAMWIDTH: %3.1f  dAZ: %5.3f\n"

/**
 * Debug printf formatter
 */
#define OFORMAT \
"%6.3f %6.3f %4.1f %6d %5.1f %7.1f %7.1f %7.1f %6.2f %4d %7.1f %2d %5.1f %6.2f %4d %7.1f %2d %5.1f\n"

/* STRUCTURES */

/* all references to _so are for superob pvols */
typedef struct {
  int slice, method, useWeights, elevUsage;
  double nodata, nodataZ;
  int gotZ; /* do we have a reflectivity volume? */
  double slopew, ordw, slopez, ordz; /* scaling factors */
  double inscale, outscale, height;
  double alt0, lon0, lat0; /* Radar's position */
  int inxmax, inymax;
  int inxsize, outxsize, inrsize, outrsize, inzrsize, outzrsize;
  int maxAzim, maxElev, maxRange; /* taken from C2PWrapper */
  double azOffset, oazOffset;
  double dndh;
  int nelev, nelev_so; /* number of elevation angles */
  double elev[MAXELEV], elev_so[MAXELEV]; /* the actual elevation angles */
  double elevT[MAXELEV], elevT_so[MAXELEV];
  int scan[MAXELEV], scan_so[MAXELEV];
  int scanZ[MAXELEV], scanZ_so[MAXELEV];
  int scanT[MAXELEV], scanT_so[MAXELEV];
  int iaz, oaz, ira, ora; /* number of azimuth gates and range bins */
  char intype, outtype, inztype, outztype;
  PyArrayObject *src[MAXELEV]; /* polar volumes: input radial winds */
  PyArrayObject *srcZ[MAXELEV]; /* input reflectivity factor */
  PyArrayObject *dest[MAXELEV]; /* output radial winds */
  PyArrayObject *destZ[MAXELEV]; /* output reflectivity factor */
  double R; /* horizontal search radius in meters */
  double Rs[MAXRANGEBINS]; /* search radii for all range bins */
  double beamBroad; /* also known as the beamwidth */
  char* tmpfile; /* file to which to write superobs */
  char* obj;
  int wmo; /* this radar's WMO station number */
  double cressmanR_z; /* vertical search distance in beamwidths */
} TrafoWrapper3D;

typedef struct { /* A 3d polar coordinate with weight */
  int elev, range, azimuth; /* Indices in polar space */
  double weight; /* relevance to the current pixel */
} CoordWeight;

/* Stupid helper to shut up the compiler... */
/* void initialize_CoordWeight(CoordWeight *cw) */
/* { */
/*    cw->elev=0; */
/*    cw->range=0; */
/*    cw->azimuth=0; */
/*    cw->weight=0.0; */
/* } */

/* Slightly modified from ctop to work for ptop. */
static double getarritemP(int e, int r, int a, TrafoWrapper3D* wrap)
{
  return get_array_item_2d((unsigned char*) wrap->src[e], r, a, wrap->intype,
                           wrap->inrsize);
}

static void setarritemP(int e, int r, int a, double v, TrafoWrapper3D *wrap)
{
  set_array_item_2d((unsigned char*) wrap->dest[e], r, a, v, wrap->outtype,
                    wrap->outrsize);
}

/* Idiotic redundancy. That's what you get when you start adding on... */
static double getarritemPZ(int e, int r, int a, TrafoWrapper3D* wrap)
{
  return get_array_item_2d((unsigned char*) wrap->srcZ[e], r, a, wrap->inztype,
                           wrap->inzrsize);
}

static void setarritemPZ(int e, int r, int a, double v, TrafoWrapper3D *wrap)
{
  set_array_item_2d((unsigned char*) wrap->destZ[e], r, a, v, wrap->outztype,
                    wrap->outzrsize);
}

static int getAzindex(double az, TrafoWrapper3D* wrap, int rounding)
{
  double azOffset, azimuth;
  int azIndex;

  azOffset = 360.0 / wrap->inymax;

#ifdef TRUNC_AZIMUTH
  azimuth = az*RAD_TO_DEG/azOffset;
#else
  switch (rounding) {
  case ROUND_AZIMUTH:
    azimuth = rint(az * RAD_TO_DEG / azOffset);
    break;
  case FLOOR_AZIMUTH:
    azimuth = floor(az * RAD_TO_DEG / azOffset);
    break;
  case CEIL_AZIMUTH:
    azimuth = ceil(az * RAD_TO_DEG / azOffset);
    break;
  }
#endif

  azIndex = mytrunc(azimuth);

  azOffset *= DEG_TO_RAD;

  if (az < azOffset) {
    return 0;
  } else if (az > (2*M_PI - azOffset)) {
    return (wrap->inymax - 1);
  } else {
    return azIndex;
  }
}

static int getElevIndex(double elev, TrafoWrapper3D* wrap)
{
  int i, elevIndex;
  double elev1, elev2;

  elevIndex = 0;

  if (wrap->slice == PPI) {
    elevIndex = mytrunc(wrap->height);
    return elevIndex;
  }

  for (i = 0; i < wrap->nelev; i++) {
    if ((i < (wrap->nelev - 1)) && elev > wrap->elev[i] && elev < wrap->elev[i
        + 1]) {

#ifdef TRUNC_ELEV
      elevIndex = i;
#else
      elev1 = elev - wrap->elev[i];
      elev2 = elev - wrap->elev[i + 1];
      if (elev1 < 0.0)
        elev1 = -elev1;
      if (elev2 < 0.0)
        elev2 = -elev2;

      if (elev1 > elev2) {
        elevIndex = i + 1;
      } else {
        elevIndex = i;
      }
#endif
      break;
    } else if (elev < wrap->elev[0]) {
      elevIndex = 0;
      break;
    } else if (elev > wrap->elev[wrap->nelev - 1]) {
      elevIndex = wrap->nelev - 1;
      break;
    } else if (elev == wrap->elev[i]) {
      elevIndex = i;
      break;
    }
  }

  return elevIndex;
}

static double getCheckHeight(Position* src, Position* tgt, TrafoWrapper3D* wrap)
{
  double checkHeight, tmpRange, tmpDistance;

  int elevIdx;

  Position* aCopy;

  if (tgt->elevation < wrap->elev[0] || tgt->elevation > wrap->elev[wrap->nelev
      - 1]) {
    aCopy = copyPosStruct(src);

    tmpRange = tgt->range;
    tmpDistance = tgt->distance;

    if (tgt->elevation < wrap->elev[0])
      elevIdx = 0;
    else
      elevIdx = wrap->nelev - 1;

    aCopy->elevation = wrap->elev[elevIdx];

    aCopy->alt = src->alt;

    ehToRd(aCopy, tgt);

    checkHeight = tgt->range * wrap->beamBroad / 2.0;

    tgt->range = tmpRange;
    tgt->distance = tmpDistance;

    free(aCopy);
  } else {
    checkHeight = src->range * wrap->beamBroad / 2.0;
  }

  checkHeight *= wrap->cressmanR_z;

  return checkHeight;
}

/* start new comment */
/* static double getCressHeightWeight(int eIndex,Position* src, */
/* 				   Position* target, double checkHeight,  */
/* 				   double* dHeight, TrafoWrapper3D* wrap) */
/* { */

/*   double cmanHeightWeight; */
/*   double calcHeight; */

/*   Position* aCopy = copyPosStruct(src); */

/*Decide elevations*/

/* start new comment */
/*   aCopy->elevation = wrap->elev[eIndex]; */

/*   deToRh(aCopy,target); */

/*   if(aCopy->alt > target->alt) */
/*     calcHeight = aCopy->alt - target->alt; */
/*   else */
/*     calcHeight = target->alt - aCopy->alt; */

/*   (*dHeight)=calcHeight; */

/*The code below that is commented can be uncommented, then
 *the elevations which are inside a elevation will get
 *the weight 1.
 */

/* start new comment. rows 1 and 3 are complicated... */
/*   if(calcHeight >= checkHeight*//* && */
/*     (src->elevation > wrap->elev[eIndex]+wrap->beamBroad/2 || */
/*     src->elevation < wrap->elev[eIndex]-wrap->beamBroad/2) *//*) { */
/*     cmanHeightWeight = 0L; */
/*  }  *//*inserted 99-03-19*/
/*   else if((src->elevation > wrap->elev[wrap->nelev-1] || */
/* 	   src->elevation < wrap->elev[0]) && wrap->slice==PCAPPI) { */
/*      cmanHeightWeight = 0L; */
/*   } */
/*   else { */
/*
 if(src->elevation <= wrap->elev[eIndex]+wrap->beamBroad/2 &&
 src->elevation >= wrap->elev[eIndex]-wrap->beamBroad/2) {
 cmanHeightWeight = 1L;
 }
 else
 */
/* start new comment */
/*     if(calcHeight < checkHeight) { */
/*       if(wrap->method==CRESSMAN) { */
/* 	cmanHeightWeight = checkHeight*checkHeight - calcHeight*calcHeight; */
/* 	cmanHeightWeight/= checkHeight*checkHeight + calcHeight*calcHeight; */
/*       } */
/*       else if(wrap->method==UNIFORM) { */
/* 	cmanHeightWeight = 1L; */
/*       } */
/*       else { */
/* 	if(checkHeight!=0.0) */
/* 	  cmanHeightWeight = 1.0 - calcHeight/checkHeight; */
/* 	else */
/* 	  cmanHeightWeight = 0L; */
/*       } */
/*     } */
/*     else { */
/*       printf("ASSERTION FAILIURE IN getCressHeightWeight\n"); */
/*       printf("src->elevation = %lg, src->range=%lg\n",\ */
/* 	     src->elevation*RAD_TO_DEG,src->range); */
/*       exit(1); */
/*     } */
/*   } */

/*   free(aCopy); */

/*   return cmanHeightWeight; */
/* } */

static double getCressARWeight(int rangeIdx, int azIndex, double gamma,
  Position* src, double* xr, double* yr, TrafoWrapper3D* wrap)
{
  double arWeight;

  int rIndex;
  rIndex = rangeIdx/* + 1*/;

  src->range = rint(src->range); /* failsafe for some platforms */

  if (gamma <= wrap->beamBroad / 2 && rIndex * wrap->inscale <= src->range
      && (rIndex + 1) * wrap->inscale >= src->range) {
    /*The current pixel is surrounding the wanted point*/
    (*xr) = 0L;
    (*yr) = 0L;
  } else if (gamma > wrap->beamBroad / 2 && rIndex * wrap->inscale
      <= src->range && (rIndex + 1) * wrap->inscale >= src->range) {
    /*The current pixel is to the left or right of the wanted pixel*/
    (*yr) = 0L;
    (*xr) = src->range * gamma;
  } else if (gamma <= wrap->beamBroad / 2 && (rIndex + 1) * wrap->inscale
      < src->range) {
    /*The current pixel is below the wanted pixel*/
    (*xr) = 0L;
    (*yr) = src->range - (rIndex + 1 / 2) * wrap->inscale;
  } else if (gamma <= wrap->beamBroad / 2 && rIndex * wrap->inscale
      > src->range) {
    /*The current pixel is above the wanted pixel*/
    (*xr) = 0L;
    (*yr) = (rIndex + 1 / 2) * wrap->inscale - src->range;
  } else {
    /*Either above or below and right or left*/

    if (rIndex * wrap->inscale > src->range) {
      /*Above*/
      (*xr) = src->range * gamma;
      (*yr) = (rIndex + 1 / 2) * wrap->inscale - (src->range * cos(gamma));
    } else {
      /*Below*/
      (*xr) = src->range * gamma;
      (*yr) = src->range * cos(gamma) - (rIndex + 1 / 2) * wrap->inscale;
    }
  }

  arWeight = 1L;

  return arWeight;
}

static CoordWeight* getAllocatedCW(int maxNoOfItems)
{
  static CoordWeight* staticWeight = NULL;
  static int noOfWeights = 0;

  if (maxNoOfItems == -99) {
    if (staticWeight) {
      free(staticWeight);
      staticWeight = NULL;
    }
    noOfWeights = 0;
    return NULL;
  }

  if (maxNoOfItems > noOfWeights) {
    if (staticWeight)
      free(staticWeight);

    staticWeight = malloc(sizeof(CoordWeight) * maxNoOfItems);
    noOfWeights = maxNoOfItems;
  }

  return staticWeight;
}

static void getAdjAzimIdx(int minmaxAzim[2], double* minAzim, double* maxAzim,
  TrafoWrapper3D* wrap)
{
  double azOffset = (360.0 / wrap->inymax) * DEG_TO_RAD;
  int maxNoOfAzimuths;

  /*Adjust azimuths*/
  if ((*minAzim) < 0)
    (*minAzim) += 2.0 * M_PI;

  if ((*maxAzim) >= 2.0 * M_PI)
    (*maxAzim) -= 2.0 * M_PI;

  minmaxAzim[0] = mytrunc(floor((*minAzim) / azOffset));
  minmaxAzim[1] = mytrunc(ceil((*maxAzim) / azOffset));

  /*Decide indexes to use*/
  if (minmaxAzim[0] > minmaxAzim[1]) {
    maxNoOfAzimuths = wrap->inymax - minmaxAzim[0];
    maxNoOfAzimuths += minmaxAzim[1];
    minmaxAzim[1] = minmaxAzim[0] + maxNoOfAzimuths;
  }
}

static int handleCressmanOrigoSE(CoordWeight* staticWeight, Position* src,
  double azOffset, int maxRange, TrafoWrapper3D* wrap)
{
  int maxAzim; /* maxElev */
  int sa, se, sr;
  int widx = 0;

  double r, R;
  double zr = 0.0; /* This may not be initialized properly, but is it used? */
  double gamma; /* heightWeight,checkHeight */
  double storedLen = 100000L;

  Position target;
  Position* srcCopy;

  srcCopy = copyPosStruct(src);

  if (wrap->slice == PCAPPI) {
    srcCopy->elevation = wrap->elev[wrap->nelev - 1];
    deToRh(srcCopy, srcCopy);
  }

  dhToRe(srcCopy, &target);

  se = wrap->height;

  maxAzim = wrap->inymax;

  R = wrap->R;

  for (sa = 0; sa < maxAzim; sa++) {
    /*
     *Decide the angle between the src->azimuth
     *and the current indexed azimuth
     */
    gamma = sa * azOffset - srcCopy->azimuth;
    if (gamma < 0)
      gamma += 2*M_PI;

    for (sr = 0; sr <= maxRange; sr++) {
      r = srcCopy->range * srcCopy->range + (sr + 1 / 2) * wrap->inscale * (sr
          + 1 / 2) * wrap->inscale;
      r = r - 2* srcCopy ->range * (sr + 1 / 2) * wrap->inscale * cos(gamma);
      r = sqrt(r);

      if (r < R) {
        staticWeight[widx].elev = se;
        staticWeight[widx].azimuth = sa;
        staticWeight[widx].range = sr;
        if (wrap->method == CRESSMAN)
          staticWeight[widx].weight = sqrt(((R * R - r * r) / (R * R + r * r)));
        else if (wrap->method == UNIFORM)
          staticWeight[widx].weight = 1L;
        else
          staticWeight[widx].weight = (1.0 - r / R);

        widx++;
      } else {
        r = sqrt(r * r + zr * zr);
        if (r < storedLen) {
          staticWeight[0].elev = se;
          staticWeight[0].azimuth = sa;
          staticWeight[0].range = sr;
          staticWeight[0].weight = 1.0;
          storedLen = r;
        }
      }
    }
  }

  if (storedLen != 100000L && widx == 0)
    widx = 1;

  free(srcCopy);

  return widx;
}

/*
 * Get cressman with single elevation, only reasonable to use for PPI.
 */
static CoordWeight* getCressmanSE(Position* src, int* weights,
  TrafoWrapper3D* wrap)
{
  int minmaxAzim[2], minmaxElev[2], minmaxRange[2];
  double azOffset;
  double yr, xr, r; /* yprim */
  double zr = 0.0; /* This may not be initialized properly, but is it used? */
  double gamma, OC, R, minAzim, maxAzim;
  double arWeight;
  int sa, se, sr, widx, azIdx;
  int maxNoOfItems = 0;

  Position* srcCopy;
  Position target;

  double checkHeight;
  //  double heightWeight;

  int storedR = -1, storedE = -1, storedA = -1;

  double storedLen = 100000L;

  CoordWeight* staticWeight = NULL;

  if (src->range > wrap->inxmax * wrap->inscale) {
    /*Out of bounds, just return*/
    (*weights) = 0;
    return staticWeight;
  }

  azOffset = (360.0 / wrap->inymax) * DEG_TO_RAD;

  if (src->distance <= wrap->R) { /*Wrap arround origo*/
    minmaxRange[1] = mytrunc(ceil(wrap->R / wrap->inscale));
    minmaxRange[1] *= 2;
    minmaxRange[1] += 1;
    maxNoOfItems = wrap->inymax * (minmaxRange[1]);
    staticWeight = getAllocatedCW(maxNoOfItems);
    (*weights) = handleCressmanOrigoSE(staticWeight, src, azOffset,
                                       minmaxRange[1], wrap);
    return staticWeight;
  }

  srcCopy = copyPosStruct(src);

  OC = src->range;
  R = wrap->R;
  gamma = asin(R / OC);
  minAzim = src->azimuth - gamma;
  maxAzim = src->azimuth + gamma;

  getAdjAzimIdx(minmaxAzim, &minAzim, &maxAzim, wrap);

  dhToRe(srcCopy, &target);

  free(srcCopy);

  checkHeight = getCheckHeight(src, &target, wrap);

  se = minmaxElev[0] = minmaxElev[1] = wrap->height; /* !!! */

  /*Determine min and max range index*/
  minmaxRange[0] = mytrunc(floor((src->range - wrap->R) / wrap->inscale)) - 1;
  minmaxRange[1] = mytrunc(ceil((src->range + wrap->R) / wrap->inscale)) - 1;

  if (minmaxRange[0] == -1)
    minmaxRange[0] = 0;

  if (minmaxRange[1] == -1)
    minmaxRange[1] = 0;

  maxNoOfItems = minmaxAzim[1] - minmaxAzim[0] + 1;
  maxNoOfItems *= minmaxRange[1] - minmaxRange[0] + 1;

  staticWeight = getAllocatedCW(maxNoOfItems);

  widx = 0;

  for (sa = minmaxAzim[0]; sa <= minmaxAzim[1]; sa++) {

    if (maxAzim > minAzim) {
      gamma = src->azimuth - sa * azOffset;
      azIdx = sa;
      if (azIdx >= wrap->inymax)
        azIdx = azIdx % wrap->inymax;
      if (gamma < 0)
        gamma = -gamma;
    } else if (sa >= wrap->inymax) {
      azIdx = sa % wrap->inymax;
      if (src->azimuth > M_PI) {
        gamma = src->azimuth - sa * azOffset;
      } else {
        gamma = src->azimuth - azIdx * azOffset;
      }
      if (gamma < 0)
        gamma = -gamma;
    } else {
      gamma = src->azimuth - sa * azOffset;
      azIdx = sa;
      if (gamma < 0)
        gamma += 2.0 * M_PI;
    }

    for (sr = minmaxRange[0]; sr <= minmaxRange[1]; sr++) {
      /* This can be optimized, when all azimuth/range
       * parts has been calculated for one elevation, the same values
       * will be calculated for the rest, this means that cashing
       * the first calculated values might be quite a good idea ;-)
       */
      arWeight = getCressARWeight(sr, azIdx, gamma, src, &xr, &yr, wrap);

      r = sqrt(xr * xr + yr * yr);

      if (r < R && arWeight != 0L) {
        staticWeight[widx].elev = se;
        staticWeight[widx].azimuth = azIdx;
        staticWeight[widx].range = sr;
        if (wrap->method == CRESSMAN) {
          staticWeight[widx].weight = (R * R - r * r) / (R * R + r * r)
              * arWeight;
        } else if (wrap->method == UNIFORM) {
          staticWeight[widx].weight = 1L;
        } else {
          staticWeight[widx].weight = 1.0 - r / R;
        }

        if (staticWeight[widx].range >= wrap->inxmax)
          continue; /*Dont take this range in account*/

        widx++;
      } else {
        r = sqrt(r * r + zr * zr);

        if (r < storedLen) {
          storedR = sr;
          storedA = azIdx;
          storedE = se;
          storedLen = r;
        }
      }
    }
  }

  if (widx == 0 && storedR != -1) {
    staticWeight[0].elev = storedE;
    staticWeight[0].azimuth = storedA;
    staticWeight[0].range = storedR;
    staticWeight[0].weight = 1.0; /*sqrt(storedHW*storedARWeight);*/
    widx = 1;
  }

  (*weights) = widx;
  return staticWeight;

}

/*------------------Nearest------------------------*/
static CoordWeight* nearestP2P(Position src, int* noItems, TrafoWrapper3D* wrap)
{
  CoordWeight* retw;

  int ie, ia, ir; /* indices for elevation, azimuth, and range */

  /* Determine range index. */
  ir = (int) (src.range / wrap->inscale);
  if (ir >= wrap->ira) {
    (*noItems) = 0;

    return NULL;
  }

  /* Determine elevation index. */
  ie = getElevIndex(src.elevation, wrap);

  /* Determine azimuth index. */
  ia = getAzindex(src.azimuth, wrap, ROUND_AZIMUTH);

  (*noItems) = 1;
  retw = getAllocatedCW(1);

  retw[0].elev = ie;
  retw[0].azimuth = ia;
  retw[0].range = ir;
  retw[0].weight = 1L;

  return retw;
}

/* ----------------------------------------------------------------
 Get an array containing search radii for each range bin. Required for
 all forms of interpolation.
 */
static void getRs(TrafoWrapper3D* wrap)
{
  Position first, second; /* two virtual azimuth gates beside each other */
  double range, zd, z;
  int r;

  if (MAXRANGEBINS < wrap->maxRange) {
    PyErr_SetString(PyExc_TypeError, "Too many range bins.\n");
    return;
  }

  switch (wrap->method) {
  case CRESSMAN:
  case INVERSE:
  case UNIFORM:
    first.elevation = wrap->elev_so[0];
    first.alt0 = wrap->alt0;
    first.lat0 = wrap->lat0;
    first.lon0 = wrap->lon0;
    first.dndh = wrap->dndh;
    /*       first.azimuth = wrap->oazOffset; */
    first.azimuth = 45.0 * DEG_TO_RAD - wrap->oazOffset / 2.0;
    first.dndh = wrap->dndh;
    second.elevation = wrap->elev_so[0];
    second.alt0 = wrap->alt0;
    second.lat0 = wrap->lat0;
    second.lon0 = wrap->lon0;
    second.dndh = wrap->dndh;
    /*       second.azimuth = 2*wrap->oazOffset; */
    second.azimuth = 45.0 * DEG_TO_RAD + wrap->oazOffset / 2.0;
    second.dndh = wrap->dndh;

    for (r = 0; r < wrap->maxRange; r++) {
      range = r * wrap->outscale + 0.5 * wrap->outscale;
      first.range = range;
      second.range = range;

      reToDh(&first, &first);
      daToLl(&first, &first);
      reToDh(&second, &second);
      daToLl(&second, &second);

      /* Arc distance. No checks on whether first and second are on
       opposite sides of the equator. */
      zd = (sin(first.lat) * sin(second.lat)) + (cos(first.lat)
          * cos(second.lat) * cos(fabs(second.lon - first.lon)));
      z = 6370997.0 * acos(zd);

      wrap->Rs[r] = (z < wrap->outscale) ? 0.5 * z : 0.5 * wrap->outscale;
    }
    break;
  default:
    break;
  }
}

/* ----------------------------------------------------------------
 Administrates the actual transformation in two steps:
 1. Determine input bins and their weights used in the interpolation.
 This is performed once only and the weights are recycled for
 each volume.
 2. Perform the interpolation using the weights.
 */
static void doP2P(int e, int a, int r, TrafoWrapper3D* wrap)
{

  Position src, tgt; /* src = input, tgt = output */
  Position above, below; /* used to calculate vertical search radius */

  CoordWeight *weights; /* *zweights */
  int noweight = 0; /* noweight = number of weights... */
  int i = 0;

  //   int debug=1;

  /* Get vars for calculating vertical search radius and other vars. */
  double vsr = 0L, lon = 0L, lat = 0L, beamwidth = 0L, thiselev = 0L, under =
      0L, over = 0L;
  double elevation = 0L, azimuth = 0L, range = 0L;

  //   initialize_CoordWeight(weights);

  wrap->height = (double) e; /* stupid fix... */
  beamwidth = wrap->beamBroad * RAD_TO_DEG;
  thiselev = wrap->elev_so[e] * RAD_TO_DEG;
  under = DEG_TO_RAD * (thiselev - 0.5 * beamwidth);
  over = DEG_TO_RAD * (thiselev + 0.5 * beamwidth);

  /* What are the current elev, az and range? Angles in radians. */
  elevation = wrap->elev_so[e];
  azimuth = a * wrap->oazOffset;
  range = r * wrap->outscale + 0.5 * wrap->outscale;

  /* Assign them to the tgt Position. */
  tgt.elevation = elevation;
  tgt.alt0 = wrap->alt0;
  tgt.lat0 = wrap->lat0;
  tgt.lon0 = wrap->lon0;
  tgt.dndh = wrap->dndh;
  tgt.range = range;
  tgt.azimuth = azimuth;

  /* This strategy assumes that src and tgt are the same radar.
   FIXME: make more general. */
  src.alt0 = wrap->alt0;
  src.lat0 = wrap->lat0;
  src.lon0 = wrap->lon0;
  src.dndh = wrap->dndh;

  /* More management of vars for calculating vertical search radius. */
  above.elevation = over;
  above.alt0 = wrap->alt0;
  above.lat0 = wrap->lat0;
  above.lon0 = wrap->lon0;
  above.dndh = wrap->dndh;
  above.range = range;
  above.azimuth = azimuth;
  below.elevation = under;
  below.alt0 = wrap->alt0;
  below.lat0 = wrap->lat0;
  below.lon0 = wrap->lon0;
  below.dndh = wrap->dndh;
  below.range = range;
  below.azimuth = azimuth;

  reToDh(&tgt, &tgt);
  daToLl(&tgt, &src);
  dhToRe(&tgt, &src);
  llToDa(&src, &src);
  reToDh(&src, &src);

  reToDh(&above, &above);
  reToDh(&below, &below);

  vsr = above.alt - below.alt;
  lon = src.lon * RAD_TO_DEG;
  lat = src.lat * RAD_TO_DEG;

  /* Determine weights depending on interpolation algorithm. */
  switch (wrap->method) {
  case NEAREST:
    weights = nearestP2P(src, &noweight, wrap);
    break;
  case CRESSMAN:
  case UNIFORM:
  case INVERSE: {
    wrap->R = wrap->Rs[r]; /* sets the right search radius */
    weights = getCressmanSE(&src, &noweight, wrap);
    break;
  }
  default:
    printf("No such method in PTOP, %d.\nDefaulting to NEAREST\n", wrap->method);
    weights = nearestP2P(src, &noweight, wrap);
    break;
  }

  /* If at least one input bin was found, then weight the final value
   appropriately before setting the output value.
   This is done first for wind and then for reflectivity since the same
   weights are used for each.
   */
  if (noweight) {
    double value = 0L; /* mean values */
    double valuez = 0L;
    double sum = 0L; /* these are weight sums */
    double sumz = 0L;
    double item = 0L; /* temp variable for holding bin values */
    int nw = 0; /* counters */
    int nz = 0;
    double totsumw = 0L; /* these are total sums for calculating variances */
    double totsumz = 0L;
    double sqsumw = 0L; /* sum of squares */
    double sqsumz = 0L;
    double varw = 0L; /* variances */
    double varz = 0L;

    /* wind */
    for (i = 0; i < noweight; i++) {
      if ((weights[i].weight != 0.0) && (item = getarritemP(weights[i].elev,
                                                            weights[i].range,
                                                            weights[i].azimuth,
                                                            wrap))
          != wrap->nodata) {
        if ((wrap->useWeights == NO_ZERO_WEIGHTS && item != 0.0)
            || wrap->useWeights == ALL_WEIGHTS) {
          sum += weights[i].weight;
        }
        totsumw += item * weights[i].weight;
      }
    }
    if ((sum) && (totsumw)) {
      value = totsumw / sum;
    }

    for (i = 0; i < noweight; i++) {
      if ((weights[i].weight != 0.0) && (item = getarritemP(weights[i].elev,
                                                            weights[i].range,
                                                            weights[i].azimuth,
                                                            wrap))
          != wrap->nodata) {
        if ((wrap->useWeights == NO_ZERO_WEIGHTS && item != 0.0)
            || wrap->useWeights == ALL_WEIGHTS) {
          //sqsumw+=pow(item-value, 2.0);
          sqsumw += pow((wrap->slopew * item + wrap->ordw) - (wrap->slopew
              * value + wrap->ordw), 2.0);
          nw += 1;
        }
      }
    }
    setarritemP(e, r, a, value, wrap);

    /* reflectivity */
    item = 0L;
    if (wrap->gotZ) {
      for (i = 0; i < noweight; i++) {
        if ((weights[i].weight != 0.0) && (item
            = getarritemPZ(weights[i].elev, weights[i].range,
                           weights[i].azimuth, wrap)) != wrap->nodataZ) {
          if ((wrap->useWeights == NO_ZERO_WEIGHTS && item != 0.0)
              || wrap->useWeights == ALL_WEIGHTS) {
            sumz += weights[i].weight;
          }
          totsumz += item * weights[i].weight;
        }
      }
      if ((sumz) && (totsumz)) {
        valuez = totsumz / sumz;
      }

      for (i = 0; i < noweight; i++) {
        if ((weights[i].weight != 0.0) && (item
            = getarritemPZ(weights[i].elev, weights[i].range,
                           weights[i].azimuth, wrap)) != wrap->nodataZ) {
          if ((wrap->useWeights == NO_ZERO_WEIGHTS && item != 0.0)
              || wrap->useWeights == ALL_WEIGHTS) {
            sqsumz += pow(item - valuez, 2.0);
            nz += 1;
          }
        }
      }
      setarritemPZ(e, r, a, valuez, wrap);
    }

    /* Calculate variances only if sample size allows it. */
    if (nw - 1 > 0) {
      varw = sqsumw / (nw - 1);
    }

    if (nz - 1 > 0) {
      varz = sqsumz / (nz - 1);
      if (varz < 1e-05) {
        varz = -50.0;
      } else {
        varz = 10* log10 (varz);
      }
    }

    /* Only write actual measurements to tmpfile. This is checked through
     the 'sum' parameter. If sum=0.0 then NaN will be written. */
    if ((wrap->gotZ) && (sum)) {
      if (valuez < 1e-05) {
        valuez = 1e-05; /* failsafe before logging */
      }
      fprintf(staticFD, OFORMAT, lon, lat, elevation * RAD_TO_DEG, (int) range,
              azimuth * RAD_TO_DEG, tgt.alt, wrap->R, vsr, wrap->slopew * value
                  + wrap->ordw, nw, varw, 0, 0.0, 10* log10 (valuez), nz, varz,
              0, 0.0);

    } else if ((!wrap->gotZ) && (sum)) {
      fprintf(staticFD, OFORMAT, lon, lat, elevation * RAD_TO_DEG, (int) range,
              azimuth * RAD_TO_DEG, tgt.alt, wrap->R, vsr, wrap->slopew * value
                  + wrap->ordw, nw, varw, 0, 0.0, 255.0, 0, 0.0, 0, 0.0);
    }
  }
}

/* ----------------------------------------------------------------
 The function for administrating the transformation.
 All input volumes are assumed to have the same geometry.
 All output volumes are also assumed to have the same geometry.
 In other words, some header attributes are assumed to be valid for
 both input volumes and some for both output volumes.
 */

static PyObject* _ptop_transform(PyObject* self, PyObject* args)
{
  //  char **argv;
  int e, a, r; /* elevation, azimuth, range indices */
  int i, j; /* n,m */
  int wasok; /* Was OK, ie. it worked... */
  TrafoWrapper3D tw; /* this struct contains just about everything needed */
  //  int no_of_src=0;
  //  int no_of_dest=0;

  PyObject *in, *inz; /* input pvols */
  PyObject *out, *outz; /* output pvols */
  //  PyObject *in_info;     /* input wind data info */
  //  PyObject *inz_info;    /* input reflectivity (Z) data info */
  //  PyObject *out_info;	 /* output data info */
  PyObject *ipo, *opo; /* generic Python Objects for mapping diverse vars */
  PyObject *ipoz = Py_None, *opoz = Py_None; /* more of the same */
  //  PyObject *ip_area, *op_area; /* polar areas */

  RaveObject inrave, inzrave, outrave, outzrave;
  char* tmpchar;
  char tmpstr[100];

  setbuf(stdout, NULL);

  /* Check args: */
  if (!PyArg_ParseTuple(args, "OOOO", &in, &out, &inz, &outz)) {
    return NULL;
  }

  if ((in == Py_None) || (out == Py_None)) {
    PyErr_SetString(PyExc_TypeError, "Arguments 1 or 2 are of type None.\n");
    return NULL;
  }

  if ((inz != Py_None) && (outz != Py_None)) {
    tw.gotZ = 1;
  } else {
    tw.gotZ = 0;
  }

  /* Get elevation angles for each object */
  ipo = GetSequenceFromINFO(in, "/how/scan");
  if (!ipo) {
    PyErr_SetString(PyExc_TypeError, "No elevations in source info.\n");
    return NULL;
  }

  opo = GetSequenceFromINFO(out, "/how/scan");
  if (!opo) {
    Py_DECREF(ipo);
    PyErr_SetString(PyExc_TypeError, "No elevations in destination info.\n");
    return NULL;
  }

  if (tw.gotZ) {
    ipoz = GetSequenceFromINFO(inz, "/how/scan");
    if (!ipoz) {
      Py_DECREF(ipo);
      Py_DECREF(opo);
      PyErr_SetString(PyExc_TypeError, "No elevations in source info.\n");
      return NULL;
    }

    opoz = GetSequenceFromINFO(outz, "/how/scan");
    if (!opoz) {
      Py_DECREF(ipo);
      Py_DECREF(opo);
      Py_DECREF(ipoz);
      PyErr_SetString(PyExc_TypeError, "No elevations in destination info.\n");
      return NULL;
    }
  }

  /* Ensure they are sequences */
  if (!PySequence_Check(ipo)) {
    Py_DECREF(ipo);
    Py_DECREF(opo);
    PyErr_SetString(PyExc_TypeError, "Input elevations must be a sequence.\n");
    return NULL;
  }

  if (!PySequence_Check(opo)) {
    Py_DECREF(ipo);
    Py_DECREF(opo);
    PyErr_SetString(PyExc_TypeError, "Output elevations must be a sequence.\n");
    return NULL;
  }

  if (tw.gotZ) {
    if (!PySequence_Check(ipoz)) {
      Py_DECREF(ipo);
      Py_DECREF(opo);
      Py_DECREF(ipoz);
      Py_DECREF(opoz);
      PyErr_SetString(PyExc_TypeError, "Input elevations must be a sequence.\n");
      return NULL;
    }

    if (!PySequence_Check(opoz)) {
      Py_DECREF(ipo);
      Py_DECREF(opo);
      Py_DECREF(ipoz);
      Py_DECREF(opoz);
      PyErr_SetString(PyExc_TypeError,
                      "Output elevations must be a sequence.\n");
      return NULL;
    }
  }

  /* Get number of elevation angles for each object */
  tw.nelev = PyObject_Length(ipo);
  tw.nelev_so = PyObject_Length(opo);

  if (tw.gotZ) {
    if ((PyObject_Length(ipoz) != tw.nelev) || (PyObject_Length(opoz)
        != tw.nelev_so)) {
      Py_DECREF(ipo);
      Py_DECREF(opo);
      Py_DECREF(ipoz);
      Py_DECREF(opoz);
      PyErr_SetString(PyExc_TypeError,
                      "# reflectivity scans differs from # wind scans.\n");
      return NULL;
    }
  }

  /* More than MAXELEV scans are truncated to MAXELEV */
  if (tw.nelev > MAXELEV) {
    printf("  Too many input elevations: %d - using first %d.\n", tw.nelev,
           MAXELEV);
    tw.nelev = MAXELEV;
  }

  if (tw.nelev_so > MAXELEV) {
    printf("  Too many output elevations: %d - using first %d.\n", tw.nelev_so,
           MAXELEV);
    tw.nelev_so = MAXELEV;
  }

  if (GetStringFromINFO(out, "/what/object", &tmpchar)) {
    tw.obj = strdup(tmpchar);
  } else {
    PyErr_SetString(PyExc_IOError, "Object not specified.\n");
    return NULL;
  }

  /* Add each elevation angle to tw and convert to radians */
  for (i = 0, wasok = 0; i < tw.nelev && wasok == 0; i++) {
    wasok |= !GetIntFromSequence(ipo, i, &tw.scan[i]);
    if (strcmp(tw.obj, "PVOL") == 0) {
      sprintf(tmpstr, "/scan%d/where/angle", tw.scan[i]);
    } else if (strcmp(tw.obj, "SCAN") == 0) {
      sprintf(tmpstr, "/where/angle");
    }
    wasok |= !GetDoubleFromINFO(in, tmpstr, &tw.elev[i]);
    //    printf("in(w): scan %d, angle %f\n",tw.scan[i],tw.elev[i]);
    tw.elev[i] *= DEG_TO_RAD;
  }

  for (i = 0, wasok = 0; i < tw.nelev_so && wasok == 0; i++) {
    wasok |= !GetIntFromSequence(opo, i, &tw.scan_so[i]);
    if (strcmp(tw.obj, "PVOL") == 0) {
      sprintf(tmpstr, "/scan%d/where/angle", tw.scan_so[i]);
    } else if (strcmp(tw.obj, "SCAN") == 0) {
      sprintf(tmpstr, "/where/angle");
    }
    wasok |= !GetDoubleFromINFO(out, tmpstr, &tw.elev_so[i]);
    //    printf("so(w): scan %d, angle %f\n",tw.scan_so[i],tw.elev_so[i]);
    tw.elev_so[i] *= DEG_TO_RAD;
  }

  if (wasok) {
    Py_DECREF(ipo);
    Py_DECREF(opo);
    Py_DECREF(ipoz);
    Py_DECREF(opoz);
    PyErr_SetString(PyExc_TypeError,
                    "Strange input or output elevation definition.\n");
    return NULL;
  }

  Py_DECREF(ipo);
  Py_DECREF(opo);

  if (tw.gotZ) {
    for (i = 0, wasok = 0; i < tw.nelev && wasok == 0; i++) {
      wasok |= !GetIntFromSequence(ipoz, i, &tw.scanT[i]);
      if (strcmp(tw.obj, "PVOL") == 0) {
        sprintf(tmpstr, "/scan%d/where/angle", tw.scanT[i]);
      } else if (strcmp(tw.obj, "SCAN") == 0) {
        sprintf(tmpstr, "/where/angle");
      }
      wasok |= !GetDoubleFromINFO(inz, tmpstr, &tw.elevT[i]);
      tw.elevT[i] *= DEG_TO_RAD;
    }

    for (i = 0, wasok = 0; i < tw.nelev_so && wasok == 0; i++) {
      wasok |= !GetIntFromSequence(opoz, i, &tw.scanT_so[i]);
      if (strcmp(tw.obj, "PVOL") == 0) {
        sprintf(tmpstr, "/scan%d/where/angle", tw.scanT_so[i]);
      } else if (strcmp(tw.obj, "SCAN") == 0) {
        sprintf(tmpstr, "/where/angle");
      }
      wasok |= !GetDoubleFromINFO(outz, tmpstr, &tw.elevT_so[i]);
      tw.elevT_so[i] *= DEG_TO_RAD;
    }

    if (wasok) {
      Py_DECREF(ipo);
      Py_DECREF(opo);
      Py_DECREF(ipoz);
      Py_DECREF(opoz);
      PyErr_SetString(PyExc_TypeError,
                      "Strange input or output elevation definition.\n");
      return NULL;
    }

    /* Ensure that index i points to the same elevation angle in the wind
     and reflectivity scans. */
    for (i = 0; i < tw.nelev; i++) {
      for (j = 0; j < tw.nelev; j++) {
        if (tw.elevT[i] == tw.elev[j]) {
          tw.scanZ[j] = tw.scanT[i];
          //	  printf("in(z): scanZ %d, angleZ %f\n",tw.scanZ[j],tw.elevT[i]*RAD_TO_DEG);
        }
      }
    }
    for (i = 0; i < tw.nelev_so; i++) {
      for (j = 0; j < tw.nelev_so; j++) {
        if (tw.elevT_so[i] == tw.elev_so[j]) {
          tw.scanZ_so[j] = tw.scanT_so[i];
          //	  printf("so(z): scanZ %d, angleZ %f\n",tw.scanZ_so[j],tw.elevT_so[i]*RAD_TO_DEG);
        }
      }
    }

    Py_DECREF(ipoz);
    Py_DECREF(opoz);
  }

  /* Get all arguments needed for interpolation. */
  if (!GetIntFromINFO(out, "/how/i_method", &tw.method)) {
    PyErr_SetString(PyExc_TypeError, "i_method not specified\n");
    return NULL;
  }

  if (!GetIntFromINFO(out, "/how/transform_weighting", &tw.useWeights)) {
    PyErr_SetString(PyExc_TypeError, "transform_weighting not specified\n");
    return NULL;
  }

  if (!GetIntFromINFO(out, "/how/elev_usage", &tw.elevUsage)) {
    PyErr_SetString(PyExc_TypeError, "elev_usage not specified\n");
    return NULL;
  }

  if (!GetIntFromINFO(out, "/how/WMO", &tw.wmo)) {
    PyErr_SetString(PyExc_TypeError, "wmo not specified\n");
    return NULL;
  }

  if (!GetDoubleFromINFO(out, "/how/beamwidth", &tw.beamBroad)) {
    PyErr_SetString(PyExc_TypeError, "beamwidth not specified\n");
    return NULL;
  }
  tw.beamBroad *= DEG_TO_RAD;

  tw.slice = PPI; /* failsafe, since we're only interested in PPI */
  tw.height = 0L;

  if (tw.slice == PPI && tw.height >= tw.nelev) {
    /*Do not allow ppi index to be higher than existing no of elevs*/
    PyErr_SetString(PyExc_AttributeError, "Wanted ppi index not found.");
    return NULL;
  }

  if (!GetDoubleFromINFO(out, "/where/height", &tw.alt0)) {
    PyErr_SetString(PyExc_TypeError, "alt_0 not specified\n");
    return NULL;
  }

  if (!GetDoubleFromINFO(out, "/where/lon", &tw.lon0)) {
    PyErr_SetString(PyExc_TypeError, "lon_0 not specified\n");
    return NULL;
  }
  tw.lon0 *= DEG_TO_RAD;

  if (!GetDoubleFromINFO(out, "/where/lat", &tw.lat0)) {
    PyErr_SetString(PyExc_TypeError, "lat_0 not specified\n");
    return NULL;
  }
  tw.lat0 *= DEG_TO_RAD;

  tw.cressmanR_z = 1L;

  if (!GetDoubleFromINFO(out, "/how/dndh", &tw.dndh)) {
    tw.dndh = (-3.9e-5) / 1000; /*To get same value in m^-1*/
  }

  sprintf(tmpstr, "/scan%d/what/gain", tw.scan[0]);
  if (!GetDoubleFromINFO(out, tmpstr, &tw.slopew)) {
    PyErr_SetString(PyExc_TypeError, "store_slope not specified\n");
    return NULL;
  }

  sprintf(tmpstr, "/scan%d/what/offset", tw.scan[0]);
  if (!GetDoubleFromINFO(out, tmpstr, &tw.ordw)) {
    PyErr_SetString(PyExc_TypeError, "store_ord not specified\n");
    return NULL;
  }

  if (tw.gotZ) {
    sprintf(tmpstr, "/scan%d/what/gain", tw.scanZ[0]);
    if (!GetDoubleFromINFO(outz, tmpstr, &tw.slopez)) {
      PyErr_SetString(PyExc_TypeError, "store_slope (Z) not specified\n");
      return NULL;
    }

    sprintf(tmpstr, "/scan%d/what/offset", tw.scanZ[0]);
    if (!GetDoubleFromINFO(outz, tmpstr, &tw.ordz)) {
      PyErr_SetString(PyExc_TypeError, "store_ord (Z) not specified\n");
      return NULL;
    }
  }

  /* Select function for calculations. This just confirms trans func is OK. */
  switch (tw.method) {
  case NEAREST:
    break;
  case CRESSMAN:
  case INVERSE:
  case UNIFORM:
    break;
  default:
    PyErr_SetString(PyExc_TypeError, "No such interpolation method\n");
    return NULL;
    break;
  }

  /* Load src and dest arrays into the transform wrapper */
  for (i = 0; i < tw.nelev; i++) {
    initialize_RaveObject(&inrave);
    if (!fill_rave_object(in, &inrave, tw.scan[i], "scan")) {
      if (!inrave.info || !inrave.data) {
        return NULL;
      }
    }
    tw.src[i] = (PyArrayObject *) array_data_2d(inrave.data);

    if (i == 0) {
      tw.intype = array_type_2d(inrave.data);
      tw.inxmax = inrave.data->dimensions[1];
      tw.inymax = inrave.data->dimensions[0];
      tw.inrsize = inrave.data->strides[0] / inrave.data->strides[1];
      /*       printf("in(w): intype %c, inxmax %d, inymax %d, inrsize %d/%d\n", */
      /*              tw.intype,tw.inxmax,tw.inymax,inrave.data->strides[0],inrave.data->strides[1]); */
    }

    if (tw.gotZ) {
      initialize_RaveObject(&inzrave);
      if (!fill_rave_object(inz, &inzrave, tw.scanZ[i], "scan")) {
        if (!inzrave.info || !inzrave.data) {
          return NULL;
        }
      }
      tw.srcZ[i] = (PyArrayObject *) array_data_2d(inzrave.data);

      if (i == 0) {
        tw.inztype = array_type_2d(inzrave.data);
        tw.inzrsize = inzrave.data->strides[0] / inzrave.data->strides[1];
        /* 	printf("in(z): inztype %c, inzrsize %d/%d\n", */
        /* 	       tw.inztype,inzrave.data->strides[0],inzrave.data->strides[1]); */
      }
    }
  }
  for (i = 0; i < tw.nelev_so; i++) {
    initialize_RaveObject(&outrave);
    if (!fill_rave_object(out, &outrave, tw.scan_so[i], "scan")) {
      if (!outrave.info || !outrave.data) {
        return NULL;
      }
    }
    tw.dest[i] = (PyArrayObject *) array_data_2d(outrave.data);

    /* Assume that all scans are of the same type */
    if (i == 0) {
      tw.outtype = array_type_2d(outrave.data);
      //      printf("out(w): outtype %c\n",tw.outtype);
    }

    if (tw.gotZ) {
      initialize_RaveObject(&outzrave);
      if (!fill_rave_object(outz, &outzrave, tw.scanZ_so[i], "scan")) {
        if (!outzrave.info || !outzrave.data) {
          return NULL;
        }
      }
      tw.destZ[i] = (PyArrayObject *) array_data_2d(outzrave.data);

      /* Assume that all scans are of the same type */
      if (i == 0) {
        tw.outztype = array_type_2d(outzrave.data);
        //	printf("out(z): outztype %c\n",tw.outztype);
      }
    }
  }

  /* Assume that nodata is the same for src and dest pvols. */
  /* The following might leak some references - check! */
  sprintf(tmpstr, "/scan%d/what/nodata", tw.scan[0]);
  if (!GetDoubleFromINFO(in, tmpstr, &tw.nodata)) {
    PyErr_SetString(PyExc_TypeError, "nodata for input winds not specified\n");
    return NULL;
  }

  if (tw.gotZ) {
    sprintf(tmpstr, "/scan%d/what/nodata", tw.scanZ[0]);
    if (!GetDoubleFromINFO(inz, tmpstr, &tw.nodataZ)) {
      PyErr_SetString(PyExc_TypeError,
                      "nodata for input reflectivities not specified\n");
      return NULL;
    }
  }

  /* Extract polar area variables. */
  if (!GetDoubleFromINFO(in, "/where/xscale", &tw.inscale)) {
    PyErr_SetString(PyExc_TypeError, "No range size in source p_area.\n");
    return NULL;
  }
  if (!GetDoubleFromINFO(out, "/where/xscale", &tw.outscale)) {
    PyErr_SetString(PyExc_TypeError, "No range size in destination p_area.\n");
    return NULL;
  }

  if (!GetIntFromINFO(in, "/where/ysize", &tw.iaz)) {
    PyErr_SetString(PyExc_TypeError,
                    "Unknown number of azimuth gates in source p_area.\n");
    return NULL;
  }
  if (!GetIntFromINFO(out, "/where/ysize", &tw.oaz)) {
    PyErr_SetString(PyExc_TypeError,
                    "Unknown number of azimuth gates in destination p_area.\n");
    return NULL;
  }

  if (!GetIntFromINFO(in, "/where/xsize", &tw.ira)) {
    PyErr_SetString(PyExc_TypeError,
                    "Unknown number of range bins in source p_area.\n");
    return NULL;
  }
  if (!GetIntFromINFO(out, "/where/xsize", &tw.ora)) {
    PyErr_SetString(PyExc_TypeError,
                    "Unknown number of range bins in destination p_area.\n");
    return NULL;
  }

  PyErr_Clear(); /* Don't quite understand why this has to be here... */

  if (GetStringFromINFO(out, "/how/tmpfile", &tmpchar)) {
    tw.tmpfile = strdup(tmpchar);
  } else {
    PyErr_SetString(PyExc_IOError, "No output file.\n");
    return NULL;
  }

  if (tw.tmpfile) {
    if (!(staticFD = fopen(tw.tmpfile, "w"))) {
      free(tw.tmpfile);
      PyErr_SetString(PyExc_IOError, "Error opening output file.\n");
      return NULL;
    }
  }

  tw.maxElev = tw.nelev_so;
  tw.maxAzim = tw.oaz;
  tw.maxRange = tw.ora;
  tw.outrsize = tw.ora / 1;
  if (tw.gotZ) {
    tw.outzrsize = tw.ora / 1;
  }
  tw.oazOffset = (360.0 / tw.maxAzim) * DEG_TO_RAD; /* dest   pvol */
  tw.azOffset = (360.0 / tw.inymax) * DEG_TO_RAD; /* source pvol */
  tw.dndh = (-3.9e-5) / 1000;
  /* tw.R must be derived at each range to avoid overlapping scans at close
   range. This should be set up as an array containing R for each bin before
   entering the main loop, below. */
  tw.R = tw.outscale; /* default */
  switch (tw.method) {
  case CRESSMAN:
  case INVERSE:
  case UNIFORM:
    getRs(&tw);
    opo = GetSequenceFromINFO(out, "/how/rs");
    if (!opo) {
      Py_DECREF(ipo);
      PyErr_SetString(PyExc_TypeError, "No Rs in output info.\n");
      return NULL;
    }
    if (!PySequence_Check(opo)) {
      Py_DECREF(ipo);
      Py_DECREF(opo);
      PyErr_SetString(PyExc_TypeError, "Output Rs must be a sequence.\n");
      return NULL;
    }
    for (i = 0; i < tw.maxRange; i++) {
      ipo = PyFloat_FromDouble(tw.Rs[i]);
      if (!PyFloat_Check(ipo)) {
        Py_DECREF(ipo);
        Py_DECREF(opo);
        PyErr_SetString(PyExc_TypeError, "Couldn't access Rs[i].\n");
        return NULL;
      }
      if (!PySequence_SetItem(opo, i, ipo)) {
        Py_DECREF(ipo);
        Py_DECREF(opo);
        PyErr_SetString(PyExc_TypeError, "Couldn't write Rs to sequence.\n");
        return NULL;
      }
    }
    if (!PyMapping_SetItemString(out, "/how/rs", opo)) {
      Py_DECREF(ipo);
      Py_DECREF(opo);
      PyErr_SetString(PyExc_TypeError, "Couldn't set Rs in out_info.\n");
      return NULL;
    }

    /* DEBUG: Cast an exception to artificially terminate. */
    /*      Py_DECREF(ipo); */
    /*      Py_DECREF(opo); */
    /*      Py_DECREF(in_info); */
    /*      Py_DECREF(out_info); */
    /*      if (tw.gotZ) { */
    /* 	Py_DECREF(inz_info); */
    /*      } */
    /*      for(i=0;i<no_of_src;i++) { */
    /* 	Py_DECREF(tw.src[i]); */
    /* 	if (tw.gotZ) { */
    /* 	   Py_DECREF(tw.srcZ[i]); */
    /* 	} */
    /*      } */
    /*      for(i=0;i<no_of_dest;i++) { */
    /* 	Py_DECREF(tw.dest[i]); */
    /* 	if (tw.gotZ) { */
    /* 	   Py_DECREF(tw.destZ[i]); */
    /* 	} */
    /*      } */
    /*      if(staticFD) { */
    /* 	fclose(staticFD); */
    /*      } */
    /*      PyErr_SetString(PyExc_TypeError,"Time to die.\n"); */
    /*      return NULL; */
    /*      break; */
  default:
    break;
  }

  /* Caching has been removed! No lookup tables. */

  /* Before entering the main loop, write this radar's header to tmpfile. */
  /*   printf("nodata: %f (w), %f (z)\n",tw.nodata,tw.nodataZ); */
  /*   printf("scale in source file: %f\n",tw.inscale); */
  /*   printf("scale in destination file: %f\n",tw.outscale); */
  /*   printf("# azimuth gates in source file: %d\n",tw.iaz); */
  /*   printf("# azimuth gates in destination file: %d\n",tw.oaz); */
  /*   printf("# range bins in source file: %d\n",tw.ira); */
  /*   printf("# range bins in destination file: %d\n",tw.ora); */
  fprintf(staticFD, HFORMAT, tw.lon0 * RAD_TO_DEG, tw.lat0 * RAD_TO_DEG,
          (int) tw.alt0, tw.wmo, tw.outscale, tw.beamBroad * RAD_TO_DEG,
          tw.oazOffset * RAD_TO_DEG);

  /* THE MAIN EVENT! */
  for (e = 0; e < tw.maxElev; e++) {
    for (a = 0; a < tw.maxAzim; a++) {
      for (r = 0; r < tw.maxRange; r++) {
        doP2P(e, a, r, &tw);
      }
    }
  }

  getAllocatedCW(-99);

  if (staticFD) {
    fclose(staticFD);
  }

  /* DECREF all temporary items! */
  if (tw.tmpfile) {
    free(tw.tmpfile);
  }
  if (tw.obj) {
    free(tw.obj);
  }

  Py_DECREF(ipo);
  Py_DECREF(opo);

  /* these two should be equal, but you never know... */
  for (i = 0; i < tw.nelev; i++) {
    Py_DECREF(tw.src[i]);
    if (tw.gotZ) {
      Py_DECREF(tw.srcZ[i]);
    }
  }
  for (i = 0; i < tw.nelev_so; i++) {
    Py_DECREF(tw.dest[i]);
    if (tw.gotZ) {
      Py_DECREF(tw.destZ[i]);
    }
  }

  PyErr_Clear();
  Py_INCREF(Py_None); /* Return nothing explicitly */
  return Py_None;
}

static PyObject* _ptop_test(PyObject* self, PyObject* args)
{
  printf("PTOP: Test to verify that ptop works\n");

  Py_INCREF(Py_None);
  return Py_None;
}

/* Collect the functions in this module and give them external names */
static struct PyMethodDef _ptop_functions[] =
{
{ "transform", (PyCFunction) _ptop_transform, METH_VARARGS },
{ "test", (PyCFunction) _ptop_test, METH_VARARGS },
{ NULL, NULL } };

/**
 * Initializes the python module _ptop
 */
PyMODINIT_FUNC init_ptop(void)
{
  PyObject* m;
  m = Py_InitModule("_ptop", _ptop_functions);
  ErrorObject = PyString_FromString("_ptop.error");
  if (ErrorObject == NULL || PyDict_SetItemString(PyModule_GetDict(m), "error",
                                                  ErrorObject) != 0)
    Py_FatalError("Can't define _ptop.error");

  import_array(); /* Make sure I access to the Numeric PyArray functions */
}

/*
 Note:           Some variables and functions are not being used at present,
 so they have been commented. They may be activated later, but
 for now we just want to shut up the compiler...
 */
