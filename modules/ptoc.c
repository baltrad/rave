#include <Python.h>
#include <arrayobject.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "raveutil.h"
#include "rave.h"

static PyObject *PyTransform_Error;

#define raiseException(type,message)\
{PyErr_SetString(type,message);return NULL;}

#define pySetError(type,message)\
{PyErr_SetString(type,message);}

/*The same as raiseException, just wanna have other def for this.*/
#define raiseErrorWI(type,message)\
{PyErr_SetString(type,message);return 0;}

/* -------------------------------------------------------------------- */
/* Constructors                                                         */

static FILE* staticFD;

static void
_ptoc_error(void)
{
    PyErr_SetString(PyTransform_Error, "Something went wrong...");
}

#define PPI 1
#define CAPPI 2
#define PCAPPI 3
#define ECHOTOP 4

#define ALL_WEIGHTS 1
#define NO_ZERO_WEIGHTS 2

#define ROUND_AZIMUTH 0
#define FLOOR_AZIMUTH 1
#define CEIL_AZIMUTH 2

#define USE_SINGLE_ELEV 1
#define USE_MANY_ELEV 2

#define MAXELEV 32

#define NO_QC 0
#define STD_DEVIATION_MEANVALUE 1
#define STD_DEVIATION_MEDIAN 2

#define ET_LH_RCE 0 /*Lower height and recalculate elevation*/
#define ET_LE_HALF 1 /*Lower elevation and recalculate height*/
#define ET_LH_HALF  2 /*Lower height*/
#define ET_NO_LOWER 3 /*Do not decrease anything*/

typedef struct {
  UV inUL;
  PJ *outpj;
  int nodata, slice, method, useWeights, elevUsage;
  double inscale, height;
  double alt0, lon0, lat0; /* Pos of radar */
  int inxmax, inymax;
  int inxsize, outxsize;
  double dndh;
  int nelev;
  double elev[MAXELEV];
  /*  char type; */
  char intype,outtype;
  PyArrayObject *src[MAXELEV];
  unsigned char *desta;
  double R;
  double beamBroad;
  char* cashfile;

  double cressmanR_xy;
  double cressmanR_z;

  int iqc;	/*Filtering flag*/
  double iqcvalue; /*Contains value when to ignore value*/

  /* used by echotop */
  int outdimx;
  int outdimy;

  double echonodata;
  double echomaxheight;
  double echoscale;

  int echodecrtype;
  double echodecrfactor;

  /*Used by max,maxheight and max_and_height methods*/
  char topotype;
  unsigned char* topodata;
  int topooutxsize;

   /*Used for setting compactness in array*/
   int set_compactness;

   /*Used for checking if the nearest value is 0 || nodata or not*/
   int check_nearest;
   UV outUL;
   double outxscale;
   double outyscale;

} TrafoWrapper3D;

static void displayWrap(TrafoWrapper3D* wrap)
{
  printf("slice=%d,method=%d,useWeights=%d,elevUsage=%d\n",\
	 wrap->slice,wrap->method,wrap->useWeights,wrap->elevUsage);
  printf("inscale=%lg,height=%lg\n",wrap->inscale,wrap->height);
  printf("inxsize=%d,inysize=%d\n",wrap->inxsize,wrap->outxsize);
  printf("inxmax=%d,inymax=%d\n",wrap->inxmax,wrap->inymax);

}

#define MAXCOORD 128

typedef struct {  /* A 3d polar coordinate with weight */
  int elev, range, azimuth; /* Index in 3d matrix */
  double weight; /* relevance to the current pixel */
} CoordWeight;

static int getAzindex(double az, TrafoWrapper3D* wrap, int rounding)
{
  double azOffset, azimuth;
  int azIndex;

  azOffset = 360.0/wrap->inymax;

#ifdef TRUNC_AZIMUTH
  azimuth = az*RAD_TO_DEG/azOffset;
#else
  switch(rounding) {
  case ROUND_AZIMUTH:
    azimuth = rint(az*RAD_TO_DEG/azOffset);
    break;
  case FLOOR_AZIMUTH:
    azimuth = floor(az*RAD_TO_DEG/azOffset);
    break;
  case CEIL_AZIMUTH:
    azimuth = ceil(az*RAD_TO_DEG/azOffset);
    break;
  }
#endif

  azIndex = mytrunc(azimuth);

  azOffset*=DEG_TO_RAD;

  if(az < azOffset) {
    return 0;
  }
  else if(az > (2*M_PI - azOffset)) {
    return (wrap->inymax-1);
  }
  else {
    return azIndex;
  }
}


static double getarritem3d(int e,int x, int y, TrafoWrapper3D *tw)
{
  double ret=0;
  switch(tw->intype) {
  case 'c': /* PyArray_CHAR */
  case '1': /* PyArray_SBYTE */ {
    char *a = (char *)tw->src[e]->data;
    ret = a[y*tw->inxsize+x];
    break;
  }
  case 'b': /* PyArray_UBYTE */ {
    unsigned char *a = (unsigned char *)tw->src[e]->data;
    ret = a[y*tw->inxsize+x];
    break;
  }

  case 's': /* PyArray_SHORT */  {
    short *a = (short *)tw->src[e]->data;
    ret = a[y*tw->inxsize+x];
    break;
  }

  case 'H': /* PyArray_USHORT */  {
    unsigned short *a = (unsigned short *)tw->src[e]->data;
    ret = a[y*tw->inxsize+x];
    break;
  }

  case 'i': /* PyArray_INT */ {
    int *a = (int *)tw->src[e]->data;
    ret = a[y*tw->inxsize+x];
    break;
  }

  case 'I': /* PyArray_UINT */  {
    unsigned int *a = (unsigned int *)tw->src[e]->data;
    ret = a[y*tw->inxsize+x];
    break;
  }

  case 'l': /* PyArray_LONG */ {
    long *a = (long *)tw->src[e]->data;
    ret = a[y*tw->inxsize+x];
    break;
  }

  case 'f': /* PyArray_FLOAT */ {
    float *a = (float *)tw->src[e]->data;
    ret = a[y*tw->inxsize+x];
    break;
  }

  case 'd': /* PyArray_DOUBLE */ {
    double *a = (double *)tw->src[e]->data;
    ret = a[y*tw->inxsize+x];
    break;
  }

  case 'F': /* PyArray_CFLOAT */
  case 'D': /* PyArray_CDOUBLE */
  case 'O': /* PyArray_OBJECT */
  default:
    /* Ignored for now => 0.0 */
    printf("Unsupported type: '%c'\n", tw->intype);
  }
  return ret;
}


/* Generic storage of data in the destination matrix */
/* NOTE: Type conversion limitations hardcoded in here! */
/* In particualar, 64 bit longs may be troublesome to port! */

static void setarritem3d(int x, int y, double v, TrafoWrapper3D *tw)
{
  switch(tw->outtype) {
  case 'c': /* PyArray_CHAR */
  case '1': /* PyArray_SBYTE */ {
    char *a = (char *)tw->desta;
    int c = mytrunc(v);

    if (c < -128)
      c = -128;
    if (c>127)
      c = 127;
    a[y*tw->outxsize+x] = c;
    break;
  }
  case 'b': /* PyArray_UBYTE */ {
    unsigned char *a = (unsigned char *)tw->desta;
    unsigned char c;

    if (v<0)  /* Oops: Not allowed!*/
      v=0;
    if (v>255)
      v=255;

    c=mytrunc(v);

    a[y*tw->outxsize+x]=c;
    break;
  }

  case 's': /* PyArray_SHORT */  {
    short *a = (short *)tw->desta;
    int c = mytrunc(v);
    if (c<-32768)  /* Oops: Not allowed!*/
      c=-32768;
    if (c>32767)
      c=32767;
    a[y*tw->outxsize+x]=c;
    break;
  }

  case 'H': /* PyArray_USHORT */  {
    unsigned short *a = (unsigned short *)tw->desta;
    int c = mytrunc(v);
    if (c<0)  /* Oops: Not allowed!*/
      c=0;
    if (c>USHRT_MAX)
      c=USHRT_MAX;
    a[y*tw->outxsize+x]=c;
    break;
  }

  case 'i': /* PyArray_INT */ {
    int *a = (int *)tw->desta;
    int c;
    if (v > MAXINT)
      v = MAXINT;
    if (v < -MAXINT)
      v = -MAXINT;
    c = mytrunc(v);
    a[y*tw->outxsize+x] = c;
    break;
  }

  case 'I': /* PyArray_UINT */  {
    unsigned int *a = (unsigned int *)tw->desta;
    int c = mytrunc(v);
    if (c<0)  /* Oops: Not allowed!*/
      c=0;
    if (c>UINT_MAX)
      c=UINT_MAX;
    a[y*tw->outxsize+x]=c;
    break;
  }

  case 'l': /* PyArray_LONG */ {
    long *a = (long *)tw->desta;
    long c;
    if (v > MAXLONG)
      v = MAXLONG;
    if (v < -MAXLONG)
      v = -MAXLONG;
    c = v;  /* Should work on 64bit boxes after above preparations. */
    a[y*tw->outxsize+x] = c;
    break;
  }

  case 'f': /* PyArray_FLOAT */ {
    float *a = (float *)tw->desta;
    if (v > MAXFLOAT)
      v = MAXFLOAT;
    if (v < -MAXFLOAT)
      v = -MAXFLOAT;
    a[y*tw->outxsize+x] = v;
    break;
  }

  case 'd': /* PyArray_DOUBLE */ {
    double *a = (double *)tw->desta;
    a[y*tw->outxsize+x] = v;
    break;
  }

  case 'F': /* PyArray_CFLOAT */
  case 'D': /* PyArray_CDOUBLE */
  case 'O': /* PyArray_OBJECT */
  default:
    /* Ignored for now => 0.0 */
    printf("Unsupported type: '%c'\n", tw->outtype);
  }
}

static void settopoitem(int x, int y, double v, TrafoWrapper3D* tw)
{
  switch(tw->topotype) {
  case 'c': /* PyArray_CHAR */
  case '1': /* PyArray_SBYTE */ {
    char *a = (char *)tw->topodata;
    int c = mytrunc(v);

    if (c < -128)
      c = -128;
    if (c>127)
      c = 127;
    a[y*tw->topooutxsize+x] = c;
    break;
  }
  case 'b': /* PyArray_UBYTE */ {
    unsigned char *a = (unsigned char *)tw->topodata;
    unsigned char c;
    if (v<0)  /* Oops: Not allowed!*/
      v=0;
    if (v>255)
      v=255;
    c=mytrunc(v);
    a[y*tw->topooutxsize+x]=c;
    break;
  }

  case 's': /* PyArray_SHORT */  {
    short *a = (short *)tw->topodata;
    int c = mytrunc(v);
    if (c<-32768)  /* Oops: Not allowed!*/
      c=-32768;
    if (c>32767)
      c=32767;
    a[y*tw->topooutxsize+x]=c;
    break;
  }

  case 'H': /* PyArray_USHORT */  {
    unsigned short *a = (unsigned short *)tw->topodata;
    int c = mytrunc(v);
    if (c<0)  /* Oops: Not allowed!*/
      c=0;
    if (c>USHRT_MAX)
      c=USHRT_MAX;
    a[y*tw->topooutxsize+x]=c;
    break;
  }

  case 'i': /* PyArray_INT */ {
    int *a = (int *)tw->topodata;
    int c;
    if (v > MAXINT)
      v = MAXINT;
    if (v < -MAXINT)
      v = -MAXINT;
    c = mytrunc(v);
    a[y*tw->topooutxsize+x] = c;
    break;
  }

  case 'I': /* PyArray_UINT */  {
    unsigned int *a = (unsigned int *)tw->topodata;
    int c = mytrunc(v);
    if (c<0)  /* Oops: Not allowed!*/
      c=0;
    if (c>UINT_MAX)
      c=UINT_MAX;
    a[y*tw->topooutxsize+x]=c;
    break;
  }
  case 'l': /* PyArray_LONG */ {
    long *a = (long *)tw->topodata;
    long c;
    if (v > MAXLONG)
      v = MAXLONG;
    if (v < -MAXLONG)
      v = -MAXLONG;
    c = v;  /* Should work on 64bit boxes after above preparations. */
    a[y*tw->topooutxsize+x] = c;
    break;
  }

  case 'f': /* PyArray_FLOAT */ {
    float *a = (float *)tw->topodata;
    if (v > MAXFLOAT)
      v = MAXFLOAT;
    if (v < -MAXFLOAT)
      v = -MAXFLOAT;
    a[y*tw->topooutxsize+x] = v;
    break;
  }

  case 'd': /* PyArray_DOUBLE */ {
    double *a = (double *)tw->topodata;
    a[y*tw->topooutxsize+x] = v;
    break;
  }

  case 'F': /* PyArray_CFLOAT */
  case 'D': /* PyArray_CDOUBLE */
  case 'O': /* PyArray_OBJECT */
  default:
    /* Ignored for now => 0.0 */
    printf("Unsupported type: '%c'\n", tw->topotype);
  }
}

static int getElevIndex(double elev, TrafoWrapper3D* wrap)
{
  int i,elevIndex;
  double elev1, elev2;

  if(wrap->slice == PPI) {
    elevIndex = mytrunc(wrap->height);
    return elevIndex;
  }

  for(i=0;i<wrap->nelev;i++) {
    if( (i<(wrap->nelev-1)) && elev > wrap->elev[i] && elev < wrap->elev[i+1]) {

#ifdef TRUNC_ELEV
      elevIndex = i;
#else
      elev1 = elev - wrap->elev[i];
      elev2 = elev - wrap->elev[i+1];
      if(elev1<0.0)
	elev1 = -elev1;
      if(elev2 < 0.0)
	elev2 = -elev2;

      if( elev1 > elev2) {
	elevIndex = i+1;
      }
      else {
	elevIndex = i;
      }
#endif
      break;
    }
    else if(elev < wrap->elev[0]) {
      elevIndex=0;
      break;
    }
    else if(elev > wrap->elev[wrap->nelev-1]) {
      elevIndex=wrap->nelev-1;
      break;
    }
    else if(elev == wrap->elev[i]) {
      elevIndex=i;
      break;
    }
  }

  return elevIndex;
}

/* ----------------------------------------------------------------
   Find nearest neighbour
*/
static void nearest3d(int x, int y, UV here_s, TrafoWrapper3D *tw)
{
  int gx, gy, ge;
  UV here, there_s;
  Position source;
  Position target;

  /* inverse transform ps surface coords to long/lat */
  here = pj_inv(here_s, tw->outpj);

  /* transform long/lat using alt0&co to elev, az, height etc */

  source.lon0=tw->lon0;
  source.lat0=tw->lat0;
  source.lon=here.u;
  source.lat=here.v;


  source.alt0=tw->alt0; /*Radar originates from alt0 meters above sea-surface*/
  source.alt=tw->alt0+200; /*Interested in the surface at xxx above origin*/
  source.dndh=(-3.9e-5); /*-3.9*10^(-5) km^(-1)*/
  source.dndh=source.dndh/1000; /*To get same value in m^(-1) I think*/

  llToDa(&source, &target);  /* LonLat => DistAzim */

  /*  printf("%d,%d = %lg,%lg (m) = %lg, %lg (ll)=> dist=%lg, az=%lg\n",x,y,here_s.u, here_s.v,here.u*RAD_TO_DEG, here.v*RAD_TO_DEG, target.distance, target.azimuth*RAD_TO_DEG);*/

  source.distance = target.distance;
  source.azimuth = target.azimuth;

  dhToRe(&source,&target);  /* DistHeight => RangeElev */

  /* trunc to nearest pixel */
#ifdef fake
  ge = 4*x/(tw->inymax);
  gx = x%(tw->inxmax);  /* Fake for now */
  gy = y%tw->inymax;
#else
  ge = mytrunc(tw->height);
  gx = mytrunc(target.range/tw->inscale);
  gy = getAzindex(target.azimuth, tw, ROUND_AZIMUTH);
#endif

  /*  printf("%d,%d => range=%d, az=%d\n",x,y,gx,gy);*/
  /* Copy if within bounds - never mind "nodata" */
  if ((ge >= 0 && ge < tw->nelev) &&
      (gx >= 0 && gx < tw->inxmax) && (gy >= 0 && gy < tw->inymax)) {
    setarritem3d(x,y,getarritem3d(ge,gx,gy,tw), tw);
  }
}

int getNearest(Position* src, CoordWeight* weight, TrafoWrapper3D* wrap)
{
  weight[0].azimuth = getAzindex(src->azimuth,wrap,ROUND_AZIMUTH);

  weight[0].elev = getElevIndex(src->elevation,wrap);

#ifndef ELEV_BOUNDARY_CHECK
  if(wrap->slice == CAPPI) {
    if(src->elevation < wrap->elev[0] || src->elevation > wrap->elev[wrap->nelev-1]) {
      /*Outside boundary*/
      return 0;
    }
  }
#endif

  if (src->range > wrap->inxmax*wrap->inscale || src->range < 0.0) {
    /* Outside boundary */
    return 0;
  }

#ifdef TRUNC_RANGE
  /* If TRUNC_RANGE is set, trunk the range, otherwise round it. */
  weight[0].range = mytrunc(src->range/wrap->inscale);
#else
  weight[0].range = mytrunc(rint(src->range/wrap->inscale));
#endif

  weight[0].weight = 1.0;

  if(weight[0].range >= wrap->inxmax)
    return 0;

  return 1;
}



static void getSurrElev(int* tgt, double elevation, TrafoWrapper3D* wrap)
{
  int elevIndex;

  elevIndex = getElevIndex(elevation, wrap);

  if ( wrap->elev[elevIndex] == elevation ) {
    tgt[0] = elevIndex;
    tgt[1] = elevIndex;
  }
  else if ( elevIndex == 0 ) {
    if( elevation > wrap->elev[0] ) {
      /* elevation lies between elev[0] and elev[1] */
      tgt[0] = elevIndex;
      tgt[1] = elevIndex + 1;
    }
    else {
      /* elevation lies below elev[0] */
      tgt[0] = tgt[1] = elevIndex;
    }
  }
  else if ( elevIndex == (wrap->nelev-1) ) {
    if( elevation < wrap->elev[wrap->nelev-1] ) {
      /* elevation lies between the highest elevation and the elevation below */
      tgt[0] = elevIndex - 1;
      tgt[1] = elevIndex;
    }
    else {
      /* elevation lies above the highest elevation */
      tgt[0] = tgt[1] = elevIndex;
    }
  }
  else {
    /* the elevation lies between two elevations, decide which */
    if(elevation < wrap->elev[elevIndex] &&
       elevation > wrap->elev[elevIndex-1] ) {
      tgt[0] = elevIndex - 1;
      tgt[1] = elevIndex;
    }
    else {
      tgt[0] = elevIndex;
      tgt[1] = elevIndex + 1;
    }
  }
}

static int getBilinear(Position* src, CoordWeight* weight, TrafoWrapper3D* wrap)
{
  int    azLowHigh[2];   /* MIN, MAX */
  int    elevMinMax[2]; /* MIN, MAX */
  int    rangeMinMax[2];
  int    weightIndex;

  int    rIndex,eIndex,aIndex;

  double azWeight,elevWeight,rangeWeight,azOffset,elevDiff;

  /*
   * first, decide what limits that should be used,
   * i.e. max, min elevation, max, min azimuth and max,min range
   */
  azLowHigh[0] = getAzindex(src->azimuth,wrap,FLOOR_AZIMUTH);
  azLowHigh[1] = getAzindex(src->azimuth,wrap,CEIL_AZIMUTH);


  if(wrap->slice == PPI) {
    elevMinMax[0] = elevMinMax[1] = mytrunc(wrap->height);
  }
  else {
    getSurrElev(elevMinMax, src->elevation, wrap);
  }

  /*
   * SHOULD I DO IT LIKE THIS WHEN IT IS CAPPI?
   */
  if(wrap->slice == CAPPI) {
    if(src->elevation < wrap->elev[0] || src->elevation > wrap->elev[wrap->nelev-1]) {
      /*Outside boundary*/
      return 0;
    }
  }

  rangeMinMax[0] = mytrunc(floor(src->range/wrap->inscale));
  rangeMinMax[1] = mytrunc(ceil(src->range/wrap->inscale));

  if (src->range> wrap->inxmax*wrap->inscale)
    return 0;/* Out of range */

  azOffset = (360.0/wrap->inymax)*DEG_TO_RAD;

  elevDiff = wrap->elev[elevMinMax[0]]-wrap->elev[elevMinMax[1]];
  if(elevDiff < 0.0)
    elevDiff = -elevDiff;

  /*
   * start calculating the different points that are affected by
   * the current max,min values.
   */
  weightIndex = 0;

  for(rIndex=0;rIndex<2;rIndex++) {
    for(eIndex=0;eIndex<2;eIndex++) {
      for(aIndex=0;aIndex<2;aIndex++) {

	weight[weightIndex].range = rangeMinMax[rIndex];
	weight[weightIndex].elev = elevMinMax[eIndex];
	weight[weightIndex].azimuth = azLowHigh[aIndex];

	if(rangeMinMax[0] == rangeMinMax[1] && rIndex == 1) {
	  /* The range is evenly dividable with scale. */
	    rangeWeight = 0.0;
	}
	else {
	  rangeWeight = rangeMinMax[rIndex]*wrap->inscale - src->range;
	  rangeWeight /= wrap->inscale;

	  if(rangeWeight < 0.0)
	    rangeWeight=-rangeWeight;

	  rangeWeight=1.0-rangeWeight;
	}

	if(azLowHigh[0] == azLowHigh[1] && aIndex == 1) {
	  /* The ray lies exactly on one azimuth */
	  azWeight = 0.0;
	}
	else {
	  azWeight = azLowHigh[aIndex]*azOffset-src->azimuth;
	  azWeight /= azOffset;

	  if(azWeight < 0.0)
	    azWeight=-azWeight;

	  azWeight = 1-azWeight;
	}

	if(elevMinMax[0] == elevMinMax[1]) {
	  /* The src elevation must lie below/above or exactly on the ray*/
	  if(eIndex == 1) {
	    elevWeight = 0.0;
	  }
	  else {
	    elevWeight = 1.0;
	  }
	}
	else {
	  elevWeight = wrap->elev[elevMinMax[eIndex]]-src->elevation;
	  elevWeight/=elevDiff;
	  if(elevWeight<0.0)
	    elevWeight=-elevWeight;
	}

	weight[weightIndex].weight = rangeWeight*azWeight*elevWeight;

	weightIndex++;
      }
    }
  }

  return 8;
}

/* Help funs for cubic */

static double cubf01(double x)
{
  double y = (x*x*x-2*x*x+1);
  return y;
}

static double cubf12(double x)
{
  double y = (-x*x*x+5*x*x-8*x+4);
  return y;
}

static double cubf2(double x)
{
  if (x<0)
    x = -x; /* No sign */
  if (x<1)
    return cubf01(x);
  if (x<=2)
    return cubf12(x);
  return 0;
}

static int getCubic(Position* src, CoordWeight* weight, TrafoWrapper3D* wrap)
{
  int rangeArray[4];
  int elevArray[4];
  int azArray[4];

  int sr,sa,se,weightIndex,i;

  double azimuthWeight, rangeWeight, elevWeight;

  double azOffset,elevDiff,sum;

  double elevHeight[4];

  Position* tmpPst;
  Position  tmpPst2;

  sum = 0.0;

  sr = mytrunc(src->range/wrap->inscale);
  se = getElevIndex(src->elevation,wrap);
  sa = getAzindex(src->azimuth,wrap,ROUND_AZIMUTH);

  if (src->range> wrap->inxmax*wrap->inscale)
    return 0;/* Out of range */

  azOffset = (360.0/wrap->inymax)*DEG_TO_RAD;


  /*
   * Initiate arrays with indexes.
   */
  for(i=-1;i<3;i++) {
    rangeArray[i+1]=sr+i;
    elevArray[i+1]=se+i;
    azArray[i+1]=sa+i;
  }


  /*
   * Make sure the arrays contains correct information.
   */
  for(i=0;i<4;i++) {
    if(rangeArray[i] > wrap->inxmax)
      rangeArray[i] = -1;
    if(elevArray[i] > wrap->nelev-1)
      elevArray[i] = -1;

    if(azArray[i] < 0) {
      azArray[i]+=wrap->inymax;
    }
    else if(azArray[i] > wrap->inymax) {
      azArray[i]=azArray[i]%wrap->inymax;
    }
  }

  /*MUST HAVE AN EQUIDISTANT SCALE, THEREFOR CHANGE TO HEIGHT */
  tmpPst = copyPosStruct(src);
  elevDiff=0.0;
  se = 0;
  for(i=0;i<4;i++) {
    if(elevArray[i] != -1) {
      tmpPst->elevation = wrap->elev[elevArray[i]];
      deToRh(tmpPst,&tmpPst2);
      elevHeight[i]=tmpPst2.alt;
      se++;
    }
  }

  free(tmpPst);

  weightIndex=0;

  for(sr=0;sr<4;sr++) {
    if(rangeArray[i]>=0) {/*LOOP OVER RANGES*/
      rangeWeight = rangeArray[sr]*wrap->inscale - src->range;
      rangeWeight /= wrap->inscale;
      rangeWeight = cubf2(rangeWeight);
    }
    else
      rangeWeight = 0.0;

    for(sa=0;sa<4;sa++) {
      if(azArray[sa]>=0) {/*LOOP OVER AZIMUTHS*/
	azimuthWeight = azArray[sa]*azOffset-src->azimuth;
	azimuthWeight /= azOffset;
	azimuthWeight = cubf2(azimuthWeight);
      }
      else {
	azimuthWeight = 0.0;
      }

      for(se=0;se<4;se++) { /*LOOP OVER ELEVATIONS*/
	if(elevArray[se]>=0) {
	  elevWeight = elevHeight[se]-src->alt;
	  elevWeight /= wrap->inscale;
	  elevWeight = cubf2(elevWeight);
	}
	else
	  elevWeight = 0.0;

	weight[weightIndex].elev=elevArray[se];
	weight[weightIndex].azimuth=azArray[sa];
	weight[weightIndex].range=rangeArray[sr];
	weight[weightIndex].weight=rangeWeight*azimuthWeight*elevWeight;

	sum+=weight[weightIndex].weight;


	weightIndex++;
      }
    }
  }

  if(sum==0.0)
    return 0;

  return 64;
}

/*METHODS BELOW ARE FOR CRESSMAN INTERPOLATION*/

static CoordWeight* getAllocatedCW(int maxNoOfItems)
{
  static CoordWeight* staticWeight = NULL;
  static int noOfWeights = 0;

  if(maxNoOfItems==-99) {
     if(staticWeight) {
	free(staticWeight);
	staticWeight=NULL;
     }
     noOfWeights=0;
     return NULL;
  }

  if(maxNoOfItems>noOfWeights) {
    if(staticWeight)
      free(staticWeight);

    staticWeight=malloc(sizeof(CoordWeight)*maxNoOfItems);
    noOfWeights = maxNoOfItems;
  }

  return staticWeight;
}

static void getMinMaxElevation(int minmaxElev[2], Position* src,
			double checkHeight, TrafoWrapper3D* wrap)
{
  Position* srcCopy;
  Position target;
  int elevIdx;
  int i;

  int debug=(checkHeight>1826.1 && checkHeight<1826.2)?0:0;

  if(debug) {
    printf("Debug enabled\n");
  }

  srcCopy = copyPosStruct(src);
  if(debug) {
    printf("Getting elevation index\n");
  }
  elevIdx = getElevIndex(srcCopy->elevation,wrap);

  if(debug) {
    printf("Elevation idx = %d\n",elevIdx);
  }

  minmaxElev[0] = minmaxElev[1] = -1;

  /* Look upwards for highest possible elevation, use
   * all elevations where "elev-beamBroad/2" generates a height
   * less than or equal to R + origin height.
   */
  if(debug) {
    printf("Upwards\n");
  }

  if(elevIdx < wrap->nelev-1) {
    for(i=elevIdx+1;i<wrap->nelev;i++) {
      srcCopy->elevation = wrap->elev[i]-wrap->beamBroad/2;
      deToRh(srcCopy,&target);
      if(target.alt > src->alt + checkHeight)
	break;
      else
	minmaxElev[1]=i;
    }
  }

  /* Look downwards for lowest possible elevation, use
   * all elevations where elev+beamBroad/2" generates a height
   * higher than or equal to R.
   */
  if(debug) {
    printf("Downwards\n");
  }
  if(elevIdx > 0) {
    for(i=elevIdx-1;i>=0;i--) {
      srcCopy->elevation = wrap->elev[i] + wrap->beamBroad/2;
      deToRh(srcCopy,&target);
      if(target.alt < src->alt - checkHeight)
	break;
      else
	minmaxElev[0]=i;
    }
  }

  if(minmaxElev[0] == -1)
    minmaxElev[0] = 0;

  if(minmaxElev[1] == -1)
    minmaxElev[1] = wrap->nelev -1;

  free(srcCopy);
}

static void getAdjAzimIdx(int minmaxAzim[2],double* minAzim,
		   double* maxAzim,TrafoWrapper3D* wrap)
{
  double azOffset = (360.0/wrap->inymax)*DEG_TO_RAD;
  int maxNoOfAzimuths;

  /*Adjust azimuths*/
  if((*minAzim) < 0)
    (*minAzim) += 2.0*M_PI;

  if( (*maxAzim) >= 2.0*M_PI)
    (*maxAzim)-=2.0*M_PI;

  minmaxAzim[0] = mytrunc(floor((*minAzim)/azOffset));
  minmaxAzim[1] = mytrunc(ceil((*maxAzim)/azOffset));

  /*Decide indexes to use*/
  if(minmaxAzim[0] > minmaxAzim[1]) {
    maxNoOfAzimuths = wrap->inymax - minmaxAzim[0];
    maxNoOfAzimuths += minmaxAzim[1];
    minmaxAzim[1] = minmaxAzim[0] + maxNoOfAzimuths;
  }
}

static double getCheckHeight(Position* src, Position* tgt,TrafoWrapper3D* wrap)
{
  double checkHeight,tmpRange,tmpDistance;

  int elevIdx;

  Position* aCopy;

  /*
  if(tgt->elevation < wrap->elev[0] ||
     tgt->elevation > wrap->elev[wrap->nelev-1]) {
  */
  /* Commented out after discussion with D.M. 99-03-11
   *if(tgt->elevation < wrap->elev[0]-wrap->beamBroad/2.0 ||
   *  tgt->elevation > wrap->elev[wrap->nelev-1]+wrap->beamBroad/2.0) {
   */
  if(tgt->elevation < wrap->elev[0] ||
     tgt->elevation > wrap->elev[wrap->nelev-1]) {
    aCopy = copyPosStruct(src);

    tmpRange = tgt->range;
    tmpDistance = tgt->distance;

    if(tgt->elevation < wrap->elev[0])
      elevIdx = 0;
    else
      elevIdx = wrap->nelev-1;

    /*aCopy->elevation = wrap->elev[elevIdx];*/

    /*Might be better, dont, know*/
    /*    aCopy->elevation = wrap->elev[elevIdx]-wrap->beamBroad/2.0;*/

    /*Might be even better*/
    /*Commented out after discussion with D.M. 99-03-11
     *if(elevIdx == 0)
     * aCopy->elevation = wrap->elev[elevIdx]-wrap->beamBroad/2.0;
     *else
     *  aCopy->elevation = wrap->elev[elevIdx]+wrap->beamBroad/2.0;
     */
    aCopy->elevation=wrap->elev[elevIdx]; /*This instead*/

    aCopy->alt = src->alt;

    ehToRd(aCopy,tgt);

    checkHeight = tgt->range*wrap->beamBroad/2.0;

    tgt->range = tmpRange;
    tgt->distance = tmpDistance;

    free(aCopy);
  }
  else {
    checkHeight = src->range*wrap->beamBroad/2.0;
  }

  checkHeight*=wrap->cressmanR_z;

  return checkHeight;
}

static double getCressHeightWeight(int eIndex,Position* src,
			    Position* target, double checkHeight, double* dHeight,
			    TrafoWrapper3D* wrap)
{

  double cmanHeightWeight;
  double calcHeight;

  Position* aCopy = copyPosStruct(src);

  /*Decide elevations*/

  aCopy->elevation = wrap->elev[eIndex];

  deToRh(aCopy,target);

  if(aCopy->alt > target->alt)
    calcHeight = aCopy->alt - target->alt;
  else
    calcHeight = target->alt - aCopy->alt;

  (*dHeight)=calcHeight;

  /*The code below that is commented can be uncommented, then
   *the elevations which are inside a elevation will get
   *the weight 1.
   */

  if(calcHeight >= checkHeight/* &&
     (src->elevation > wrap->elev[eIndex]+wrap->beamBroad/2 ||
     src->elevation < wrap->elev[eIndex]-wrap->beamBroad/2) */) {
    cmanHeightWeight = 0L;
  } /*inserted 99-03-19*/
  else if((src->elevation > wrap->elev[wrap->nelev-1] ||
	   src->elevation < wrap->elev[0]) && wrap->slice==PCAPPI) {
     cmanHeightWeight = 0L;
  }
  else {
    /*
    if(src->elevation <= wrap->elev[eIndex]+wrap->beamBroad/2 &&
       src->elevation >= wrap->elev[eIndex]-wrap->beamBroad/2) {
      cmanHeightWeight = 1L;
    }
    else
    */
    if(calcHeight < checkHeight) {
      if(wrap->method==CRESSMAN) {
	cmanHeightWeight = checkHeight*checkHeight - calcHeight*calcHeight;
	cmanHeightWeight/= checkHeight*checkHeight + calcHeight*calcHeight;
      }
      else if(wrap->method==UNIFORM) {
	cmanHeightWeight = 1L;
      }
      else {
	if(checkHeight!=0.0)
	  cmanHeightWeight = 1.0 - calcHeight/checkHeight;
	else
	  cmanHeightWeight = 0L;
      }
    }
    else {
      printf("ASSERTION FAILIURE IN getCressHeightWeight\n");
      printf("src->elevation = %lg, src->range=%lg\n",\
	     src->elevation*RAD_TO_DEG,src->range);
      exit(1);
    }
  }

  free(aCopy);

  return cmanHeightWeight;
}

static double getCressARWeight(int rangeIdx, int azIndex,\
			       double gamma,Position* src, \
			       double* xr,double* yr,TrafoWrapper3D* wrap)
{
  double arWeight;

  int rIndex;
  rIndex=rangeIdx/* + 1*/;

  if(gamma <= wrap->beamBroad/2 &&
     rIndex*wrap->inscale <= src->range &&
     (rIndex+1)*wrap->inscale >= src->range){
    /*The current pixel is surrounding the wanted point*/
    (*xr)=0L;
    (*yr)=0L;
  }
  else if(gamma > wrap->beamBroad/2 &&
	  rIndex*wrap->inscale <= src->range &&
	  (rIndex+1)*wrap->inscale >= src->range) {
    /*The current pixel is to the left or right of the wanted pixel*/
    (*yr)=0L;
    (*xr)=src->range*gamma;
  }
  else if(gamma <= wrap->beamBroad/2 &&
	  (rIndex+1)*wrap->inscale<src->range) {
    /*The current pixel is below the wanted pixel*/
    (*xr)=0L;
    (*yr)=src->range - (rIndex+1/2)*wrap->inscale;
  }
  else if (gamma <= wrap->beamBroad/2 &&
	   rIndex*wrap->inscale >src->range) {
    /*The current pixel is above the wanted pixel*/
    (*xr)=0L;
    (*yr) = (rIndex+1/2)*wrap->inscale - src->range;
  }
  else {
    /*Either above or below and right or left*/

    if(rIndex*wrap->inscale > src->range ) {
      /*Above*/
      (*xr) = src->range * gamma;
      (*yr) = (rIndex+1/2)*wrap->inscale - (src->range*cos(gamma));
    }
    else {
      /*Below*/
      (*xr) = src->range * gamma;
      (*yr) = src->range*cos(gamma) - (rIndex+1/2)*wrap->inscale;
    }
  }

  arWeight = 1L;

  return arWeight;
}

static int handleCressmanOrigo(CoordWeight* staticWeight,Position* src,
			double azOffset, int maxRange, TrafoWrapper3D* wrap)
{
  int maxElev, maxAzim;
  int sa, se, sr;
  int widx=0;

  double zr,r,R;
  double heightWeight,checkHeight,gamma;
  double storedLen = 100000L;

  Position target;
  Position* srcCopy;

  srcCopy = copyPosStruct(src);

  if(wrap->slice == PCAPPI) {
    srcCopy->elevation = wrap->elev[wrap->nelev - 1];
    deToRh(srcCopy,srcCopy);
  }

  dhToRe(srcCopy,&target);

  checkHeight = getCheckHeight(srcCopy,&target,wrap);

  maxElev = wrap->nelev;
  maxAzim = wrap->inymax;

  R = wrap->R;

  for(se=0;se<maxElev;se++) {

    heightWeight = getCressHeightWeight(se,srcCopy,&target,checkHeight,&zr,wrap);

    if(heightWeight == 0L) {
      if(wrap->slice == CAPPI)
	continue;
      else {
	/* For PPI handling, usually PCAPPI
	 * If the elevation is outside the current ppi(s) both max and min
	 * fetch the nearest PPI and set the current hight to that one
	 * instead, then calculate the heightWeight with this information
	 * instead.
	 */
	srcCopy->elevation = wrap->elev[getElevIndex(srcCopy->elevation,wrap)];
	deToRh(srcCopy,&target);
	srcCopy->alt = target.alt;
	checkHeight = getCheckHeight(srcCopy,&target,wrap);
	heightWeight = getCressHeightWeight(se,srcCopy,&target,checkHeight,&zr,wrap);
      }
    }

    for(sa=0;sa<maxAzim;sa++) {
      /*
       *Decide the angle between the src->azimuth
       *and the current indexed azimuth
       */
      gamma = sa*azOffset - srcCopy->azimuth;
      if(gamma < 0)
	gamma += 2*M_PI;

      for(sr=0;sr<=maxRange;sr++) {
	r = srcCopy->range*srcCopy->range + (sr+1/2)*wrap->inscale*(sr+1/2)*wrap->inscale;
	r = r - 2*srcCopy->range*(sr+1/2)*wrap->inscale*cos(gamma);
	r = sqrt(r);

	if(r < R && heightWeight!=0L) {
	  staticWeight[widx].elev=se;
	  staticWeight[widx].azimuth = sa;
	  staticWeight[widx].range = sr;
	  if(wrap->method==CRESSMAN)
	    staticWeight[widx].weight = sqrt(((R*R - r*r)/(R*R+r*r))*heightWeight);
	  else if(wrap->method==UNIFORM)
	    staticWeight[widx].weight=1L;
	  else
	    staticWeight[widx].weight=(1.0 - r/R)*heightWeight;

	  widx++;
	}
	else {
	  r = sqrt(r*r + zr*zr);
	  if(r < storedLen) {
	    staticWeight[0].elev=se;
	    staticWeight[0].azimuth = sa;
	    staticWeight[0].range = sr;
	    staticWeight[0].weight = 1.0;
	    storedLen = r;
	  }
	}
      }
    }
  }

  if(storedLen!=100000L && widx == 0)
    widx = 1;

  free(srcCopy);

  return widx;
}

static int handleCressmanOrigoSE(CoordWeight* staticWeight,Position* src,
			double azOffset, int maxRange, TrafoWrapper3D* wrap)
{
  int maxElev, maxAzim;
  int sa, se, sr;
  int widx=0;

  double zr,r,R;
  double heightWeight,checkHeight,gamma;
  double storedLen = 100000L;

  Position target;
  Position* srcCopy;

  srcCopy = copyPosStruct(src);

  if(wrap->slice == PCAPPI) {
    srcCopy->elevation = wrap->elev[wrap->nelev - 1];
    deToRh(srcCopy,srcCopy);
  }

  dhToRe(srcCopy,&target);

  se = wrap->height;

  maxAzim = wrap->inymax;

  R = wrap->R;

  for(sa=0;sa<maxAzim;sa++) {
    /*
     *Decide the angle between the src->azimuth
     *and the current indexed azimuth
     */
    gamma = sa*azOffset - srcCopy->azimuth;
    if(gamma < 0)
      gamma += 2*M_PI;

    for(sr=0;sr<=maxRange;sr++) {
      r = srcCopy->range*srcCopy->range + (sr+1/2)*wrap->inscale*(sr+1/2)*wrap->inscale;
      r = r - 2*srcCopy->range*(sr+1/2)*wrap->inscale*cos(gamma);
      r = sqrt(r);

      if(r < R) {
	staticWeight[widx].elev=se;
	staticWeight[widx].azimuth = sa;
	staticWeight[widx].range = sr;
	if(wrap->method==CRESSMAN)
	  staticWeight[widx].weight = sqrt(((R*R - r*r)/(R*R+r*r)));
	else if(wrap->method==UNIFORM)
	  staticWeight[widx].weight=1L;
	else
	  staticWeight[widx].weight=(1.0 - r/R);

	widx++;
      }
      else {
	r = sqrt(r*r + zr*zr);
	if(r < storedLen) {
	  staticWeight[0].elev=se;
	  staticWeight[0].azimuth = sa;
	  staticWeight[0].range = sr;
	  staticWeight[0].weight = 1.0;
	  storedLen = r;
	}
      }
    }
  }

  if(storedLen!=100000L && widx == 0)
    widx = 1;

  free(srcCopy);

  return widx;
}

static CoordWeight* getCressman(Position* src, int* weights, TrafoWrapper3D* wrap)
{
  int minmaxAzim[2], minmaxElev[2], minmaxRange[2];
  double azOffset;
  double yr,xr,zr,r,yprim;
  double gamma,OC,OT,R,minAzim,maxAzim;
  double arWeight;
  int sa,se,sr,widx,azIdx;
  int maxNoOfItems=0;

  Position* srcCopy;
  Position target;
  Position* secCopy;
  Position secTgt;

  double checkHeight;
  double heightWeight;

  int storedR=-1, storedE=-1, storedA=-1;

  double storedLen = 100000L;

  CoordWeight* staticWeight = NULL;

  int debug=0;
  int printedDebug=0;

  if(src->range > wrap->inxmax*wrap->inscale) {
    /*Out of bounds, just return*/
    (*weights)=0;
    return staticWeight;
  }

  /*debug=(src->azimuth*RAD_TO_DEG>359.5 || src->azimuth*RAD_TO_DEG<0.5)?0:0;*/

  azOffset = (360.0/wrap->inymax)*DEG_TO_RAD;

  if(src->distance <= wrap->R) { /*Wrap arround origo*/
    minmaxRange[1] = mytrunc(ceil(wrap->R/wrap->inscale));
    minmaxRange[1] *=2;
    minmaxRange[1] +=1;
    maxNoOfItems = wrap->inymax*wrap->nelev*(minmaxRange[1]);
    staticWeight = getAllocatedCW(maxNoOfItems);
    (*weights)=handleCressmanOrigo(staticWeight,src,azOffset,minmaxRange[1],wrap);
    return staticWeight;
  }

  srcCopy = copyPosStruct(src);

  OC = src->range;
  R = wrap->R;
  OT = sqrt(OC*OC+R*R);
  gamma = asin(R/OT);
  minAzim = src->azimuth - gamma;
  maxAzim = src->azimuth + gamma;

  getAdjAzimIdx(minmaxAzim,&minAzim,&maxAzim,wrap);

  dhToRe(srcCopy,&target);

  checkHeight = getCheckHeight(src,&target,wrap);

  getMinMaxElevation(minmaxElev, src, checkHeight, wrap);

  /*Determine min and max range index*/
  minmaxRange[0] = mytrunc(floor((src->range - wrap->R)/wrap->inscale)) - 1;
  minmaxRange[1] = mytrunc(ceil((src->range + wrap->R)/wrap->inscale)) - 1;

  if(minmaxRange[0] == -1)
     minmaxRange[0] = 0;

  if(minmaxRange[1] == -1)
     minmaxRange[1] = 0;

  maxNoOfItems = minmaxAzim[1] - minmaxAzim[0] + 1;
  maxNoOfItems *= minmaxRange[1] - minmaxRange[0] + 1;
  maxNoOfItems *= minmaxElev[1] - minmaxElev[0] + 1;

  staticWeight = getAllocatedCW(maxNoOfItems);

  widx = 0;

  for(se=minmaxElev[0];se<=minmaxElev[1];se++) {

    heightWeight = getCressHeightWeight(se,src,&target,checkHeight,&zr,wrap);

    if(heightWeight==0L) {
      if(wrap->slice == CAPPI &&
	 /*	 (src->elevation > wrap->elev[wrap->nelev-1]+wrap->beamBroad/2 ||
		 src->elevation < wrap->elev[0]-wrap->beamBroad/2)) {*/
	 (src->elevation > wrap->elev[wrap->nelev-1] ||
	  src->elevation < wrap->elev[0])) {
	continue;
      }
      else if(wrap->slice == PPI) {
	continue;
      }
      else if(wrap->slice == PCAPPI) {
	int regenerateHW = 1;

	secCopy = copyPosStruct(srcCopy);
	secCopy->alt = wrap->height;
	dhToRe(secCopy,&secTgt);

	/*if(secTgt.elevation > wrap->elev[wrap->nelev-1] + wrap->beamBroad/2) {*/
	if(secTgt.elevation > wrap->elev[wrap->nelev-1]) {
	  srcCopy->elevation = wrap->elev[wrap->nelev - 1];
	}
	/*else if (secTgt.elevation < wrap->elev[0] - wrap->beamBroad/2) {*/
	else if (secTgt.elevation < wrap->elev[0]) {
	  srcCopy->elevation = wrap->elev[0];
	}
	else {
	  regenerateHW = 0;
	}

	if(regenerateHW) {
	  double storedCheckHeight=checkHeight;
	  deToRh(srcCopy,&target);
	  srcCopy->alt = target.alt;
	  checkHeight = getCheckHeight(srcCopy,&target,wrap);
	  heightWeight = getCressHeightWeight(se,srcCopy,&target,checkHeight,&zr,wrap);

	  if(debug && !printedDebug) {
	     printf("%f\t%f\t%f\t%f\n",\
		    src->distance,srcCopy->alt,\
		    (srcCopy->alt-checkHeight),\
		    (srcCopy->alt+checkHeight));
	     printedDebug=1;
	  }

	  checkHeight = storedCheckHeight;

	}

	free(secCopy);
      }
      else {
	continue;
      }
    }

    for(sa=minmaxAzim[0];sa<=minmaxAzim[1];sa++) {

      if(maxAzim > minAzim) {
	gamma = src->azimuth - sa*azOffset;
	azIdx = sa;
	if(azIdx >= wrap->inymax)
	  azIdx = azIdx%wrap->inymax;
	if(gamma<0)
	  gamma = -gamma;
      }
      else if(sa >= wrap->inymax) {
	azIdx = sa%wrap->inymax;
	if(src->azimuth > M_PI) {
	  gamma = src->azimuth - sa*azOffset;
	}
	else {
	  gamma = src->azimuth - azIdx*azOffset;
	}
	if(gamma < 0)
	  gamma = -gamma;
      }
      else {
	gamma = src->azimuth - sa*azOffset;
	azIdx = sa;
	if(gamma<0)
	  gamma+=2.0*M_PI;
      }

      for(sr=minmaxRange[0];sr<=minmaxRange[1];sr++) {
	/* This can be optimized, when all azimuth/range
	 * parts has been calculated for one elevation, the same values
	 * will be calculated for the rest, this means that cashing
	 * the first calculated values might be quite a good idea ;-)
	 */
	arWeight = getCressARWeight(sr,azIdx,gamma,src,&xr,&yr,wrap);

	r = sqrt(xr*xr+yr*yr);

	if(r<R && arWeight!=0L && heightWeight!=0L) {
	  staticWeight[widx].elev = se;
	  staticWeight[widx].azimuth = azIdx;
	  staticWeight[widx].range = sr;
	  if(wrap->method==CRESSMAN) {
	    staticWeight[widx].weight = (R*R - r*r)/(R*R+r*r)*arWeight;
	  }
	  else if(wrap->method==UNIFORM){
	    staticWeight[widx].weight = 1L;
	  }
	  else {
	    staticWeight[widx].weight = 1.0 - r/R;
	  }

	  staticWeight[widx].weight = sqrt(staticWeight[widx].weight*heightWeight);

	  if(staticWeight[widx].range>=wrap->inxmax)
	    continue;

	  widx++;
	}
	else {
	  r = sqrt(r*r+zr*zr);

	  if( r < storedLen) {
	    storedR=sr;
	    storedA=azIdx;
	    storedE=se;

	    storedLen=r;
	  }
	}
      }
    }
  }

  if(debug) {
     if(!printedDebug) {
	printf("%f\t%f\t%f\t%f\n",\
	       src->distance,srcCopy->alt,\
	       (srcCopy->alt-checkHeight),\
	       (srcCopy->alt+checkHeight));
	printedDebug=1;
     }
  }

  if(widx==0 && storedR!=-1) {
    staticWeight[0].elev = storedE;
    staticWeight[0].azimuth = storedA;
    staticWeight[0].range = storedR;
    staticWeight[0].weight = 1.0; /*sqrt(storedHW*storedARWeight);*/
    widx = 1;
  }

  free(srcCopy);

  (*weights)=widx;
  return staticWeight;
}

/*
 * Get cressman with single elevation, only reasonable to use
 * for PPI
 */
static CoordWeight* getCressmanSE(Position* src, int* weights, TrafoWrapper3D* wrap)
{
  int minmaxAzim[2], minmaxElev[2], minmaxRange[2];
  double azOffset;
  double yr,xr,zr,r,yprim;
  double gamma,OC,OT,R,minAzim,maxAzim;
  double arWeight;
  int sa,se,sr,widx,azIdx;
  int maxNoOfItems=0;

  Position* srcCopy;
  Position target;
  Position* secCopy;
  Position secTgt;

  double checkHeight;
  double heightWeight;

  int storedR=-1, storedE=-1, storedA=-1;

  double storedLen = 100000L;

  CoordWeight* staticWeight = NULL;

  if(src->range > wrap->inxmax*wrap->inscale) {
    /*Out of bounds, just return*/
    (*weights)=0;
    return staticWeight;
  }

  azOffset = (360.0/wrap->inymax)*DEG_TO_RAD;

  if(src->distance <= wrap->R) { /*Wrap arround origo*/
    minmaxRange[1] = mytrunc(ceil(wrap->R/wrap->inscale));
    minmaxRange[1] *=2;
    minmaxRange[1] +=1;
    maxNoOfItems = wrap->inymax*(minmaxRange[1]);
    staticWeight = getAllocatedCW(maxNoOfItems);
    (*weights)=handleCressmanOrigoSE(staticWeight,src,azOffset,minmaxRange[1],wrap);
    return staticWeight;
  }

  srcCopy = copyPosStruct(src);

  OC = src->range;
  R = wrap->R;
  OT = sqrt(OC*OC+R*R);
  gamma = asin(R/OT);
  minAzim = src->azimuth - gamma;
  maxAzim = src->azimuth + gamma;

  getAdjAzimIdx(minmaxAzim,&minAzim,&maxAzim,wrap);

  dhToRe(srcCopy,&target);

  free(srcCopy);

  checkHeight = getCheckHeight(src,&target,wrap);

  /*  getMinMaxElevation(minmaxElev, src, checkHeight, wrap);*/
  se=minmaxElev[0]=minmaxElev[1]=wrap->height;

  /*Determine min and max range index*/
  minmaxRange[0] = mytrunc(floor((src->range - wrap->R)/wrap->inscale)) - 1;
  minmaxRange[1] = mytrunc(ceil((src->range + wrap->R)/wrap->inscale)) - 1;

  if(minmaxRange[0] == -1)
     minmaxRange[0] = 0;

  if(minmaxRange[1] == -1)
     minmaxRange[1] = 0;

  maxNoOfItems = minmaxAzim[1] - minmaxAzim[0] + 1;
  maxNoOfItems *= minmaxRange[1] - minmaxRange[0] + 1;

  staticWeight = getAllocatedCW(maxNoOfItems);

  widx = 0;

  for(sa=minmaxAzim[0];sa<=minmaxAzim[1];sa++) {

    if(maxAzim > minAzim) {
      gamma = src->azimuth - sa*azOffset;
      azIdx = sa;
      if(azIdx >= wrap->inymax)
	azIdx = azIdx%wrap->inymax;
      if(gamma<0)
	gamma = -gamma;
    }
    else if(sa >= wrap->inymax) {
      azIdx = sa%wrap->inymax;
      if(src->azimuth > M_PI) {
	gamma = src->azimuth - sa*azOffset;
      }
      else {
	gamma = src->azimuth - azIdx*azOffset;
      }
      if(gamma < 0)
	gamma = -gamma;
    }
    else {
      gamma = src->azimuth - sa*azOffset;
      azIdx = sa;
      if(gamma<0)
	gamma+=2.0*M_PI;
    }

    for(sr=minmaxRange[0];sr<=minmaxRange[1];sr++) {
      /* This can be optimized, when all azimuth/range
       * parts has been calculated for one elevation, the same values
       * will be calculated for the rest, this means that cashing
       * the first calculated values might be quite a good idea ;-)
       */
      arWeight = getCressARWeight(sr,azIdx,gamma,src,&xr,&yr,wrap);

      r = sqrt(xr*xr+yr*yr);

      if(r<R && arWeight!=0L) {
	staticWeight[widx].elev = se;
	staticWeight[widx].azimuth = azIdx;
	staticWeight[widx].range = sr;
	if(wrap->method==CRESSMAN) {
	  staticWeight[widx].weight = (R*R - r*r)/(R*R+r*r)*arWeight;
	}
	else if(wrap->method==UNIFORM){
	  staticWeight[widx].weight = 1L;
	}
	else {
	  staticWeight[widx].weight = 1.0 - r/R;
	}

	if(staticWeight[widx].range>=wrap->inxmax)
	  continue; /*Dont take this range in account*/

	widx++;
      }
      else {
	r = sqrt(r*r+zr*zr);

	if( r < storedLen) {
	  storedR=sr;
	  storedA=azIdx;
	  storedE=se;
	  storedLen=r;
	}
      }
    }
  }


  if(widx==0 && storedR!=-1) {
    staticWeight[0].elev = storedE;
    staticWeight[0].azimuth = storedA;
    staticWeight[0].range = storedR;
    staticWeight[0].weight = 1.0; /*sqrt(storedHW*storedARWeight);*/
    widx = 1;
  }

  (*weights)=widx;
  return staticWeight;

}

#ifdef OLD_ONE
static CoordWeight* getCressmanSE(Position* src, int* weights, TrafoWrapper3D* wrap)
{
  int noOfItems, elevIdx,maxNoOfItems;
  double azOffset;
  double gamma,OC,R,minAzim,maxAzim,OT;

  double yr,xr,r,yprim;

  int widx, sa,sr,i,azIdx;

  int maxNoOfAzimuths;

  int minmaxAzim[2], minmaxRange[2];

  int debug = 0;

  /*Static members, used for holding enough space*/
  static CoordWeight* staticWeight = NULL;
  static int noOfWeights = 0;

  if(src->range > wrap->inxmax*wrap->inscale) {
    /*Out of bounds, just return*/
    (*weights)=0;
    return staticWeight;
  }

  azOffset = (360.0/wrap->inymax)*DEG_TO_RAD;

  elevIdx = getElevIndex(wrap->height,wrap);

#ifdef DEBUG_TRANS
  if(src->range > 233287.0 && src->range < 233287.9)
    debug = 1;
#endif

  if(src->range <= wrap->R) {
    double x1,y1,x2,y2;

    widx = 0;

    minmaxRange[1] = mytrunc(ceil(wrap->R/wrap->inscale));
    maxNoOfItems = wrap->inymax*mytrunc(minmaxRange[1]+1);

    if(maxNoOfItems>noOfWeights) {
      if(staticWeight)
	free(staticWeight);
      staticWeight = malloc(sizeof(CoordWeight)*maxNoOfItems);
      noOfWeights = maxNoOfItems;
    }

    for(sa=0;sa<wrap->inymax;sa++) {
      for(sr=0;sr<minmaxRange[1];sr++) {
	x1 = sin(sa*azOffset)*sr;
	y1 = cos(sa*azOffset)*sr;
	x2 = src->range*sin(src->azimuth);
	y2 = src->range*cos(src->azimuth);
	r = sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
	if(r < wrap->R) {
	  staticWeight[widx].elev = elevIdx;
	  staticWeight[widx].azimuth = sa;
	  staticWeight[widx].range = sr;
	  staticWeight[widx].weight = (wrap->R*wrap->R -r*r)/(wrap->R*wrap->R + r*r);
	  widx++;
	}
      }
    }
    (*weights) = widx;
    return staticWeight;
  }

  OC = src->range;
  R=wrap->R;
  gamma = asin(R/OC);
  minAzim = src->azimuth - gamma;
  maxAzim = src->azimuth + gamma;

  if(minAzim < 0)
    minAzim+=2*M_PI;

  minmaxAzim[0] = mytrunc(minAzim/azOffset);
  minmaxAzim[1] = mytrunc(maxAzim/azOffset);

  minmaxRange[0] = mytrunc(floor((src->range - wrap->R)/wrap->inscale));
  minmaxRange[1] = mytrunc(floor((src->range + wrap->R)/wrap->inscale));

  if(minmaxAzim[0] > minmaxAzim[1]){
    maxNoOfItems = wrap->inymax - minmaxAzim[0];
    maxNoOfItems += minmaxAzim[1];
  }
  else {
    maxNoOfItems = minmaxAzim[1]-minmaxAzim[0]+1;
  }

  maxNoOfItems *= minmaxRange[1] - minmaxRange[0] + 1;

  if(maxNoOfItems>noOfWeights) {
    if(staticWeight)
      free(staticWeight);
    staticWeight=malloc(sizeof(CoordWeight)*maxNoOfItems);
    noOfWeights = maxNoOfItems;
  }

  /* Reset counters */
  widx = 0;

  if(minmaxAzim[0] > minmaxAzim[1]) {
    maxNoOfAzimuths = wrap->inymax - minmaxAzim[0];
    maxNoOfAzimuths += minmaxAzim[1];
    minmaxAzim[1] = minmaxAzim[0] + maxNoOfAzimuths;
  }

  for(sa=minmaxAzim[0];sa<=minmaxAzim[1];sa++) {
    if(maxAzim > minAzim) {
      if(sa >= wrap->inymax)
	azIdx = sa%wrap->inymax;
      else
	azIdx = sa;

      gamma = src->azimuth - azIdx*azOffset;

      if(gamma<0)
	gamma = -gamma;
    }
    else if(sa >= wrap->inymax) {
      azIdx = sa%wrap->inymax;
      gamma = src->azimuth - azIdx*azOffset;
      if(gamma<0)
	gamma = -gamma;
    }
    else {
      if(sa >= wrap->inymax)
	azIdx = sa%wrap->inymax;
      else
	azIdx = sa;

      gamma = src->azimuth -azIdx*azOffset;

      if(gamma<0)
	gamma+=2*M_PI;
    }

    for(sr=minmaxRange[0];sr<=minmaxRange[1];sr++) {
      double A,B;

      A = src->range;
      B = sr*wrap->inscale+1000.0; /*Adjust to get in midle of range*/
      r=sqrt(A*A + B*B -2*A*B*cos(gamma));

      if(r<=R) {
	staticWeight[widx].elev = elevIdx;
	staticWeight[widx].range = sr;
	staticWeight[widx].azimuth = azIdx;
	staticWeight[widx].weight = (R*R-r*r)/(R*R+r*r);
	widx++;
      }
    }
  }

  (*weights) = widx;
  return staticWeight;
}
#endif

static CoordWeight* getCressmanCW(UV coord, int* noOfItems, TrafoWrapper3D* wrap)
{
  Position source;
  int tmp;
  CoordWeight* tmpcw;

  source.lon0 = wrap->lon0;
  source.lat0 = wrap->lat0;
  source.alt0 = wrap->alt0;
  source.lon = coord.u;
  source.lat = coord.v;

  source.alt = wrap->height;
  source.dndh = wrap->dndh;

  wrap->R = wrap->cressmanR_xy*wrap->inscale;

  llToDa(&source,&source);

  if(wrap->slice == PPI) {
    /* if slice is PPI, the elevation is known, calculate the range */
    source.elevation = wrap->elev[mytrunc(wrap->height)];
    deToRh(&source,&source);
  }
  else {
    dhToRe(&source,&source);
  }

  if(wrap->elevUsage == USE_SINGLE_ELEV && wrap->slice==PPI) {
    tmpcw = getCressmanSE(&source,noOfItems,wrap);
  } else {
    tmpcw = getCressman(&source,noOfItems,wrap);
  }

  return tmpcw;
}

static int getCoordWeights(UV coord, CoordWeight* weight, TrafoWrapper3D* wrap)
{
  Position source;
  int tmp;

  source.lon0 = wrap->lon0;
  source.lat0 = wrap->lat0;
  source.alt0 = wrap->alt0;
  source.lon = coord.u;
  source.lat = coord.v;

  source.alt = wrap->height;
  source.dndh = wrap->dndh;

  wrap->R = wrap->cressmanR_xy*wrap->inscale;

  llToDa(&source,&source);

  if(wrap->slice == PPI) {
    /* if slice is PPI, the elevation is known, calculate the range */
    source.elevation = wrap->elev[mytrunc(wrap->height)];
    deToRh(&source,&source);
  }
  else {
    dhToRe(&source,&source);
  }

  switch (wrap->method) {
  case NEAREST:
    tmp = getNearest(&source,weight,wrap);
    return tmp;
    break;
  case BILINEAR:
    return getBilinear(&source, weight, wrap);
    break;
  case CUBIC:
    tmp = getCubic(&source,weight,wrap);
    return tmp;
    break;
  case CRESSMAN:
  default:
    printf("Oops: nyi %d", wrap->method);
    return 0;
    break;
  }
}

static void storeCash(CoordWeight* cw, int x, int y, int noi)
{
  fwrite(&x,sizeof(x),1,staticFD);
  fwrite(&y,sizeof(y),1,staticFD);
  fwrite(&noi,sizeof(noi),1,staticFD);
  fwrite(cw,sizeof(cw[0]),noi,staticFD);
}

/*Method for comparing doubles, used by qsort.*/
static int compareDoubles(const void* a,const void* b)
{
  if( (*(double*)a) < (*(double*)b) ) {
    return -1;
  }
  else if( (*(double*)a) > (*(double*)b)) {
    return 1;
  }
  else {
    return 0;
  }
}

static double sumWeights(CoordWeight* cw, int n, int* noofitems,TrafoWrapper3D* tw)
{
  int i;
  int noi=0;
  double ret=0.0;
  double item;
  for(i=0;i<n;i++) {
    item=getarritem3d(cw[i].elev,cw[i].range,cw[i].azimuth,tw);
    if(item!=tw->nodata) {
      if(tw->useWeights==NO_ZERO_WEIGHTS && item==0.0)
	continue;

      ret+=item;
      noi++;
    }
  }
  (*noofitems)=noi;
  return ret;
}

static double getMedian(CoordWeight* cw, int n, int* noofitems, TrafoWrapper3D* tw)
{
  double* values;
  int i;
  int midpoint=-1;
  double meanvalue=0.0;

  int minidx=0, maxidx=0;

  values=malloc(sizeof(double)*n);
  for(i=0;i<n;i++) {
    values[i]=getarritem3d(cw[i].elev,cw[i].range,cw[i].azimuth,tw);
  }

  qsort(values,n,sizeof(double),compareDoubles);

  if(tw->useWeights==NO_ZERO_WEIGHTS) {
    for(i=0;i<n;i++) {
      if(values[i]!=0.0) {
	minidx=i;
	break;
      }
    }
  }
  for(i=n-1;i>=0;i--) {
    if(values[i]!=tw->nodata) {
      maxidx=i;
      break;
    }
  }

  (*noofitems)=maxidx-minidx;
  midpoint=(int)minidx+((*noofitems)/2);

  if(midpoint==-1) {
    free(values);
    return 0.0;
  }
  else {
    meanvalue=values[midpoint];
  }

  free(values);

  if( (*noofitems)==0)
    return 0.0;

  return meanvalue;

}

static double calculateStdDeviation(CoordWeight* cw,int n,TrafoWrapper3D* tw)
{
  double sum=0.0;
  int noitems=0;
  double meanvalue=0.0;
  double deviation=0.0;
  double item;
  double minvalue,maxvalue;
  double refr;
  int i;
  int noofhits=0;
  double v=0.0;
  int totalnitems=0;

  if(tw->iqc==STD_DEVIATION_MEDIAN) {
    meanvalue=getMedian(cw,n,&noitems,tw);

    if(noitems==0)
      return 0.0; /*Hmm, no hits, better leave*/
  }
  else if(tw->iqc==STD_DEVIATION_MEANVALUE) {
    sum=sumWeights(cw,n,&noitems,tw);

    if(noitems==0)
      return 0.0; /*Hmm, no hits, better return*/

    meanvalue=sum/noitems;
  }
  else {
    printf("Wow,  Deviation alg, not supp, ABORTING IN ptoc::calculateStdDeviation\n");
    abort();
  }

  /*Calculate deviation*/
  for(i=0;i<n;i++) {
    item=getarritem3d(cw[i].elev,cw[i].range,cw[i].azimuth,tw);
      if(item!=tw->nodata &&
	 ((tw->useWeights==NO_ZERO_WEIGHTS&&item!=0.0)||
	  tw->useWeights==ALL_WEIGHTS)) {
	deviation+=(item-meanvalue)*(item-meanvalue);
      }
  }

  deviation=sqrt(deviation/noitems);

  minvalue=meanvalue-deviation;
  maxvalue=meanvalue+deviation;

  sum=0.0;

  /*Calculate the sum of weights so that all weights added with
   *each other will result in the sum 1.
   */
  /*The sum of all weights will be based on:
   * 1) All positions where the refraction of the position not is nodata
   * 2) The refraction of the position has to reside within the meanvalue
   *    +/- the standard deviation
   * 3) If NO_ZERO_WEIGHTS flag is set, then no refraction == 0.0 will
   *    Be used either.
   */

  for(i=0;i<n;i++) {
    if((cw[i].weight != 0.0) &&
       ((refr=getarritem3d(cw[i].elev,cw[i].range,cw[i].azimuth,tw))!=tw->nodata) &&
       ( (tw->useWeights == NO_ZERO_WEIGHTS && refr != 0.0) ||
	 tw->useWeights == ALL_WEIGHTS)) {
      if(refr>=minvalue && refr<=maxvalue) {
	sum+=cw[i].weight;
      }
      totalnitems++;
    }
  }

  if(tw->useWeights == NO_ZERO_WEIGHTS && sum == 0.0) {
    v=0.0;
  }
  else {
    for(i=0;i<n;i++) {
      if(cw[i].weight!=0.0)
	item=getarritem3d(cw[i].elev,cw[i].range,cw[i].azimuth,tw);
      else
	item=tw->nodata;

      if(item!=tw->nodata && item >= minvalue && item <=maxvalue) {
	if(tw->useWeights==NO_ZERO_WEIGHTS && item!=0.0 ||
	   tw->useWeights==ALL_WEIGHTS) {
	  v+=item*cw[i].weight/sum;
	  noofhits++;
	}
      }
    }
  }

  if(tw->iqcvalue!=-1 && totalnitems!=0) {
    double percentage;
    percentage=(double)noofhits/totalnitems;
    if(percentage<=tw->iqcvalue) {
      v=0.0;
    }
  } else if(totalnitems==0) {
    v=0.0;
  }

  return v;
}

static int checkCompactness(int r,int a,int e,TrafoWrapper3D* tw,int debug)
{
   /*Aha, check the surrounding pixels*/
   /*If r=1,a=7,e=2 =>
    * check r=1, a=7, e=1
    * check r=1, a=7, e=3
    * check r=1, a=6, e=2
    * check r=1, a=8, e=2
    * check r=0, a=7, e=2
    * check r=2, a=7, e=2
    */
   int minr=r-1;
   int maxr=r+1;
   int mine=e-1;
   int maxe=e+1;
   int mina=a-1;
   int maxa=a+1;
   double val;
   int noofzeros=0;

   /*   int debug=(r>90&&r<100&&a>140&&a<160)?1:0;*/

   val=getarritem3d(e,minr,a,tw);
   noofzeros+=( val==0 || val==tw->nodata)?1:0;

   if(debug) {
      printf("MINR: val=%f,noofzeros=%d\n",val,noofzeros);
   }

   val=getarritem3d(e,maxr,a,tw);
   noofzeros+=( val==0 || val==tw->nodata)?1:0;
   if(debug) {
      printf("MAXR: val=%f,noofzeros=%d\n",val,noofzeros);
   }

   if(mine>=0) {
      val=getarritem3d(mine,r,a,tw);
      noofzeros+=( val==0 || val==tw->nodata)?1:0;
      if(debug)
	 printf("MINE: val=%f,noofzeros=%d\n",val,noofzeros);
   }

   if(maxe<tw->nelev) {
      val=getarritem3d(maxe,r,a,tw);
      noofzeros+=( val==0 || val==tw->nodata)?1:0;
      if(debug)
	 printf("MAXE: val=%f,noofzeros=%d\n",val,noofzeros);
   }

   if(mina<0) {
      mina=tw->inymax-1;
   }
   val=getarritem3d(e,r,mina,tw);
   noofzeros+=( val==0 || val==tw->nodata)?1:0;
   if(debug)
      printf("MINA: val=%f,noofzeros=%d\n",val,noofzeros);

   if(maxa>=tw->inymax) {
      maxa=0;
   }
   val=getarritem3d(e,r,maxa,tw);
   noofzeros+=( val==0 || val==tw->nodata)?1:0;
   if(debug)
      printf("MAXA: val=%f,noofzeros=%d\n",val,noofzeros);

   return noofzeros;
}


static void useCash(TrafoWrapper3D* tw)
{
  FILE* f;
  int x,y,n,i,noe;
  double tmpelist[MAXELEV];
  CoordWeight* cw;

  int acta,actr,acte;

  if(!(f=fopen(tw->cashfile,"r"))) {
    return;
  }

  if(!feof(f)) {
    int tempint;
    double tempdouble;

    /*Remove header from cash file*/

    fread(&noe,sizeof(noe),1,f); /*No of elevations*/
    fread(tmpelist,sizeof(tmpelist[0]),noe,f); /*Elevation list*/
    fread(&tempint,sizeof(tempint),1,f); /*Range uppb*/
    fread(&tempint,sizeof(tempint),1,f); /*Azimuth uppb*/
    fread(&tempdouble,sizeof(tempdouble),1,f); /*Range size*/
    fread(&tempdouble,sizeof(tempdouble),1,f); /*cressmanR_xy*/
    fread(&tempdouble,sizeof(tempdouble),1,f); /*cressmanR_z*/
  }

  while(!feof(f)) {
    fread(&x, sizeof(x), 1,f);
    fread(&y, sizeof(y), 1,f);
    fread(&n, sizeof(n), 1,f);
    cw = getAllocatedCW(n);
    fread(cw, sizeof(cw[0]), n, f);
    if (n) {
      double v=0;
      double sum=0, item;
      double refr;

#ifdef OLD_CHECK_NEAREST
      if(tw->check_nearest) {
	 Position source;
	 UV here_s,here;
	 /* printf("Working with x=%d,y=%d\n",x,y); */
	 here_s.v = (tw->outUL.v-tw->outyscale*y);
	 here_s.u = (tw->outUL.u+tw->outxscale*x);
	 here = pj_inv(here_s, tw->outpj);
	 /* printf("Got pjinv\n"); */

	 source.lon0 = tw->lon0;
	 source.lat0 = tw->lat0;
	 source.alt0 = tw->alt0;
	 /* printf("Got lon/lat/alt\n"); */

	 source.lon = here.u;
	 source.lat = here.v;
	 source.alt = tw->height;
	 source.dndh = tw->dndh;
	 llToDa(&source,&source);
	 /* printf("Got lltoda\n"); */
	 acta=getAzindex(source.azimuth, tw, ROUND_AZIMUTH);
	 if(tw->slice==PPI) {
	    source.elevation=tw->elev[mytrunc(tw->height)];
	    deToRh(&source,&source);
	 } else {
	    dhToRe(&source,&source);
	 }
	 actr=mytrunc(source.range/tw->inscale)-1;
	 actr=(actr<0)?0:actr;
	 acte=getElevIndex(source.elevation, tw);
	 /* printf("a=%d,e=%d,r=%d\n",acta,acte,actr); */
      }
#endif

      for (i=0; i<n; i++) {
	 if ( (cw[i].weight != 0.0) &&
	      (refr=getarritem3d(cw[i].elev,cw[i].range, \
				 cw[i].azimuth,tw)) != tw->nodata) {

	    if( (tw->useWeights == NO_ZERO_WEIGHTS && refr != 0.0) ||
		tw->useWeights == ALL_WEIGHTS)
	       sum +=cw[i].weight;
	 }
      }

      if(tw->useWeights == NO_ZERO_WEIGHTS && sum == 0.0) {
	 /*
	  *Ouch, all possible points has got zero refraction, set
	  *v to refraction 0
	  */
	 v=0.0;
      }
      else {
	 if(tw->iqc==STD_DEVIATION_MEANVALUE ||
	    tw->iqc==STD_DEVIATION_MEDIAN) {
	    v=calculateStdDeviation(cw,n,tw);
	 } else {
	    for(i=0;i<n;i++) {
	       if(cw[i].weight!=0.0)
		  item=getarritem3d(cw[i].elev,cw[i].range,cw[i].azimuth,tw);
	       else
		  item=tw->nodata;
	       if(item!=tw->nodata)
		  v+=item*cw[i].weight/sum;
	    }
	 }
      }

      if(tw->set_compactness==1) {
	 double cval,cval2;
	 int noofv=0;
	 int ee,aa,rr;
	 int noofzeros=0;

	 for(i=0;i<n;i++) {
	    cval=getarritem3d(cw[i].elev,cw[i].range,cw[i].azimuth,tw);
	    if(cval!=0.0 && cval!=tw->nodata && cw[i].range>0) {
	       if(cw[i].elev==0 || cw[i].elev==tw->nelev-1) {
		  noofv+=5;
	       }
	       else {
		  noofv+=6;
	       }

	       noofzeros+=checkCompactness(cw[i].range,
					   cw[i].azimuth,cw[i].elev,tw,0);
	    }
	 }

	 if(noofv==0) {
	    v=0.0;
	 } else {
	    cval2=(double)noofzeros/(double)noofv;
	    v=cval2;
	 }
      }

      if(tw->check_nearest) {
	 int nidx=0;
	 double highw=0.0;
	 double aval;
	 for(i=0;i<n;i++) {
	    if(highw<cw[i].weight) {
	       nidx=i;
	       highw=cw[i].weight;
	    }
	 }
	 aval=getarritem3d(cw[nidx].elev,cw[nidx].range,cw[nidx].azimuth,tw);
	 if(aval==0 || aval==tw->nodata)
	    setarritem3d(x,y,aval,tw);
	 else
	    setarritem3d(x,y,v,tw);
      } else {
	 setarritem3d(x,y,v,tw);
      }
#ifdef OLD_CHECK_NEAREST
      if(tw->check_nearest) {
	 for(i=0;i<n;i++) {
	    if(cw[i].range==actr && cw[i].azimuth==acta) {
	       if(cw[i].elev<tw->nelev-1 && acte+1==cw[i].elev) {
		  double aval=getarritem3d(cw[i].elev,cw[i].range,cw[i].azimuth,tw);
		  if(aval==tw->nodata || aval==0)
		     setarritem3d(x,y,aval,tw);
		  else
		     setarritem3d(x,y,v,tw);
	       }
	    }
	 }
      } else {
	 setarritem3d(x,y,v,tw);
      }
#endif
    /*  setarritem3d(x,y,v,tw);*/
    }
  }

  fclose(f);
}

/* ----------------------------------------------------------------
   bilinear3d: interpolation of 4 nearest neighbours
*/
static void bilinear3d(int x, int y, UV here_s, TrafoWrapper3D *tw)
{
  int gx, gy, ge, n, i;
  UV here;
  CoordWeight cw[MAXCOORD];
  int show = (x==y)?0:0;

  /* inverse transform ps surface coords to long/lat */
  here = pj_inv(here_s, tw->outpj);


  n = getCoordWeights(here, cw, tw);

  /* Assign result if there was data - keep NO_DATA otherwise */
  if (n) {
    double v=0;
    double sum=0, item;

    for (i=0; i<n; i++) {
      if (getarritem3d(cw[i].elev,cw[i].range,\
		       cw[i].azimuth,tw) != tw->nodata)
		sum +=cw[i].weight;
    }
    for(i=0; i<n; i++) {
      item = getarritem3d(cw[i].elev,cw[i].range,cw[i].azimuth,tw);
      if (item != tw->nodata)
		v+=item*cw[i].weight/sum;
    }

    setarritem3d(x,y,v,tw);
  }
}

/* ----------------------------------------------------------------
   cubic3d: interpolation of 64 nearest neighbours
*/
static void cubic3d(int x, int y, UV here_s, TrafoWrapper3D *tw)
{
  int gx, gy, ge, n, i;
  UV here;
  CoordWeight cw[MAXCOORD];
  int show = (x==y)?0:0;

  /* inverse transform ps surface coords to long/lat */
  here = pj_inv(here_s, tw->outpj);

  /* Compute relevant coordinates with weight */
  n = getCoordWeights(here, cw, tw);


  /* Assign result if there was data - keep NO_DATA otherwise */
  if (n) {
    double v=0;
    double sum=0, item;

    for (i=0; i<n; i++) {
      if ( (cw[i].weight != 0.0) &&
		   getarritem3d(cw[i].elev,cw[i].range, \
						cw[i].azimuth,tw) != tw->nodata)
		sum +=cw[i].weight;
    }

    for(i=0; i<n; i++) {
      if(cw[i].weight!=0.0)
		item = getarritem3d(cw[i].elev,cw[i].range,cw[i].azimuth,tw);
      else
		item = tw->nodata;

      if (item != tw->nodata)
		v+=item*cw[i].weight/sum;
    }

    setarritem3d(x,y,v,tw);
  }
}

/* ----------------------------------------------------------------
   cressman3d: interpolation of 64 nearest neighbours
*/

static void cressman3d(int x, int y, UV here_s, TrafoWrapper3D *tw)
{
  int gx, gy, ge, n, i;
  UV here;
  CoordWeight* cw;
  /*  int show = (x==y)?1:0;*/
  int show = (x>118&&x<122&&y<120)?0:0;

  /*int acta,actr,acte;*/

  /* inverse transform ps surface coords to long/lat */
  here = pj_inv(here_s, tw->outpj);

  /* Compute relevant coordinates with weight */

  cw = getCressmanCW(here, &n, tw);

#ifdef OLD_CHECK_NEAREST
  if(tw->check_nearest) {
     Position source;
     source.lon0 = tw->lon0;
     source.lat0 = tw->lat0;
     source.alt0 = tw->alt0;
     source.lon = here.u;
     source.lat = here.v;
     source.alt = tw->height;
     source.dndh = tw->dndh;
     llToDa(&source,&source);
     acta=getAzindex(source.azimuth, tw, ROUND_AZIMUTH);
     if(tw->slice==PPI) {
	source.elevation=tw->elev[mytrunc(tw->height)];
	deToRh(&source,&source);
     } else {
	dhToRe(&source,&source);
     }
     actr=mytrunc(source.range/tw->inscale)-1;
     actr=(actr<0)?0:actr;
     acte=getElevIndex(source.elevation, tw);
  }
#endif

  if(tw->cashfile && n>0) {
     if(!ferror(staticFD))
	storeCash(cw,x,y,n);
  }

  /* Assign result if there was data - keep NO_DATA otherwise */
  if (n) {
    double v=0;
    double sum=0, item;

    double refr;

    for (i=0; i<n; i++) {
      if ( (cw[i].weight != 0.0) &&
	   (refr=getarritem3d(cw[i].elev,cw[i].range, \
			      cw[i].azimuth,tw)) != tw->nodata) {

	 if( (tw->useWeights == NO_ZERO_WEIGHTS && refr != 0.0) ||
	     tw->useWeights == ALL_WEIGHTS)
	    sum +=cw[i].weight;
      }
    }

    if(tw->useWeights == NO_ZERO_WEIGHTS && sum == 0.0) {
      /*
       *Ouch, all possible points has got zero refraction, set
       *v to refraction 0
       */
      v=0.0;
    }
    else {
      if(tw->iqc==STD_DEVIATION_MEANVALUE ||
	 tw->iqc==STD_DEVIATION_MEDIAN) {
	 v=calculateStdDeviation(cw,n,tw);
      } else {
	 for(i=0;i<n;i++) {
	    if(cw[i].weight!=0.0)
	       item=getarritem3d(cw[i].elev,cw[i].range,cw[i].azimuth,tw);
	    else
	       item=tw->nodata;
	    if(item!=tw->nodata)
	       v+=item*cw[i].weight/sum;
	 }
      }
    }
    if(tw->set_compactness==1) {
       double cval,cval2;
       int noofv=0;
       int ee,aa,rr;
       int noofzeros=0;
       for(i=0;i<n;i++) {
	  cval=getarritem3d(cw[i].elev,cw[i].range,cw[i].azimuth,tw);
	  if(cval!=0.0 && cval!=tw->nodata && cw[i].range>0) {
	     if(cw[i].elev==0 || cw[i].elev==tw->nelev-1)
		noofv+=5;
	     else
		noofv+=6;
	     /*noofv+=6;*/
	     /*Aha, check the surrounding pixels*/
	     noofzeros+=checkCompactness(cw[i].range,cw[i].azimuth,cw[i].elev,tw,0);
	  }
       }
       if(noofv==0) {
	  v=0.0;
       } else {
	  cval2=(double)noofzeros/(double)(noofv);
	  v=cval2;
       }
    }

    if(tw->check_nearest) {
       int nidx=0;
       double highw=0.0;
       double aval;
       for(i=0;i<n;i++) {
	  if(highw<cw[i].weight) {
	     nidx=i;
	     highw=cw[i].weight;
	  }
       }
       aval=getarritem3d(cw[nidx].elev,cw[nidx].range,cw[nidx].azimuth,tw);
       if(aval==0 || aval==tw->nodata)
	  setarritem3d(x,y,aval,tw);
       else
	  setarritem3d(x,y,v,tw);

    } else {
       setarritem3d(x,y,v,tw);
    }
#ifdef OLD_CHECK_NEAREST
    if(tw->check_nearest) {
       for(i=0;i<n;i++) {
	  if(cw[i].range==actr && cw[i].azimuth==acta) {
	     if(cw[i].elev<tw->nelev-1 && acte+1==cw[i].elev) {
		double aval=getarritem3d(cw[i].elev,cw[i].range,cw[i].azimuth,tw);
		if(aval==tw->nodata || aval==0)
		   setarritem3d(x,y,aval,tw);
		else
		   setarritem3d(x,y,v,tw);
	     }
	  }
       }
    } else {
       setarritem3d(x,y,v,tw);
    }
#endif
  }

}

/*
 *static int getEchotop(int* ctr, double* height, CoordWeight* cw,
 */

static int calcECwithCW(double* height,UV* here, CoordWeight* cw, int noitems,
			TrafoWrapper3D* wrap)
{
  int j,ctr=0;
  double refr=0.0;
  double sumHeight=0.0;

  for(j=0;j<noitems;j++) {
    if(cw[j].weight==0.0)
      continue;

    refr=getarritem3d(cw[j].elev,cw[j].range,cw[j].azimuth,wrap);
    if(refr!=0.0 && refr!=wrap->nodata)
      ctr++;
  }

  if(ctr!=0) {
    double a=ctr;
    double b=noitems;
    double perc=a/b;

    if(wrap->iqcvalue==-1.0 || perc>wrap->iqcvalue) {
      double heightSum=0.0;
      Position hSource,hTarget;
      hSource.lon0 = wrap->lon0;
      hSource.lat0 = wrap->lat0;
      hSource.alt0 = wrap->alt0;
      hSource.lon = here->u;
      hSource.lat = here->v;
      hSource.dndh = wrap->dndh;

      for(j=0;j<noitems;j++) {
	 if(cw[j].weight==0.0)
	    continue;

	 refr=getarritem3d(cw[j].elev,cw[j].range,cw[j].azimuth,wrap);
	 if(refr!=0.0 && refr!=wrap->nodata) {
	    hSource.elevation=wrap->elev[cw[j].elev];
	    hSource.range=wrap->inscale*cw[j].range;
	    reToDh(&hSource,&hTarget);
	    sumHeight+=(hTarget.alt-hSource.alt0);
	 }
      }

      sumHeight/=ctr;

      if(sumHeight>wrap->echomaxheight) {
	 ctr=-1;
	 /*	continue;*/
      }

      if(wrap->echoscale) {
	 sumHeight/=wrap->echoscale;
      }

      (*height)=sumHeight;
    }
  }

  return ctr;
}

static void calculateEchotop(int x, int y, UV here_s, TrafoWrapper3D* wrap)
{
  UV here;
  Position source;
  CoordWeight* cw;
  int noitems,ctr;
  int ai,ri,i;
  double height;
  double refr;
  double azOffset = (360.0/wrap->inymax)*DEG_TO_RAD;

  double modHeight=0.0;

  here=pj_inv(here_s,wrap->outpj);
  source.lon0 = wrap->lon0;
  source.lat0 = wrap->lat0;
  source.alt0 = wrap->alt0;
  source.lon = here.u;
  source.lat = here.v;
  source.dndh = wrap->dndh;

  llToDa(&source,&source);

  wrap->R = wrap->cressmanR_xy*wrap->inscale;

  ai=mytrunc(ceil(source.azimuth/azOffset));

  source.elevation=wrap->elev[0];
  deToRh(&source,&source);

  if(source.range > wrap->inxmax*wrap->inscale) {
    setarritem3d(x,y,wrap->echonodata,wrap);
    return;
  }

  for(i=wrap->nelev-1;i>=0;i--) {
    source.elevation=wrap->elev[i];
    deToRh(&source,&source);
    ri=mytrunc(source.range/wrap->inscale);
    refr=getarritem3d(i,ri,ai,wrap);

    if(wrap->echodecrtype==ET_LH_RCE) {
      modHeight=source.range*((wrap->beamBroad/2.0)*wrap->echodecrfactor);
      source.alt-=modHeight;
      dhToRe(&source,&source);
    }
    else if (wrap->echodecrtype==ET_LE_HALF) {
      source.elevation=wrap->elev[i]-((wrap->beamBroad/2.0)*wrap->echodecrfactor);
      deToRh(&source,&source);
    }
    else if (wrap->echodecrtype==ET_LH_HALF) {
      modHeight=source.range*((wrap->beamBroad/2.0)*wrap->echodecrfactor);;
      source.alt-=modHeight;
    }
    else if (wrap->echodecrtype==ET_NO_LOWER) {
      /*AHH DO NOTHING*/
    }

#ifdef HALF_BEAMWIDTH
    source.elevation = wrap->elev[i]-wrap->beamBroad/2.0;
    deToRh(&source,&source);
#endif

#ifdef OTHER_HALF_BEAMWIDTH
    modHeight=source.range*(wrap->beamBroad/2.0);
    source.alt-=modHeight;
    dhToRe(&source,&source); /*Guess it should be same behaviour as for CAPPI*/
#endif

    if(refr==0.0 || refr==wrap->nodata)
      continue;

    cw=getCressman(&source,&noitems,wrap);

    ctr=calcECwithCW(&height,&here,cw,noitems,wrap);

    if(ctr==-1) { /*Less hits that minimum*/
      continue;
    }

    if(ctr!=0) {
      setarritem3d(x,y,height,wrap);
      return;
    }
  }
  setarritem3d(x,y,0.0,wrap);

}

static int storeEchotop(FILE* fp, CoordWeight* cw, int x, int y, int e, int noi)
{
  if(fwrite(&x,sizeof(x),1,fp)<0)
    return 0;
  if(fwrite(&y,sizeof(y),1,fp)<0)
    return 0;
  if(fwrite(&e,sizeof(e),1,fp)<0)
    return 0;
  if(fwrite(&noi,sizeof(noi),1,fp)<0)
    return 0;
  if(fwrite(cw,sizeof(cw[0]),noi,fp)<0)
    return 0;

  return 1;
}

static int cashEchotopEntry(FILE* fp,int x, int y, UV here_s, TrafoWrapper3D *wrap)
{
  UV here;
  int i,j,ctr=0;
  int ai,ri;
  int noitems;
  Position source;
  CoordWeight* cw;
  double refr;
  double modHeight=0.0;
  double azOffset = (360.0/wrap->inymax)*DEG_TO_RAD;

  here = pj_inv(here_s, wrap->outpj);

  source.lon0 = wrap->lon0;
  source.lat0 = wrap->lat0;
  source.alt0 = wrap->alt0;
  source.lon = here.u;
  source.lat = here.v;
  source.dndh = wrap->dndh;

  llToDa(&source,&source);

  wrap->R = wrap->cressmanR_xy*wrap->inscale;

  ai=mytrunc(ceil(source.azimuth/azOffset));

  for(i=wrap->nelev-1;i>=0;i--) {
    source.elevation=wrap->elev[i];
    deToRh(&source,&source);
    ri=mytrunc(source.range/wrap->inscale);

    if(wrap->echodecrtype==ET_LH_RCE) {
      modHeight=source.range*((wrap->beamBroad/2.0)*wrap->echodecrfactor);
      source.alt-=modHeight;
      dhToRe(&source,&source);
    }
    else if (wrap->echodecrtype==ET_LE_HALF) {
      source.elevation=wrap->elev[i]-((wrap->beamBroad/2.0)*wrap->echodecrfactor);
      deToRh(&source,&source);
    }
    else if (wrap->echodecrtype==ET_LH_HALF) {
      modHeight=source.range*((wrap->beamBroad/2.0)*wrap->echodecrfactor);;
      source.alt-=modHeight;
    }
    else if (wrap->echodecrtype==ET_NO_LOWER) {
      /*AHH DO NOTHING*/
    }

#ifdef HALF_BEAMWIDTH
    source.elevation = wrap->elev[i]-wrap->beamBroad/2.0;
    deToRh(&source,&source);
#endif

    cw=getCressman(&source,&noitems,wrap);

    if(!storeEchotop(fp,cw,x,y,i,noitems)) {
      /*      printf("Failed writing data to file\n");*/
      return 0;
    }
  }

  return 1;
}

static int getEchotopCash(FILE* fp, int* x, int* y, int* e, int* n/*, CoordWeight** cw*/)
{
  if(fread(x, sizeof(*x), 1,fp)<0)
    return 0;
  if(fread(y, sizeof(*y), 1,fp)<0)
    return 0;
  if(fread(e, sizeof(*e), 1,fp)<0)
    return 0;
  if(fread(n, sizeof(*n), 1,fp)<0)
    return 0;
  return 1;
}

static int getEchotopCashedCW(FILE* fp, CoordWeight* cw, int noitems)
{
  if(fread(cw,sizeof(CoordWeight),noitems,fp)<0)
    return 0;
  return 1;
}


static int calcEchotopWithCash(FILE* fp,UV* outUL,double outyscale, double outxscale,\
			       TrafoWrapper3D* wrap)
{
  CoordWeight* cw;
  Position source;
  int noitems,ctr;
  int ai,ri,i;
  int x,y,e;
  int sx=0,sy=0;
  double height;
  double refr;
  int columnhandled=0;
  UV place_s;
  UV place;
  double azOffset = (360.0/wrap->inymax)*DEG_TO_RAD;

  /*Initialization*/
  source.lon0 = wrap->lon0;
  source.lat0 = wrap->lat0;
  source.alt0 = wrap->alt0;
  source.dndh = wrap->dndh;


  while(!feof(fp)) {
    if(!getEchotopCash(fp,&x,&y,&e,&noitems))
      return 0;

    if(e==wrap->nelev-1) {
      columnhandled=0;
      sx=x;
      sy=y;
    }

    cw=getAllocatedCW(noitems);
    if(!getEchotopCashedCW(fp,cw,noitems)) {
      return 0;
    }

    if(columnhandled) {
      continue;
    }

    if(e==wrap->nelev-1) {
      place_s.v=(outUL->v-outyscale*y);
      place_s.u=(outUL->u+outxscale*x);
      place=pj_inv(place_s, wrap->outpj);
      source.lon=place.u;
      source.lat=place.v;
      llToDa(&source,&source);
    }

    source.elevation=wrap->elev[e];
    ai=mytrunc(ceil(source.azimuth/azOffset));
    deToRh(&source,&source);

    if(e==0) {
      if(source.range>wrap->inxmax*wrap->inscale) {
	setarritem3d(x,y,wrap->echonodata,wrap);
	continue;
      }
    }

    ri=mytrunc(source.range/wrap->inscale);
    refr=getarritem3d(e,ri,ai,wrap);

    if( (refr==0.0 || refr==wrap->nodata) && e!=0)
      continue;
    else if(refr==0.0 || refr==wrap->nodata) {
      setarritem3d(x,y,0.0,wrap);
      continue;
    }

    ctr=calcECwithCW(&height,&place,cw,noitems,wrap);

    if(ctr==-1) { /*Less hits than minimum*/
      setarritem3d(x,y,0.0,wrap);
    }
    else if(ctr==0) { /*no hits*/
      setarritem3d(x,y,0.0,wrap);
    }
    else {
      setarritem3d(x,y,height,wrap);
      columnhandled=1;
    }
  }

  if(sx==wrap->outdimx-1 && sy==wrap->outdimy-1)
    return 1;
  else
    return 0;
}

static double getMaxecho(int x,int y,UV here_s, TrafoWrapper3D* wrap)
{
  UV here;
  Position source;
  double azOffset = (360.0/wrap->inymax)*DEG_TO_RAD;
  double refr;
  double maxrefr=0;
  int i,ri,ai;
  double height=0.0;

  here=pj_inv(here_s,wrap->outpj);
  source.lon0 = wrap->lon0;
  source.lat0 = wrap->lat0;
  source.alt0 = wrap->alt0;
  source.lon = here.u;
  source.lat = here.v;
  source.dndh = wrap->dndh;

  llToDa(&source,&source);

  ai=mytrunc(ceil(source.azimuth/azOffset));
  if(ai>=wrap->inymax) {
    ai=ai%wrap->inymax;
  }

  source.elevation=wrap->elev[0];
  deToRh(&source,&source);

  height=source.alt-source.alt0;

  if(source.range > wrap->inxmax*wrap->inscale) {
    /*Out of bounds, just return*/
    setarritem3d(x,y,wrap->nodata,wrap);
    return wrap->echonodata;
  }

  for(i=0;i<wrap->nelev;i++) {
    source.elevation=wrap->elev[i];
    deToRh(&source,&source);
    if(source.range>wrap->inxmax*wrap->inscale)
      break;
    ri=mytrunc(source.range/wrap->inscale);

    if(i>=wrap->nelev || (ri>=wrap->inxmax) || (ai<0 || ai>=wrap->inymax)) {
      printf("Indexes would be outside array, baad\n");
      abort();
    }

    refr=getarritem3d(i,ri,ai,wrap);
    if(refr!=wrap->nodata && refr>maxrefr) {
      if( (source.alt-source.alt0) <= wrap->echomaxheight) {
	maxrefr=refr;
	height=source.alt-source.alt0;
      }
    }
  }

  if(maxrefr==0.0)
    height=0.0;

  setarritem3d(x,y,maxrefr,wrap);

  if(wrap->echoscale) {
    height/=wrap->echoscale;
  }

  return height;
}


/*Some useful python methods for building a trafowrapper*/
static int buildElevList(PyObject* info,TrafoWrapper3D* wrap)
{
  PyObject* po;
  int wasok=0;
  int i;

  po = PyMapping_GetItemString(info,"elev");
  if(!po) {
    pySetError(PyExc_AttributeError,"No elevations in source info\n");
    return 0;
  }

  if(!PySequence_Check(po)) {
    pySetError(PyExc_AttributeError,"Elevations must be sequence\n");
    return 0;
  }

  wrap->nelev = PyObject_Length(po);

  if (wrap->nelev > MAXELEV) {
    printf("  too many elevations: %d - using first %d\n", wrap->nelev, MAXELEV);
    wrap->nelev = MAXELEV;
  }

  for(i=0,wasok=0;i<wrap->nelev && wasok==0;i++) {
    wasok |= !getIdxDoubleFromTuple(i,&wrap->elev[i],po);
    wrap->elev[i]*=DEG_TO_RAD;
  }

  if(wasok) {
    Py_DECREF(po);
    pySetError(PyExc_AttributeError,"Strange elev definition\n");
    return 0;
  }

  Py_DECREF(po);

  return 1;
}

static int getAreaExtent(PyObject* info,UV* uv)
{
  /*Trying to get area extent from in_info structure*/
  int wasok;
  PyObject* po;
  po = PyMapping_GetItemString(info,"area_extent");

  if(po) {
    wasok=0;
    wasok|=!getIdxDoubleFromTuple(0,&uv->u,po);
    wasok|=!getIdxDoubleFromTuple(3,&uv->v,po);
    Py_DECREF(po);

    if(wasok) {
      pySetError(PyExc_AttributeError,"Area extent corrupt\n");
      return 0;
    }
  } else {
    pySetError(PyExc_AttributeError,"No area_extent definition\n");
    return 0;
  }

  Py_DECREF(po);

  return 1;
}

/* ----------------------------------------------------------------
   The actual transformation function.
   NOTE: we rely on a python wrapper/glue function to find the
   related projections for us first.
*/

static PyObject*  _ptoc_transform(PyObject* self, PyObject* args)
{
  char **argv;
  int i,ii, n,x,y;
  int wasok;
  double outxscale, outyscale;
  UV inLL;
  UV inUR;
  UV outUL;
  TrafoWrapper3D tw;
  void (*methfun)(int, int, UV, TrafoWrapper3D *);
  int no_of_src=0;
  int cashfile_error=0;

  PJ *pj;
  PyObject* in;          /* rave_image.image_2d */
  PyObject* out;
  PyObject* pcs;
  PyObject* elev;
  PyObject* inpcs;       /* in pcs definition */
  PyObject* outpcs;      /* our pcs definition */
  PyArrayObject* src;    /* data array with readings */
  PyArrayObject* dest;
  PyObject* in_info;     /* details about the image */
  PyObject* out_info;
  PyObject* po;

  PyObject* p_area;

  tw.set_compactness=0;

  /* Check args: */
  if(!PyArg_ParseTuple(args,"OOO",&in,&out,&outpcs)) {
    return NULL;
  }

  /* First set up projections to be used - only one for now! */
  for (ii=0;ii<1; ii++) {
    switch (ii) {
    default:
      pcs = outpcs;
      break;
    }
    if (!PySequence_Check(pcs)) {
      PyErr_SetString(PyExc_TypeError, "argument must be sequence");
      return NULL;
    }

    n = PyObject_Length(pcs);

    /* fetch argument array */
    argv = malloc(n * sizeof(char*));
    for (i = 0; i < n; i++) {
      PyObject* op = PySequence_GetItem(pcs, i);
      PyObject* str = PyObject_Str(op);
      argv[i] = PyString_AsString(str);
      Py_DECREF(str);
      Py_DECREF(op);
    }

    pj = pj_init(n, argv);

    free(argv);

    if (!pj) {
      _ptoc_error();
      return NULL;
    }

    /*For future use
    switch (ii) {
     break;
    default:
      tw.outpj = pj;
      break;
    }
    */
    tw.outpj=pj;

  }

  in_info = PyObject_GetAttrString(in, "info");
  out_info = PyObject_GetAttrString(out, "info");

  po = PyMapping_GetItemString(in_info,"elev");
  if(!po) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"No elevations in source info mapping\n");
    return NULL;
  }

  if(!PySequence_Check(po)) {
    Py_DECREF(po);
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"Elevations must be sequence\n");
    return NULL;
  }

  tw.nelev = PyObject_Length(po);

  if (tw.nelev > MAXELEV) {
    printf("  too many elevations: %d - using first %d\n", tw.nelev, MAXELEV);
    tw.nelev = MAXELEV;
  }

  for(i=0,wasok=0;i<tw.nelev && wasok==0;i++) {
    wasok |= !getIdxDoubleFromTuple(i,&tw.elev[i],po);
    tw.elev[i]*=DEG_TO_RAD;
  }

  if(wasok) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    Py_DECREF(po);
    PyErr_SetString(PyExc_TypeError,"Strange elevation definition\n");
    return NULL;
  }

  Py_DECREF(po);

  /*Trying to get area extent from in_info structure*/
  po = PyMapping_GetItemString(out_info,"area_extent");

  if(po) {
    wasok=0;
    wasok|=!getIdxDoubleFromTuple(0,&outUL.u,po);
    wasok|=!getIdxDoubleFromTuple(3,&outUL.v,po);
    Py_DECREF(po);

    if(wasok) {
      Py_DECREF(in_info);
      Py_DECREF(out_info);
      PyErr_SetString(PyExc_TypeError,"Area extent corrupt\n");
      return NULL;
    }
  } else {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"No area_extent definition\n");
    return NULL;
  }


  /*Fetch all arguments needed for interpolation*/
  if(!getIntFromDictionary("i_method",&tw.method,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);

    PyErr_SetString(PyExc_TypeError,"i_method not specified\n");
    return NULL;
  }

  if(!getIntFromDictionary("i_slice",&tw.slice,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"i_slice not specified\n");
    return NULL;
  }

  if(!getIntFromDictionary("transform_weighting",&tw.useWeights,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"transform_weighting not specified\n");
    return NULL;
  }

  if(!getIntFromDictionary("elev_usage",&tw.elevUsage,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"elev_usage not specified\n");
    return NULL;
  }

  if(!getDoubleFromDictionary("beamwidth",&tw.beamBroad,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"beamwidth not specified\n");
    return NULL;
  }
  tw.beamBroad*=DEG_TO_RAD;

  if(!getDoubleFromDictionary("i_height",&tw.height,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"i_height not specified\n");
    return NULL;
  }

  if(tw.slice==PPI && tw.height>=tw.nelev) {
     /*Do not allow ppi index to be higher than existing no of elevs*/
     Py_DECREF(in_info);
     Py_DECREF(out_info);
     PyErr_SetString(PyExc_AttributeError,"Wanted ppi index does not exist");
     return NULL;
  }

  if(!getDoubleFromDictionary("alt_0",&tw.alt0,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"alt_0 not specified\n");
    return NULL;
  }

  if(!getDoubleFromDictionary("lon_0",&tw.lon0,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"lon_0 not specified\n");
    return NULL;
  }
  tw.lon0*=DEG_TO_RAD;

  if(!getDoubleFromDictionary("lat_0",&tw.lat0,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"lat_0 not specified\n");
    return NULL;
  }
  tw.lat0*=DEG_TO_RAD;

  if(!getDoubleFromDictionary("cressman_xy",&tw.cressmanR_xy,out_info)) {
    tw.cressmanR_xy = 1L;
  }

  if(!getDoubleFromDictionary("cressman_z",&tw.cressmanR_z,out_info)) {
    tw.cressmanR_z = 1L;
  }

  if(!getDoubleFromDictionary("dndh",&tw.dndh,out_info)) {
    tw.dndh = (-3.9e-5)/1000; /*To get same value in m^-1*/
  }

  if(!getIntFromDictionary("i_qc",&tw.iqc,out_info)) {
    tw.iqc=NO_QC;
  }

  if(!getIntFromDictionary("check_nearest",&tw.check_nearest,out_info)) {
     tw.check_nearest=0;
  }

  if(!getDoubleFromDictionary("i_qcvalue",&tw.iqcvalue,out_info)) {
    tw.iqcvalue=-1.0;
  }
  else {
    if(tw.iqcvalue <0.0 || tw.iqcvalue>1.0) {
      Py_DECREF(out_info);
      Py_DECREF(in_info);
      PyErr_SetString(PyExc_AttributeError,"i_qc_value must be between 0.0 and 1.0\n");
      return NULL;
    }
  }

  switch (tw.method) { /* Select function for calculations */
  case NEAREST:
    methfun = bilinear3d;
    break;
  case CRESSMAN:
  case INVERSE:
  case UNIFORM:
    methfun = cressman3d;
    break;
  default:
    PyErr_SetString(PyExc_TypeError, "No such interpolation method\n");
    return NULL;
    break;
  }

  po = PyObject_GetAttrString(in, "data");
  if (!PySequence_Check(po)) {
    PyErr_SetString(PyExc_TypeError, "source.data must be sequence");
    return NULL;
  }
  n = PyObject_Length(po);
  no_of_src=n;
  if (n != tw.nelev ) {
    printf(" n = %d nelev = %d", n, tw.nelev);
    PyErr_SetString(PyExc_TypeError, "inconsistant number of elevs");
    return NULL;
  }

  for (i = 0; i < n; i++) {
    tw.src[i] = (PyArrayObject *)PySequence_GetItem(po, i);
  }
  Py_DECREF(po);

  dest = (PyArrayObject *)PyObject_GetAttrString(out, "data");

  tw.desta = (unsigned char *)(dest->data);

  /*  tw.type = dest->descr->type;*/
  tw.outtype=dest->descr->type;
  tw.intype=((PyArrayObject*)tw.src[0])->descr->type;

  /* The following might leak some references - check! */
  po = PyMapping_GetItemString(in_info, "nodata");
  if (po) {
    tw.nodata = PyInt_AsLong(po);
    Py_DECREF(po);
  }
  else {
    tw.nodata = 0;
    PyErr_Clear();
  }

  p_area = PyObject_GetAttrString(in,"p_area");
  if(!p_area) {
    PyErr_SetString(PyExc_TypeError, "source.p_area does not exist");
    return NULL;
  }

  po = PyObject_GetAttrString(p_area,"range_size");
  if(po) {
    tw.inscale = PyFloat_AsDouble(po);
    Py_DECREF(po);
  } else {
    printf("No range size in p_area=>defaulting to 2000 meters\n");
    tw.inscale=2000L;
    PyErr_Clear();
  }

  Py_DECREF(p_area);

  po = PyMapping_GetItemString(out_info,"cashfile");
  if(po) {
      PyObject* str = PyObject_Str(po);
      char* tmpchar;
      tmpchar=PyString_AsString(str);
      tw.cashfile=strdup(tmpchar);
      Py_DECREF(str);
      Py_DECREF(po);
  } else {
    tw.cashfile=NULL;
    PyErr_Clear();
  }

  po = PyMapping_GetItemString(out_info, "xscale");
  if (po) {
    outxscale = PyFloat_AsDouble(po);
    Py_DECREF(po);
  } else {
    printf(" ** No info['xscale'] => 1\n");
    outxscale = 1;
    PyErr_Clear();
  }

  po = PyMapping_GetItemString(out_info, "yscale");
  if (po) {
    outyscale = PyFloat_AsDouble(po);
    Py_DECREF(po);
  } else {
    printf(" ** No info['yscale'] => 1\n");
    outyscale = 1;
    PyErr_Clear();
  }

  tw.outyscale=outyscale;
  tw.outxscale=outxscale;
  tw.outUL.u=outUL.u;
  tw.outUL.v=outUL.v;

  tw.inxsize = tw.src[0]->strides[0]/tw.src[0]->strides[1];
  tw.outxsize = dest->strides[0]/dest->strides[1];
  tw.inxmax = tw.src[0]->dimensions[1];
  tw.inymax = tw.src[0]->dimensions[0];

  if(tw.cashfile!=NULL &&
     (tw.method==CRESSMAN || tw.method==INVERSE || tw.method==UNIFORM)) {
    FILE* tmpf;
    int recash=0;
    double checkelev[MAXELEV];
    int noofelev;
    int range_uppb,azim_uppb;
    double range_size;

    double cressmanRxy,cressmanRz;

    recash=(tmpf=fopen(tw.cashfile,"r"))?0:1;

    if(recash==0) {
      fread(&noofelev,sizeof(noofelev),1,tmpf);
      fread(checkelev,sizeof(checkelev[0]),noofelev,tmpf);

      if(noofelev==tw.nelev) {
	 for(i=0;i<noofelev;i++) {
	    if(checkelev[i]!=tw.elev[i]) {
	       recash = 1;
	    }
	 }
      }
      else {
	 recash = 1;
      }
      if(recash==0) {
	 fread(&range_uppb,sizeof(range_uppb),1,tmpf);
	 fread(&azim_uppb,sizeof(azim_uppb),1,tmpf);
	 fread(&range_size,sizeof(range_size),1,tmpf);
	 fread(&cressmanRxy,sizeof(cressmanRxy),1,tmpf);
	 fread(&cressmanRz,sizeof(cressmanRz),1,tmpf);

	 if(range_uppb!=tw.inymax ||
	    azim_uppb!=tw.inxmax ||
	    range_size!=tw.inscale ||
	    cressmanRxy!=tw.cressmanR_xy ||
	    cressmanRz!=tw.cressmanR_z) {
	    recash=1;
	 }
      }
    }

    if(!tmpf || recash==1) {
      if(tmpf) {
	 fclose(tmpf);
	 remove(tw.cashfile);
      }

      if(!(staticFD=fopen(tw.cashfile,"a"))) {
	 printf("Could not create cashfile for writing\n");
	 /*perror("Could not create cashfile for writing\n");*/
	 cashfile_error=1;
	 free(tw.cashfile);
	 tw.cashfile=NULL;
      }

      if(!cashfile_error) {
	 fwrite(&tw.nelev,sizeof(tw.nelev),1,staticFD);
	 fwrite(tw.elev,sizeof(tw.elev[0]),tw.nelev,staticFD);

	 fwrite(&tw.inymax,sizeof(tw.inymax),1,staticFD);
	 fwrite(&tw.inxmax,sizeof(tw.inxmax),1,staticFD);
	 fwrite(&tw.inscale,sizeof(tw.inscale),1,staticFD);

	 fwrite(&tw.cressmanR_xy,sizeof(tw.cressmanR_xy),1,staticFD);
	 fwrite(&tw.cressmanR_z,sizeof(tw.cressmanR_z),1,staticFD);
      }

      for(y=0;y<dest->dimensions[0]; y++) {
	 UV here_s;
	 here_s.v = (outUL.v-outyscale*y);
	 for(x=0;x<dest->dimensions[1]; x++) {
	    here_s.u = (outUL.u+outxscale*x);
	    methfun(x,y,here_s, &tw);
	 }
      }

      if(ferror(staticFD)) {
	 cashfile_error=1;
      }

      if(staticFD)
	 fclose(staticFD);
    }
    else {
      fclose(tmpf);
    }

    if(!cashfile_error)
       useCash(&tw);
    else {
       if(tw.cashfile) {
	  printf("Error occured while writing cashfile, removing it\n");
	  remove(tw.cashfile);
       }
    }
  }
  else {
    for(y=0;y<dest->dimensions[0]; y++) {/* do it! */
      UV here_s;
      here_s.v = (outUL.v-outyscale*y);
      for(x=0;x<dest->dimensions[1]; x++) {
	 here_s.u = (outUL.u+outxscale*x);
	 methfun(x,y,here_s, &tw); /* Call appropriate function to do the job*/
      }
    }
  }

  getAllocatedCW(-99);

  /* Also: DECREF all temporary items!! */
  pj_free(tw.outpj);

  if(tw.cashfile)
    free(tw.cashfile);

  Py_DECREF(in_info);
  Py_DECREF(out_info);
  Py_DECREF((PyObject*)dest);

  for(i=0;i<no_of_src;i++) {
    Py_DECREF(tw.src[i]);
  }

  PyErr_Clear();
  Py_INCREF(Py_None); /* Return nothing explicitly */
  return Py_None;
}

static int getEchotopVars(PyObject* info, TrafoWrapper3D* wrap)
{
  /*Fetch all arguments needed for interpolation*/
  PyObject* po;

  if(!getIntFromDictionary("i_method",&wrap->method,info)) {
    wrap->method=CRESSMAN;
  }

  wrap->slice=ECHOTOP;

  if(!getDoubleFromDictionary("beamwidth",&wrap->beamBroad,info)) {
    raiseErrorWI(PyExc_AttributeError,"beamwidth not specified\n");
  }
  wrap->beamBroad*=DEG_TO_RAD;

  if(!getDoubleFromDictionary("alt_0",&wrap->alt0,info)) {
    raiseErrorWI(PyExc_AttributeError,"alt_0 not specified\n");
  }

  if(!getDoubleFromDictionary("lon_0",&wrap->lon0,info)) {
    raiseErrorWI(PyExc_AttributeError,"lon_0 not specified\n");
  }
  wrap->lon0*=DEG_TO_RAD;

  if(!getDoubleFromDictionary("lat_0",&wrap->lat0,info)) {
    raiseErrorWI(PyExc_AttributeError,"lat_0 not specified\n");
  }
  wrap->lat0*=DEG_TO_RAD;

  if(!getDoubleFromDictionary("cressman_xy",&wrap->cressmanR_xy,info)) {
    wrap->cressmanR_xy = 1L;
  }

  if(!getDoubleFromDictionary("cressman_z",&wrap->cressmanR_z,info)) {
    wrap->cressmanR_z = 1L;
  }

  if(!getDoubleFromDictionary("dndh",&wrap->dndh,info)) {
    wrap->dndh = (-3.9e-5)/1000; /*To get same value in m^-1*/
  }

  if(!getIntFromDictionary("i_qc",&wrap->iqc,info)) {
    wrap->iqc=NO_QC;
  }

  if(!getDoubleFromDictionary("i_qcvalue",&wrap->iqcvalue,info)) {
    wrap->iqcvalue=-1.0;
  }
  else {
    if(wrap->iqcvalue <0.0 || wrap->iqcvalue>1.0) {
      raiseErrorWI(PyExc_AttributeError, \
		   "i_qcvalue must be between 0.0 and 1.0\n");
    }
  }

#ifdef ECHO_TOP_CASHING
  po = PyMapping_GetItemString(info,"cashfile");
  if(po) {
      PyObject* str = PyObject_Str(po);
      wrap->cashfile = PyString_AsString(str);
      Py_DECREF(str);
      Py_DECREF(po);
  } else {
    wrap->cashfile=NULL;
    PyErr_Clear();
  }
#else
  wrap->cashfile=NULL;
#endif

  if(!getDoubleFromDictionary("echo_nodata",&wrap->echonodata,info)) {
    wrap->echonodata=wrap->nodata;
  }

  if(!getDoubleFromDictionary("echo_maxheight",&wrap->echomaxheight,info)) {
    wrap->echomaxheight=25400;
  }

  if(!getDoubleFromDictionary("echo_scale",&wrap->echoscale,info)) {
    wrap->echoscale=100.0;
  }

  if(!getDoubleFromDictionary("echo_decr_factor",&wrap->echodecrfactor,info)) {
    wrap->echodecrfactor=1.0;
  }

  if(!getIntFromDictionary("echo_decrtype",&wrap->echodecrtype,info)) {
    wrap->echodecrtype=ET_LH_RCE;
  }

  return 1;
}

static int doEchoTopCashing(FILE* fp,UV* outUL, \
			    double outyscale,double outxscale,\
			    TrafoWrapper3D* tw)
{
  int x,y;

  fwrite(&tw->nelev,sizeof(tw->nelev),1,fp);
  fwrite(tw->elev,sizeof(tw->elev[0]),tw->nelev,fp);
  fwrite(&tw->inymax,sizeof(tw->inymax),1,fp);
  fwrite(&tw->inxmax,sizeof(tw->inxmax),1,fp);
  fwrite(&tw->inscale,sizeof(tw->inscale),1,fp);

  fwrite(&tw->cressmanR_xy,sizeof(tw->cressmanR_xy),1,fp);
  fwrite(&tw->cressmanR_z,sizeof(tw->cressmanR_z),1,fp);

  fwrite(&tw->echodecrtype,sizeof(tw->echodecrtype),1,fp);
  fwrite(&tw->echodecrfactor,sizeof(tw->echodecrfactor),1,fp);

  for(y=0;y<tw->outdimy;y++) {
    UV here_s;
    here_s.v = (outUL->v-outyscale*y);
    for(x=0;x<tw->outdimx;x++) {
      here_s.u = (outUL->u+outxscale*x);
      if(!cashEchotopEntry(fp,x,y,here_s,tw))
		return 0;
    }
  }

  return 1;
}

static int shouldRecashEcho(FILE* fp, TrafoWrapper3D* wrap)
{
  int recash=0;
  double checkelev[MAXELEV];
  int noofelev;
  int range_uppb,azim_uppb;
  double range_size;
  int i;

  double cressmanRxy,cressmanRz;

  int echotype;
  double echofactor;


  if(fread(&noofelev,sizeof(noofelev),1,fp)<0)
    return -1;
  if(fread(checkelev,sizeof(checkelev[0]),noofelev,fp)<0)
    return -1;

  if(noofelev!=wrap->nelev)
    return 1;

  for(i=0;i<noofelev;i++) {
    if(checkelev[i]!=wrap->elev[i]) {
      return 1;
    }
  }

  if(fread(&range_uppb,sizeof(range_uppb),1,fp)<0)
    return -1;

  if(fread(&azim_uppb,sizeof(azim_uppb),1,fp)<0)
    return -1;

  if(fread(&range_size,sizeof(range_size),1,fp)<0)
    return -1;

  if(range_uppb!=wrap->inymax ||
     azim_uppb!=wrap->inxmax ||
     range_size!=wrap->inscale) {
    return 1;
  }

  if(fread(&cressmanRxy,sizeof(cressmanRxy),1,fp)<0)
    return -1;
  if(cressmanRxy!=wrap->cressmanR_xy)
    return 1;

  if(fread(&cressmanRz,sizeof(cressmanRz),1,fp)<0)
    return -1;
  if(cressmanRz!=wrap->cressmanR_z)
    return 1;

  if(fread(&echotype,sizeof(echotype),1,fp)<0)
    return -1;
  if(fread(&echofactor,sizeof(echofactor),1,fp)<0)
    return -1;

  if(echotype!=wrap->echodecrtype ||
     echofactor !=wrap->echodecrfactor)
    return 1;

  return 0;
}

static PyObject* _ptoc_echotop(PyObject* self, PyObject* args)
{
  PyObject *in,*out,*outpcs;
  PyObject *in_info,*out_info;
  PyObject *po,*p_area;
  PyArrayObject* dest;
  UV outUL;
  int x,y,n,i;
  TrafoWrapper3D tw;
  double outxscale,outyscale;
  int errorset=0;

  if(!PyArg_ParseTuple(args,"OOO",&in,&out,&outpcs)) {
    return NULL;
  }

  tw.outpj=initProjection(outpcs);
  if(!tw.outpj) {
    raiseException(PyExc_AttributeError,"Erroneous PCS definition\n");
  }

  in_info = PyObject_GetAttrString(in, "info");

  if(!in_info) {
	raiseException(PyExc_AttributeError,"No in info exists");
  }

  if(!buildElevList(in_info,&tw)) {
    Py_DECREF(in_info);
    return NULL;
  }

  if(!getIntFromDictionary("nodata",&tw.nodata,in_info))
    tw.nodata=0;

  Py_DECREF(in_info);

  out_info=PyObject_GetAttrString(out,"info");
  if(!out_info) {
    raiseException(PyExc_AttributeError,"No out info exists");
  }

  if(!getAreaExtent(out_info,&outUL)) {
    Py_DECREF(out_info);
    return NULL;
  }

  if(!getEchotopVars(out_info,&tw)) {
    Py_DECREF(out_info);
    return NULL;
  }

  if(!getDoubleFromDictionary("xscale",&outxscale,out_info))
    outxscale=2000.0;
  if(!getDoubleFromDictionary("yscale",&outyscale,out_info))
    outyscale=2000.0;

  Py_DECREF(out_info);

  p_area = PyObject_GetAttrString(in,"p_area");
  if(!p_area) {
    raiseException(PyExc_AttributeError,"Source.p_area does not exist");
  }

  if(!getDoubleFromDictionary("range_size",&tw.inscale,p_area))
    tw.inscale=2000.0;

  Py_DECREF(p_area);

  /*Fetch indata*/
  po = PyObject_GetAttrString(in, "data");
  if (!PySequence_Check(po)) {
    raiseException(PyExc_TypeError,"source.data must be sequence");
  }

  n = PyObject_Length(po);
  if (n != tw.nelev ) {
    raiseException(PyExc_AttributeError,"inconsistent number of elevs");
  }

  for (i = 0; i < n; i++) {
    tw.src[i] = (PyArrayObject *)PySequence_GetItem(po, i);
  }
  Py_DECREF(po);

  /*Fetch outdata*/
  dest = (PyArrayObject *)PyObject_GetAttrString(out, "data");

  tw.desta = (unsigned char *)(dest->data);
  /*  tw.type = dest->descr->type;*/
  tw.outtype = dest->descr->type;
  tw.intype=((PyArrayObject*)tw.src[0])->descr->type;


  tw.outdimx=dest->dimensions[1];
  tw.outdimy=dest->dimensions[0];

  tw.inxsize = tw.src[0]->strides[0]/tw.src[0]->strides[1];
  tw.outxsize = dest->strides[0]/dest->strides[1];
  tw.inxmax = tw.src[0]->dimensions[1];
  tw.inymax = tw.src[0]->dimensions[0];

#ifdef ECHO_TOP_CASHING
  if(tw.cashfile!=NULL) {
    FILE* tmpf;
    int writeok=1;
    if(!(tmpf=fopen(tw.cashfile,"r"))) {
      if(tmpf=fopen(tw.cashfile,"a")) {
	int recash;

	if(!doEchoTopCashing(tmpf,&outUL,outyscale,outxscale,&tw)) {
	  errorset=1;
	  pySetError(PyExc_IOError,"Failed to cash entries\n");
	}
	fclose(tmpf);

	if(errorset==0&&(!(tmpf=fopen(tw.cashfile,"r")))) {
	  errorset=1;
	  pySetError(PyExc_IOError,"Could not open cashfile\n");
	} else if(errorset==0) {
	  recash=shouldRecashEcho(tmpf,&tw);
	  if(recash==-1 || recash==1) {
	    pySetError(PyExc_IOError,\
		       "Have cashed file, but need to recash, which is bad\n");
	    errorset=1;
	  }
	}

	if(errorset==0 &&
	   (!calcEchotopWithCash(tmpf,&outUL,outyscale,outxscale,&tw))) {
	  pySetError(PyExc_IOError,"Failed to use cashfile\n");
	  errorset=1;
	}
	fclose(tmpf);
      }
    } else {
      int recash=shouldRecashEcho(tmpf,&tw);
      if(recash==-1 || recash==1) {
	fclose(tmpf);
	remove(tw.cashfile);
	if(!(tmpf=fopen(tw.cashfile,"a"))) {
	  pySetError(PyExc_IOError,"Could not create cashfile\n");
	  errorset=1;
	}
	if(!doEchoTopCashing(tmpf,&outUL,outyscale,outxscale,&tw)) {
	  pySetError(PyExc_IOError,"Could not recash cashfile\n");
	  errorset=1;
	}
	fclose(tmpf);
	if(!(tmpf=fopen(tw.cashfile,"r"))) {
	  pySetError(PyExc_IOError,"Could not open cashfile for reading");
	  errorset=1;
	}
	recash=shouldRecashEcho(tmpf,&tw);
	if(recash==1 || recash==-1) {
	  pySetError(PyExc_IOError,\
		     "Have cashed file, but need to recash, something is wrong\n");
	  errorset=1;
	}
      }

      if(errorset==0 && !calcEchotopWithCash(tmpf,&outUL,outyscale,outxscale,&tw)) {
	pySetError(PyExc_IOError,"Could not use cashed file\n");
	errorset=1;
      }
      fclose(tmpf);
    }
  } else {
    for(y=0;y<dest->dimensions[0]; y++) {
      UV here_s;
      here_s.v = (outUL.v-outyscale*y);
      for(x=0;x<dest->dimensions[1]; x++) {
	here_s.u = (outUL.u+outxscale*x);
	calculateEchotop(x,y,here_s, &tw);
      }
    }
  }
#else
  for(y=0;y<dest->dimensions[0]; y++) {
    UV here_s;
    here_s.v = (outUL.v-outyscale*y);
    for(x=0;x<dest->dimensions[1]; x++) {
      here_s.u = (outUL.u+outxscale*x);
      calculateEchotop(x,y,here_s, &tw);
    }
  }
#endif

  pj_free(tw.outpj);

  if(tw.cashfile)
    free(tw.cashfile);

  Py_DECREF(dest);
  for(i=0;i<n;i++) {
    Py_DECREF(tw.src[i]);
  }

  if(errorset)
    return NULL;

  PyErr_Clear();
  Py_INCREF(Py_None);
  return Py_None;
}

static int getMaxechoVars(PyObject* info, TrafoWrapper3D* wrap)
{
  PyObject* po;

  if(!getDoubleFromDictionary("beamwidth",&wrap->beamBroad,info)) {
    raiseErrorWI(PyExc_AttributeError,"beamwidth not specified\n");
  }
  wrap->beamBroad*=DEG_TO_RAD;

  if(!getDoubleFromDictionary("alt_0",&wrap->alt0,info)) {
    raiseErrorWI(PyExc_AttributeError,"alt_0 not specified\n");
  }

  if(!getDoubleFromDictionary("lon_0",&wrap->lon0,info)) {
    raiseErrorWI(PyExc_AttributeError,"lon_0 not specified\n");
  }
  wrap->lon0*=DEG_TO_RAD;

  if(!getDoubleFromDictionary("lat_0",&wrap->lat0,info)) {
    raiseErrorWI(PyExc_AttributeError,"lat_0 not specified\n");
  }
  wrap->lat0*=DEG_TO_RAD;

  if(!getDoubleFromDictionary("cressman_xy",&wrap->cressmanR_xy,info)) {
    wrap->cressmanR_xy = 1L;
  }

  if(!getDoubleFromDictionary("cressman_z",&wrap->cressmanR_z,info)) {
    wrap->cressmanR_z = 1L;
  }

  if(!getDoubleFromDictionary("dndh",&wrap->dndh,info)) {
    wrap->dndh = (-3.9e-5)/1000; /*To get same value in m^-1*/
  }

  if(!getDoubleFromDictionary("echo_nodata",&wrap->echonodata,info)) {
    wrap->echonodata=wrap->nodata;
  }

  if(!getDoubleFromDictionary("echo_maxheight",&wrap->echomaxheight,info)) {
    wrap->echomaxheight=25400;
  }

  if(!getDoubleFromDictionary("echo_scale",&wrap->echoscale,info)) {
    wrap->echoscale=100.0;
  }

  return 1;
}

static PyObject* _ptoc_max(PyObject* self, PyObject* args)
{
  PyObject *in,*out,*outpcs;
  PyObject *in_info,*out_info;
  PyObject *po,*p_area;
  PyArrayObject* dest;
  UV outUL;
  int x,y,n,i;
  TrafoWrapper3D tw;
  double outxscale,outyscale;
  int errorset=0;
  double height;
  PyArrayObject* topoarr;

  if(!PyArg_ParseTuple(args,"OOO",&in,&out,&outpcs)) {
    return NULL;
  }

  tw.outpj=initProjection(outpcs);
  if(!tw.outpj) {
    raiseException(PyExc_AttributeError,"Erroneous PCS definition\n");
  }

  in_info = PyObject_GetAttrString(in, "info");
  if(!in_info) {
	raiseException(PyExc_AttributeError,"No in info exists");
  }

  if(!buildElevList(in_info,&tw)) {
    Py_DECREF(in_info);
    return NULL;
  }

  if(!getIntFromDictionary("nodata",&tw.nodata,in_info)) {
    printf("No data was not set\n");
    tw.nodata=0;
  }

  Py_DECREF(in_info);

  out_info=PyObject_GetAttrString(out,"info");
  if(!out_info) {
    raiseException(PyExc_AttributeError,"No out info exists");
  }

  if(!getAreaExtent(out_info,&outUL)) {
    Py_DECREF(out_info);
    return NULL;
  }

  if(!getMaxechoVars(out_info,&tw)) {
    Py_DECREF(out_info);
    return NULL;
  }

  if(!getDoubleFromDictionary("xscale",&outxscale,out_info))
    outxscale=2000.0;
  if(!getDoubleFromDictionary("yscale",&outyscale,out_info))
    outyscale=2000.0;

  Py_DECREF(out_info);

  p_area = PyObject_GetAttrString(in,"p_area");
  if(!p_area) {
    raiseException(PyExc_AttributeError,"Source.p_area does not exist");
  }

  if(!getDoubleFromDictionary("range_size",&tw.inscale,p_area))
    tw.inscale=2000.0;

  Py_DECREF(p_area);

  /*Fetch indata*/
  po = PyObject_GetAttrString(in, "data");
  if (!PySequence_Check(po)) {
    raiseException(PyExc_TypeError,"source.data must be sequence");
  }

  n = PyObject_Length(po);
  if (n != tw.nelev ) {
    raiseException(PyExc_AttributeError,"inconsistent number of elevs");
  }

  for (i = 0; i < n; i++) {
    tw.src[i] = (PyArrayObject *)PySequence_GetItem(po, i);
  }
  Py_DECREF(po);

  /*Fetch outdata*/
  dest = (PyArrayObject *)PyObject_GetAttrString(out, "data");

  tw.desta = (unsigned char *)(dest->data);
  /*  tw.type = dest->descr->type;*/
  tw.outtype = dest->descr->type;
  tw.intype=((PyArrayObject*)tw.src[0])->descr->type;

  tw.outdimx=dest->dimensions[1];
  tw.outdimy=dest->dimensions[0];

  tw.inxsize = tw.src[0]->strides[0]/tw.src[0]->strides[1];
  tw.outxsize = dest->strides[0]/dest->strides[1];
  tw.inxmax = tw.src[0]->dimensions[1];
  tw.inymax = tw.src[0]->dimensions[0];

  topoarr=(PyArrayObject*)PyObject_GetAttrString(out,"topo");
  if(!topoarr) {
    PyErr_Clear();
    tw.topodata=NULL;
  } else if(!PyArray_Check(topoarr)) {
    PyErr_Clear();
    Py_DECREF(topoarr);
    topoarr=NULL;
    tw.topodata=NULL;
  } else {
    tw.topodata=(unsigned char*)topoarr->data;
    tw.topotype=topoarr->descr->type;
    tw.topooutxsize=topoarr->strides[0]/topoarr->strides[1];
  }

  /* printf("Going to calculate\n");*/

  for(y=0;y<dest->dimensions[0]; y++) {
    UV here_s;
    /*printf("Working with y=%d\n",y);*/
    here_s.v = (outUL.v-outyscale*y);
    for(x=0;x<dest->dimensions[1]; x++) {
      here_s.u = (outUL.u+outxscale*x);
      height=getMaxecho(x,y,here_s, &tw);
      if(tw.topodata) {
	settopoitem(x,y,height,&tw);
      }
    }
  }

  /*printf("Finished with calculation\n");*/

  pj_free(tw.outpj);
  Py_DECREF(dest);

  if(topoarr)
    Py_DECREF(topoarr);

  for(i=0;i<n;i++) {
    Py_DECREF(tw.src[i]);
  }

  if(errorset)
    return NULL;

  PyErr_Clear();
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject*  _ptoc_compactness(PyObject* self, PyObject* args)
{
   char **argv;
   int i,ii, n,x,y;
   int wasok;
   double outxscale, outyscale;
   UV inLL;
   UV inUR;
   UV outUL;
   TrafoWrapper3D tw;
   void (*methfun)(int, int, UV, TrafoWrapper3D *);
   int no_of_src=0;
   int cashfile_error=0;

   PJ *pj;
   PyObject* in;          /* rave_image.image_2d */
   PyObject* out;
   PyObject* pcs;
   PyObject* elev;
   PyObject* inpcs;       /* in pcs definition */
   PyObject* outpcs;      /* our pcs definition */
   PyArrayObject* src;    /* data array with readings */
   PyArrayObject* dest;
   PyObject* in_info;     /* details about the image */
   PyObject* out_info;
   PyObject* po;

   PyObject* p_area;

   tw.set_compactness=1;

   /* Check args: */
   if(!PyArg_ParseTuple(args,"OOO",&in,&out,&outpcs)) {
      return NULL;
   }

   /* First set up projections to be used - only one for now! */
   for (ii=0;ii<1; ii++) {
      switch (ii) {
      default:
	 pcs = outpcs;
	 break;
      }
      if (!PySequence_Check(pcs)) {
	 PyErr_SetString(PyExc_TypeError, "argument must be sequence");
	 return NULL;
      }

      n = PyObject_Length(pcs);

      /* fetch argument array */
      argv = malloc(n * sizeof(char*));
      for (i = 0; i < n; i++) {
	 PyObject* op = PySequence_GetItem(pcs, i);
	 PyObject* str = PyObject_Str(op);
	 argv[i] = PyString_AsString(str);
	 Py_DECREF(str);
	 Py_DECREF(op);
      }

      pj = pj_init(n, argv);

      free(argv);

      if (!pj) {
	 _ptoc_error();
	 return NULL;
      }
      /*For future use
      switch (ii) {
	 break;
      default:
	 tw.outpj = pj;
	 break;
      }
      */
      tw.outpj=pj;
   }

   in_info = PyObject_GetAttrString(in, "info");
   out_info = PyObject_GetAttrString(out, "info");

   po = PyMapping_GetItemString(in_info,"elev");
   if(!po) {
      Py_DECREF(in_info);
      Py_DECREF(out_info);
      PyErr_SetString(PyExc_TypeError,"No elevations in source info mapping\n");
      return NULL;
   }

   if(!PySequence_Check(po)) {
      Py_DECREF(po);
      Py_DECREF(in_info);
      Py_DECREF(out_info);
      PyErr_SetString(PyExc_TypeError,"Elevations must be sequence\n");
      return NULL;
   }

   tw.nelev = PyObject_Length(po);

   if (tw.nelev > MAXELEV) {
      printf("  too many elevations: %d - using first %d\n", tw.nelev, MAXELEV);
      tw.nelev = MAXELEV;
   }

   for(i=0,wasok=0;i<tw.nelev && wasok==0;i++) {
      wasok |= !getIdxDoubleFromTuple(i,&tw.elev[i],po);
      tw.elev[i]*=DEG_TO_RAD;
   }

   if(wasok) {
      Py_DECREF(in_info);
      Py_DECREF(out_info);
      Py_DECREF(po);
      PyErr_SetString(PyExc_TypeError,"Strange elevation definition\n");
      return NULL;
   }

   Py_DECREF(po);

   /*Trying to get area extent from in_info structure*/
   po = PyMapping_GetItemString(out_info,"area_extent");

   if(po) {
      wasok=0;
      wasok|=!getIdxDoubleFromTuple(0,&outUL.u,po);
      wasok|=!getIdxDoubleFromTuple(3,&outUL.v,po);
      Py_DECREF(po);

      if(wasok) {
	 Py_DECREF(in_info);
	 Py_DECREF(out_info);
	 PyErr_SetString(PyExc_TypeError,"Area extent corrupt\n");
	 return NULL;
      }
   } else {
      Py_DECREF(in_info);
      Py_DECREF(out_info);
      PyErr_SetString(PyExc_TypeError,"No area_extent definition\n");
      return NULL;
   }

  /*Fetch all arguments needed for interpolation*/
   if(!getIntFromDictionary("i_method",&tw.method,out_info)) {
      Py_DECREF(in_info);
      Py_DECREF(out_info);

      PyErr_SetString(PyExc_TypeError,"i_method not specified\n");
      return NULL;
   }

  if(!getIntFromDictionary("i_slice",&tw.slice,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"i_method not specified\n");
    return NULL;
  }

  if(!getIntFromDictionary("transform_weighting",&tw.useWeights,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"transform_weighting not specified\n");
    return NULL;
  }

  if(!getIntFromDictionary("elev_usage",&tw.elevUsage,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"elev_usage not specified\n");
    return NULL;
  }

  if(!getDoubleFromDictionary("beamwidth",&tw.beamBroad,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"beamwidth not specified\n");
    return NULL;
  }
  tw.beamBroad*=DEG_TO_RAD;

  if(!getDoubleFromDictionary("i_height",&tw.height,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"i_height not specified\n");
    return NULL;
  }

  if(!getDoubleFromDictionary("alt_0",&tw.alt0,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"alt_0 not specified\n");
    return NULL;
  }

  if(!getDoubleFromDictionary("lon_0",&tw.lon0,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"lon_0 not specified\n");
    return NULL;
  }
  tw.lon0*=DEG_TO_RAD;

  if(!getDoubleFromDictionary("lat_0",&tw.lat0,out_info)) {
    Py_DECREF(in_info);
    Py_DECREF(out_info);
    PyErr_SetString(PyExc_TypeError,"lat_0 not specified\n");
    return NULL;
  }
  tw.lat0*=DEG_TO_RAD;

  if(!getDoubleFromDictionary("cressman_xy",&tw.cressmanR_xy,out_info)) {
    tw.cressmanR_xy = 1L;
  }

  if(!getDoubleFromDictionary("cressman_z",&tw.cressmanR_z,out_info)) {
    tw.cressmanR_z = 1L;
  }

  if(!getDoubleFromDictionary("dndh",&tw.dndh,out_info)) {
    tw.dndh = (-3.9e-5)/1000; /*To get same value in m^-1*/
  }

  if(!getIntFromDictionary("i_qc",&tw.iqc,out_info)) {
    tw.iqc=NO_QC;
  }

  if(!getDoubleFromDictionary("i_qcvalue",&tw.iqcvalue,out_info)) {
    tw.iqcvalue=-1.0;
  }
  else {
    if(tw.iqcvalue <0.0 || tw.iqcvalue>1.0) {
      Py_DECREF(out_info);
      Py_DECREF(in_info);
      PyErr_SetString(PyExc_AttributeError,"i_qc_value must be between 0.0 and 1.0\n");
      return NULL;
    }
  }

  switch (tw.method) { /* Select function for calculations */
  case NEAREST:
    methfun = bilinear3d;
    break;
  case CRESSMAN:
  case INVERSE:
  case UNIFORM:
    methfun = cressman3d;
    break;
  default:
    PyErr_SetString(PyExc_TypeError, "No such interpolation method\n");
    return NULL;
    break;
  }

  po = PyObject_GetAttrString(in, "data");
  if (!PySequence_Check(po)) {
    PyErr_SetString(PyExc_TypeError, "source.data must be sequence");
    return NULL;
  }
  n = PyObject_Length(po);
  no_of_src=n;
  if (n != tw.nelev ) {
    printf(" n = %d nelev = %d", n, tw.nelev);
    PyErr_SetString(PyExc_TypeError, "inconsistant number of elevs");
    return NULL;
  }

  for (i = 0; i < n; i++) {
    tw.src[i] = (PyArrayObject *)PySequence_GetItem(po, i);
  }
  Py_DECREF(po);

  dest = (PyArrayObject *)PyObject_GetAttrString(out, "data");

  tw.desta = (unsigned char *)(dest->data);

  /*  tw.type = dest->descr->type;*/
  tw.outtype=dest->descr->type;
  tw.intype=((PyArrayObject*)tw.src[0])->descr->type;

  /* The following might leak some references - check! */
  po = PyMapping_GetItemString(in_info, "nodata");
  if (po) {
    tw.nodata = PyInt_AsLong(po);
    Py_DECREF(po);
  }
  else {
    tw.nodata = 0;
    PyErr_Clear();
  }

  p_area = PyObject_GetAttrString(in,"p_area");
  if(!p_area) {
    PyErr_SetString(PyExc_TypeError, "source.p_area does not exist");
    return NULL;
  }

  po = PyObject_GetAttrString(p_area,"range_size");
  if(po) {
    tw.inscale = PyFloat_AsDouble(po);
    Py_DECREF(po);
  } else {
    printf("No range size in p_area=>defaulting to 2000 meters\n");
    tw.inscale=2000L;
    PyErr_Clear();
  }

  Py_DECREF(p_area);

  po = PyMapping_GetItemString(out_info,"cashfile");
  if(po) {
      PyObject* str = PyObject_Str(po);
      char* tmpchar;
      tmpchar=PyString_AsString(str);
      tw.cashfile=strdup(tmpchar);
      Py_DECREF(str);
      Py_DECREF(po);
  } else {
    tw.cashfile=NULL;
    PyErr_Clear();
  }

  po = PyMapping_GetItemString(out_info, "xscale");
  if (po) {
    outxscale = PyFloat_AsDouble(po);
    Py_DECREF(po);
  } else {
    printf(" ** No info['xscale'] => 1\n");
    outxscale = 1;
    PyErr_Clear();
  }

  po = PyMapping_GetItemString(out_info, "yscale");
  if (po) {
    outyscale = PyFloat_AsDouble(po);
    Py_DECREF(po);
  } else {
    printf(" ** No info['yscale'] => 1\n");
    outyscale = 1;
    PyErr_Clear();
  }

  tw.outyscale=outyscale;
  tw.outxscale=outxscale;
  tw.outUL.u=outUL.u;
  tw.outUL.v=outUL.v;

  tw.inxsize = tw.src[0]->strides[0]/tw.src[0]->strides[1];
  tw.outxsize = dest->strides[0]/dest->strides[1];
  tw.inxmax = tw.src[0]->dimensions[1];
  tw.inymax = tw.src[0]->dimensions[0];

  if(tw.cashfile!=NULL &&
     (tw.method==CRESSMAN || tw.method==INVERSE || tw.method==UNIFORM)) {
    FILE* tmpf;
    int recash=0;
    double checkelev[MAXELEV];
    int noofelev;
    int range_uppb,azim_uppb;
    double range_size;

    double cressmanRxy,cressmanRz;

    recash=(tmpf=fopen(tw.cashfile,"r"))?0:1;

    if(recash==0) {
      fread(&noofelev,sizeof(noofelev),1,tmpf);
      fread(checkelev,sizeof(checkelev[0]),noofelev,tmpf);

      if(noofelev==tw.nelev) {
		for(i=0;i<noofelev;i++) {
		  if(checkelev[i]!=tw.elev[i]) {
			recash = 1;
		  }
		}
      }
      else {
		recash = 1;
      }
      if(recash==0) {
	 fread(&range_uppb,sizeof(range_uppb),1,tmpf);
	 fread(&azim_uppb,sizeof(azim_uppb),1,tmpf);
	 fread(&range_size,sizeof(range_size),1,tmpf);
	 fread(&cressmanRxy,sizeof(cressmanRxy),1,tmpf);
	 fread(&cressmanRz,sizeof(cressmanRz),1,tmpf);

	 if(range_uppb!=tw.inymax ||
	    azim_uppb!=tw.inxmax ||
	    range_size!=tw.inscale ||
	    cressmanRxy!=tw.cressmanR_xy ||
	    cressmanRz!=tw.cressmanR_z) {
	    recash=1;
	 }
      }
    }

    if(!tmpf || recash==1) {
      if(tmpf) {
	 fclose(tmpf);
	 remove(tw.cashfile);
      }

      if(!(staticFD=fopen(tw.cashfile,"a"))) {
	 printf("Could not create cashfile for writing\n");
	 /*perror("Could not create cashfile for writing\n");*/
	 cashfile_error=1;
	 free(tw.cashfile);
	 tw.cashfile=NULL;
      }

      if(!cashfile_error) {
	 fwrite(&tw.nelev,sizeof(tw.nelev),1,staticFD);
	 fwrite(tw.elev,sizeof(tw.elev[0]),tw.nelev,staticFD);

	 fwrite(&tw.inymax,sizeof(tw.inymax),1,staticFD);
	 fwrite(&tw.inxmax,sizeof(tw.inxmax),1,staticFD);
	 fwrite(&tw.inscale,sizeof(tw.inscale),1,staticFD);

	 fwrite(&tw.cressmanR_xy,sizeof(tw.cressmanR_xy),1,staticFD);
	 fwrite(&tw.cressmanR_z,sizeof(tw.cressmanR_z),1,staticFD);
      }

      for(y=0;y<dest->dimensions[0]; y++) {
	 UV here_s;
	 here_s.v = (outUL.v-outyscale*y);
	 for(x=0;x<dest->dimensions[1]; x++) {
	    here_s.u = (outUL.u+outxscale*x);
	    methfun(x,y,here_s, &tw);
	 }
      }

      if(ferror(staticFD)) {
	 cashfile_error=1;
      }

      if(staticFD)
	 fclose(staticFD);
    }
    else {
      fclose(tmpf);
    }

    if(!cashfile_error)
       useCash(&tw);
    else {
       if(tw.cashfile) {
	  printf("Error occured while writing cashfile, removing it\n");
	  remove(tw.cashfile);
       }
    }
  }
  else {
    for(y=0;y<dest->dimensions[0]; y++) {/* do it! */
      UV here_s;
      here_s.v = (outUL.v-outyscale*y);
      for(x=0;x<dest->dimensions[1]; x++) {
	 here_s.u = (outUL.u+outxscale*x);
	 methfun(x,y,here_s, &tw); /* Call appropriate function to do the job*/
      }
    }
  }

  /* Also: DECREF all temporary items!! */
  pj_free(tw.outpj);

  if(tw.cashfile)
    free(tw.cashfile);

  Py_DECREF(in_info);
  Py_DECREF(out_info);
  Py_DECREF((PyObject*)dest);

  for(i=0;i<no_of_src;i++) {
    Py_DECREF(tw.src[i]);
  }

  PyErr_Clear();
  Py_INCREF(Py_None); /* Return nothing explicitly */
  return Py_None;
}

static PyObject* _ptoc_test(PyObject* self, PyObject* args)
{
  printf("PTOC: test to see that interface works\n");

  Py_INCREF(Py_None);
  return Py_None;
}

static struct PyMethodDef _ptoc_methods[] = {
    {"transform", (PyCFunction)_ptoc_transform, METH_VARARGS},
    {"echotop", (PyCFunction)_ptoc_echotop, METH_VARARGS},
    {"max",(PyCFunction)_ptoc_max,METH_VARARGS},
    {"compactness",(PyCFunction)_ptoc_compactness,METH_VARARGS},
    {"test",(PyCFunction)_ptoc_test,1},
    {NULL, NULL} /* sentinel */
};

static PyObject*
_ptoc_getattr(PyObject* s, char *name)
{
    PyObject* res;

    res = Py_FindMethod(_ptoc_methods, (PyObject*  ) s, name);
    if (res)
	return res;

    PyErr_Clear();

    /* no attributes */

    PyErr_SetString(PyExc_AttributeError, name);
    return NULL;
}

static struct PyMethodDef _ptoc_functions[] = {
    {"transform", (PyCFunction)_ptoc_transform,METH_VARARGS },
    {"echotop", (PyCFunction)_ptoc_echotop, METH_VARARGS},
    {"max",(PyCFunction)_ptoc_max,METH_VARARGS},
    {"compactness",(PyCFunction)_ptoc_compactness,METH_VARARGS},
    {"test",(PyCFunction)_ptoc_test,1},
    {NULL, NULL} /* sentinel */
};

init_ptoc()
{
    PyObject* m;

    /* Patch object type */
    /*    Proj_Type.ob_type = &PyType_Type;*/

    /* Initialize module object */
    m = Py_InitModule("_ptoc", _ptoc_functions);

    /* Create error object */
    PyTransform_Error = PyString_FromString("_ptoc.error");
    if (PyTransform_Error == NULL ||
        PyDict_SetItemString(PyModule_GetDict(m), "error", PyTransform_Error) != 0)
        Py_FatalError("can't define _ptoc.error");

    import_array(); /*To make sure I get access to the Numeric PyArray functions*/
}
