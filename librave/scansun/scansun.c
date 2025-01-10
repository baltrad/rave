/* --------------------------------------------------------------------
Copyright (C) 2010 Royal Netherlands Meteorological Institute, KNMI and
Swedish Meteorological and Hydrological Institute, SMHI

This file is now part of RAVE.

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
 * KNMI's sun scanning functionality
 * @file
 * @author Original algorithm and code: Iwan Holleman, KNMI, and Integration: Daniel Michelson, SMHI
 * @date 2010-10-29
 */
/*Original (but updated) preamble**********************************************/
/*This program scans volume reflectivity data in ODIM_H5 format for solar     */
/*interferences using SMHI's HL-HDF library. Added features:                  */
/*1) Use of minimum elevation and range is changed to use of minimum height   */
/*and range                                                                   */
/*2) Received solar power is normalized to the band width                     */
/******************************************************************************/

/*Program: scansun_h5.c*/
/*Authors: Iwan Holleman (KNMI) and Daniel Michelson (SMHI)*/
/*Date: May 2007*/  
/*6 November 2007: Update*/
/*20 May 2008: Change from anaysis of Z to uncorrected Z, more precision in stddev*/
/*29 October 2010: started integration with RAVE and BALTRAD*/
/*6 April 2015: Improved extraction of sunhit power (with A. Huuskomen).*/
/*May 2015: New integration with RAVE for OPERA */
/*December 2015: Minor refactor in order to enable the creation of a quality plugin for
 *chaining scansun in memory like other algorithms.*/
/*October 2018: original code using Numerical Recipes has been replaced.*/
/*April 2019: Now ODIM v2.1 files can be processed too. Additionally, the use of astart has been
 *adjusted to be ODIM compliant, and stopaz is estimated in case stopazA is not available.*/
/* Note that Iwan's code is largely left 'as is', except for the 'fill_meta' and
 * 'scansun' function which are restructured and modularized. Definitions and structures have been
 * placed in their own header file. */

#include "scansun.h"


/******************************************************************************/
/*LOCAL FUNCTIONS:                                                            */
/******************************************************************************/

int getDoubleAttribute(RaveCoreObject* obj, const char* aname, double* tmpd) {
	RaveAttribute_t* attr = NULL;
	int ret = 0;

	if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)) {
		attr = PolarVolume_getAttribute((PolarVolume_t*)obj, aname);
	} else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)) {
		attr = PolarScan_getAttribute((PolarScan_t*)obj, aname);
	} else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScanParam_TYPE)) {
		attr = PolarScanParam_getAttribute((PolarScanParam_t*)obj, aname);
	}
	if (attr != NULL) {
		ret = RaveAttribute_getDouble(attr, tmpd);
	}
	RAVE_OBJECT_RELEASE(attr);
	return ret;
}


int getDoubleArrayAttribute(PolarScan_t* scan, const char* aname, double** array, PolarScanParam_t* param) {
  RaveAttribute_t* attr = NULL;
  int len = 0;
  int ret = 0;

  attr = PolarScan_getAttribute(scan, aname);
  if (attr == NULL && param != NULL) {
    if (PolarScanParam_hasAttribute(param, aname)) {
      attr = PolarScanParam_getAttribute(param, aname);
    }
  }

  if (attr != NULL) {
    len = (int)PolarScan_getNbins(scan);
    ret = RaveAttribute_getDoubleArray(attr, array, &len);
  } else {
    *array = NULL;
  }
  RAVE_OBJECT_RELEASE(attr);
  return ret;
}


/* Unfortunate duplication of functionality to determine the location of required metadata.
 * This is, however, the nature of ODIM. */

void fill_toplevelmeta(RaveCoreObject* object, SCANMETA *meta)
{
  double tmpd;
  PolarScan_t* scan = NULL;

  if (!getDoubleAttribute(object, "how/wavelength", &tmpd)) {
    RAVE_WARNING1("No /how/wavelength attribute. Using default %2.3f m.\n", WAVELENGTH);
    meta->wavelength = WAVELENGTH;
  } else meta->wavelength = tmpd/100.0;  /* cm to m */

  if (!getDoubleAttribute(object, "how/rpm", &tmpd)) {
    RAVE_WARNING1("No /how/rpm attribute. Using default %2.1f deg/s.\n", ANTVEL);
    meta->antvel = ANTVEL;
  } else meta->antvel = tmpd * 6.0;  /* 360degrees/60sec */

  if (!getDoubleAttribute(object, "how/astart", &tmpd)) {
    RAVE_WARNING1("No /how/astart attribute. Using default %1.1f deg.\n", ASTART);
    meta->astart = ASTART;
  } else meta->astart = tmpd;

  if (!getDoubleAttribute(object, "how/pulsewidth", &tmpd)) {
    RAVE_WARNING1("No /how/pulsewidth attribute. Using default %1.2f microseconds.\n", PULSEWIDTH);
    meta->pulse = PULSEWIDTH;
  } else meta->pulse = tmpd;

  if (!getDoubleAttribute(object, "how/RXbandwidth", &tmpd)) {
    RAVE_WARNING1("No /how/RXbandwidth attribute. Using default %1.2f MHz.\n", CWIDTH/meta->pulse);
    meta->bandwidth = CWIDTH/meta->pulse;
  } else meta->bandwidth = tmpd;

  if (RAVE_OBJECT_CHECK_TYPE(object, &PolarVolume_TYPE)) {
    scan = PolarVolume_getScan((PolarVolume_t*)object, 0);  /* Assume first scan contains what we want */
  } else if (RAVE_OBJECT_CHECK_TYPE(object, &PolarScan_TYPE)) {
    scan = (PolarScan_t*)RAVE_OBJECT_COPY(object);
  }

  /* Use availability of quantities as a qualifier for determining which radar constant to look for */
  if ( (PolarScan_hasParameter(scan, "TH")) || (PolarScan_hasParameter(scan, "DBZH")) || (PolarScan_hasParameter(scan, "ZDR")) ) {
    if (!getDoubleAttribute(object, "how/radconstH", &tmpd)) {
      RAVE_WARNING1("No /how/radconstH attribute. Using default %2.1f dB.\n", RADCNST);
      meta->radcnst = RADCNST;
    } else meta->radcnst = tmpd;

    if (!getDoubleAttribute(object, "how/RXloss", &tmpd) && !getDoubleAttribute(object, "how/RXlossH", &tmpd)) {
      RAVE_WARNING1("No /how/RXloss or /how/RXlossH attribute. Using default %2.1f dB.\n", RXLOSS);
      meta->RXLoss = RXLOSS;
    } else meta->RXLoss = tmpd;

    if (!getDoubleAttribute(object, "how/antgain", &tmpd) && !getDoubleAttribute(object, "how/antgainH", &tmpd)) {
      RAVE_WARNING1("No /how/antgain or /how/antgainH attribute. Using default %2.1f dB.\n", ANTGAIN);
      meta->AntGain = ANTGAIN;
    } else meta->AntGain = tmpd;

  } else if ( (PolarScan_hasParameter(scan, "TV")) || (PolarScan_hasParameter(scan, "DBZV")) ) {
    if (!getDoubleAttribute(object, "how/radconstV", &tmpd)) {
      RAVE_WARNING1("No /how/radconstV attribute. Using default %2.1f dB.\n", RADCNST);
      meta->radcnst = RADCNST;
    } else meta->radcnst = tmpd;

    if (!getDoubleAttribute(object, "how/RXloss", &tmpd) && !getDoubleAttribute(object, "how/RXlossV", &tmpd)) {
      RAVE_WARNING1("No /how/RXloss or /how/RXlossV attribute. Using default %2.1f dB.\n", RXLOSS);
      meta->RXLoss = RXLOSS;
    } else meta->RXLoss = tmpd;

    if (!getDoubleAttribute(object, "how/antgain", &tmpd) && !getDoubleAttribute(object, "how/antgainV", &tmpd)) {
      RAVE_WARNING1("No /how/antgain or /how/antgainV attribute. Using default %2.1f dB.\n", ANTGAIN);
      meta->AntGain = ANTGAIN;
    } else meta->AntGain = tmpd;
  }

  RAVE_OBJECT_RELEASE(scan);
  return;
}


void fill_meta(PolarScan_t* scan, PolarScanParam_t* param, SCANMETA *meta)
{
   const char *date, *time, *quant;
   double tmpd = 0.0;

   quant = PolarScanParam_getQuantity(param);
   date = PolarScan_getStartDate(scan);
   time = PolarScan_getStartTime(scan);
   sscanf(date,"%ld",&meta->date);
   sscanf(time,"%ld",&meta->time);
   meta->lon = PolarScan_getLongitude(scan)*RAD2DEG; /* RAVE quirk */
   meta->lat = PolarScan_getLatitude(scan)*RAD2DEG;
   meta->elev = PolarScan_getElangle(scan)*RAD2DEG;
   meta->nrang = PolarScan_getNbins(scan);
   meta->nazim = PolarScan_getNrays(scan);
   meta->azim0 = PolarScan_getA1gate(scan);
   meta->rscale = PolarScan_getRscale(scan)/1000.0; /* Scale back to km */
   meta->ascale = 360.0 / meta->nazim;              /* First guess. FIXME?? */

   /* Don't overwrite metadata attributes in "meta" because they may already be populated by fill_toplevelmeta */
   if (!getDoubleAttribute((RaveCoreObject*)scan, "how/wavelength", &tmpd)) {
     RAVE_WARNING2("Scan elevation %2.1f: No how/wavelength attribute. Using %2.3f m.\n", meta->elev, meta->wavelength);
   } else meta->wavelength = tmpd/100.0;  /* cm to m */

   if (!getDoubleAttribute((RaveCoreObject*)scan, "how/rpm", &tmpd)) {
     RAVE_WARNING2("Scan elevation %2.1f: No how/rpm attribute. Using %2.1f deg/s.\n", meta->elev, meta->antvel);
   } else meta->antvel = tmpd * 6.0;  /* 360degrees/60sec */

   if (!getDoubleAttribute((RaveCoreObject*)scan, "how/astart", &tmpd)) {
     RAVE_WARNING2("Scan elevation %2.1f: No how/astart attribute. Using %1.2f deg.\n", meta->elev, meta->astart);
   } else meta->astart = tmpd;

   if (!getDoubleAttribute((RaveCoreObject*)scan, "how/pulsewidth", &tmpd)) {
     RAVE_WARNING2("Scan elevation %2.1f: No how/pulsewidth attribute. Using %2.2f microseconds.\n", meta->elev, meta->pulse);
   } else meta->pulse = tmpd;

   if (!getDoubleAttribute((RaveCoreObject*)scan, "how/RXbandwidth", &tmpd)) {
     RAVE_WARNING2("Scan elevation %2.1f: No how/RXbandwidth attribute. Using %1.2f MHz.\n", meta->elev, meta->CWIDTH/meta->pulse);
     meta->bandwidth = CWIDTH/meta->pulse;
   } else meta->bandwidth = tmpd;

   /* Would be possible that radar constants are found in individual quantity how,
    * but this implies unnecessarily replicated information, so we will only look at the scan level. */
   if ( (!strcmp(quant, "TH")) || (!strcmp(quant, "DBZH")) || (!strcmp(quant, "ZDR")) ) {
     if (!getDoubleAttribute((RaveCoreObject*)scan, "how/radconstH", &tmpd)) {
       RAVE_WARNING2("Scan elevation %2.1f: No how/radconstH attribute. Using %2.1f dB.\n", meta->elev, meta->radcnst);
     } else meta->radcnst = tmpd;

     if (!getDoubleAttribute((RaveCoreObject*)scan, "how/RXloss", &tmpd) && !getDoubleAttribute((RaveCoreObject*)scan, "how/RXlossH", &tmpd)) {
       RAVE_WARNING2("Scan elevation %2.1f: No how/RXloss or how/RXlossH attribute. Using %2.1f dB.\n", meta->elev, meta->RXLoss);
     } else meta->RXLoss = tmpd;

     if (!getDoubleAttribute((RaveCoreObject*)scan, "how/antgain", &tmpd) && !getDoubleAttribute((RaveCoreObject*)scan, "how/antgainH", &tmpd)) {
       RAVE_WARNING2("Scan elevation %2.1f: No how/antgain or how/antgainH attribute. Using %2.1f dB.\n", meta->elev, meta->AntGain);
     } else meta->AntGain = tmpd;

   } else if ( (!strcmp(quant, "TV")) || (!strcmp(quant, "DBZV")) ) {
     if (!getDoubleAttribute((RaveCoreObject*)scan, "how/radconstV", &tmpd)) {
       RAVE_WARNING2("Scan elevation %2.1f: No how/radconstV attribute. Using %2.1f dB.\n", meta->elev, meta->radcnst);
     } else meta->radcnst = tmpd;

     if (!getDoubleAttribute((RaveCoreObject*)scan, "how/RXloss", &tmpd) && !getDoubleAttribute((RaveCoreObject*)scan, "how/RXlossV", &tmpd)) {
       RAVE_WARNING2("Scan elevation %2.1f: No how/RXloss or how/RXlossV attribute. Using %2.1f dB.\n", meta->elev, meta->RXLoss);
     } else meta->RXLoss = tmpd;

     if (!getDoubleAttribute((RaveCoreObject*)scan, "how/antgain", &tmpd) && !getDoubleAttribute((RaveCoreObject*)scan, "how/antgainV", &tmpd)) {
       RAVE_WARNING2("Scan elevation %2.1f: No how/antgain or how/antgainV attribute. Using %2.1f dB.\n", meta->elev, meta->AntGain);
     } else meta->AntGain = tmpd;
   }

   /* Additional metadata arrays containing read-out angles and times */
   if (!getDoubleArrayAttribute(scan, "how/startazA", &meta->startazA, param)) {
     RAVE_WARNING1("Scan elevation %2.1f: No how/startazA array attribute. Estimating azimuth angle from polar geometry.\n", meta->elev);
   } else {
     if (!getDoubleArrayAttribute(scan, "how/stopazA", &meta->stopazA, param)) {
       RAVE_WARNING1("Scan elevation %2.1f: how/startazA found but not how/stopazA array attribute.\n", meta->elev);
     }
   }
   if (!getDoubleArrayAttribute(scan, "how/elangles", &meta->elangles, param)) {
     RAVE_WARNING1("Scan elevation %2.1f: No how/elangles array attribute. Using commanded elevation angle.\n", meta->elev);
   }
   if (!getDoubleArrayAttribute(scan, "how/startazT", &meta->startazT, param)) {
     RAVE_WARNING1("Scan elevation %2.1f: No how/startazT array attribute.\n", meta->elev);
   }
   if (!getDoubleArrayAttribute(scan, "how/stopazT", &meta->stopazT, param)) {
     RAVE_WARNING1("Scan elevation %2.1f: No how/stopazT array attribute.\n", meta->elev);
   }

return;
}


double ElevHeig2Rang(double elev,float heig)
{
double rang;
rang=RADIUS43*sin(DEG2RAD*elev);
rang=sqrt(rang*rang+2*RADIUS43*heig+heig*heig)-RADIUS43*sin(DEG2RAD*elev);
return rang;
}


void datetime(long date1, long time1, long ss, long *date2, long *time2)
{
struct tm timestruct;

/*Initializing of time structure.*/

timestruct.tm_year=(date1/10000)-1900;
timestruct.tm_mon=((date1/100)%100)-1;
timestruct.tm_mday=(date1%100);
timestruct.tm_hour=(int)(time1/10000);
timestruct.tm_min=(int)((time1/100)%100);
timestruct.tm_sec=(int)(time1%100)+ss;
timegm(&timestruct);

/*Calculation of new date.*/

*date2=(timestruct.tm_year+1900)*10000+(timestruct.tm_mon+1)*100;
(*date2)+=timestruct.tm_mday;

/*Calculation of new time.*/

*time2=10000*timestruct.tm_hour+100*timestruct.tm_min+timestruct.tm_sec;

/*End*/

return;
}


double refraction(double *elev)
{
double refr;

/*Using newly derived formula for refraction consistent with equivalent earth*/
/*model.*/

refr=sin(DEG2RAD*(*elev))*sin(DEG2RAD*(*elev));
refr+=(4*KFACT-2)*N0*1e-6/(KFACT-1);
refr=sqrt(refr)-sin(DEG2RAD*(*elev));
refr*=cos(DEG2RAD*(*elev))*(KFACT-1)/(2*KFACT-1);

/*Return refraction in degrees.*/

return refr*RAD2DEG;
}


void solar_elev_azim(double lon, double lat, long yyyymmdd, long hhmmss, double *elev, double *azim, double *relev)
{
float MeanLon,MeanAnom,EclipLon,Obliquity,RightAsc,Declinat;
float GMST,angleH;
double days,hour;
struct tm timestruct;
time_t time1, time0;

/*Conversion of lon,lat.*/

lon*=DEG2RAD;
lat*=DEG2RAD;

/*Initializing of time structure with actual date and time.*/

timestruct.tm_year=(yyyymmdd/10000)-1900;
timestruct.tm_mon=((yyyymmdd/100)%100)-1;
timestruct.tm_mday=(yyyymmdd%100);
timestruct.tm_hour=(int)(hhmmss/10000);
timestruct.tm_min=(int)((hhmmss/100)%100);
timestruct.tm_sec=(int)(hhmmss%100);
timestruct.tm_isdst=0;
time1=mktime(&timestruct);

/*Initializing of time structure with reference (noon 1 Jan 2000).*/

timestruct.tm_year=100;
timestruct.tm_mon=0;
timestruct.tm_mday=1;
timestruct.tm_hour=12;
timestruct.tm_min=0;
timestruct.tm_sec=0;
timestruct.tm_isdst=0;
time0=mktime(&timestruct);

/*Calculation of fractional days.*/

days=difftime(time1,time0)/(24*3600);

/*Calculation of eclips coordinates.*/

MeanLon=280.460+0.9856474*days;
MeanAnom=357.528+0.9856003*days;
EclipLon=MeanLon+1.915*sin(MeanAnom*DEG2RAD)+0.020*sin(2*MeanAnom*DEG2RAD);
EclipLon*=DEG2RAD;
Obliquity=23.439-0.0000004*days;
Obliquity*=DEG2RAD;

/*Calculation of the celestial coordinates of the sun.*/

RightAsc=atan2(cos(Obliquity)*sin(EclipLon),cos(EclipLon));
Declinat=asin(sin(Obliquity)*sin(EclipLon));

/*Calculation of current, local hour angle.*/

hour=(double)(hhmmss/10000)+((hhmmss/100)%100)/60.0+(hhmmss%100)/3600.0;
GMST=hour+6.697375+0.0657098242*days;
angleH=GMST*15*DEG2RAD+lon-RightAsc;

/*Calculation of elevation and azimuth.*/

*elev=asin(sin(Declinat)*sin(lat)+cos(Declinat)*cos(lat)*cos(angleH));
*azim=atan2(-sin(angleH),tan(Declinat)*cos(lat)-cos(angleH)*sin(lat));

/*Scaling and shifting of values.*/

(*elev)*=RAD2DEG;
(*azim)*=RAD2DEG;
if ((*azim)<0) (*azim)+=360;

/* Refraction */
*relev = (*elev) + refraction(elev);
}


void readoutTiming(SCANMETA* meta, int ia, long* date, long* time, double* timer) {
  double timed, timef;
  char datebuf[9];  /* YYYYMMDD\0 */
  char timebuf[7];  /* HHMMSS\0 */
  char *ptr;

  if ( (!meta->stopazT) && (meta->startazT) ) timed = meta->startazT[ia];
  else if ( (!meta->startazT) && (meta->stopazT) ) timed = meta->stopazT[ia];
  else timed = (meta->startazT[ia] + meta->stopazT[ia]) / 2.0;
  *timer = modf(timed, &timef);

  const time_t timet = (time_t)timef;
  const struct tm *timetm = gmtime(&timet);
  struct tm buf;
  timetm = gmtime_r(&timet, &buf);
  strftime(datebuf, sizeof datebuf, "%Y%m%d", timetm);
  strftime(timebuf, sizeof timebuf, "%H%M%S", timetm);
  *date = strtol(datebuf, &ptr, 10);
  *time = strtol(timebuf, &ptr, 10);
}


int processData(PolarScan_t* scan, SCANMETA* meta, RaveList_t* list) {
  int ret = 1;
  int ia, ir, irn1, irn2, n;
  long date,time,addtime;
  double HeigMin1=HEIGMIN1,HeigMin2=HEIGMIN2,FracData=FRACDATA,dBdifX=DBDIFX,AngleDif=ANGLEDIF;
  double RayMin=RAYMIN,GasAttn=GASATTN,lonlat[2],Elevation,Azimuth,Range,SunFirst;
  double SunMean,SunStdd,dBSunFlux,ZdrMean,ZdrStdd,Signal,DSignal,timer;
  PolarScanParam_t* Zparam = NULL;
  PolarScanParam_t* Dparam = NULL;
  RaveValueType t;

  /* Note that RAVE reads coordinates directly into radians. */
  lonlat[0] = PolarScan_getLongitude(scan)*RAD2DEG;
  lonlat[1] = PolarScan_getLatitude(scan)*RAD2DEG;

  Zparam = PolarScan_getParameter(scan, meta->quant1);  /* Extract reflectivity */
  if (meta->Zdr) Dparam = PolarScan_getParameter(scan, meta->quant2);  /* Extract quant for ZDR */

  fill_meta(scan,Zparam,meta);

  if (meta->nrang*meta->rscale<RayMin) {
	  ret = 0;
	  goto fail_scan;
  }
  irn1=(int)(ElevHeig2Rang(meta->elev,HeigMin1)/meta->rscale);
  if ((meta->nrang-irn1)*meta->rscale<RayMin) irn1=meta->nrang-RayMin/meta->rscale;
  irn2=(int)(ElevHeig2Rang(meta->elev,HeigMin2)/meta->rscale);
  if ((meta->nrang-irn2)*meta->rscale<RayMin) irn2=meta->nrang-RayMin/meta->rscale;

  for (ia=0 ; ia<meta->nazim ; ia++) {
    RVALS* rvals = RAVE_MALLOC(sizeof(RVALS));
    memset(rvals, 0, sizeof(RVALS));
    timer = 0.0;

    /* Use exact azimuth and elevation angles if available */
    if (meta->startazA) {
      double startaz = meta->startazA[ia];
      double stopaz;
      if (meta->stopazA) {
        stopaz = meta->stopazA[ia];
      } else {
        stopaz = startaz + meta->ascale;
      }
      /* Most radars scan clockwise, but negative antvel indicates otherwise */
      if (meta->antvel > 0.0) {
        if (startaz > stopaz) startaz = -(360.0-startaz);
      } else {
        if (stopaz > startaz) stopaz = -(360.0-stopaz);
      }
      Azimuth = (startaz + stopaz) / 2.0;
    } else {
      Azimuth = ia * meta->ascale + meta->astart + meta->ascale / 2.0;
    }

//    if (meta->startelA) Elevation = (meta->startelA[ia] + meta->stopelA[ia]) / 2.0;
    if (meta->elangles) Elevation = meta->elangles[ia];
    else Elevation = meta->elev;

    /*First analysis to estimate sun power at higher altitudes (less rain contamination).*/

    n=0;
    SunFirst=0.0;
    for (ir=irn1 ; ir<meta->nrang ; ir++) {
      /* Reads out the scaled value directly. If 't' isn't
         RaveValueType_NODATA or RaveValueType_UNDETECT, then process. */
      t = PolarScanParam_getConvertedValue(Zparam,ir,ia,&Signal);
      if (t == RaveValueType_DATA) {
        Range=(ir+0.5)*meta->rscale;
        Signal-=meta->radcnst+20*log10(Range)+GasAttn*Range;
        SunFirst+=Signal;
        n++;
      }
    }
    if (!n||n<FracData*(meta->nrang-irn1)) {
      RAVE_FREE(rvals);
      continue;
    }
    SunFirst/=n;

    /*Second analysis with removal of outliers, if available also ZDR analysis.*/

    n=0;
    SunMean=SunStdd=dBSunFlux=0.0;
    ZdrMean=ZdrStdd=0.0;
    for (ir=irn2 ; ir<meta->nrang ; ir++) {
      t = PolarScanParam_getConvertedValue(Zparam,ir,ia,&Signal);
      if (t == RaveValueType_DATA) {
        Range=(ir+0.5)*meta->rscale;
        Signal-=meta->radcnst+20*log10(Range)+GasAttn*Range;
        if (fabs(Signal-SunFirst)>dBdifX) continue;
        SunMean+=Signal;
        SunStdd+=Signal*Signal;
        if (meta->Zdr) {
          t = PolarScanParam_getConvertedValue(Dparam,ir,ia,&DSignal);
          if (t == RaveValueType_DATA) {  /* This condition can affect the validity of n, but it will also guarantee no zero-division errors. */
            if (meta->Zdr == ZdrType_CALCULATE) {
              double H = pow(10.0,(Signal/10.0));
              double V = pow(10.0,(DSignal/10.0));
              DSignal = 10*log10(H/V);
            }
            ZdrMean+=DSignal;
            ZdrStdd+=DSignal*DSignal;
          }
        }
        n++;
      }
    }
    if (!n||n<FracData*(meta->nrang-irn2)) {
      RAVE_FREE(rvals);
      continue;
    }
    SunMean/=n;
    SunStdd=sqrt(SunStdd/n-SunMean*SunMean+1e-8);
    SunMean-=10*log10(meta->bandwidth);
    if (meta->Zdr) {
       ZdrMean/=n;
       ZdrStdd=sqrt(ZdrStdd/n-ZdrMean*ZdrMean+1e-8);
    }

    /*Conversion to SunFlux*/
    meta->LAntGain = pow(10.0,meta->AntGain/10.0);
    meta->AntArea = (pow(meta->wavelength,2.0) * meta->LAntGain) / (4*M_PI);
    dBSunFlux=130+SunMean+meta->RXLoss-10*log10(meta->AntArea)+AVGATTN+ONEPOL;

    /*Appending of results to return list.*/

    /* First timing approximation */
    addtime=((ia-meta->azim0+meta->nazim)%meta->nazim)*meta->ascale/fabs(meta->antvel);
    datetime(meta->date,meta->time,addtime,&date,&time);

    /* Replace with exact timing if available */
    if ( (meta->startazT) || (meta->stopazT) ) {
      readoutTiming(meta, ia, &date, &time, &timer);
    }

    solar_elev_azim(lonlat[0],lonlat[1],date,time,&rvals->ElevSun,&rvals->AzimSun,&rvals->RelevSun);
    if (fabs(Elevation-rvals->RelevSun)>AngleDif||fabs(Azimuth-rvals->AzimSun)>AngleDif) {
      RAVE_FREE(rvals);
      continue;
    }
    rvals->date = date;
    rvals->time = time;
    rvals->timer = time + timer;
    rvals->Elev = Elevation;
    rvals->Azimuth = Azimuth;
    rvals->dBSunFlux = dBSunFlux;
    rvals->SunMean = SunMean;
    rvals->SunStdd = SunStdd;
    rvals->n = n;
    strcpy(rvals->quant1, meta->quant1);
    if (meta->Zdr) {
      rvals->ZdrMean = ZdrMean;
      rvals->ZdrStdd = ZdrStdd;
      strcpy(rvals->quant2, meta->quant2);
    } else {
      rvals->ZdrMean = nan("NaN");
      rvals->ZdrStdd = nan("NaN");
      strcpy(rvals->quant2, "NA");
    }
    RaveList_add(list, rvals);  /* No checking */
    if (Rave_getDebugLevel() <= RAVE_DEBUG) outputMeta(meta);
  }
fail_scan:
  RAVE_OBJECT_RELEASE(Zparam);
  RAVE_OBJECT_RELEASE(Dparam);
  return ret;
}


/* Useful for debugging */
void outputMeta(SCANMETA* meta) {
  FILE *fd;
  fd=fopen("meta.txt", "w");
  if (fd != NULL) {
    fprintf(fd, "Date = %ld\n", meta->date);
    fprintf(fd, "Time = %ld\n", meta->time);
    fprintf(fd, "Tilt = %f\n", meta->elev);
    fprintf(fd, "Quant1 = %s\n", (strcmp(meta->quant1,"")==0?"NULL":meta->quant1));
    fprintf(fd, "Quant2 = %s\n", (strcmp(meta->quant2,"")==0?"NULL":meta->quant2));
    fprintf(fd, "nrays = %ld\n", meta->nazim);
    fprintf(fd, "nbins = %ld\n", meta->nrang);
    fprintf(fd, "rscale = %f\n", meta->rscale);
    fprintf(fd, "ascale = %f\n", meta->ascale);
    fprintf(fd, "astart = %f\n", meta->astart);
    fprintf(fd, "azim0 = %ld\n", meta->azim0);
    fprintf(fd, "Pulse width = %f\n", meta->pulse);
    fprintf(fd, "Receiver bandwidth = %f\n", meta->bandwidth);
    fprintf(fd, "Radar constant = %f\n", meta->radcnst);
    fprintf(fd, "Antvel = %f\n", meta->antvel);
    fprintf(fd, "Longitude = %f\n", meta->lon);
    fprintf(fd, "Latitude = %f\n", meta->lat);
    fprintf(fd, "RXLoss = %f\n", meta->RXLoss);
    fprintf(fd, "Antenna gain = %f\n", meta->AntGain);
    fprintf(fd, "Linear antenna gain = %f\n", meta->LAntGain);
    fprintf(fd, "Antenna area = %f\n", meta->AntArea);
    fprintf(fd, "Wavelength = %f\n", meta->wavelength);
    fprintf(fd, "Zdr type = %d\n", meta->Zdr);
  }
  fclose(fd);
}


int processScan(PolarScan_t* scan, SCANMETA* meta, RaveList_t* list) {
  int ret = 0;

  /* Quantities are queried in order of priority */

  if (PolarScan_hasParameter(scan, "TH")) strcpy(meta->quant1, "TH");
  else if (PolarScan_hasParameter(scan, "DBZH")) strcpy(meta->quant1, "DBZH");
  else if (PolarScan_hasParameter(scan, "TV")) strcpy(meta->quant1, "TV");
  else if (PolarScan_hasParameter(scan, "DBZV")) strcpy(meta->quant1, "DBZV");
  else strcpy(meta->quant1,"");

  if (PolarScan_hasParameter(scan, "ZDR")) {
    strcpy(meta->quant2, "ZDR");
    meta->Zdr = ZdrType_READ;
  }
  else if ( (PolarScan_hasParameter(scan, "TV")) && (!strncmp(meta->quant1, "TH", 2)) ) {
    strcpy(meta->quant2, "TV");
    meta->Zdr = ZdrType_CALCULATE;
  }
  else if ( (PolarScan_hasParameter(scan, "DBZV")) && (!strncmp(meta->quant1, "DBZH", 4)) ) {
    strcpy(meta->quant2, "DBZV");
    meta->Zdr = ZdrType_CALCULATE;
  }
  else meta->Zdr = ZdrType_None;

  if (strcmp(meta->quant1, "") != 0) {
    ret = processData(scan, meta, list);
  }

  return ret;
}


int scansunFromObject(RaveCoreObject* object, Rave_ObjectType ot, RaveList_t* list, char** source) {
	int ret = 0;
	int Nscan, id;
	SCANMETA meta;
	PolarVolume_t* volume = NULL;
	PolarScan_t* scan = NULL;
	memset(&meta, 0, sizeof(SCANMETA));
	fill_toplevelmeta(object, &meta);

	/* Individual scan */
	if (ot == Rave_ObjectType_SCAN) {
	  scan = (PolarScan_t*)object;
	  if (source != NULL) *source = RAVE_STRDUP(PolarScan_getSource(scan));
	  ret = processScan(scan, &meta, list);

	/* Polar volume */
	} else if (ot == Rave_ObjectType_PVOL) {
	  volume = (PolarVolume_t*)object;
	  if (source != NULL) *source = RAVE_STRDUP(PolarVolume_getSource(volume));
	  Nscan = PolarVolume_getNumberOfScans(volume);

	  for (id=0 ; id<Nscan ; id++) {
	    scan = PolarVolume_getScan(volume, id);
	    ret = processScan(scan, &meta, list);  /* can fail for some scans and succeed for others */
	    RAVE_OBJECT_RELEASE(scan);
	  }
	}

	return ret;
}


int scansun(const char* filename, RaveList_t* list, char** source) {
	int ret = 0;
	RaveIO_t* raveio = RaveIO_open(filename, 0, NULL);
	if (raveio == NULL) {
	  goto done;
	}
	RaveCoreObject* object = NULL;
	Rave_ObjectType ot = Rave_ObjectType_UNDEFINED;

	/* Accessing contents of input file. */
	ot = RaveIO_getObjectType(raveio);
	if ( (ot == Rave_ObjectType_PVOL) || (ot == Rave_ObjectType_SCAN) ) {
		object = (RaveCoreObject*)RaveIO_getObject(raveio);
	}else{
		printf("Input file is neither a polar volume nor a polar scan. Giving up ...\n");
		RAVE_OBJECT_RELEASE(object);
		RAVE_OBJECT_RELEASE(raveio);
		return 0;
	}

	ret = scansunFromObject(object, ot, list, source);

	//if (ot == Rave_ObjectType_PVOL) {
	//	RaveIO_close(raveio);
	//}
  RAVE_OBJECT_RELEASE(raveio);

	RAVE_OBJECT_RELEASE(object);

done:
	return ret;
}
