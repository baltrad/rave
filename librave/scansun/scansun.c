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
/* Note that Iwan's code is largely left 'as is', except for the 'fill_meta' and
 * 'scansun' function which are restructured. Definitions and structures have been
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


int fill_meta(PolarScan_t* scan, PolarScanParam_t* dbzh, SCANMETA *meta)
{
   const char *date, *time;
   double tmpd = 0.0;

   date = PolarScan_getStartDate(scan);
   time = PolarScan_getStartTime(scan);
   sscanf(date,"%ld",&meta->date);
   sscanf(time,"%ld",&meta->time);
   meta->elev = PolarScan_getElangle(scan)*RAD2DEG; /* RAVE quirk */
   meta->nrang = PolarScan_getNbins(scan);
   meta->nazim = PolarScan_getNrays(scan);
   meta->azim0 = PolarScan_getA1gate(scan);
   meta->rscale = PolarScan_getRscale(scan)/1000.0; /* Scale back to km */
   meta->ascale = 360.0 / meta->nazim;              /* First guess. FIXME?? */

   if (!getDoubleAttribute((RaveCoreObject*)scan, "how/rpm", &tmpd)) meta->antvel = 18.;
   else meta->antvel = tmpd * 6.0;  /* 360¡/60sec */

   if (!getDoubleAttribute((RaveCoreObject*)scan, "how/pulsewidth", &tmpd)) meta->pulse = 2.0;
   else meta->pulse = tmpd;

   meta->zscale = PolarScanParam_getGain(dbzh);
   meta->zoffset = PolarScanParam_getOffset(dbzh);
   meta->nodata = PolarScanParam_getNodata(dbzh);

return 1;
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
int ss1,ss2,Nday;
ss1=3600*(time1/10000)+60*((time1/100)%100)+time1%100;
ss2=ss+ss1;
Nday=(int)floor((float)ss2/86400);
ss2-=Nday*86400;
*date2=julday2date(Nday+date2julday(date1));
*time2=10000*(ss2/3600)+100*((ss2/60)%60)+ss2%60;
return;
}


double refraction(double* elev)
{
double refr;
refr=4.5e-3*REFPRES/REFTEMP;
refr/=tan(DEG2RAD*((*elev)+8.0/((*elev)+4.23)));
return refr;
}


void solar_elev_azim(double lon, double lat, long yyyymmdd, long hhmmss, double *elev, double *azim, double *relev)
{
float MeanLon,MeanAnom,EclipLon,Obliquity,RightAsc,Declinat;
float GMST,angleH;
double julday,julday0,hour;

/*Conversion of lon,lat.*/

lon*=DEG2RAD;
lat*=DEG2RAD;

/*Calculation of fractional julian day, 2000 Jan 1 at noon is 2451545.*/

hour=(double)(hhmmss/10000)+((hhmmss/100)%100)/60.0+(hhmmss%100)/3600.0;
julday=(double)date2julday(yyyymmdd)+(hour-12)/24.0;
julday0=(double)date2julday(20000101);

/*Calculation of eclips coordinates.*/

MeanLon=280.460+0.9856474*(julday-julday0);
MeanAnom=357.528+0.9856003*(julday-julday0);
EclipLon=MeanLon+1.915*sin(MeanAnom*DEG2RAD)+0.020*sin(2*MeanAnom*DEG2RAD);
EclipLon*=DEG2RAD;
Obliquity=23.439-0.0000004*(julday-julday0);
Obliquity*=DEG2RAD;

/*Calculation of the celestial coordinates of the sun.*/

RightAsc=atan2(cos(Obliquity)*sin(EclipLon),cos(EclipLon));
Declinat=asin(sin(Obliquity)*sin(EclipLon));

/*Calculation of current, local hour angle.*/

GMST=6.697375+0.0657098242*(julday-julday0)+hour;
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

#define IGREG (15+31L*(10+12L*1582))
long date2julday(long yyyymmdd)
{
long jul,dd,mm,yyyy,ja,jy,jm;
dd=yyyymmdd%100;
mm=(yyyymmdd/100)%100;
yyyy=yyyymmdd/10000;
jy=yyyy;

if (jy == 0) printf("julday: there is no year zero!\n");
if (jy < 0) ++jy;
if (mm > 2) {
   jm=mm+1;
} 
else {
   --jy;
   jm=mm+13;
}
jul = (int) (floor(365.25*jy)+floor(30.6001*jm)+dd+1720995);
if (dd+31L*(mm+12L*yyyy) >= IGREG) {
   ja=(int)(0.01*jy);
   jul += 2-ja+(int) (0.25*ja);
}
return jul;
}
#undef IGREG 


#define IGREG 2299161
long julday2date(long julian)
{
long dd,mm,yyyy,ja,jalpha,jb,jc,jd,je;
if (julian >= IGREG) {
   jalpha=(int)(((float) (julian-1867216)-0.25)/36524.25);
   ja=julian+1+jalpha-(int) (0.25*jalpha);
} 
else ja=julian;
jb=ja+1524;
jc=(int)(6680.0+((float) (jb-2439870)-122.1)/365.25);
jd=(int)(365*jc+(0.25*jc));
je=(int)((jb-jd)/30.6001);
dd=jb-jd-(int) (30.6001*je);
mm=je-1;
if (mm > 12) mm -= 12;
yyyy=jc-4715;
if (mm > 2) --(yyyy);
if (yyyy <= 0) --(yyyy);
return dd+100*(mm+100*yyyy);
}
#undef IGREG 


int scansun(const char* filename, RaveList_t* list, char** source) {
	int Nscan, id, ia, ir, irn, n;
	long date,time,addtime;
	double lonlat[2], range, Azimuth, dBmSun, dBmStdd;
	double dBm;
	double tmpd;
	SCANMETA meta;
	RaveIO_t* raveio = RaveIO_open(filename);
	PolarVolume_t* volume = NULL;
	PolarScan_t* scan = NULL;
	PolarScanParam_t* dbzh = NULL;
	RaveValueType t;

	/*Opening of HDF5 radar input file.*/
	if (RaveIO_getObjectType(raveio) == Rave_ObjectType_PVOL) {
		volume = (PolarVolume_t*)RaveIO_getObject(raveio);
	}else{
		printf("Input file is not a polar volume. Giving up ...\n");
		RAVE_OBJECT_RELEASE(volume);
		RAVE_OBJECT_RELEASE(raveio);
		return 0;
	}

	/* Reading number of scans and radar location from file.
	 * Note that RAVE reads coordinates directly into radians. */

	Nscan = PolarVolume_getNumberOfScans(volume);
	lonlat[0] = PolarVolume_getLongitude(volume)*RAD2DEG;
	lonlat[1] = PolarVolume_getLatitude(volume)*RAD2DEG;
	if (source != NULL) *source = RAVE_STRDUP(PolarVolume_getSource(volume));

	/* The following attribute doesn't (yet) exist in ODIM in January 2011.
	 * Default value is for the Den Helder radar run by KNMI from its most
	 * recent calibration and valid for long pulse (2 us). */
	if (!getDoubleAttribute((RaveCoreObject*)volume, "how/rconst", &tmpd)) meta.radcnst = 64.08;
	else meta.radcnst = tmpd;

	/*Reading of PPI data and looking for solar interferences.*/

	for (id=0 ; id<Nscan ; id++) {
		scan = PolarVolume_getScan(volume, id);  /* Extract this scan */
		dbzh = PolarScan_getParameter(scan, "DBZH");  /* Extract this parameter */
		if (!fill_meta(scan,dbzh,&meta)) goto fail_scan;
		irn=(int)(ElevHeig2Rang(meta.elev,HEIGMIN)/meta.rscale);
		if (RANGMIN>irn*meta.rscale) irn=(int)(RANGMIN/meta.rscale);
		if (irn>=meta.nrang) goto fail_scan;
		for (ia=0 ; ia<meta.nazim ; ia++) {
			RVALS* ret = RAVE_MALLOC(sizeof(RVALS));
			Azimuth=(ia+0.5)*meta.ascale;
			dBmSun=dBmStdd=0.0;
			n=0;
			for (ir=irn ; ir<meta.nrang ; ir++) {
				/* Reads out the scaled value directly. If 't' isn't
	    		   RaveValueType_NODATA or RaveValueType_UNDETECT, then process. */
				t = PolarScanParam_getConvertedValue(dbzh,ir,ia,&dBm);
				if (t == RaveValueType_DATA) {
					range=(ir+0.5)*meta.rscale;
					dBm-=meta.radcnst+20*log10(range)+GASATTN*range;
					dBmSun+=dBm;
					dBmStdd+=dBm*dBm;
					n++;
				}
			}
			if (n==0||n<FRACDATA*(meta.nrang-irn)) {
				RAVE_FREE(ret);
				continue;
			}
			dBmSun/=n;
			dBmStdd=sqrt(dBmStdd/n-dBmSun*dBmSun+1e-8);
			dBmSun-=10*log10(CWIDTH/meta.pulse);
			addtime=((ia-meta.azim0+meta.nazim)%meta.nazim)*meta.ascale/meta.antvel;
			datetime(meta.date,meta.time,addtime,&date,&time);
			solar_elev_azim(lonlat[0],lonlat[1],date,time,&ret->ElevSun,&ret->AzimSun,&ret->RelevSun);
			if (fabs(meta.elev-ret->ElevSun)>ANGLEDIF||fabs(Azimuth-ret->AzimSun)>ANGLEDIF) {
				RAVE_FREE(ret);
				continue;
			}
			ret->Azimuth = Azimuth;
			ret->dBmSun = dBmSun;
			ret->dBmStdd = dBmStdd;
			ret->date = date;
			ret->time = time;
			ret->Elev = meta.elev;
			RaveList_add(list, ret);  /* No checking */
		}
		fail_scan:
			RAVE_OBJECT_RELEASE(dbzh);
			RAVE_OBJECT_RELEASE(scan);
	}
	RAVE_OBJECT_RELEASE(volume);
	RAVE_OBJECT_RELEASE(raveio);

	return 1;
}
