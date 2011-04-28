/**

    Copyright 2002 - 2010  Harri Hohti (minor edits Markus Peura),
    Finnish Meteorological Institute (First.Last@fmi.fi)


    This file is part of Rack.

    Rack is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Rack is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser Public License for more details.

    You should have received a copy of the GNU Lesser Public License
    along with Rack.  If not, see <http://www.gnu.org/licenses/>.

*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <limits.h>
#define KOR 0
#define VAN 1
#define ANJ 2
#define IKA 3
#define KUO 4
#define UTA 5
#define LUO 6


/* 2001 (C) Harri.Hohti@fmi.fi */

static double DEG_TO_RAD,RAD_TO_DEG;

typedef struct { double RA;
                 double Dec;
               } Equatorial;

typedef struct {  double A;
                  double h;
               } Horizontal;

typedef struct { double Lat;
                 double Lon;
               } Geographical;

/* Julian date */
static double JDate(char *datestr);

/* 0-meridiaanin astronomic time (tähtiaika) */
static double Sidereal_GW(double JD);

/* Equatorial coordinates of the Sun */
static Equatorial Solar_coords(double JD);

/* Horizontal coordinates of the Sun (no refraction) */
static Horizontal Solar_pos(double Sd0,
                            Geographical Geo, 
                            Equatorial Equ);


void sunPosition(double latDeg, double lonDeg,char *timestamp,double *azimuth,double *elevation){

	double JD, Sd0;
	Equatorial Equ;
	Horizontal Hor;
	Geographical Geo;

	Geo.Lat = latDeg * DEG_TO_RAD;
	Geo.Lon = lonDeg * DEG_TO_RAD;

	JD = JDate(timestamp);

	/* T�htiaika radiaaneina auringon tuntikulman laskentaa varten */
	Sd0 = Sidereal_GW(JD)*15.0*DEG_TO_RAD;

	/* Lasketaan auringon ekvatoriaalikoordinaatit ajan funktiona */
	Equ = Solar_coords(JD);

	/* Lasketaan auringon horisontaalikoordinaatit paikan ja ajan funktiona */
	Hor = Solar_pos(Sd0,Geo,Equ);

	*azimuth = Hor.A;
	*elevation = Hor.h;

}

/// 10.11.2010
int mainOld(int argc, char *argv[])
{

   double JD,Sd0;
   int r=0;
   Equatorial Equ;
   Horizontal Hor;
   Geographical Geo;

   /* Tutkien latitudit ja longitudit radiaaneina */
   char radar[7][4]={"KOR","VAN","ANJ","IKA","KUO","UTA","LUO"};
   Geographical Radars[7]={{1.0492337686,0.37757289484},
                           {1.0518517625,0.43400520732},
                           {1.0629055144,0.47298422728},
                           {1.0780317013,0.40258928078},
                           {1.0969394348,0.47792932682},
                           {1.1303915788,0.45931248147},
                           {1.1716977045,0.46949356877}};                        

   DEG_TO_RAD=M_PI/180.0;
   RAD_TO_DEG=180.0/M_PI;

   /* JD=JDate(argv[1]); */

   switch(argc){
   case 4:
   /* sunpos 200203011547 35.0 62.0 */
     JD=JDate(argv[1]);
     Geo.Lat=atof(argv[2])*DEG_TO_RAD;
     Geo.Lon=atof(argv[3])*DEG_TO_RAD;
     break;
   case 2:
   /* sunpos KOR200203011547 */
     JD=JDate(&argv[1][3]);
     for(r=0;r<7;r++) if(strncasecmp(argv[1],radar[r],3)==0) break;
     Geo=Radars[r];
     break;
   case 3:
   /* sunpos 200203011547 KOR */
     JD=JDate(argv[1]);
     for(r=0;r<7;r++) if(strncasecmp(argv[2],radar[r],3)==0) break;
     Geo=Radars[r];
     break;
   default:
     fprintf(stderr,"%s : illegal parameter count: %d\n",argv[0],argc);
     fprintf(stderr,"usage:\n");
     fprintf(stderr,"%s KOR200203011547\n",argv[0]);
     fprintf(stderr,"%s 200203011547 KOR\n",argv[0]);
     fprintf(stderr,"%s 200203011547 35.0 62.0 \n",argv[0]);
     return (-1);
   }

   if(r==7) return(1);

   /* T�htiaika radiaaneina auringon tuntikulman laskentaa varten */
   Sd0=Sidereal_GW(JD)*15.0*DEG_TO_RAD;

   /* Lasketaan auringon ekvatoriaalikoordinaatit ajan funktiona */
   Equ=Solar_coords(JD);

   /* Lasketaan auringon horisontaalikoordinaatit paikan ja ajan funktiona */
   Hor=Solar_pos(Sd0,Geo,Equ);

   /*   printf("A = %.4f\nh = %.4f\n",Hor.A,Hor.h); */
   printf("%.1f\t%.1f\n",Hor.A,Hor.h);

   return(0);
}

static double Sidereal_GW(double JD)
{ 
   double JD0,d,T,SDR,SD_0GW,SD_GW;

   JD0=(double)((long int)(JD-0.5));
   d=(JD-0.5-JD0)*24.0;
   T=(JD0+0.5-2415020.0)/36525.0;

   /*   printf("%f %.16f\n",JD0+0.5,T); */

   /* Lasketaan t�htiaika 0-meridiaanilla 0 UTC */
   SDR=0.276919398+100.0021359*T+0.000001075*T*T;
   SD_0GW=(SDR-(double)((long int)SDR))*24.0;

   /* ja lis�t��n siihen todellinen aika t�htiajaksi muunnettuna,
      jotta saadaan t�htiaika halutulle ajanhetkelle */
   SD_GW=SD_0GW+d*1.002737908;
 
   return(SD_GW);
}


double JDate(char *datestr)
{
   int YY,MM,DD,hh,mm,A,B,y,m;
   double d,JD=0.0,fy,fm;

   sscanf(datestr,"%4d%2d%2d%2d%2d",&YY,&MM,&DD,&hh,&mm);
      
   y=YY;
   m=MM;
   if(MM<3)
   {
     y=YY-1;
     m=MM+12;
   }
   A=y/100;
   B=2-A+A/4;
   fy=(double)y;
   fm=(double)m;

   d=(double)DD+(double)hh/24.0+(double)mm/1440.0;
   JD=(double)((long int)(fy*365.25))+(double)((long int)(30.6001*(fm+1)))
      +d+1720994.5+(double)B;
      
   return(JD);
}   

static Equatorial Solar_coords(double JD)
{

   double T,TT,L,M,e,OL,C,W,y,x,OA,ep;
   Equatorial Equ;

   T=(JD-2415020.0)/36525.0;
   TT=T*T;

   /* Auringon keskilongitudi */
   L=279.69668+36000.76892*T+0.0003025*TT;

   /* Auringon keskianomalia */
   M=358.47583+35999.04975*T-0.00015*TT-0.0000033*T*TT;

   /* Maan radan eksentrisyys */
   e=0.01675104-0.0000418*T-0.000000126*TT;

   /* printf("M=%.12f\nL=%.12f\ne=%.12f\n",M,L,e); */

   M*=DEG_TO_RAD;

   /* Auringon "equation of the center" */
   C=(1.919460-0.004789*T-0.000014*TT)*sin(M);
   C+=(0.020094-0.0001*T)*sin(2*M);
   C+=0.000293*sin(3*M);

   /* Auringon longitudi */
   OL=L+C;
   /*   printf("OL=%.12f\nC=%.12f\n",OL,C); */

   /* Maan nousevan solmun longitudi */
   W=(259.18-1934.142*T)*DEG_TO_RAD;

   /* Auringon longitudi korjattuna todelliseen epookkiin */
   OA=OL-0.00569-0.00479*sin(W);
   /* printf("Oapp=%.12f\n",OA); */
   
   OA*=DEG_TO_RAD;

   /* ekliptikan kaltevuus */
   ep=23.452294-0.0130125*T-0.00000164*TT+0.000000503*T*TT+0.00256*cos(W);
   ep*=DEG_TO_RAD;

   /* Muunnos auringon ekliptikaalisista koordinaateista
      (longitudi lambda, latitudi beta (auringolle < 1.2")) 
      ekvatoriaalisiin, rektaskensioon ja deklinaatioon */
   
   y=cos(ep)*sin(OA);
   x=cos(OA);

   Equ.Dec=asin(sin(ep)*sin(OA));
   Equ.RA=atan2(y,x);

   /* printf("RA=%.12f\nDec=%.12f\n",Equ.RA*RAD_TO_DEG/15.0,Equ.Dec*RAD_TO_DEG); */
   
   return(Equ);
}


static Horizontal Solar_pos(double Sd0,Geographical Geo, 
                            Equatorial Equ)
{

   double H,x,y;
   Horizontal Hor;

   /* Tuntikulma H on 0-meridiaanin t�htiaika + longitudi - rektaskensio */
   H=Sd0+Geo.Lon-Equ.RA;

   /* Muunnos ekvatoriaalikoordinaateista halutun latitudin ja
      longitudin horisontaalisiin koordinaatteihin */
 
   y=sin(H);
   x=cos(H)*sin(Geo.Lat)-tan(Equ.Dec)*cos(Geo.Lat);

   /* Lis�t��n 180, jotta atsimuutti alkaisi pohjoisesta */
   Hor.A=atan2(y,x)*RAD_TO_DEG+180.0;

   Hor.h=(asin(sin(Geo.Lat)*sin(Equ.Dec)
              +cos(Geo.Lat)*cos(Equ.Dec)*cos(H)))*RAD_TO_DEG;

   if(Hor.A<0.0) Hor.A+=360.0;

   return(Hor);
}


