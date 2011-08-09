/**

    Copyright 2001 - 2010  Markus Peura,
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
    along with Rack.  If not, see <http://www.gnu.org/licenses/>. */




#ifndef _FMI_UTIL_
#define _FMI_UTIL_
#include <string.h>

#define MAX(x,y) ((x)>(y)?(x):(y))
#define MIN(x,y) ((x)<(y)?(x):(y))
#define SGN(x)   ((x)<0?-1:((x)>0?1:0))
#define ABS(x)   ((x)<0?-(x):(x))
#define SQRT2 1.41421356

#ifndef PI
#define PI 3.14159 
#endif

#ifndef YES
#define YES 1
#endif

#ifndef NO
#define NO 0
#endif

#define EPSILON 0.0000001
#define BIGNUMBER 32768

#define EARTH_RADIUS 6372000 /* metres, = 6372 KM */

#define TRUNC(x) ((int)(100*(x))/100.0)
/*#define SIGMOID(x,a) ((x)>0 ? (a)*(x)/(1+(a)*(x)) : (a)*(x)/(1-(a)*(x)) )  */
#define GAUSS(a,x) (1/((x*x)/(a*a+0.000001)+1))

float pseudo_gauss(float a,float x);
int   pseudo_gauss_int(int a,int x);

float pseudo_sigmoid(float a,float x);  /* between -128 and 128 */
int pseudo_sigmoid_int(int a,int x);  /* between -128 and 128 */
/*#define pseudo_sigmoid(a,x) ((x)>0 ? 255*(x)/(a+(x)) : 255*(x)/(a-(x)))  */
/*#define pseudo_gauss(a,x) (255*(a)*(a)/((a)*(a)+(x)*(x))) */

char *argument(int argc, char **argv, char *target);
int file_count(int argc, char **argv);

int param(int argc, char **argv,int index, char *key);
int paramn(int argc, char **argv,int index, char *key);

#define FMI_UTIL_COMMAND_LINE_LEN 1024
char fmi_util_command_line[FMI_UTIL_COMMAND_LINE_LEN];
char *fmi_util_comment;
char *extract_command_line(int argc, char **argv);

  
#define FMI_FILENAME_LEN 1024


/*#define BASENAME(s) (strrchr(s,'/')==NULL?s:(strrchr(s,'/'))+1) */
char *file_path(char *filename);
char *file_basename(char *filename);  /* without path and 1 extension */
char *file_extension(char *filename);
char *file_name(char *filename);      /* without path */

#define FMI_ARG(i) (argv[i])
#define FMI_IARG(i) (atoi(argv[i]))

#define FMI_PARAM(s) (argument(argc,argv,s))
#define FMI_FLAG(s) (argument(argc,argv,s)!=NULL)
#define FMI_FILES() (file_count(argc,argv))
#define FMI_HELP(n) ((argc<n)||(argument(argc,argv,"-h"))||(argument(argc,argv,"-help")))

/* DEBUG LEVELS */
/* 0 silent */
/* 1 light processing information */
/* 2 light processing information, write aux files */
/* 3 heavy processing information, write aux files  */
int fmi_debug_level;
#define FMI_DEBUG(n) (n<=fmi_debug_level)
void fmi_debug(int n,char *s);
void fmi_error(char *s);

void trchr(char *string,char from,char to);


/* METEOROLOGICAL STUFF */


#endif

