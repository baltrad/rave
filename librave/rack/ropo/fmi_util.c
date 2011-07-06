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
    along with Rack.  If not, see <http://www.gnu.org/licenses/>.

*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "fmi_util.h"

char *fmi_util_comment=NULL;


/* GENERAL argv UTILITY */
char *argument(int argc, char **argv, char *target){
  int i=0;
  while (++i<argc)
    if (!strcmp(argv[i],target)){
      if (i!=argc-1) // if just a flag
        return argv[i+1];
      else
        return argv[i];
    }
  return NULL;
}

//int param(char *s, char *key,char *shortcut_key){
int param(int argc,char **argv,int index,char *key){
  if (index<1)
    return 0;
  if (index>=argc)
    return 0;
  //    fmi_error("param arg index overflow");
  if (strcmp(argv[index],key)==0)
    return 1;
  else 
    return 0;
}

int paramn(int argc,char **argv,int index,char *key){
  if (index<1)
    return 0;
  if (index>=argc)
    return 0;
  //    fmi_error("param arg index overflow");
  if (strncmp(key,argv[index],strlen(key))==0)
    return 1;
  else 
    return 0;
}

int file_count(int argc, char **argv){
  int i=0;
  while (++i<argc)
    if (argv[i][0]=='-')
      if (strcmp(argv[i],"-")!=0)
	return (i-1);
  return (i-1);
}

char *extract_command_line(int argc, char **argv){
  int i;
  for (i=0;i<argc;i++)
    if ((strlen(fmi_util_command_line)+strlen(argv[i]))<FMI_UTIL_COMMAND_LINE_LEN){
      strcat(fmi_util_command_line,argv[i]);
      strcat(fmi_util_command_line," ");}
  return (fmi_util_command_line);
}


/*
char *basename(char *filename){
  //const int strlen=256;
  //static char result[strlen];
  char *ptr;
  return ptr=strstr
}
*/

void trchr(char *string,char from,char to){
  char *c;
  if (string==NULL) 
    return;
  for (c=string;*c!='\0';c++)
    if (*c==from) *c=to;
}

/* DEBUGGING */
int fmi_debug_level=0;

void fmi_error(char *s){
  fprintf(stderr,"fmi_error: ");
  fprintf(stderr,"%s",s);
  fprintf(stderr,"\n");
  exit(-1);
}

void fmi_debug(int n,char *s){
  int i;
  if (n<=fmi_debug_level){
    for (i=0;i<n;i++)
      fputc(':',stderr);
    fprintf(stderr," %s\n",s);
    fflush(stderr);}
}

char *file_extension(char *filename){
  return (strrchr(filename,'.'));
}

char *file_name(char *filename){
  char *s;
  s=strrchr(filename,'/');
  if (s==NULL)
    return (filename);
  else
    return (++s);
}

char *file_path(char *filename){
  char *end,c,*path; 
  end=strrchr(filename,'/');
  if (end==NULL)
    return NULL;
  end++;
  c=*end;
  *end='\0';                             /* cut! (kludge) */
  path=(char *)malloc(strlen(filename)+1);
  strcpy(path,filename);
  *end=c;                           /* fix! (kludge) */
  return path;
}

char *file_basename(char *filename){
  char *start,*end,c,*basename; 
  start=strrchr(filename,'/');
  end=strrchr(filename,'.');
  if (start==NULL)
    start=filename;
  else
    start++;
  if (end!=NULL){
    c=*end;
    *end='\0';}                             /* cut! (kludge) */
  basename=(char *)malloc(strlen(start)+1);
  strcpy(basename,start);
  if (end!=NULL){
    *end=c;}
  return basename;
}



/* function g=gauss(a,x);
   g=(1./((x.*x)/(a*a+0.000001)+1));
*/

/* function m=msigmoid(x,a)
   m=a*x./(1+abs(a*x));
*/

/* BELL CURVE in [0,1], ORIGIN=0, BELL=0.5 AT A */
float pseudo_gauss(float a,float x){
  return (255.0*a*a/(a*a+x*x));
}

int pseudo_gauss_int(int a,int x){
  return (255*a*a/(a*a+x*x));
}
  //  return (1.0 / (1+x*x/a/a));}

/* SIGMOID CURVE in [0,1], ORIGIN=0, A=DERIV AT ORGIN */
float pseudo_sigmoid(float a,float x){
  // float xa=x/a;
  if (x>0.0) 
    return ( 255.0*x/(a+x) );
  else     
    return ( 255.0*x/(a-x) );
}

int pseudo_sigmoid_int(int a,int x){
  // float xa=x/a;
  if (x>0) 
    return ( (255*x)/(a+x) );
  else     
    return ( (255*x)/(a-x) );
}

//#define pseudo_sigmoid(a,x) ((x)>0 ? 255*(x)/(a+(x)) : 255*(x)/(a-(x))) 
//#define pseudo_gauss(a,x) (255*(a)*(a)/((a)*(a)+(x)*(x)))








