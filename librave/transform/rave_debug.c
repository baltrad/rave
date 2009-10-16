/* --------------------------------------------------------------------
Copyright (C) 2009 Swedish Meteorological and Hydrological Institute, SMHI,

This file is part of HLHDF.

HLHDF is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

HLHDF is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with HLHDF.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/

#include "rave_debug.h"

#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

rave_debug_struct raveDbg;
static int initialized = 0;

/*@{ Private functions */
static void setLogTime(char* strtime, int len)
{
  time_t cur_time;
  struct tm* tu_time;

  time(&cur_time);
  tu_time = gmtime(&cur_time);
  strftime(strtime, len, "%Y/%m/%d %H:%M:%S", tu_time);
}

static void Rave_defaultDebugFunction(char* filename, int lineno, Rave_Debug lvl,
  const char* fmt, ...)
{
  char msgbuff[512];
  char dbgtype[20];
  char strtime[24];
  char infobuff[120];

  va_list alist;
  va_start(alist,fmt);

  if (raveDbg.dbgLevel == RAVE_SILENT)
    return;

  setLogTime(strtime, 24);

  strcpy(dbgtype, "");

  if (lvl >= raveDbg.dbgLevel) {
    switch (lvl) {
    case RAVE_SPEWDEBUG:
      sprintf(dbgtype, "SDEBUG");
      break;
    case RAVE_DEBUG:
      sprintf(dbgtype, "DEBUG");
      break;
    case RAVE_DEPRECATED:
      sprintf(dbgtype, "DEPRECATED");
      break;
    case RAVE_INFO:
      sprintf(dbgtype, "INFO");
      break;
    case RAVE_WARNING:
      sprintf(dbgtype, "WARNING");
      break;
    case RAVE_ERROR:
      sprintf(dbgtype, "ERROR");
      break;
    case RAVE_CRITICAL:
      sprintf(dbgtype, "CRITICAL");
      break;
    default:
      sprintf(dbgtype, "UNKNOWN");
      break;
    }
  } else {
    return;
  }
  sprintf(infobuff, "%20s : %11s", strtime, dbgtype);
  vsprintf(msgbuff, fmt, alist);

  fprintf(stderr, "%s : %s (%s:%d)\n", infobuff, msgbuff, filename, lineno);
}
/*@} End of Private functions */

/*@{ Interface functions */
void Rave_initializeDebugger()
{
  if (initialized == 0) {
    initialized = 1;
    raveDbg.dbgLevel = RAVE_SILENT;
    raveDbg.dbgfun = Rave_defaultDebugFunction;
  }
}

void Rave_setDebugLevel(Rave_Debug lvl)
{
  raveDbg.dbgLevel = lvl;
}

void Rave_setDebugFunction(void(*dbgfun)(char* filename, int lineno,
  Rave_Debug lvl, const char* fmt, ...))
{
  raveDbg.dbgfun = dbgfun;
}
/*@} End of Interface functions */
