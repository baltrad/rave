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
#include <syslog.h>
#include <fcntl.h>
#include <unistd.h>
#include <strings.h>

#ifdef PTHREAD_SUPPORTED
#include <pthread.h>
#endif

static rave_dbgfun raveDebugFunction = NULL;
static Rave_Debug raveDebugLevel = RAVE_SILENT;
static int initialized = 0;

static Rave_LogOutput raveLogOutput = RAVE_LOG_OUTPUT_STDERR;
static int raveLogFileFd = -1;
static char raveSyslogIdent[128] = "";
static int raveSyslogOpened = 0;

/* Protects some critical sections */
#ifdef PTHREAD_SUPPORTED
static pthread_mutex_t raveLogOutputMutex = PTHREAD_MUTEX_INITIALIZER;
#define RAVE_LOG_OUTPUT_LOCK() pthread_mutex_lock(&raveLogOutputMutex)
#define RAVE_LOG_OUTPUT_UNLOCK() pthread_mutex_unlock(&raveLogOutputMutex)
#else
#define RAVE_LOG_OUTPUT_LOCK()
#define RAVE_LOG_OUTPUT_UNLOCK()
#endif

/*@{ Private functions */
static void setLogTime(char* strtime, int len)
{
  time_t cur_time;
  struct tm* tu_time;

  time(&cur_time);
  tu_time = gmtime(&cur_time);
  strftime(strtime, len, "%Y/%m/%d %H:%M:%S", tu_time);
}

/**
 * Maps a Rave_Debug level onto the closest matching syslog priority.
 */
static int Rave_debugLevelToSyslogPriority(Rave_Debug lvl)
{
  switch (lvl) {
  case RAVE_SPEWDEBUG:
  case RAVE_DEBUG:
    return LOG_DEBUG;
  case RAVE_DEPRECATED:
    return LOG_NOTICE;
  case RAVE_INFO:
    return LOG_INFO;
  case RAVE_WARNING:
    return LOG_WARNING;
  case RAVE_ERROR:
    return LOG_ERR;
  case RAVE_CRITICAL:
    return LOG_CRIT;
  default:
    return LOG_DEBUG;
  }
}

/**
 * Map syslog facility name into the corresponding int value. Default
 * fallback is USER.
 */
static int Rave_syslogFacilityFromName(const char* facility)
{
  static const struct {
    const char* name;
    int value;
  } FACILITIES[] = {
    {"user", LOG_USER},
    {"local0", LOG_LOCAL0},
    {"local1", LOG_LOCAL1},
    {"local2", LOG_LOCAL2},
    {"local3", LOG_LOCAL3},
    {"local4", LOG_LOCAL4},
    {"local5", LOG_LOCAL5},
    {"local6", LOG_LOCAL6},
    {"local7", LOG_LOCAL7},
  };
  size_t i;
  if (facility != NULL) {
    for (i = 0; i < sizeof(FACILITIES) / sizeof(FACILITIES[0]); i++) {
      if (strcasecmp(facility, FACILITIES[i].name) == 0) {
        return FACILITIES[i].value;
      }
    }
  }
  return LOG_USER;
}

void Rave_printf(const char* fmt, ...)
{
  va_list alist;
  va_start(alist,fmt);
  char msgbuff[4096];
  int n = vsnprintf(msgbuff, 4096, fmt, alist);
  va_end(alist);
  if (n < 0 || n >= 1024) {
    return;
  }
#ifndef NO_RAVE_PRINTF
  fprintf(stderr, "%s", msgbuff);
#endif
}

static void Rave_defaultDebugFunction(const char* filename, int lineno, Rave_Debug lvl, const char* fmt, ...)
{
  char msgbuff[512];
  char dbgtype[20];
  char strtime[24];
  char infobuff[120];

  va_list alist;
  va_start(alist,fmt);

  Rave_initializeDebugger(); /* So that we always have an initialized debugger */

  if (raveDebugLevel == RAVE_SILENT && lvl != RAVE_CRITICAL)
    return;

  setLogTime(strtime, 24);

  strcpy(dbgtype, "");

  if (lvl >= raveDebugLevel || lvl == RAVE_CRITICAL) {
    switch (lvl) {
    case RAVE_SPEWDEBUG:
      snprintf(dbgtype, 20, "SDEBUG");
      break;
    case RAVE_DEBUG:
      snprintf(dbgtype, 20, "DEBUG");
      break;
    case RAVE_DEPRECATED:
      snprintf(dbgtype, 20, "DEPRECATED");
      break;
    case RAVE_INFO:
      snprintf(dbgtype, 20, "INFO");
      break;
    case RAVE_WARNING:
      snprintf(dbgtype, 20, "WARNING");
      break;
    case RAVE_ERROR:
      snprintf(dbgtype, 20, "ERROR");
      break;
    case RAVE_CRITICAL:
      snprintf(dbgtype, 20, "CRITICAL");
      break;
    default:
      snprintf(dbgtype, 20, "UNKNOWN");
      break;
    }
  } else {
    return;
  }
  snprintf(infobuff, 120, "%20s : %11s", strtime, dbgtype);
  vsnprintf(msgbuff, 512, fmt, alist);

  RAVE_LOG_OUTPUT_LOCK();
  if (raveLogOutput == RAVE_LOG_OUTPUT_FILE && raveLogFileFd >= 0) {
    char linebuff[1024];
    int len = snprintf(linebuff, sizeof(linebuff), "%s : %s (%s:%d)\n", infobuff, msgbuff, filename, lineno);
    if (len > 0) {
      if (len >= (int)sizeof(linebuff)) {
        len = sizeof(linebuff) - 1;
      }
      /* Write atomic since a fprintf can write chunkwise */
      if (write(raveLogFileFd, linebuff, len) < 0) {
        // No op
      }
    }
  } else if (raveLogOutput == RAVE_LOG_OUTPUT_SYSLOG && raveSyslogOpened) {
    syslog(Rave_debugLevelToSyslogPriority(lvl), "%-8s [%d] %s (%s:%d)", dbgtype, (int)getpid(), msgbuff, filename, lineno);
  } else {
    Rave_printf("%s : %s (%s:%d)\n", infobuff, msgbuff, filename, lineno);
  }
  RAVE_LOG_OUTPUT_UNLOCK();
}
/*@} End of Private functions */

/*@{ Interface functions */
void Rave_initializeDebugger(void)
{
  if (initialized == 0) {
    initialized = 1;
    raveDebugLevel = RAVE_SILENT;
    raveDebugFunction = Rave_defaultDebugFunction;
  }
}

void Rave_setDebugLevel(Rave_Debug lvl)
{
  Rave_initializeDebugger();
  if (lvl >= RAVE_SPEWDEBUG && lvl <= RAVE_SILENT) {
    raveDebugLevel = lvl;
  }
}

Rave_Debug Rave_getDebugLevel(void)
{
  Rave_initializeDebugger();
  return raveDebugLevel;
}

void Rave_setDebugFunction(rave_dbgfun dbgfun)
{
  Rave_initializeDebugger();
  raveDebugFunction = dbgfun;
}

rave_dbgfun Rave_getDebugFunction(void)
{
  Rave_initializeDebugger();
  return raveDebugFunction;
}

static void Rave_closeCurrentLogOutput(void)
{
  if (raveLogFileFd >= 0) {
    close(raveLogFileFd);
    raveLogFileFd = -1;
  }
  if (raveSyslogOpened) {
    closelog();
    raveSyslogOpened = 0;
  }
}

void Rave_setLogOutputStderr(void)
{
  Rave_initializeDebugger();
  RAVE_LOG_OUTPUT_LOCK();
  Rave_closeCurrentLogOutput();
  raveLogOutput = RAVE_LOG_OUTPUT_STDERR;
  RAVE_LOG_OUTPUT_UNLOCK();
}

int Rave_setLogOutputFile(const char* filename)
{
  int fd;
  Rave_initializeDebugger();
  if (filename == NULL) {
    return 0;
  }
  fd = open(filename, O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd < 0) {
    return 0;
  }
  RAVE_LOG_OUTPUT_LOCK();
  Rave_closeCurrentLogOutput();
  raveLogFileFd = fd;
  raveLogOutput = RAVE_LOG_OUTPUT_FILE;
  RAVE_LOG_OUTPUT_UNLOCK();
  return 1;
}

void Rave_setLogOutputSyslog(const char* logid, const char* facility)
{
  Rave_initializeDebugger();
  RAVE_LOG_OUTPUT_LOCK();
  Rave_closeCurrentLogOutput();
  if (logid != NULL) {
    strncpy(raveSyslogIdent, logid, sizeof(raveSyslogIdent) - 1);
    raveSyslogIdent[sizeof(raveSyslogIdent) - 1] = '\0';
  } else {
    strcpy(raveSyslogIdent, "rave");
  }
  openlog(raveSyslogIdent, LOG_CONS, Rave_syslogFacilityFromName(facility));
  raveSyslogOpened = 1;
  raveLogOutput = RAVE_LOG_OUTPUT_SYSLOG;
  RAVE_LOG_OUTPUT_UNLOCK();
}

Rave_LogOutput Rave_getLogOutput(void)
{
  Rave_LogOutput result;
  Rave_initializeDebugger();
  RAVE_LOG_OUTPUT_LOCK();
  result = raveLogOutput;
  RAVE_LOG_OUTPUT_UNLOCK();
  return result;
}

/*@} End of Interface functions */
