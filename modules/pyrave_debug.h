/* --------------------------------------------------------------------
Copyright (C) 2009 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Useful macros and functions when debugging python rave objects.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2009-12-14
 */
#ifndef PYRAVE_DEBUG_H
#define PYRAVE_DEBUG_H
#include <stdio.h>
#include "rave_debug.h"

typedef struct {
  long created;
  long destroyed;
  const char* name;
} PyRaveObjectDebugging;

#define PYRAVE_DEBUG_MODULE(oname) \
  static PyRaveObjectDebugging _pyravedebug = { \
      0, 0, oname }; \
  \
  static void _pyravedebug_statistics(void) { \
    if ((_pyravedebug.created - _pyravedebug.destroyed) != 0) { \
      fprintf(stderr, "\n------------------------------------------\n"); \
      fprintf(stderr, "CRITICAL ( Python module: %s ) \n", _pyravedebug.name); \
      fprintf(stderr, "Created objects: %ld\n", _pyravedebug.created); \
      fprintf(stderr, "Destroyed objects: %ld\n", _pyravedebug.destroyed); \
      fprintf(stderr, "Objects lost: %ld\n", (_pyravedebug.created - _pyravedebug.destroyed)); \
      fprintf(stderr, "\n"); \
    } \
  }

#define PYRAVE_DEBUG_OBJECT_CREATED \
  _pyravedebug.created++

#define PYRAVE_DEBUG_OBJECT_DESTROYED \
  _pyravedebug.destroyed++

#define PYRAVE_DEBUG_INITIALIZE \
  if (atexit(_pyravedebug_statistics) != 0) { \
    fprintf(stderr, "Failed to setup debug statistics for module %s\n", _pyravedebug.name); \
  } \
  Rave_initializeDebugger()

#endif /* PYRAVE_DEBUG_H */
