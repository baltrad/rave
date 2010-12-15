/* --------------------------------------------------------------------
Copyright (C) 2010 Swedish Meteorological and Hydrological Institute, SMHI,

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
 * Shows how the area registry can be used.
 * @file
 * @author Anders Henja (Swedish Meteorological and Hydrological Institute, SMHI)
 * @date 2010-12-08
 */
#include "arearegistry.h"
#include "rave_debug.h"
#include "rave_list.h"
#include "rave_utilities.h"

static void printArea(Area_t* area)
{
  if (area != NULL) {
    double llx = 0.0, lly = 0.0, urx = 0.0, ury = 0.0;
    Projection_t* proj = NULL;
    fprintf(stderr, "  AREA: %s\n", Area_getID(area));
    fprintf(stderr, " PCSID: %s\n", Area_getPcsid(area));
    proj = Area_getProjection(area);
    if (proj != NULL) {
      fprintf(stderr, "    DEF: %s\n", Projection_getDefinition(proj));
    }
    fprintf(stderr, " DESCR: %s\n", Area_getDescription(area));
    fprintf(stderr, "  SIZE: %ld x %ld\n", Area_getXSize(area), Area_getYSize(area));
    fprintf(stderr, " SCALE: %g / %g\n", Area_getXScale(area), Area_getYScale(area));
    Area_getExtent(area, &llx, &lly, &urx, &ury);
    fprintf(stderr, "EXTENT: %g, %g, %g, %g\n", llx, lly, urx, ury);
    fprintf(stderr, "\n");
    RAVE_OBJECT_RELEASE(proj);
  }
}

int main(int argc, char** argv)
{
  int result = 1;
  AreaRegistry_t* aRegistry = NULL;
  ProjectionRegistry_t* pRegistry = NULL;

  Rave_initializeDebugger();
  Rave_setDebugLevel(RAVE_INFO);

  if (argc != 2 && argc != 3) {
    RAVE_ERROR1("Usage is %s <area xml file> <proj xml file>", argv[0]);
    goto done;
  }

  if (argc == 3) {
    pRegistry = ProjectionRegistry_loadRegistry(argv[2]);
    if (pRegistry == NULL) {
      RAVE_ERROR0("Failed to read projection registry");
      goto done;
    }
  }

  aRegistry = AreaRegistry_load(argv[1], pRegistry);
  if (aRegistry != NULL) {
    int nrareas = AreaRegistry_size(aRegistry);
    int index = 0;
    for (index = 0; index < nrareas; index++) {
      Area_t* area = AreaRegistry_get(aRegistry, index);
      printArea(area);
      RAVE_OBJECT_RELEASE(area);
    }
  }

  result = 0;
done:
  RAVE_OBJECT_RELEASE(pRegistry);
  RAVE_OBJECT_RELEASE(aRegistry);
  return result;
}
