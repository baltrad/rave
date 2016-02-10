/* --------------------------------------------------------------------
Copyright (C) 2010 Royal Netherlands Meteorological Institute, KNMI and
                   Swedish Meteorological and Hydrological Institute, SMHI,

This is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with HLHDF.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/
#include "scansun.h"
#include "rave_debug.h"

/** Main function for a binary for running KNMI's sun scanning functionality
 * @file
 * @author Original algorithm and code: Iwan Holleman, KNMI, and Integration: Daniel Michelson, SMHI
 * @date 2011-01-18
 */

int main(int argc,char *argv[]) {
  char* source = NULL;
	RaveList_t* list = RAVE_OBJECT_NEW(&RaveList_TYPE);
	RVALS* ret = NULL;
	const char* FORMAT = "%08ld %010.3f  %7.3f %7.2f   %7.4f  %7.4f  %4d  %9.2f %9.2f  %6.3f %9.2f  %6.3f  %s   %s  %s\n";

	Rave_initializeDebugger();
	Rave_setDebugLevel(RAVE_WARNING);

	if (argc<2) {
	   printf("Usage: %s <ODIM_H5-file>\n",argv[0]);
	   RaveList_freeAndDestroy(&list);
	   exit(1);
	}

	if (!scansun(argv[1], list, &source)) {
		printf("Could not process %s, exiting ...\n", argv[1]);
		if (source != NULL) {
		  RAVE_FREE(source);
		}
		RaveList_freeAndDestroy(&list);
		exit(1);
	}

	if (RaveList_size(list) > 0) {
		printf("#Date    Time        Elevatn Azimuth   ElevSun   AzimSun    N  dBSunFlux   SunMean SunStdd   ZdrMean ZdrStdd  Refl ZDR  Source\n");
		while ((ret = RaveList_removeLast(list)) != NULL) {
			printf(FORMAT, ret->date,
            			   ret->timer,
            			   ret->Elev,
            			   ret->Azimuth,
            			   ret->ElevSun,
            			   ret->AzimSun,
            			   ret->dBSunFlux,
						   ret->SunMean,
						   ret->SunStdd,
						   ret->ZdrMean,
						   ret->ZdrStdd,
						   ret->n,
						   ret->quant1,
						   ret->quant2,
            			   source);
		}
	}
  RAVE_FREE(source);
	RaveList_freeAndDestroy(&list);
	exit(0);
}
