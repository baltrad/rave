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

/** Main function for a binary for running KNMI's sun scanning functionality
 * @file
 * @author Original algorithm and code: Iwan Holleman, KNMI, and Integration: Daniel Michelson, SMHI
 * @date 2011-01-18
 */

int main(int argc,char *argv[]) {
  char* source = NULL;
	RaveList_t* list = RAVE_OBJECT_NEW(&RaveList_TYPE);
	RVALS* ret = NULL;

	if (argc<2) {
	   printf("Usage: %s <ODIM_H5-volume>\n",argv[0]);
	   RAVE_OBJECT_RELEASE(list);
	   exit(1);
	}
	if (!scansun(argv[1], list, &source)) {
		printf("Could not process %s, exiting ...\n", argv[1]);
		RAVE_OBJECT_RELEASE(list);
		exit(1);
	}
	if (RaveList_size(list) > 0) {
		printf("#Date    Time   Elevatn Azimuth ElevSun RelevSun  AzimSun dBSunFlux   SunMean SunStdd Quant Source\n");
		while ((ret = RaveList_removeLast(list)) != NULL) {
			printf("%08ld %06ld %7.2f %7.2f %7.2f  %7.2f  %7.2f %9.2f %9.2f  %6.3f  %s %s\n", ret->date,
			                                                                                  ret->time,
			                                                                                  ret->Elev,
			                                                                                  ret->Azimuth,
			                                                                                  ret->ElevSun,
			                                                                                  ret->RelevSun,
			                                                                                  ret->AzimSun,
			                                                                                  ret->dBSunFlux,
			                                                                                  ret->SunMean,
			                                                                                  ret->SunStdd,
			                                                                                  ret->quant,
			                                                                                  source);
			// RAVE_FREE(ret);  /* No longer frees! */
		}
	}
	RAVE_OBJECT_RELEASE(list);
	// RAVE_FREE(source);  /* No longer frees! */

	exit(0);
}
