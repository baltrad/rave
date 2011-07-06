/**


    Copyright 2006 - 2011


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

   Created on: Apr 13, 2011
*/
#include "Rack.h"
#include "RackIf.h"
#include "RackLocalIf.h"
#include "../hi5/RaveConvert.h"

extern "C"
{
#include <rave_debug.h>
}

///////////////////////////
// prototypes
///////////////////////////

void splitString(std::string argstr, std::vector<string>& args);

///////////////////////////
// functions
///////////////////////////

/**
 * Splits a string into space separated words, stored in a vector.
 * @param[in] argstr - the string to split
 * @param[in/out] args - the vector to store the words in
 */
void splitString(std::string argstr, std::vector<string>& args)
{
  std::istringstream iss(argstr);
  std::string token;
  while(iss >> token) {
    args.push_back(token);
  }
}

RaveCoreObject* execRack(RaveCoreObject* obj, const char *argstr)
{
  RaveCoreObject* retObj = NULL;
  std::vector<string> args;
  splitString(string(argstr), args);

  if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarVolume_TYPE)){
    retObj = (RaveCoreObject*)runRack((PolarVolume_t*)obj, args);
  }
  else if (RAVE_OBJECT_CHECK_TYPE(obj, &PolarScan_TYPE)){
    retObj = (RaveCoreObject*)runRack((PolarScan_t*)obj, args);
  }
  else {
    cerr << "RackIf::execRack() called with unknown type" << endl;
  }

  return retObj;
}

PolarVolume_t* runRack(PolarVolume_t* volume, std::vector<string>& args)
{	
	PolarVolume_t* outputVolume = (PolarVolume_t*)RAVE_OBJECT_NEW(&PolarVolume_TYPE);
	RAVE_ASSERT((outputVolume != NULL),"Failed to allocate PolarVolume");
	PolarScan_t* scan = NULL;

	int numScans = PolarVolume_getNumberOfScans(volume);
	for (int i = 0; i < numScans; i++)
	{
		scan = PolarVolume_getScan(volume, i);

		rack::Rack rack;
		//Run algorithms
		double result = rack.main(scan, args);
		RAVE_OBJECT_RELEASE(scan);
		if (result < 0)
		{
			cerr << "rack failed with exit code" << result << endl;
			return NULL;
		}
		//get result
		scan = rack.getPolarScan();
		PolarVolume_addScan(outputVolume, scan);
		RAVE_OBJECT_RELEASE(scan);
	}
	//Copy all volume attributes from input volume to output volume
	rack::RaveConvert conv;
	conv.copyVolumeAttributes(volume, outputVolume);

	return outputVolume;
}

PolarScan_t* runRack(PolarScan_t* scan, std::vector<string>& args)
{
        rack::Rack rack;
	//Run algorithms
	double result = rack.main(scan, args);

	if (result < 0)
	{
		cerr << "rack failed with exit code" << result << endl;
		return NULL;
	}

	//get result
	PolarScan_t* outputScan = rack.getPolarScan();
	return outputScan;
}

