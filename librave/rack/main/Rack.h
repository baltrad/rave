/**

    Copyright 2001 - 2010

    This file is part of Rack library for C++.

    Rack is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    Rack is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Rack.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifndef __RACK__
#define __RACK__  "rack v1.94  2011.02.04  Markus.Peura@fmi.fi"


/*TODO, If both projects.h and proj_api.h are included in the same file then
 * projects.h must be included first due to bug (?) which causes redefine of
 * projUV.
 */
extern "C"
{
#include <projects.h>
#include <proj_api.h>
}

#include <drain/image/ImageView.h>
#include <drain/image/Image.h>

#include <drain/util/Options.h>
#include <drain/util/Proj4.h>

#ifdef RACK_ANORACK
#include <drain/anorack/AnoRack.h>
#else
#include <drain/radar/Andre.h>
#endif

#include <drain/radar/Composite.h>
#include <drain/radar/PolarCappi.h>
#include <drain/radar/PolarEchoTop.h>
#include <drain/radar/PolarMaxEcho.h>
#include <drain/radar/PolarAttenuation.h>

#include "../hi5/RaveConvert.h"

///// Included for accessing polarvolume and subtypes
extern "C"
{
#include <polarvolume.h>
#include <rave_io.h>
}
/////


// Old anomaly detection software written in C.
#include "../ropo/Ropo.h"

using namespace std;

namespace rack
{
 
/**
 * The toplevel class for Rack library, handling all Rack algorithms.
 */
class Rack
{
public:
	Rack();
	virtual ~Rack();

	/**
	 * The Rack main function, called when a chained result of different Rack algorithms is needed.
	 * @param[in] scan - The Polar scan to read
	 * @param[in] argv - The argument vector
	 */
	double main(PolarScan_t* scan, std::vector<string> argv);

	/**
	 * Get the result from the main function
	 * @return The polar scan.
	 */
	PolarScan_t* getPolarScan();
    
protected:
	
	/**
	 * Initialize variables
	 */
	void init();
	
	/*
	 * Convert data in PolarScan to the RACK internal image format.
	 * @param[in] scan - the PolarScan
	 */
	void convertScanToImage(PolarScan_t* scan);
	
	/**
	 * Check if default quantity is specified as argument and if that is the case, store it.
	 * @param[in] argv - the vector with arguments
	 */
	void getDefaultQuantity(std::vector<string>& argv);

	/**
	 * Set composite attributes for ODIM.
         * @param[in] composite - the composite
         * @param[in/out] p - the options class to store the attributes in
	 */
	void setCompositeAttributesODIM(const drain::radar::Composite &composite,drain::Options &p);


	//Data

	drain::image::Image<> inputImage;
	drain::image::Image<> polarProduct;
	drain::image::Image<> cartesianProduct;
	drain::image::Image<> colourProduct;
	drain::image::Image<> *currentImage; //Pointer to the image currently used
	drain::image::ImageView<> currentView; // By default this views currentImage, but can view its separate channels as well.
	drain::image::Image<> *currentPolarImage;

	drain::radar::Composite composite;
	drain::Options options;
	drain::Proj4 proj;
	drain::Options localConf;

	drain::image::Drain<> drainHandler;
	drain::image::Drain<> polarProductHandler;
	drain::radar::PolarCappi<> cappi;
	drain::radar::PolarMaxEcho<> maxEcho;
	drain::radar::PolarEchoTop<> echoTop;
	drain::radar::PolarAttenuation<> attenuation;

	///  Anomaly detection handlers
#ifdef RACK_ANORACK
	drain::anorack::AnoRack<> andre;
#else
	drain::radar::Andre<> andre;
#endif

	drain::radar::Ropo ropo;

	RaveConvert raveConv; //Class used for converting between RAVE formats like PolarScan_t and the RACK internal image format.

};

} //end namespace rack

#endif /*__RACK__*/
