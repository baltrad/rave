/**

    Copyright 2010 Markus Peura, Finnish Meteorological Institute (First.Last@fmi.fi)


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

#ifndef  __RACK_ROPO__
#define  __RACK_ROPO__



extern "C" {

#include "stdio.h"
#include "fmi_image.h"
#include "fmi_image_filter_speck.h"
#include "fmi_radar_image.h"

/*	void testRopo(const char *str); */

}


#include <drain/radar/Andre.h>

#include "convertImage.h"

namespace drain {

namespace radar {


/* This file contains C++ wrappers for the original Ropo in C. */

/*/ Detects speckle noise. */
/**
 *  Segments of intensity > 'threshold' and area=1 get intensity 255, and larger ones
 *  smaller values with descreasing slope 'slope'.
 */
/*template <class T = unsigned char,class T2 = unsigned char> */
class RopoDetector : public AndreDetector<Byte,Byte>
{
public:

	RopoDetector(const string &name = "RopoDetector",
			const string &description="Radar data processor.",
			const string & parameterNames="",
			const string & defaultValues = "") :
				AndreDetector<Byte,Byte>(name,description,parameterNames,defaultValues){
	};

	void filter(const image::Image<Byte> &src,image::Image<Byte> &dst) const;

protected:
	virtual
	void filter(FmiImage &src,FmiImage &dst) const = 0;

};

class RopoEmitter : public RopoDetector {

public:
	RopoEmitter(const string p="1,3") :
		RopoDetector("rSpeckle","Detects speckle noise. ","minIntensity,minLength",p){};

protected:
	void filter(FmiImage &src,FmiImage &dst) const {
		const int minIntensity = this->getParameter("minIntensity",1);
		const int minLength    = this->getParameter("minLength",3);
		detect_emitters(&src, &dst, minIntensity, minLength);
	};

};


class RopoShip : public RopoDetector {

public:

	RopoShip(const string p="1,3") :
		RopoDetector("RopoShip","Detects ships in radar data.","minIntensity,maxArea",p){};

protected:

	void filter(FmiImage &src,FmiImage &dst) const {
		const int minIntensity = this->getParameter("minIntensity",1);
		const int maxArea    = this->getParameter("maxArea",9);
		detect_ships(&src,&dst,minIntensity,maxArea);
	};

};


class RopoSpeckle : public RopoDetector {

public:

	RopoSpeckle(const string p="16") :
		RopoDetector("RopoSpeckle","Detects speckle noise.","minIntensity",p){};

protected:

	void filter(FmiImage &src,FmiImage &dst) const {
		/*const int minIntensity = this->getParameter("minIntensity",1); */
		detect_specks(&src, &dst, this->getParameter("minIntensity",1), NULL);
	};

};

class RopoBioMet : public RopoDetector {

public:

	RopoBioMet(const string p="16,4,500,50") :
		RopoDetector("RopoSpeckle","Detects speckle noise.",
				"maxIntensity,intensitySteepness,maxAltitude,altitudeSteepness",p){};

protected:

	void filter(FmiImage &src,FmiImage &dst) const {
		/*const int minIntensity = this->getParameter("minIntensity",1); */
		detect_biomet(&src, &dst,
				this->getParameter("maxIntensity",16),
				this->getParameter("intensitySteepness",4),
				this->getParameter("maxAltitude",500),
				this->getParameter("altitudeSteepness",50));
	};

};



/*/ The handler which contains all the Ropo detectors and processes given data with them. */
class Ropo : public Andre<Byte,Byte> {
public:

	RopoBioMet  biomet;
	RopoEmitter emitter;
	RopoShip    ship;
	RopoSpeckle speckle;

	virtual void addDefaultOps(); /*		addOperator("speckle",speckle) }; */

};



}  /* namespace radar */


}  /* namespace drain */

#endif
