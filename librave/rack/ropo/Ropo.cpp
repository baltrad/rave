/**


    Copyright 2010  Markus Peura, Finnish Meteorological Institute (First.Last@fmi.fi)


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


#include <drain/image/CopyOp.h>
#include "Ropo.h"

namespace drain {

namespace radar {

/*
RopoDetector::RopoDetector(const string p) :
	AndreDetector<Byte,Byte>("rSpeckle","Detects speckle noise.","minIntensity,minLength",p)
{
};
*/


void RopoDetector::filter(const image::Image<Byte> &src,image::Image<Byte> &dst) const {


	makeCompatible(src,dst);

	//drain::image::File::write(src,"ropo-src1I.png");

	FmiImage srcC;
	viewImage(src,srcC);

	FmiImage dstC;

	filter(srcC,dstC);
	//detect_emitters(&srcC, &dstC, minIntensity, minLength);
	//write_image((char *)"ropo-2dsti",&dstC,PGM_RAW);

	convertImage(dstC,dst);

	reset_image(&srcC);
	reset_image(&dstC);

	//drain::image::File::write(dst,"ropo-src3I.png")
	//put_pixel(&dstC,30,30,0,255);
	//CopyOp<Byte,Byte>().filter(src,dst);

	//dst.at(5,5) = 255;

	// const int threshold = this->getParameter("threshold",1);
	// const int size = this->getParameter("size",3);
	//testRopo(dst.getName().c_str());

};


/*
void RopoEmitter::filter(FmiImage &src,FmiImage &dst) const {
	const int minIntensity = this->getParameter("minIntensity",1);
	const int minLength    = this->getParameter("minLength",3);
	detect_emitters(&srcC, &dstC, minIntensity, minLength);
};
*/


void Ropo::addDefaultOps(){
	addOperator("biomet",biomet);
	addOperator("emitter",emitter);
	addOperator("ship",ship);
	addOperator("speckle",speckle);
};

}

}
