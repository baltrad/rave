/**

    Copyright 2001 - 2010  Markus Peura, Finnish Meteorological Institute (First.Last@fmi.fi)


    This file is part of Drain library for C++.

    Drain is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    Drain is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Drain.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifndef SEGMENTAREAOP_H
#define SEGMENTAREAOP_H

#include "ImageOp.h"
#include "FloodFillOp.h"

namespace drain
{
namespace image
{


/**
 *  \bugs - single-pixel segments neglected.
 */
template <typename T = unsigned char, typename T2 = unsigned char>
class SegmentAreaOp: public ImageOp<T, T2>
{
public:
	/*!
	 * 
	 *  @TODO Bug:
	 *  ./drainage radar-VAN-00.0-Z.png --segmentArea 64,155,i,18 -o segmentArea.png
	 */
    SegmentAreaOp(const string & p = "1,255,f,128,0") :
    	ImageOp<T,T2>("SegmentArea","Computes sizes of segments with intensity within [min,max] and marks them in dst image. Mapping (d|i|b:128|f:128:1) - direct, inverse,bilinearly scaled, fuzzy, inverse size.",
    	"min,max,mapping,mSlope,mPos",p) {
    };
	

    /// TODO: for each channel
    virtual void filter(const Image<T> &src,Image<T2> &dst) const {

    	if (!filterEachChannel(src,dst))
    		return;

		const unsigned int w = src.getWidth();
		const unsigned int h = src.getHeight();

		FloodFillRecursion<> floodFill(src,dst);

	   // const float dMax = Intensity::max<T2>();
   	   const int dMax = Intensity::max<T2>();

   	   const int min = this->parameters.get("min",1);
   	   const int max = this->parameters.get("max",(int)Intensity::max<T>());
   	   const char m =  this->parameters.get("mapping",'d');
   	   //const string mapping = this->parameters.get("mapping","d");
   	   //const char m = mapping.empty() ? 'd' : mapping[0];
   	   const int slope = this->parameters.get("mSlope",(min+max)/2);
   	   const int slope2 = slope*slope;
   	   const int position = this->parameters.get("mPos",1);


   	   if (drain::Debug > 2)
   		   cerr << "SegmentAreaOp range:" << (int)min  << '-' << (int)max  << '\n';
   	    
   	   long int size = 0;
   	   for (unsigned int i=0; i<w; i++)
   		   for (unsigned int j=0; j<h; j++){
   			   if ((drain::Debug > 5) && ((i&15)==0) && ((j&15)==0))
   				   cerr << i << ',' << j << "\n";
   			   if (dst.at(i,j) == 0){ // not visited
   				   size = floodFill.fill(i,j,1,min,max);  // painting with '1' does not disturb dst
   				   if (size > 0){
   					   T2 sizeMapped = 0;
   					   switch (m){
   					   case 'd':
   						   sizeMapped = Intensity::limit<T2>(size);
   						   break;
   					   case 'b':
   						   //cerr << "warning: b obsolete\n";
   						   sizeMapped = dMax - Intensity::limit<T2>((dMax*slope)/(slope+size-1));
   						   break;
   					   case 'f':
   						   size = size - position;
   						   sizeMapped = Intensity::limit<T2>(dMax*slope2 / (slope2 + size*size));
   						   //sizeMapped = Intensity::limit<T2>(dMax-(dMax*halfWidth)/(halfWidth+size-1));
   						   break;
   					   case 'i':
   						   sizeMapped = dMax-size;
   						   //sizeMapped = dMax - Intensity::limit<T2>(dMax*h2 / (h2 + size*size));
   						   break;
   					   default:
   						   throw runtime_error(string("Segment area: unknown mapping: ") + m);
   					   }
   					   //cerr << "(" << i << "," << j << "):\t";
   					   //cerr << "size " << size << " (" << (int)sizeMapped << ")\n";
   					   if (sizeMapped == 0)
   						   sizeMapped = 1;

   					   floodFill.fill(i,j,sizeMapped,min,max);
   				   }
   			   }
   		   }
	};

	
   
};



} // namespace image
} // namespace drain


#endif /* SEGMENTAREAOP_H */
