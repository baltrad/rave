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
#ifndef MARGINALSTATISTICOP_H_
#define MARGINALSTATISTICOP_H_

#include "math.h"

#include "../util/Histogram.h"


#include "ImageOp.h"


namespace drain
{

namespace image
{


/** Computes horizontal or vertical intensity statistics: average, sum, ...
 */
template <class T=unsigned char,class T2=unsigned char> //,class D=int>
class MarginalStatisticOp : public ImageOp<T,T2>
{
public:
    
    MarginalStatisticOp(const string & p = "horz,a"){
    	this->setInfo("Computes statistics on horizontal or vertical lines. ","mode,stat",p);
	};


    /*
    virtual void makeCompatible(const Image<T> &src,Image<T2> &dst) const  {

    	const string &mode = this->parameters.get("mode","horz");
    	const string &stat = this->parameters.get("stat","avg");


    	if (mode == "horz"){
			dst.setGeometry(n,src.getHeight(),src.getImageChannelCount(),src.getAlphaChannelCount());
		}
		else if (mode == "vert"){
			dst.setGeometry(src.getWidth(),n,src.getImageChannelCount(),src.getAlphaChannelCount());
		}
		else {

		}

    	/// dst.getCoordinateHandler(); TODO
    };
    */

	void filter(const Image<T> &src, Image<T2> &dst) const {

		Histogram<T,T2>  histogram(256);

		const string &mode = this->parameters.get("mode","horz");
		const string &stat = this->parameters.get("stat","a");
    	const unsigned int n = stat.size();

		const unsigned int width  = src.getWidth();
    	const unsigned int height = src.getHeight();
    	const unsigned int iChannels = src.getImageChannelCount();
    	const unsigned int aChannels = src.getAlphaChannelCount();
    	const unsigned int channels = iChannels + aChannels;

		if (mode == "horz"){
			dst.setGeometry(n,height,iChannels,aChannels);
			histogram.setSampleCount(height);
			for (unsigned int k=0; k<channels; k++){
				for (unsigned int j=0; j<height; j++){
					histogram.clearBins();
					//histogram.clear();
					for (unsigned int i=0; i<width; i++)
						histogram[src.at(i,j)]++;
					for (unsigned int l=0; l<n; l++){
						switch (stat[l]){
						case 'M':
							dst.at(l,j,k) = histogram.getMax();
							break;
						case 'm':
							dst.at(l,j,k) = histogram.getMin();
							break;
						case 'd':
							dst.at(l,j,k) = histogram.getMedian();
							break;
						case 'S':
							dst.at(l,j,k) = histogram.getStdDeviation();
							break;
						case 's':
							dst.at(l,j,k) = histogram.getSum();
							break;
						case 'a':
							dst.at(l,j,k) = histogram.getMean();
							break;
						default:
							dst.at(l,j,k) = stat[l];
							//throw runtime_error(this->name + "undefined statistic: " + stat[l]);
						}

					}
				}
			}
		}
		else if (mode == "vert"){
			//dst.setGeometry(width,1);
			dst.setGeometry(width,n,iChannels,aChannels);
			histogram.setSampleCount(width);
			for (unsigned int k=0; k<channels; k++)
				for (unsigned int i=0; i<width; i++){
					histogram.clear();
					for (unsigned int j=0; j<height; j++)
						histogram[src.at(i,j,k)]++;
					for (unsigned int l=0; l<n; l++){
						switch (stat[l]){
						case 's':
							dst.at(i,l,k) = histogram.getSum();
							break;
						case 'a':
							dst.at(i,l,k) = histogram.getMean();
							break;
						default:
							throw runtime_error(this->name + "undefined statistic: " + stat[l]);
						}

					}
				}
		}
		else {
			throw runtime_error(this->name + ": geometry not horz or vert: " + mode);
		}
		//return dst;
	};	
	

	
protected:

    
};
   
}
}

#endif /*MARGINALSTATISTICOP_H_*/
