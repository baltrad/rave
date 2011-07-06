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
#ifndef PixelVectorOP_H_
#define PixelVectorOP_H_

#include <cmath>

#include "ImageOp.h"

namespace drain
{

namespace image
{

///  A base class for operations between of intensity vectors of two images.
/**  In each point(i,j), performs  an operation for vectors constructed the intensities in different channels.
 *
 *  \see DistanceOp
 *  \see ProductOp
 *  \see MagnitudeOp
 */
template <class T=unsigned char,class T2=unsigned char>
class PixelVectorOp : public ImageOp<T,T2>
{
public:

	PixelVectorOp(const string & name = "PixelVectorOp",const string & description = "") :
		ImageOp<T,T2>(name,description + " Computes L-norm or power of channel intensities. Euclidean L=2. ",
				"mapping,l,lInv","l,2"){
		// this->setInfo(,"mapping,l,lInv",p);
	};

	virtual
	void filter(const Image<T> &src,Image<T2> &dst) const {
		filter(src,dst,dst);
	};


	//inline 	double getValue(const T &src) const { return getValue(src,src); };


	template <class T3>
	void filter(const Image<T> &src1,const Image<T2> &src2,Image<T3> &dst) const {

		// src
		const int width  = std::min(src1.getWidth(),src2.getWidth());
		const int height = std::min(src1.getHeight(),src2.getHeight());
		const int channels = std::min(src1.getChannelCount(),src2.getChannelCount());

		// dst
		bool tmpNeeded = (channels>1) && (src1.hasOverlap(dst) || src2.hasOverlap(dst));
		Image<T3> tmp;
		Image<T3> & dstR = tmpNeeded ? tmp : dst;
		dstR.setGeometry(width,height,1);

		const vector< ImageView<T> > & channels1 = src1.getChannelVector();
		const vector< ImageView<T> > & channels2 = src2.getChannelVector();

		const char mapping = this->getParameter("mapping",'l');
		const double halfWidth = this->getParameter("mapping",128.0);
		const double l    = this->getParameter("l",2.0);
		const double lInv = this->getParameter("lInv",1.0/l);

		const bool powerUp = (l != 1.0);
		const bool powerDown = (lInv != 1.0);
		const bool scaled    = (halfWidth > 0.0);
		const bool scaledInv = (halfWidth < 0.0);

		const double max = static_cast<double>(Intensity::max<T3>());
		const double maxOrigin = static_cast<double>(Intensity::maxOrigin<T3>());


		double x=0.0, sum=0.0;
		if (drain::Debug > 2){
			cerr << "PixelVectorOp:" << this->getName() << '\n';
			cerr << " mapping=" << mapping << '\t';
			cerr << " halfWidth=" << halfWidth << '\n';
			cerr << " l=" << l << '\t';
			cerr << " lInv=" << lInv << '\n';
			cerr << " scaled=" << scaled << '\t';
			cerr << " scaledInv=" << scaledInv << '\n';
		}
		// debug
		/*
		src1.debug();
		src2.debug();
		for (int k = 0; k < channels; k++){
			cout << "PixelVectorOp, k=" << k << '\t';
			channels1[k].debug();
			channels2[k].debug();
			x = getValue( channels1[k].at(0,0) , channels2[k].at(0,0) );
			cout << " oka " << x << '\n';
		}
		*/
		for (int j = 0; j < height; j++) {
			//cout << "Pixv " << j << '\n';
			for (int i = 0; i < width; i++) {

				sum = 0.0;
				for (int k = 0; k < channels; k++){
					x = getValue( channels1[k].at(i,j) , channels2[k].at(i,j) );
					if (powerUp)
						sum += pow(abs(x),l);
					else
						sum += abs(x);  // TODO ?
				}

				//if (i==j) cerr << x << '\t';

				if (powerDown)
					sum = pow(sum,lInv);

				//if (i==j) cerr << sum << '\n';

				if (scaled){
					dstR.at(i,j) = static_cast<T3>(max*(1.0-halfWidth/(halfWidth+sum)));
				}
				else if (scaledInv){
					dstR.at(i,j) = static_cast<T3>(max*(halfWidth/(halfWidth-sum)));
				}
				else {
					switch (mapping) {
					case 'l':
						dstR.at(i,j) = static_cast<T3>(sum);
						break;
					case 'i':
						dstR.at(i,j) = static_cast<T3>(maxOrigin - sum);
						break;
					default:
						string error("PixelVectorOp: unknown mapping: ");
						throw (runtime_error(error + mapping));
						break;
					}

				}
			}
		}
		if (tmpNeeded){
			dst.setGeometry(width,height,1);
			CopyOp<T3,T3>().filter(dstR,dst);
		};
	};

protected:

	virtual
	double getValue(const T &src, const T2 &src2) const = 0;



};

/**! Computes dot ChannelVector of intensities of two images.
 *    dst(i,j) = \sqrt{ src(i,j,0)*src2(i,j,0) + ... + src(i,j,k)*src2(i,j,k) }.
 *
 *  \see DistanceOp
 *  \see MagnitudeOp
 *  \see ChannelVectorOp
 */
template <class T=unsigned char,class T2=unsigned char>
class ProductOp : public PixelVectorOp<T,T2>
{
public:

	ProductOp(const string & p = "l") :
			PixelVectorOp<T,T2>("ProductOp","Computes the dot product of pixel vectors."){
		this->setParameters("l,1,1");
		this->setParameters(p);
		// this->setInfo("Computes the product of pixel vectors. Applies mapping: [l]inear, [i]nverse or bilinear > 0.",
			//	"mapping,l,lInv",p);
	};

protected:

	inline
	double getValue(const T &src, const T2 &src2) const {
		return static_cast<double>(src)*static_cast<double>(src2); };

};

/** Computes dot ChannelVector of intensities of two images.
 *    dst(i,j) = \sqrt{ src(i,j,0)*src2(i,j,0) + ... + src(i,j,k)*src2(i,j,k) }.
 *
 *  \see DistanceOp
 *  \see ProductOp
 */
template <class T=unsigned char,class T2=unsigned char>
class MagnitudeOp : public PixelVectorOp<T,T2>
{
public:

	MagnitudeOp(const string & p = "l,2"):PixelVectorOp<T,T2>("ProductOp","Computes the magnitude of a pixel vector."){
		// Assumes that user will not give lInv
		// There is a bug in ImageOp, so every param will be initl'd
		this->setParameters(p);
		this->setParameter("lInv",1.0/this->getParameter("l",1.0));
		this->setParameters(p); // yes
	};


	virtual
	void filter(const Image<T> &src,Image<T2> &dst) const {
		PixelVectorOp<T,T2>::filter(src,src,dst);
	};

protected:
	inline
	double getValue(const T &src, const T2 &src2) const {
		//cout << static_cast<double>(src2);
		return static_cast<double>(src)*static_cast<double>(src2);
		//cerr << "magn\n";
		//exit(-1);
	};

private:
	template <class T3>
	void filter(const Image<T> &src1,const Image<T2> &src2,Image<T3> &dst) const {};

};

/** Computes distance of intensity vectors of two images.
 *    dst(i,j) = \sqrt{ src(i,j,0)*src2(i,j,0) + ... + src(i,j,k)*src2(i,j,k) }.
 *
 *  \see DistanceOp
 *  \see MagnitudeOp
 */
template <class T=unsigned char,class T2=unsigned char>
class DistanceOp : public PixelVectorOp<T,T2>
{
public:

	DistanceOp(const string & p = "l,2,0.5") :
		PixelVectorOp<T,T2>("DistanceOp","Computes the distance of pixel vectors."){
		this->setParameters(p);
	};

protected:

	double getValue(const T &src, const T2 &src2) const {
		//static double d;
		return static_cast<double>(src) - static_cast<double>(src2);
		//return d*d;
	};

};


}
}


#endif /*PixelVectorOP_H_*/
