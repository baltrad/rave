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
#ifndef CUMULATOR_H_
#define CUMULATOR_H_

//#include "coordinates.h" // for site coords and bin coords.


//#include <drain/util/proj.h>  // for geographical projection of radar data bins

#include <math.h>

#include <stdexcept>

#include "Point.h"
#include "Image.h"
#include "CoordinateHandler.h" 



// TODO: image/
/** See also radar::Compositor
 * 
 */
namespace drain
{

namespace image
{


/// General-purpose image compositing.
/*!
  Main features:

  Injection. The main loop iterates radar (image) data points of type <Ti>, 
  mapping each point to a single point in the target array of type <T>;
  hence the name injector.

  Cumulation. The compositing starts by initiating the target image,
  see open().
  Each input radar image is projected on the target image cumulatively,
  see execute(). Finally, the cumulation must be completed by
  normalizing the result with the number/weight of
  cumulated inputs, see close().
  Hence, the target which should be seen.

  The weights are stored in the first alphaChannel (index 0) of the
  target image. If the target has a second image channel, also
  the (weighted) squared values are cumulated.  

  In cumulation, ...
  \f[
  x = \sqrt[p]{\frac{\sum_i {q_i}^r x_i^p}{\sum_i {q_i}^r } }
  \f]

  The default type of target image, that is, cumulation array, is
  double. If no (float valued) weighting  is applied, target image
  could be long int as well.
 */
template <class T = double>
class Cumulator 
{
public:

	/// Default constructor. The channels are DATA, COUNT, WEIGHT, WEIGHT2
	Cumulator(unsigned int width = 0, unsigned int height = 0, const string & method="WAVG", double p=1.0,double r=0.0){
		setGeometry(width,height);
		setMethod(method,p,r);
	}

	virtual ~Cumulator(){};

	virtual
	void setGeometry(unsigned int width,unsigned int height){
		this->width = width;
		this->height = height;

		data.setGeometry(width,height);
		weight.setGeometry(width,height);
		count.setGeometry(width,height);
		dataSquared.setGeometry(width,height);

		coordinateHandler.setBounds(width,height);
	}

	void clear(){
		data.clear();
		weight.clear();
		count.clear();
		dataSquared.clear();
	}

	enum Method { AVG, WAVG, MAX, MAXW, OVERWRITE, UNDEFINED};

	Method method;

	void setMethod(const string & method, double p=1,double r=0){
		if (method=="AVG")
			setMethod(AVG);
		else if (method=="WAVG")
			setMethod(WAVG,p,r);
		else if (method=="MAX")
			setMethod(MAX);
		else if (method=="MAXW")
			setMethod(MAXW);
		else if (method=="OVERWRITE")
			setMethod(OVERWRITE);
		else
			setMethod(UNDEFINED);
	}

	void setMethod(Method m, double p=1,double r=0){
			this->method = m;
			this->p = p;
			this->r = r;	
			if (m == UNDEFINED)
				throw runtime_error(method + ": undefined method");
	}

	inline
	const unsigned int & getWidth() const {return width;};

	inline
	const unsigned int & getHeight() const {return height;};

	//void add(const Image &src,int x=0,int y=0);

	//template <class T>
	///
	inline
	void add(const int &i, const int &j,double value, double weight);

	//template <class T>	void putWeighted(int i, int j,int k,T value, T weight);
	/// \deprecated Getting obsolete.ss
	inline
	//void addDirect(const T &s,const T &w,const int &i=0, const int &j=0);
	void addDirect(const int &i, const int &j,const T &s,const T &w,const unsigned int &c,const T &s2);

	//void add(const Image &src,int x=0,int y=0);

	/// Copies the contents of the cumulation array to the target image.
	// COnsider template based such that INTERNAL variables are of type T ? 
	/*! d = data, scaled
	 *  w = weight, scaled
	 *  c = count
	 *  p = data, sum of squared ("power"), scaled
	 *  s = standard deviation, scaled
	 *  D = data, cumulated  (debugging)
	 *  W = weight, cumulated
	 *  C = count, scaled
	 *  S = standard deviation, unscaled
	 */
	//void extract(Image &target,string channels = "dw") const;
	template <class T2>
	void extractTo(Image<T2> &dst,const string & channels="dw") const;

	/// Drops an image
	template <class T2>
	void addImage(const Image<T2> &src,const Image<T2> &srcWeight,
			float globalWeight = 1.0, int i=0, int j=0);

	// TODO: different projections



	/// Overall weight for source image.
	//double priorWeight;

	/// Data weight (power)
	double p;

	/// Quality weight (power)
	double r;

	bool debug;


	CoordinateHandler coordinateHandler;

protected:
	unsigned int width;
	unsigned int height;

	//private:
	Image<T> data;
	Image<T> weight;
	Image<unsigned int> count;
	Image<T> dataSquared;

	Options properties;
};


template <class T>
void Cumulator<T>::add(const int &i, const int &j, double s, double w){

	//if (coordinateHandler.handle(int(i)));
	if (i<0)
		return;

	if (j<0)
		return;

	if (i>=static_cast<int>(width))
		return;

	if (j>=static_cast<int>(height))
		return;


	const long int it = data.address(i,j);

	static const T wd = 128;
	//if (r != 1)
	//  w = pow(w,r);
	//cerr << (float)s << '\t';

	switch (method) {
	case AVG:
		if (w > 0){ // Skip no-data areas
			data.at(it)   += wd*s;
			weight.at(it) += wd;
			count.at(it)  += 1;
			dataSquared.at(it) += wd*s*s;
		}
		break;
	case WAVG:
		if (p != 1)
			s = pow(s,p);
		if ((r > 0.0)&&(r != 1.0))
			w = pow(w,r);
		data.at(it)   += w * s;
		weight.at(it) += w;
		count.at(it)  += 1;
		dataSquared.at(it) += w * s*s;
		break;		
	case MAX:
		if ((weight.at(it)==0)||(s >= (data.at(it)/weight.at(it)))){
			data.at(it) = w * s;
			weight.at(it) = w;
			count.at(it)  = 1;
		}
		break;		
	case MAXW:
		if (w >= weight.at(it)){
			data.at(it) = w * s;
			weight.at(it) = w;
			count.at(it)  = 1;
		}
		break;
	case OVERWRITE:
		data.at(it) = w*s;
		weight.at(it) = w;
		count.at(it)  = 1;
		dataSquared.at(it) = w*s*s;
	//case UNDEFINED:
	//	break;
	default:
		throw runtime_error(method + ": undefined method");
	}

	//dataSquared.at(it) = 57;

}

template <class T>
void Cumulator<T>::addDirect(const int &i, const int &j,const T &s,const T &w,const unsigned int &c,const T &s2){

	if ((i<0)||(j<0)||(i>=width)||(j>=height))
		return;

	const long int it = data.address(i,j);
	data.at(it)   += s;
	weight.at(it) += w;
	count.at(it)  += c;
	dataSquared.at(it) += s2;
}

template <class T>
template <class T2>
void Cumulator<T>::addImage(const Image<T2> &src,const Image<T2> &srcWeight,
		float globalWeight, int iOffset, int jOffset){

	const unsigned int w = src.getWidth();
	const unsigned int h = src.getHeight();
	//const unsigned int wCum = getWidth();
	//const unsigned int hCum = getHeight();

	int a;
	Point2D<int> p;
	//CoordinateHandler handler(getWidth(),getHeight());
	//const int r = 64+(rand()&127);
	for (unsigned int i = 0; i < w; ++i) {
		for (unsigned int j = 0; j < h; ++j) {
			p.setLocation(iOffset+i,jOffset+j);
			if (coordinateHandler.handle(p) == CoordinateHandler::UNCHANGED){
				a = src.address(i,j);
				add(p.x,p.y,src.at(a),globalWeight * srcWeight.at(a));
			}
			// else overflow=true;

			//add(iOffset+i,jOffset+j,src.at(i,j),100);
			//add(i,j,src.at(i,j),globalWeight*10.0);
			//add(i,j,r,globalWeight*10.0);
			//debug
			//dataSquared.at(i,j) = (i+j)&255;
			//data.at(i,j) = 123;
			//weight.at(i,j) = 246;
		}
	}
}


template <class T>
template <class T2>
void Cumulator<T>::extractTo(Image<T2> &dst,const string &channels) const {


	// const Geometry g = getGeometry();
	// const int width  = g.getWidth();
	// const int height = g.getHeight();
	const int area = width * height;
	const unsigned int channelCount = channels.length();
	//const int area = g.getArea();
	//const int alphaChannels =  g.getAlphaChannelCount();
	unsigned int dstImageChannelCount = channelCount;

	if ((dst.getWidth() != width) || (dst.getHeight() != height) ||
			(dst.getChannelCount() != channelCount)){
		dst.setGeometry(width,height,channelCount,0);
		// cerr warning...
	}


	if (drain::Debug > 2){
		cerr << " Extracting cumulated image To dst:\n";
		dst.debug(cerr);
	}

	double w = 0;
	double c = 0;

	for (unsigned int k=0; k<channelCount; ++k) {

		char ch = channels.at(k);

		if (drain::Debug > 3)
			cerr << "Extracting channel: " << ch << endl;

		Image<T2> &channel = dst.getChannel(k);

		if ((ch != 'D') && (ch != 'd') && (k<dstImageChannelCount))
			dstImageChannelCount = k;

		switch (ch) {
		case 'D':  // data, unscaled
			for (int i = 0; i < area; ++i) {
				channel.at(i) = static_cast<T2>(data.at(i));
			}
			break;
		case 'd':  // data, scaled
			for (int i = 0; i < area; ++i) {
				w = weight.at(i);
				if (w > 0){  // or w!=0 ?
					if ((p == 1)||(p == 0)) // unity??
						channel.at(i) = static_cast<T2>(data.at(i)/w);
					else //if (p > 0)
						channel.at(i) = static_cast<T2>(pow(data.at(i)/w,1.0/p));
				}
				else
					channel.at(i) = 0;
			}
			break;
		case 'W':  // weight, scaled (average weight)
			for (int i = 0; i < area; ++i) {
				channel.at(i) = static_cast<T2>(weight.at(i));
			}
			break;
		case 'w':  // weight, scaled (average weight)
			for (int i = 0; i < area; ++i) {
				c = count.at(i);
				if (c > 0){
					if ((r == 1)||(r == 0)) // unity?
						channel.at(i) = static_cast<T2>(weight.at(i)/c);
					else //if (r>0)
						channel.at(i) = static_cast<T2>(pow(weight.at(i)/c,1.0/r));
				}
				else
					channel.at(i) = 0;
			}
			break;
		case 'C':  // count
			for (int i = 0; i < area; ++i) {
				channel.at(i) = static_cast<T2>(count.at(i));
			}
			break;
		case 'c':  // count, scaled
			for (int i = 0; i < area; ++i) {
				channel.at(i) = 255 - 255/(1+static_cast<T2>(count.at(i)));
			}
			break;
		case 'S':  // st.deviation, unscaled (and quite good so) // TODO other is count-based?
			for (int i = 0; i < area; ++i) {
				double d  = data.at(i);
				double d2 = dataSquared.at(i);
				double w  = weight.at(i);
				//unsigned int c = count.at(i);
				if (w > 0){
					//channel.at(i) = static_cast<T2>(d2);
					channel.at(i) = static_cast<T2>(sqrt((double)(d2/w - (d*d)/(w*w))));
				}
				else
					channel.at(i) = 0;
			}
			break;
		case 's':  // count, scaled
		default:
			throw runtime_error(string("Error: Cumulator: undefined channel extrator: ") +  ch);
			break;
		}
	}
	dst.setChannelCount(dstImageChannelCount,channelCount-dstImageChannelCount);
	//cerr << "setting dst geometry" << dst.getGeometry() << '\n';
	dst.properties = properties;
}






}

}

#endif /* Cumulator_H_ */
