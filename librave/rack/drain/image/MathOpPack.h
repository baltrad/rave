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
#ifndef MATHOP_H_
#define MATHOP_H_

#include <cmath>

#include "SequentialImageOp.h"


namespace drain
{
namespace image
{

/// A set of for mathematical operations.
template <class T=unsigned char,class T2=unsigned char>
class MathOp: public SequentialImageOp<T,T2>
{
public:

	MathOp(const string &name = "MathOp",
			const string &description="Mathematical operator.",
			const string & parameterNames="scale,bias",
			const string & defaultValues = "1.0,0.0") :
				SequentialImageOp<T,T2>(name,description,parameterNames,defaultValues) {
	}

	MathOp(const string &name, const string &description, double scale, double bias = 0.0) :
		SequentialImageOp<T,T2>(name,description,"scale,bias","1,0") {
		this->setParameter("scale",scale);
		this->setParameter("bias",bias);
	}

	virtual void setScale(float scale,float bias = 0.0){
		this->setParameter("scale",scale);
		this->setParameter("bias",bias);
	}


	void initialize() const {
		scale = this->parameters.get("scale",scale);
		bias  = this->parameters.get("bias",bias);
	}

	void filter(const Image<T> &src,Image<T2> &dst) const {
		filter(src,dst,dst);
	};


	// TODO: consider template <class T3> for src2
	void filter(const Image<T> &src,const Image<T2> &src2,Image<T2> &dst) const {

		initialize();
		makeCompatible(src,dst);

		typename std::vector<T>::const_iterator s  = src.begin();
		typename std::vector<T2>::const_iterator s2 = src2.begin();
		typename std::vector<T2>::iterator d;
		typename std::vector<T2>::iterator dEnd = dst.end();

		for (d=dst.begin(); d!=dEnd; d++,s++, s2++)
			filterValue(*s,*s2,*d);

	}


	inline
	virtual void filterValue(const T &src, T2 &dst) const {
		dst = static_cast<T2>(src);
	};

	///
	virtual void filterValue(const T &src, const T2 &src2, T2 &dst) const {
		if (sizeof(src2) > 0) // dummy operation to skip "unused parameter" wraning
			filterValue(src,dst);
	}

	//void filterValue(const T &src, T2 &dst) = 0; //{ dst 


	// This is problematic. Float could be limited to 0...1.0
	inline 
	static T2 limit(double x){
		//static const double xMax = static_cast<double>( Intensity::max<T2>() );
		//static const double xMin = static_cast<double>( Intensity::min<T2>() );
		// TODO consts OUTSIDE
		static const double xMin = static_cast<double>(
				numeric_limits<T2>::is_integer ?  std::numeric_limits<T2>::min() : -std::numeric_limits<T2>::max() );
		static const double xMax = static_cast<double>( std::numeric_limits<T2>::max() );
		x = max(xMin,x);
		x = min(x,xMax);
		return static_cast<T2>(x);
	};

protected:


	mutable float bias;
	mutable float scale;

}; 

/// Rescales intensities: f' = scale*f + bias.
/**
 *   Inversely: f = (f'-bias)/scale = a*f+b, where a=1/scale and b=-bias/scale.
 */
template <class T=unsigned char,class T2=unsigned char>
class ScaleOp: public MathOp<T,T2>
{
public:

	ScaleOp() : MathOp<unsigned char,T2>("ScaleOp","Scales intensities.",1,0) {
		setDefaultScale();
	};

	ScaleOp(float scale,float bias = 0.0) :
		MathOp<T,T2>("ScaleOp","Scales intensities.",scale,bias) {
	};

	ScaleOp(const string & p) : MathOp<T,T2>("ScaleOp","Scales intensities.","scale,bias",p) {
	};


	//void filterValue(const T &src, const T2 &src2, T2 &dst) const {
	void filterValue(const T &src, T2 &dst) const {
		dst = MathOp<T,T2>::limit(this->scale*static_cast<double>(src) + this->bias);
	};

protected:

	void setDefaultScale(){
		const int s1 = sizeof(T);
		const int s2 = sizeof(T2);
		// TODO const type_info &x = typeid(T);
		if ((s1 > 2) || (s2 > 2)){
			MathOp<T,T2>::setScale(1,0);
		}
		else {
			MathOp<T,T2>::setScale(pow(2.0,8.0*(s2-s1)),0);
		}
	};

};


//template <>

/*
template <class T2>
ScaleOp<unsigned short int,T2>::ScaleOp() :
	MathOp<unsigned short int,T2>("ScaleOp","Scales intensities.",static_cast<float>(sizeof(T2))/2.0f,0) {
};

template <class T,class T2>
ScaleOp<T,T2>::ScaleOp() : MathOp<T,T2>("ScaleOp","Scales intensities.",1.0,0){

};
*/


template <class T=unsigned char,class T2=unsigned char>
class NegateOp: public MathOp<T,T2>
{
public:

	NegateOp(const string & max = "") : MathOp<T,T2>("NegateOp","Inverts intensities.","max",max) {
		// debuggery...
		try {
			if (max.empty())
				this->setParameter("max",static_cast<double>(Intensity::maxOrigin<T2>()));
			return;
		}
		catch (runtime_error &e){
			cerr << e.what() << '\n';
		}
		catch (exception e) {
			cerr << e.what() << '\n';
		}
		cerr << this->getParameterNames() << '\n';
			//this->setParameter("max",255);
	};

	void initialize() const {
		this->bias  = this->parameters.get("max",static_cast<double>( Intensity::maxOrigin<T2>() ));
		this->scale = -1;
		// debug:
		//this->help(cout);
		//cout << "bias:" << this->bias << '\n';
	}

	virtual void filterValue(const T &src, T2 &dst) const {
		dst = MathOp<T,T2>::limit(this->scale*static_cast<double>(src) + this->bias);
		//dst = -static_cast<T2>(src);
	};
};


/// Maps a single intensity value to another value.
template <class T=unsigned char,class T2=unsigned char>
class RemapOp: public MathOp<T,T2>
{
public:

	RemapOp(const string & p = "0,0") :
		MathOp<T,T2>("RemapOp","Changes an intensity.","from,to",p) {
	};

	void initialize() const {
		this->fromValue = static_cast<T>(this->parameters.get("from",0.0));
		this->toValue   = static_cast<T2>(this->parameters.get("to",0.0));
	}

	void filterValue(const T &src, T2 &dst) const {
		if (src == fromValue)
			dst = toValue;
		else
			dst = static_cast<T2>(src);
	};

private:
	mutable T fromValue;
	mutable T2 toValue;
};

template <class T=unsigned char,class T2=unsigned char>
class AdditionOp: public MathOp<T,T2>
{
public:
	AdditionOp(const string & p = "") :
		MathOp<T,T2>("AdditionOp","Adds intensities.","scale,bias",p) {
	};

	inline void filterValue(const T &src, const T2 &src2, T2 &dst) const {
		dst = MathOp<T,T2>::limit(this->scale*(src+src2) + this->bias);
	};
};

template <class T=unsigned char,class T2=unsigned char>
class SubtractionOp: public MathOp<T,T2>
{
public:

	SubtractionOp(const string & p = "1.0,0.0") :
		MathOp<T,T2>("SubtractionOp","Subtracts intensities.","scale,bias","1,0"){
	};

	SubtractionOp(float scale,float bias) :
		MathOp<T,T2>("SubtractionOp","Subtracts intensities.",scale,bias) {
	};

	inline void filterValue(const T &src, const T2 &src2, T2 &dst) const {
		//dst = (this->scale*(static_cast<double>(src)-static_cast<double>(src2)) + this->bias);
		dst = MathOp<T,T2>::limit(this->scale*(static_cast<double>(src)-static_cast<double>(src2)) + this->bias);
		//dst = MathOp<T,T2>::limit(this->scale*(src-src2) + this->bias);
	};
};

/// Multiplies two images, scaling with the scale of the target image.
template <class T=unsigned char,class T2=unsigned char>
class MultiplicationOp: public MathOp<T,T2>
{
public:
	//MultiplicationOp(const string & p = "") : MathOp<T,T2>(p) {
	MultiplicationOp(const string & p = "255.0,0") :
		MathOp<T,T2>("MultiplicationOp","Multiplies intensities.","scale,bias",p) {
		//this->scale = Intensity::max<T>();
	}

	inline void filterValue(const T &src, const T2 &src2, T2 &dst) const {
		//dst = MathOp<T,T2>::limit(this->scale*(src*src2)/Intensity::max<T2>() + this->bias);
		dst = MathOp<T,T2>::limit((src*src2)/this->scale + this->bias);
		//dst = static_cast<T2>((src*src2) + this->bias);
	};
};

template <class T=unsigned char,class T2=unsigned char>
class DivisionOp: public MathOp<T,T2>
{
public:

	DivisionOp(const string & p = "1,0") :
		MathOp<T,T2>("DivisionOp","Divides intensities of an image with those of another.","coeff,bias",p){
	};

	inline void filterValue(const T &src, const T2 &src2, T2 &dst) const {
		if (src2 != 0)
			dst = static_cast<T2>((this->bias*src)/src2);
	};
};

template <class T=unsigned char,class T2=unsigned char>
class MixerOp: public MathOp<T,T2>
{
public:

	MixerOp(const string & p = "0.5") :
		MathOp<T,T2>("MixerOp","Blends an image to another with given proportion.","coeff",p) {
	}


	MixerOp(double coeff) :
			MathOp<T,T2>("MixerOp","Blends an image to another with given proportion.","coeff","0.5") {
		this->setParameter("coeff",coeff);
	}


	void initialize() const {
		coeff  = this->getParameter("coeff",coeff);
		coeff2 = 1.0-coeff;
	}

	inline void filterValue(const T &src, const T2 &src2, T2 &dst) const {
		dst = static_cast<T2>(coeff*src + coeff2*src2);
	};

protected:
	mutable float coeff;
	mutable float coeff2;
};

/*
 * replaced by NegateOp
template <class T=unsigned char,class T2=unsigned char>
class InversionOp: public MathOp<T,T2>
{
public:
   InversionOp(){
   		this->scale = -1;
   		this->bias = Intensity::max<T>();
   }
   inline void filterValue(const T &src, const T2 &src2, T2 &dst){
   		dst = static_cast<T2>(this->bias + this->scale * src);
   }; 

};
 */


template <class T=unsigned char,class T2=unsigned char>
class MaximumOp: public MathOp<T,T2>
{
public:
	MaximumOp(){
		this->setInfo("Maximum intensity.","","");
		this->name="MaximumOp";
	}

	/// WARNING BUGS!
	//MaximumOp(const string & p = "scale") : MathOp<T,T2>(p) {};
	inline void filterValue(const T &src, const T2 &src2, T2 &dst) const {
		dst = static_cast<T2>(src);
		if (src2 > dst)  // to avoid multiple casts
			dst = src2;
	};
};

template <class T=unsigned char,class T2=unsigned char>
class MinimumOp: public MathOp<T,T2>
{
public:
	MinimumOp(){
		this->setInfo("Minimum intensity.","","");
		this->name="MinimumOp";
	}

	inline void filterValue(const T &src, const T2 &src2, T2 &dst) const {
		dst = static_cast<T2>(src);
		if (src2 < dst)  // to avoid multiple casts
			dst = src2;
		//dst = static_cast<T2>(src<src2 ? src : src2);
	};
};



}
}

#endif /*THRESHOLD_H_*/
