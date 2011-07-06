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
#ifndef FASTAVERAGEOP_H_
#define FASTAVERAGEOP_H_

#include "SlidingStripeOp.h"
#include "WindowOp.h"

namespace drain
{

namespace image
{


template <class T=unsigned char,class T2=unsigned char>
class SlidingStripeAverage : public SlidingStripe<T,T2>
//public SlidingWindowAverage<T,T2>
{
public:

	double sum;
	float areaF;

	 SlidingStripeAverage(int width=1,int height=1) {
	   this->setSize(width,height);
	 };

	 //virtual ~SlidingWindowStripe(){};


	/**
	 *   TODO: weight = > weightSUm
	 *   TODO: multiple channels (banks)?
	 */
	 virtual void initialize(){

		this->location.setLocation(0,0);;
		this->areaF = this->getArea();

		cerr << "area" << this->areaF << "\n";
		cerr << this->dst.getCoordinateHandler() << '\n';

		this->sum = 0;
		if (this->horizontal){
			for (int i = this->iMin; i <= this->iMax; i++) {
				this->p.setLocation(i,0);
				this->dst.getCoordinateHandler().handle(this->p);
				this->sum += this->src.at(this->p);
			}
		}
		else {
			for (int j = this->jMin; j <= this->jMax; j++) {
				this->p.setLocation(0,j);
				this->dst.getCoordinateHandler().handle(this->p);
				this->sum += this->src.at(this->p);
			}
		}
	}



	/// Kesken
	virtual void updateHorz(int dx){

		const int xOld = this->getXOld(dx);
		const int xNew = this->getXNew(dx);

		if (this->horizontal){
			this->p.setLocation(xOld,this->location.y);
			this->dst.getCoordinateHandler().handle(this->p);
			this->sum -= this->src.at(this->p);

			this->p.setLocation(xNew,this->location.y);
			this->dst.getCoordinateHandler().handle(this->p);
			this->sum += this->src.at(this->p);

		}
		else
		{
			this->sum = 0;
			for (int j = this->jMin; j <= this->jMax; j++) {
				this->p.setLocation(xNew,this->location.y+j);
				this->dst.getCoordinateHandler().handle(this->p);
				this->sum += this->src.at(this->p);
			}

		}

	}

	/// Kesken
	virtual void updateVert(int dy){

		const int yOld = this->getYOld(dy);
		const int yNew = this->getYNew(dy);

		if (this->horizontal){
			this->sum = 0;
			for (int i = this->iMin; i <= this->iMax; i++) {
				this->p.setLocation(this->location.x+i,yNew);
				this->dst.getCoordinateHandler().handle(this->p);
				this->sum += this->src.at(this->p);
			}
		}
		else {

			this->p.setLocation(this->location.x,yOld);
			this->dst.getCoordinateHandler().handle(this->p);
			this->sum -= this->src.at(this->p);

			this->p.setLocation(this->location.x,yNew);
			this->dst.getCoordinateHandler().handle(this->p);
			this->sum += this->src.at(this->p);
		}

	}

	virtual void write(){
		this->dst.at(this->location) =  static_cast<T2>(sum/areaF);
	}

	//protected:
	//	bool horizontal;

};

/// Supports using another image (channel) for weighting the pixels of the source image.
/**
 *
 *
 *
 */
template <class T=unsigned char,class T2=unsigned char>
class SlidingStripeAverageWeighted : public SlidingStripeAverage<T,T2>
{
public:

	double sum;
	float sumW;
	float areaF;

	SlidingStripeAverageWeighted(int width=1,int height=1) : SlidingStripeAverage<T,T2>(width,height) {};


	/**
	 *   TODO: weight = > weightSUm
	 *   TODO: multiple channels (banks)?
	 */
	 virtual void initialize(){

		static T w;
	
		this->location.setLocation(0,0);
		this->sum = 0;
		this->sumW = 0;
		this->areaF = this->getArea();
		
		if (this->horizontal){
			for (int i = this->iMin; i <= this->iMax; i++) {
				this->p.setLocation(i,0);
				this->dst.getCoordinateHandler().handle(this->p);
				w = this->srcWeight.at(this->p);
				this->sum  += w*this->src.at(this->p);
				this->sumW += w;
			}
		}
		else {
			for (int j = this->jMin; j <= this->jMax; j++) {
				this->p.setLocation(0,j);
				this->dst.getCoordinateHandler().handle(this->p);
				w = this->srcWeight.at(this->p);
				this->sum  += w*this->src.at(this->p);
				this->sumW += w;
			}
		}
	}



	/// Kesken
	virtual void updateHorz(int dx){
		
		static T w;
		const int xOld = this->getXOld(dx);
		const int xNew = this->getXNew(dx);

		if (this->horizontal){
			this->p.setLocation(xOld,this->location.y);
			this->dst.getCoordinateHandler().handle(this->p);
			w = this->srcWeight.at(this->p);
			this->sum  -= w*this->src.at(this->p);
			this->sumW -= w;

			this->p.setLocation(xNew,this->location.y);
			this->dst.getCoordinateHandler().handle(this->p);
			w = this->srcWeight.at(this->p);
			this->sum  += w*this->src.at(this->p);
			this->sumW += w;
		}
		else
		{
			this->sum  = 0;
			this->sumW = 0;
			for (int j = this->jMin; j <= this->jMax; j++) {
				this->p.setLocation(xNew,this->location.y+j);
				this->dst.getCoordinateHandler().handle(this->p);
				w = this->srcWeight.at(this->p);
				this->sum  += w*this->src.at(this->p);
				this->sumW += w;
			}

		}

	}

	/// Kesken
	virtual void updateVert(int dy){

		const int yOld = this->getYOld(dy); // location.y + (dy>0 ? jMin-1 : jMax+1);
		const int yNew = this->getYNew(dy); // location.y + (dy>0 ? jMax     : jMin);
		static T w;

		if (this->horizontal){
			this->sum  = 0;
			this->sumW = 0;
			for (int i = this->iMin; i <= this->iMax; i++) {
				this->p.setLocation(this->location.x+i,yNew);
				this->dst.getCoordinateHandler().handle(this->p);
				w = this->srcWeight.at(this->p);
				this->sum  += w*this->src.at(this->p);
				this->sumW += w;
			}
		}
		else {

			this->p.setLocation(this->location.x,yOld);
			this->dst.getCoordinateHandler().handle(this->p);
			w = this->srcWeight.at(this->p);
			this->sum  -= w*this->src.at(this->p);
			this->sumW -= w;

			this->p.setLocation(this->location.x,yNew);
			this->dst.getCoordinateHandler().handle(this->p);
			w = this->srcWeight.at(this->p);
			this->sum  += w*this->src.at(this->p);
			this->sumW += w;
		}

	}

	virtual void write(){
		if (sumW > 0)
			this->dst.at(this->location) =  static_cast<T2>(sum/sumW);
		else
			this->dst.at(this->location) = 0;
		this->dstWeight.at(this->location) =  static_cast<T2>(sumW/areaF);
		//this->dst.at(this->location) =  128; 
	}

	/// Sets the source channel from which the pixel weights will be read.
	/// Sets the target channel in which resulting weight will be written.

};



/// If source images contain alpha channels, adopts to weighted mode.

template <class T=unsigned char,class T2=unsigned char>
class FastAverageOp : public WindowOp<T,T2> //public SlidingStripeOp<T,T2>
{
public:
	
	FastAverageOp(const string & p = "")  { //: SlidingStripeOp<T,T2>(stripeAvg) {
		this->setInfo("Weighted window averaging operator.","width,height",p);
	};
	
	FastAverageOp(int width,int height) { //: SlidingStripeOp<T,T2>(stripeAvg) {
		this->setInfo("Weighted window averaging operator.","width,height","0,0");
		this->setParameter("height",height);
		this->setParameter("width",width);
	};

	virtual void filter(const Image<T> &src,Image<T2> &dst) const {
		
		makeCompatible(src,dst);
		
		// UNWEIGHTED
		if (src.getAlphaChannelCount() == 0){
			cerr << "ssaw:filter0\n";
			unsigned int w = this->getParameter("width",3);
			unsigned int h = this->getParameter("height",w);
			SlidingStripeAverage<T,T2> window1; //(w,h);
			SlidingStripeAverage<T2,T2> window2; //(w,h);
			SlidingStripeOp<T,T2> op(window1,window2,"width,height");
			op.setParameter("width",w);
			op.setParameter("height",h);
			//cerr << "SSAW calling SlidingStripeOp/f2 with " << window1 << "\n";
			//cerr << "SSAW calling SlidingStripeOp/f2 with " << window2 << "\n";
			for (unsigned int i=0; i<src.getChannelCount(); i++){
				op.filter(src.getChannel(i),dst.getChannel(i));	
			}
		}
		// WEIGHTED
		else {
			cerr << "SSAW calling f2->f4\n";
			filter(src.getImageChannels(),src.getAlphaChannels(),
				dst.getImageChannels(),dst.getAlphaChannels());
		}
		
	};
	
	/// Calls SlidingStripeOp<T,T2>::filter() separately for each image channel. This is natural for many operations, such as averaging.

	// Raise filterUnweighted	
	virtual void filter(const Image<T> &src,const Image<T> &srcWeight,
		Image<T2> &dst,Image<T2> &dstWeight) const {
		
		makeCompatible(src,dst);	
		makeCompatible(srcWeight,dstWeight);	
		
		//src.setName("src");
		//srcWeight.setName("srcWeight");
		//dst.setName("dst");
		//dstWeight.setName("dstWeight");
		
		unsigned int w = this->getParameter("width",3);
		unsigned int h = this->getParameter("height",w);
		SlidingStripeAverageWeighted<T,T2> window1; //(w,h);
		SlidingStripeAverageWeighted<T2,T2> window2; //(w,h);
		SlidingStripeOp<T,T2> op(window1,window2,"width,height");
		op.setParameter("width",w);
		op.setParameter("height",h);
			
		const unsigned int imageChannels = src.getImageChannelCount();	
		const unsigned int alphaChannels = srcWeight.getChannelCount();
		for (unsigned int i=0; i<imageChannels; i++){	
			const unsigned int a = i % alphaChannels;
			cerr << "SSAW calling SlidingStripeOp/4 " << i << ':' << a << " \n";
			op.filter(src.getChannel(i),srcWeight.getChannel(a),
						dst.getChannel(i),dstWeight.getChannel(a));
		}	
		
	}
	
protected:
	SlidingStripeAverageWeighted<T,T2> stripeAvg;	
};

}

}

#endif /*SLIDINGWINDOWAVERAGESTRIPEWEIGHTED_H_*/
