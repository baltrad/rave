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
#ifndef SLIDINGWINDOWOPTICALFLOWOP_H_
#define SLIDINGWINDOWOPTICALFLOWOP_H_

#include <sstream>

#include "SlidingWindow.h"
#include "SlidingWindowOp.h"

//#include "FastAverageOp.h"
#include "DoubleSmootherOp.h"
#include "FuzzyPeakOp.h"
#include "GradientOp.h"
#include "PixelVectorOp.h"
#include "QuadraticSmootherOp.h"
#include "RecursiveRepairerOp.h"

namespace drain
{

namespace image
{

/// Consider using SlidingStripeOpticalFlow instead
/**
 *  T  = class of the input
 *  T2 = class of the derivative  images, weight, and output (motion field and weight)
 */
 template <class T=double> // ,class T2=double> // NOTICE
class SlidingOpticalFlow : public SlidingWindow<T,T> {

  public:
	
	
	double wXX;
	double wXY;
	double wYY;
	double wXT;
	double wYT;
	//double area;

	/// Precomputed horizontal diffential 
	ImageView<T> Dx;
	
	/// Precomputed vertical diffential 
	ImageView<T> Dy;
	
	/// Precomputed time diffential 
	ImageView<T> Dt;
	
	/// Precomputed weight field (optional) 
	ImageView<T> w;
	
	ImageView<T> u;
	ImageView<T> v;
	ImageView<T> q;
	

	SlidingOpticalFlow(int width = 0, int height = 0){
		this->setSize(width,height);
	}

	///
	void setDx(Image<T> &image, unsigned int k = 0){
		Dx.viewChannel(image,k);
	}

	void setDy(Image<T> &image, unsigned int k = 0){
		Dy.viewChannel(image,k);
	}


	void setDt(Image<T> &image, unsigned int k = 0){
		Dt.viewChannel(image,k);
	}

	/// Sets quality channel
	void setW(Image<T> &image, unsigned int k = 0){
		w.viewChannel(image,k);
	}

	/// Given 4-channel image, associates Dx, Dy, Dt and w to the channels.
	void setDiff(const Image<T> &image){
		Dx.viewChannel(image,0);
		Dy.viewChannel(image,1);
		Dt.viewChannel(image,2);
		w.viewChannel(image,3);
	}


	// protected:
	double areaD;
	
	
	/** 
	 *   TODO: weight = > weightSUm
	 *   TODO: multiple channels (banks)?
	 */
	virtual void initialize() {

		u.viewChannel(this->dst,0);
		v.viewChannel(this->dst,1);
		q.viewChannel(this->dst,2);

		areaD = this->getArea();
		
		//------------------------------
		wXX = 0.0;
    	wXY = 0.0;
    	wYY = 0.0;
    	wXT = 0.0;
    	wYT = 0.0;
		
    	static T DX, DY, DT, W;

		this->location.setLocation(0,0);
		// sum = 0;
		// areaF = this->getArea();
		// cerr << "area" << areaF << "\n";
		cerr << this->dst.getCoordinateHandler();
		
		static unsigned int address;
		
		for (int i = this->iMin; i <= this->iMax; i++) {
			for (int j = this->jMin; j <= this->jMax; j++) {

				this->p.x = i;
				this->p.y = j;
				this->dst.getCoordinateHandler().handle(this->p);

				address = this->dst.address(this->p.x,this->p.y);

				DX = Dx.at(address);
				DY = Dy.at(address);
				DT = Dt.at(address);
				//W  = 1; // w.at(address);
				W = w.at(address);
				
				wXX += W*DX*DX;
            	wXY += W*DX*DX;
            	wYY += W*DY*DY;
            	wXT += W*DX*DT;
            	wYT += W*DY*DT;
				
			}
		}
		
	};
	
	
	
	virtual void updateHorz(int dx){
		
		static unsigned int address;
		static T DX, DY, DT, W;
		static int xOld, xNew;

		xOld = this->getXOld(dx); // location.x + (dx>0 ? iMin-1 : iMax+1);
		xNew = this->getXNew(dx); // location.x + (dx>0 ? iMax     : iMin);

		for (int j = this->jMin; j <= this->jMax; j++) {
			
			this->p.x = xOld;
			this->p.y = this->location.y+j;
			
			this->dst.getCoordinateHandler().handle(this->p);

			address = this->dst.address(this->p.x,this->p.y);

			DX = Dx.at(address);
			DY = Dy.at(address);
			DT = Dt.at(address);
			//W = 1;
			W  = w.at(address);
			
			wXX -= W*DX*DX;
    		wXY -= W*DX*DY;
    		wYY -= W*DY*DY;
    		wXT -= W*DX*DT;
    		wYT -= W*DY*DT;
			// sum -= this->src.at(this->p);
			
			this->p.x = xNew;
			this->p.y = this->location.y+j;
			
			this->dst.getCoordinateHandler().handle(this->p);

			address = this->dst.address(this->p.x,this->p.y);

			DX = Dx.at(address);
			DY = Dy.at(address);
			DT = Dt.at(address);
			W  =  w.at(address);

			wXX += W*DX*DX;
    		wXY += W*DX*DY;
    		wYY += W*DY*DY;
    		wXT += W*DX*DT;
    		wYT += W*DY*DT;
			
		}
	};
	
	// KESKEN
	virtual void updateVert(int dy){
		
		static unsigned int address;
		static T DX, DY, DT, W;
		static int yOld, yNew;

		yOld = this->getYOld(dy); // location.y + (dy>0 ? jMin-1 : jMax+1);
		yNew = this->getYNew(dy); // location.y + (dy>0 ? jMax     : jMin);
		
		for (int i=this->iMin; i<=this->iMax; i++) {

			this->p.x = this->location.x+i;
			this->p.y = yOld;
			this->dst.getCoordinateHandler().handle(this->p);
			
			// tsekkaamatta!
			address = this->dst.address(this->p.x,this->p.y);

			DX = Dx.at(address);
			DY = Dy.at(address);
			DT = Dt.at(address);
			//W  =  1;
			W  =  w.at(address);

			wXX -= W*DX*DX;
			wXY -= W*DX*DY;
			wYY -= W*DY*DY;
			wXT -= W*DX*DT;
			wYT -= W*DY*DT;

			this->p.x = this->location.x+i;
			this->p.y = yNew;
			this->dst.getCoordinateHandler().handle(this->p);

			address = this->dst.address(this->p.x,this->p.y);

			DX = Dx.at(address);
			DY = Dy.at(address);
			DT = Dt.at(address);
			W  =  w.at(address);

			wXX += W*DX*DX;
			wXY += W*DX*DY;
			wYY += W*DY*DY;
			wXT += W*DX*DT;
			wYT += W*DY*DT;
		}
		
	}

	inline T nominator() const
    {
        return static_cast<T>(wXX*wYY - wXY*wXY + .01);  // TODO "epsilon", ensures positivity
    };

    /// Returns the horizontal component of motion. Must be scaled by nominator().
    inline T uDenominator() const
    {
        return static_cast<T>(wXY*wYT - wYY*wXT);
    };

	/// Returns the vertical component of motion. Must be scaled by nominator().
    inline T vDenominator() const
    {
        return static_cast<T>(wXY*wXT - wXX*wYT);
    };


	virtual void write(){
		static T nom;
		static double quality;

		nom = nominator();
		quality = sqrt(nom/(areaD*areaD*2048));  // hihasta kerroin

		if (quality > 1.0){  // todo threshold
			u.at(this->location) = uDenominator()/nom;
			v.at(this->location) = vDenominator()/nom;
			q.at(this->location) = static_cast<T>( 255.0 - 255.0 / (1.0+quality) );
		}
		else {
			u.at(this->location) = 0;
			v.at(this->location) = 0;
			q.at(this->location) = 0;
		}

	}
	
	
  };


//------------------------------------------------------------------------------------------


/// Detects motion between two subsequent images. Does not extrapolate images.
/// Applies recursive correction for smoothing motion field.
/**  Notice:
 *   T  input data (images)
 *   T2 all the other arrays (differentials, motion)
 *
 *   - gradPow - highlighting the role of gradient magnitude
 *   - gradWidth -
 */
template <class T=unsigned char,class T2=double>
class FastOpticalFlowOp : public SlidingWindowOp<T2,T2>
{
public:

	FastOpticalFlowOp(const string & p = "5,5,0.5,2,16") :
		SlidingWindowOp<T2,T2>(opticalFlowWindow,"OpticalFlow","A pipeline implementation of optical flow. ","width,height,smoothing,gradPow,gradWidth",p){

	};

	/// Creates a 2+1 channel target image for storing motion (u,v) and quality (q).
	//  Notice T2,T2.
	virtual void makeCompatible(const Image<T2> &src,Image<T2> &dst) const  {
		unsigned int width = src.getWidth();
		unsigned int height = src.getHeight();
		dst.setGeometry(width,height,2,1);
    	dst.getCoordinateHandler().setBounds(width,height);  // ??
	};

	/// Computes an image with channels dx, dy, dt and w (quality of gradients). User may wish to redefine this.
	///
	virtual void computeDerivativeImage(const Image<T> &src1,const Image<T> &src2,Image<T2> &dst) const {

		// TODO: concentrate on the "middle image". Skip grad stability, let oflow use

		// Window properties
		const unsigned int width  = this->getParameter("width",3);
		const unsigned int height = this->getParameter("height",width);
		const double smoothingCoeff  = this->getParameter("smoothing",0.5);

		const unsigned int imageWidth = src1.getWidth();
		const unsigned int imageHeight = src1.getHeight();
		dst.setGeometry(imageWidth,imageHeight,3,1);

		const ImageView<T2> grad(dst,0,2);  // dx & dy
		Image<T2> & dx = dst.getChannel(0);
		Image<T2> & dy = dst.getChannel(1);
		Image<T2> & dt = dst.getChannel(2);
		Image<T2> &  w = dst.getChannel(3);

		Image<T2> src1Smooth; //(width,height,2);
		Image<T2> src2Smooth; //(width,height,2);

		// Notice that weighted smoothing applies, if applicable
		QuadraticSmootherOp<T,T2> smoother("5,5,0.5"); // bug
		// FastAverageOp<T,T2> smoother("5,5"); // bug
		//DoubleSmootherOp<T,T2> smoother("5,5,0.5");
		smoother.setSize(width/2,height/2);  // half of oflow windows
		smoother.setParameter("coeff",smoothingCoeff);
		smoother.filter(src1,src1Smooth);
		smoother.filter(src2,src2Smooth);

		if (drain::Debug > 3){
			Image<T> tmp;
			CopyOp<T2,T>().filter(src1Smooth,tmp); File::write(tmp,"grad-s1.png");
			CopyOp<T2,T>().filter(src2Smooth,tmp); File::write(tmp,"grad-s2.png");
		}

		// Time derivative, dt
		SubtractionOp<T2,T2>().filter(src2Smooth.getChannel(0),src1Smooth.getChannel(0),dt);


		/// Gradient quality = stability * magnitude
		// (part 1: gradient unchangedness)
		// TEST ERAD

		GradientHorizontalOp<T2,T2>().filter(src2Smooth.getChannel(0),dx);
		GradientVerticalOp<T2,T2>().filter(src2Smooth.getChannel(0),dy);
		Image<T2> gradTemp(imageWidth,imageHeight,2);
		GradientHorizontalOp<T2,T2>().filter(src1Smooth.getChannel(0),gradTemp.getChannel(0));
		GradientVerticalOp<T2,T2>().filter(src1Smooth.getChannel(0),gradTemp.getChannel(1));
		//DistanceOp<T2,T2>("5.0,2").filter(grad,gradTemp,w);
		DistanceOp<T2,T2> gradQ;
		gradQ.setParameter("mapping","l"); // linear
		const double gradPow = this->getParameter("gradPow",2.0); // TODO always 2 => skip
		gradQ.setParameter("l",gradPow);
		gradQ.setParameter("lInv",1.0/gradPow);
		gradQ.filter(grad,gradTemp,w);

		FuzzyPeakOp<T2,T2> peak;
		peak.setParameter("location",0);
		const double gradWidth = this->getParameter("gradWidth",16.0);
		peak.setParameter("width",gradWidth);
		peak.setParameter("scaleDst",255.0); // 255
		peak.filter(w,w);


		if (drain::Debug > 3){
			Image<T> tmp;
			ImageView<T2> view;

			view.viewImageFlat(grad); //
			ScaleOp<T2,T>("4.0,128").filter(view,tmp);
			File::write(tmp,"grad-raw1.png");
			view.viewImageFlat(gradTemp);
			ScaleOp<T2,T>("4.0,128").filter(view,tmp);
			File::write(tmp,"grad-raw2.png");
			//ScaleOp<T2,T>("1.0,0").filter(w,tmp);
			CopyOp<T2,T>().filter(w,tmp);
			File::write(tmp,"grad-stab.png");
		}

		// Intensity gradients should not be computed for the 1st or 2nd image, but "between" them.
		// So images will be mixed here. To save memory, src2Smooth is recycled.
		MixerOp<T2,T2>(0.5).filter(src2Smooth,src1Smooth,src2Smooth);
		GradientHorizontalOp<T2,T2>().filter(src2Smooth.getChannel(0),dx);
		GradientVerticalOp<T2,T2>().filter(src2Smooth.getChannel(0),dy);

		// Update Gradient stability as well (emphasize strong gradients)
		Image<T2> gradMagnitude;
		MagnitudeOp<T2,T2>("l,2").filter(grad,gradMagnitude);
		MultiplicationOp<T2,T2>("255").filter(w,gradMagnitude,w);

		//MagnitudeOp<T2,T2>("l,2").filter(grad,w);
		//FuzzyPeakOp<T2,T2>(0,width,8.0).filter(w,w);

		if (drain::Debug > 3){
			Image<T> tmp;
			CopyOp<T2,T>().filter(src2Smooth,tmp); File::write(tmp,"grad-smix.png");
			ScaleOp<T2,T>(0.5,128).filter(dt,tmp); File::write(tmp,"grad-dt.png");
			ScaleOp<T2,T>(1.5,128).filter(dx,tmp); File::write(tmp,"grad-dx.png");
			ScaleOp<T2,T>(1.5,128).filter(dy,tmp); File::write(tmp,"grad-dy.png");
			ScaleOp<T2,T>(1.0,0).filter(gradMagnitude,tmp); File::write(tmp,"grad-magnitude.png");
			ScaleOp<T2,T>(1.0,0).filter(w,tmp); File::write(tmp,"grad-quality.png");
		}


	}

	///  Cre
	/**
	 *   Notice T2, T2
	 *   @param src - difference image (dx,dy,dt,q)
	 *   @param dst - motion field (u,v,q);
	 */
	virtual void filter(const Image<T2> &src,Image<T2> &dst) const {

		const unsigned int width  = this->getParameter("width",3);
		const unsigned int height = this->getParameter("height",width);
		this->opticalFlowWindow.setSize(width,height);
		this->opticalFlowWindow.setDiff(src);

		SlidingWindowOp<T2,T2>::filter(src,dst);

		if (drain::Debug > 3){
			Image<T> tmp;
			ScaleOp<T2,T>(2,128.0).filter(dst,tmp);
			File::write(tmp,"oflow-M.png");
			ImageView<T> tmpView;
			tmpView.viewImageFlat(tmp);
			File::write(tmpView,"oflow-MF.png");
			for (unsigned int i = 0; i < dst.getChannelCount(); ++i) {
				stringstream filename;
				filename << "oflow-M" << i << ".png";
				ScaleOp<T2,T>(0.5,128.0).filter(dst.getChannel(i),tmp);
				File::write(tmp,filename.str());
			};
		}

	};

protected:

	mutable SlidingOpticalFlow<T2> opticalFlowWindow;


};

}
}

#endif
