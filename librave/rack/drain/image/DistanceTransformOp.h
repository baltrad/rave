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
#ifndef DISTANCETRANSFORMOP_H_
#define DISTANCETRANSFORMOP_H_


#include <math.h>


#include "ImageOp.h"

namespace drain
{

namespace image
{

template <class T=unsigned char> 
class DistanceModel {
	
public:
	/** Sets the horizontal and vertical radius for the distance function.
	 *  It is recommended to make \c vert and \c diag optional. ie. apply default values.
	 *  By default, the geometry is octagonal, applyin 8-distance, but setting diag 
	 *  a large value makes it square (4-distance, diamond). 
	*/
	virtual ~DistanceModel(){};

	/// The values are in float, as eg. half-width radius may be sharp, under 1.0.
	virtual 
	void setGeometry(float horz,float vert,float diag) = 0;
	
	virtual 
	T decreaseHorz(const T &x) const = 0; 

	virtual 
	T decreaseVert(const T &x) const = 0;

	virtual 
	T decreaseDiag(const T &x) const = 0;


};

template <class T=unsigned char> 
class DistanceModelLinear : public DistanceModel<T> {

	

	virtual 
	void setGeometry(float horz,float vert = -1,float diag = -1){
		
		if (horz <= 1)
		  horzDecrement = Intensity::max<T>();
		else
		  horzDecrement = static_cast<T>(Intensity::max<T>()/horz);
		
		if (vert < 0)
			vertDecrement = horzDecrement;
		else if (vert == 0)
			vertDecrement = Intensity::max<T>();
		else // (vert > 0)	
		  	vertDecrement = static_cast<T>(Intensity::max<T>()/vert);
		
 		if (diag < 0)
 			diagDecrement =
 			static_cast<T>(sqrt(horzDecrement*horzDecrement + vertDecrement*vertDecrement)) ;
 		else if (diag == 0)  
 			diagDecrement = Intensity::max<T>();
 		else
 			diagDecrement = static_cast<T>(Intensity::max<T>()/diag);
				
		//cerr << "Linear: " << (float)horzDecrement << ", " << (float)vertDecrement << ", " << (float)diagDecrement << "\n";
		 	
	} 
	
	// TODO setDecrements alternative...
	
	virtual 
	inline T decreaseHorz(const T &x) const { return Intensity::limit<T>(x - horzDecrement); }; 

	virtual 
	inline T decreaseVert(const T &x) const { return Intensity::limit<T>(x - vertDecrement); };

	virtual 
	inline T decreaseDiag(const T &x) const { return Intensity::limit<T>(x - diagDecrement); };
	
protected:
	
	T horzDecrement;
	T vertDecrement;
	T diagDecrement;
	
};


template <class T=unsigned char> 
class DistanceModelExponential : public DistanceModel<T> {
	
	virtual 
	void setGeometry(float horz,float vert = -1,float diag = -1){
		
		// TODO: interpret handle 0 and -1 better
		if (horz <= 0.0)
			horzDecay = 0;
		  //horz = 1;
		else
		  horzDecay = pow(0.5,1.0/horz);
 		
 		if (vert < 0)
 		  vertDecay = horzDecay;
		else if (vert == 0)  // label => spread to infinity ??
		  vertDecay = 1.0;
		else   
		  vertDecay = pow(0.5,1.0/vert);
		

 		// TODO Fix a math bug here
 		if (diag == -1){
 			//if ((horzDecay > 0) && (horzDecay > 0))
 			const double hLog = log(horzDecay);
 			const double vLog = log(vertDecay);
 			diagDecay = exp(-sqrt(hLog*hLog + vLog*vLog));
 		}
 		else if (diag == 0)
 			diagDecay = 1.0;
 		else
 			diagDecay = pow(0.5,1.0/diag);
			
		cerr << "Exponential: " << horzDecay << ", " << vertDecay << ", " << diagDecay << "\n";
	} 
	
	// TODO setDecays, as alternative...
	
	virtual 
	inline T decreaseHorz(const T &x) const { return static_cast<T>(horzDecay * x); }; 

	virtual 
	inline T decreaseVert(const T &x) const { return static_cast<T>(vertDecay * x); };

	virtual 
	inline T decreaseDiag(const T &x) const { return static_cast<T>(diagDecay * x); };
 	
protected:
	
	double horzDecay;
    double vertDecay;
    double diagDecay;
};



 		//const double d = this->parameters.get("diag",sqrt(1.0/(h*h) + 1.0/(v*v)));
 		

/** Fast distance transform using 8-directional distance function.
 *  Class D is used for computing distances?
 */
template <class T=unsigned char,class T2=unsigned char> //,class D=int>
class DistanceTransformOp : public ImageOp<T,T2>
{

public:
    
    DistanceTransformOp(DistanceModel<T2> & model) : distanceModel(model) {};
    
    /*
    DistanceTransformOp(const string & p = "5,5"){
    	this->setInfo("Creates areas of decreasing intensities.","horz,vert,diag",p);
		//this->parameters.setDefaultKeys("horz,vert,diag");
		//this->parameters.set(p);
	};
	*/
	    
    virtual ~DistanceTransformOp(){};


	virtual void initialize() const {
		float h = this->parameters.get("horz",3.0);
		float v = this->parameters.get("vert",-1.0);
		float d = this->parameters.get("diag",-1.0); //sqrt(h*h + v*v));
		distanceModel.setGeometry(h,v,d);
 	}

	

	void filter(const Image<T> &src, Image<T2> &dst) const ;

	void filterDownRight(const Image<T> &src, Image<T2> &dst) const ;
	void filterUpLeft(const Image<T> &src, Image<T2> &dst) const ;

protected:
	mutable DistanceModel<T2> & distanceModel;
	mutable DistanceModelLinear<T2> distanceLinear;
 	mutable DistanceModelExponential<T2> distanceExponential;
};
  
  
  
template <class T,class T2>
void DistanceTransformOp<T,T2>::filter(const Image<T> &src, Image<T2> &dst) const
{
		if (filterWithTmp(src,dst))
			return;

        //cerr << "DTrf" << endl; // << (float)rectDecrement << endl;
        initialize();
		makeCompatible(src,dst);
		        
        //this->coordinateHandler->setBounds(src.getWidth(), src.getHeight());

        filterDownRight(src,dst);
        filterUpLeft(dst,dst);
        //return dst;
}
  
template <class T,class T2>
void DistanceTransformOp<T,T2>::filterDownRight(const Image<T> &src, Image<T2> &dst) const
{


    const int width  = src.getWidth();
    const int height = src.getHeight();
	const CoordinateHandler & coordinateHandler = src.getCoordinateHandler();
			
        // proximity (inverted distance)
		float d;
		float dPrev;
			
		//int k = 0;

        cerr << "UR" << (int)src.at(0,0) << endl;
        cerr << "UR dst" << (int)dst.at(0,0) << endl;


		// Point in the source image
        Point2D<> p; 

		// Point in the target image
        Point2D<int> t;
		int &tx = t.x;
        int &ty = t.y;
        
        for (ty=0; ty<height; ty++)
        {
            for (tx=0; tx<width; tx++)
            {

                // Take source value as default
                d = src.at(t);

                // Compare to previous value
                dPrev = dst.at(t);
                if (dPrev > d)
                {
                    d = dPrev;
                }

                // Compare to upper left neighbour
                p.setLocation(tx-1,ty-1);
                coordinateHandler.handle(p);
                //dPrev = dst.at(p) - diagdistanceModel.decrement;
                dPrev = distanceModel.decreaseDiag(dst.at(p));
                if (dPrev > d)
                {
                    d = dPrev;
                }

                // Compare to upper neighbour
                p.setLocation(tx,ty-1);
                coordinateHandler.handle(p);
                //dPrev = dst.at(p) - vertdistanceModel.decrement;
                dPrev = distanceModel.decreaseVert(dst.at(p));
                if (dPrev > d)
                {
                    d = dPrev;
                }

                // Compare to upper right neighbour
                p.setLocation(tx+1,ty-1);
                coordinateHandler.handle(p);
                //dPrev = dst.at(p) - diagdistanceModel.decrement;
                dPrev = distanceModel.decreaseDiag(dst.at(p));
                if (dPrev > d)
                {
                    d = dPrev;
                }

                // Compare to left neighbour
                p.setLocation(tx-1,ty);
                coordinateHandler.handle(p);
                //dPrev = dst.at(p) - horzdistanceModel.decrement;
                dPrev = distanceModel.decreaseHorz(dst.at(p));
                if (dPrev > d)
                {
                    d = dPrev;
                }

                if (d>0)
                	dst.at(t) = static_cast<T2>(d);
                    

            };
        };
        //return dst;

};
    

    //public WritableRaster filterUpLeft(Raster srcDist, WritableRaster dst){
    //template <class T=unsigned char,class T2=unsigned char>
template <class T,class T2>
void DistanceTransformOp<T,T2>::filterUpLeft(const Image<T> &src, Image<T2> &dst) const {

        const int width  = src.getWidth();
        const int height = src.getHeight();
		const CoordinateHandler & coordinateHandler = src.getCoordinateHandler();
			
        // proximity (inverted distance)
      	float d;
		float dPrev;
		
        Point2D<> p(0,0);
        //coordinateHandler.setBounds(width,height);
        
        // TODO:  target:
        Point2D<> t;
        int &tx = t.x;
        int &ty = t.y;
        
        
        for (ty=height-1; ty>=0; ty--)
        {
            for (tx=width-1; tx>=0; tx--)
            {
                // Source
                d = src.at(t);

                // Compare to previous value
                dPrev = dst.at(t);
                if (dPrev > d)
                {
                    d = dPrev;
                }

                // Compare to lower left neighbour
                p.setLocation(tx-1,ty+1);
                coordinateHandler.handle(p);
                //dPrev = dst.at(p) - diagdistanceModel.decrement;
                dPrev = distanceModel.decreaseDiag(dst.at(p));
                if (dPrev > d)
                {
                    d = dPrev;
                }

                // Compare to lower neighbour
                p.setLocation(tx,ty+1);
                coordinateHandler.handle(p);
                //dPrev = dst.at(p) - vertdistanceModel.decrement;
                dPrev = distanceModel.decreaseVert(dst.at(p));
                if (dPrev > d)
                {
                    d = dPrev;
                }

                // Compare to lower right neighbour
                p.setLocation(tx+1,ty+1);
                coordinateHandler.handle(p);
                //dPrev = dst.at(p) - diagdistanceModel.decrement;
                dPrev = distanceModel.decreaseDiag(dst.at(p));
                if (dPrev > d)
                {
                    d = dPrev;
                }

                // Compare to right neighbour
                p.setLocation(tx+1,ty);
                coordinateHandler.handle(p);
                //dPrev = dst.at(p) - horzDecrement;
                dPrev = distanceModel.decreaseHorz(dst.at(p));
                if (dPrev > d)
                {
                    d = dPrev;
                }

                if (d>0)
                    dst.at(t) = static_cast<T2>(d);


            }
        }
        //return dst;
};

template <class T=unsigned char,class T2=unsigned char> //,class D=int>
class DistanceTransformLinearOp : public DistanceTransformOp<T,T2>
{
	public:
	DistanceTransformLinearOp(const string & p = "5,5,7") : DistanceTransformOp<T,T2>(this->distanceLinear) {
    	this->setInfo("Creates areas of linearly decreasing intensities.","horz,vert,diag",p);
	};

};

template <class T=unsigned char,class T2=unsigned char> //,class D=int>
class DistanceTransformExponentialOp : public DistanceTransformOp<T,T2>
{
public:
   DistanceTransformExponentialOp(const string & p = "5,5,7") :
   	DistanceTransformOp<T,T2>(this->distanceExponential) {
    	this->setInfo("Creates areas of exponentially decreasing intensities. Set half-widths.",
    	"horz,vert,diag",p);
   };
};	    

}
}
	
#endif /*DISTANCETRANSFORMOP_H_*/
