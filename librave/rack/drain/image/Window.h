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
#ifndef WINDOW_H_
#define WINDOW_H_

namespace drain
{

namespace image
{

template <class T,class T2>
class Window
{
public:
	
	/// Constructor with geometry setting option.
	Window(unsigned int width = 0,unsigned int height = 0){
  		setSize(width,height);
		location.setLocation(0,0); 
	};
	
	virtual ~Window(){}; 
	
	/// Sets the size of the window; setSize(width) equals setSize(width,width);
	virtual void setSize(unsigned int width,unsigned int height = 0){
	   if (height == 0)
	    	height = width;
	   this->width = width;
	   this->height = height; 
	   iMin = -(static_cast<int>(width)-1)/2;   // note unsigned risk!
       iMax = width/2;
       jMin = -(static_cast<int>(height)-1)/2;
       jMax = height/2;
         
	};

	inline
    unsigned int getArea(){ return width * height;};

	void setSrc(const Image<T> &image){src.viewImage(image);};
    void setDst(Image<T2> &image){dst.viewImage(image);};
    
    // Optional
    // TODO VIRHE TÄSSÄ
    void setSrcWeight(const Image<T> &image){srcWeight.viewImage(image);};
    void setDstWeight(Image<T2> &image){dstWeight.viewImage(image);};
        	
    virtual void initialize(){};
        	
	Point2D<> location;
	
	// experimental
	/*
	Options parameters;
	
	void setInfo(const string & description, const string & parameterNames, const string & defaultValues = ""){
    	parameters.info.description = description;
    	parameters.info = parameterNames;
    	parameters.set(defaultValues);
    };
    */
    
    // TODO: protect
    int width;  // Important! Not unsigned
	int height;
    int iMin;
    int iMax;
    int jMin;
    int jMax;
    
 // protected:

	ImageView<T> src;
	ImageView<T2> dst;
	ImageView<T> srcWeight;
	ImageView<T2> dstWeight;
	
};
    
    
template <class T,class T2>
ostream & operator<<(ostream &ostr,const Window<T,T2> &w){
	ostr << "Window: " << w.width << 'x' <<  w.height << ' ';
	ostr << '[' << w.iMin << ',' << w.jMin << ',' << w.iMax << ',' << w.jMax << ']';
	ostr << w.src.getName() << '\n';
	ostr << w.dst.getName() << '\n';
	ostr << w.srcWeight.getName() << '\n';
	ostr << w.dstWeight.getName() << '\n';
	return ostr;
};
 
}

}

#endif /*WINDOW_H_*/
