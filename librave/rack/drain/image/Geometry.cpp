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

#include <algorithm>
#include <sstream>

#include "Geometry.h"
//#include "DistanceTransformOp.h"

namespace drain
{

namespace image
{
	
using namespace std;
	
// ref ei onnistunut (width, height, chC)
Geometry::Geometry() : 
//	std::vector<int>(3), width(at(0)), height(at(1)), channelCount(at(2)), alphaChannelCount(0) 
	std::vector<unsigned int>(3)
{
	setGeometry(0,0,0,0);
}

Geometry::Geometry(unsigned int width, unsigned int height, unsigned int imageChannelCount, unsigned int alphaChannelCount) :
	vector<unsigned int>(max(3u,imageChannelCount+alphaChannelCount))
	//vector<int>(max(3,channelCount)), width(at(0)), height(at(1)), channelCount(at(2)), alphaChannelCount(0)
{
	setGeometry(width,height,imageChannelCount,alphaChannelCount);
	//DistanceTransformLinearOp<> foo;
}

Geometry::~Geometry()
{
}

void Geometry::setGeometry(const Geometry &g){
	setGeometry(g.getWidth(),g.getHeight(),g.getImageChannelCount(),g.getAlphaChannelCount());
} 

void Geometry::setGeometry(unsigned int width, unsigned int height, unsigned int imageChannelCount, unsigned int alphaChannelCount){
  resize(std::max(static_cast<size_type>(3),size()));
	
	at(0) = width;
	at(1) = height;
	at(2) = imageChannelCount + alphaChannelCount;
	this->alphaChannelCount = alphaChannelCount;
	update();
	//area = width * height;
	//volume = area*(imageChannelCount + alphaChannelCount);
} 
	
void Geometry::setWidth(unsigned int w){
	//resize(std::max(3u,size()));
	
	at(0) = w;
	update();
};

void Geometry::setHeight(unsigned int h){
	//resize(std::max(3u,size()));
	
	at(1) = h;
	update(); 
};

void Geometry::setChannelCount(unsigned int imageChannels, unsigned int alphaChannels){
	
	//resize(std::max(3u,size()));
	
	at(2) = imageChannels + alphaChannels;
	alphaChannelCount = alphaChannels;
	update();
};


void Geometry::setAlphaChannelCount(unsigned int alphaChannels){
	setChannelCount(getImageChannelCount(),alphaChannels);
};


void Geometry::update(){
	
	//resize(std::max(3u,size()));
	
	width = at(0);
	height = at(1);
	channelCount = at(2); 
	
	imageChannelCount = channelCount - alphaChannelCount; 
	area = width*height;
	volume = 1;
	for (unsigned int i = 0; i < this->size(); ++i) {
		volume *= (*this)[i];
	}
}



Geometry & Geometry::operator=(const Geometry &g)
{
	(*this) = (const std::vector<int> &)g;
	return (*this); 
};
    

ostream & operator<<(ostream &ostr,const Geometry &geometry) {

	string separator("");
	
	ostr << "[";
    for (unsigned int i=0; i<geometry.size(); i++)
    {
    	ostr << separator;
        ostr << geometry[i];
        separator = "×";
    }
    ostr << "] ";
    ostr << "iC=" << geometry.getImageChannelCount() << ", aC=" <<  geometry.getAlphaChannelCount()  << ", ";
    ostr << "w=" << geometry.getWidth() << ", h=" <<  geometry.getHeight()  << ", ";
    ostr << "A=" << geometry.getArea() << ", V=" <<  geometry.getVolume()  << " ";
		
	return ostr;
}
 
string &Geometry::toString(string & s) const
{
    stringstream sstr;
	sstr << *this;
	s = sstr.str();
    return s;
};


}

}
