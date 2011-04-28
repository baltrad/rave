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
#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include <iostream>
#include <vector>

namespace drain
{

namespace image
{

using namespace std;

/*! The basic idea is to encode dimensions directly as a vector<int>; the number of elements is 
 *  the number of dimensions. Each element states the discrete coordinate space.
 * 
 *  Three standard "copies" are defined for convenience; KLUDGE!
 *  # geometry.width  = geometry[0]
 *  # geometry.height = geometry[1]
 *  # geometry.channelCount = geometry[2]  
 *
 *  To guarantee this, the minimum size of the vector will always be at least three (3).
 *  
 * 
 *  For example, a three-channel image having width 640 and height 400 would have dimension vector 
 *  [640,400,3]. The coordinates will be: x \in {0,1,...,639}, y \in {0,1,...,399} and z \in {0,1,2}.
 *  The size of the vector of unlimited, allowing hypermatrices of any size. 
 * 
 */
class Geometry : public std::vector<unsigned int> 
// SP: itse en käyttäisi perintää vaan kompositiota, ts. vector-olio on luokan jäsenenä. 
// MP: OK, muunnan kun ehdin. Muistaakseni ei ollut erityistä syytä tälle suoralle.
{
public:

    Geometry();
    
	/*! Width, height, channels, imageChannels (alphaChannels)
	 */
    Geometry(unsigned int width, unsigned int height, unsigned int channelCount=1, unsigned int alphaChannelCount=0);
	
	//Geometry(const std::vector<int> &vector, int a = 0); 

    virtual ~Geometry();

	void setGeometry(const Geometry &g);
    
    void setGeometry(unsigned int width,unsigned int height,
		unsigned int imageChannelCount = 1,unsigned int alphaChannelCount = 0);
    
    void setWidth(unsigned int weight);
    void setHeight(unsigned int height);
    void setChannelCount(unsigned int imageChannelCount,unsigned  int alphaChannelCount = 0);
    void setAlphaChannelCount(unsigned  int alphaChannelCount);
    


    // SP: Miksi palautetaan referenssit? Eikö alkeistyypeillä riitä arvon palautus?
    // Hyvä kommentti tämäkin. En muista. Liittyneekö siihen, että jos kuva elää
    // kesken suoritusta, muuttujakin pysyy ajan tasalla.
	inline const unsigned int & getWidth() const { return width; };
	inline const unsigned int & getHeight() const { return height; };
	inline const unsigned int & getChannelCount() const { return channelCount; };
	inline const unsigned int & getImageChannelCount() const { return imageChannelCount; };
	inline const unsigned int & getAlphaChannelCount() const { return alphaChannelCount; };
	
	inline const unsigned int & getArea() const { return area; };
	inline const unsigned int & getVolume() const { return volume; };

    // to-be-protected? :


    template <class T>
    Geometry & operator=(const std::vector<T> &v);
    
    //inline
    Geometry & operator=(const Geometry &g);
    
    //Geometry & operator=(const Geometry &g);
    
    
    string & toString(string & s) const;
    
	
    protected:
    	// alphaChannelCount?
    	// area and volume
    	void update();
    	
    unsigned int width;
    unsigned int height;
    unsigned int channelCount;
	
	unsigned int imageChannelCount;
	unsigned int alphaChannelCount;

    unsigned int area;
    unsigned int volume;
    
    	
};



/// Not ready.
template <class T>
Geometry & Geometry::operator=(const std::vector<T> &v){
	int d = v.size();
	for (int i = 0; i < d; ++i) {
		if (i<d)
			(*this)[i] = v[i];
		else
			(*this)[i] = 0;
	}
	update();
	return (*this);
}
    
ostream & operator<<(ostream &ostr,const Geometry &geometry);
    

}

}

#endif /*GEOMETRY_H_*/
