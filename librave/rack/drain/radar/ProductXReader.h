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
#ifndef PRODUCTXREADER_H_
#define PRODUCTXREADER_H_

#include <stdexcept>

#include "../util/Options.h"
#include "../image/Image.h"
#include "../image/Intensity.h"

namespace drain
{

namespace radar
{

class ProductXReader
{
public:
	ProductXReader(const string &parameters){
		this->parameters.set(parameters);
	};
	
	inline
	void setParameters(const string & p){
    	parameters.set(p);
    }
    
    inline
    bool hasParameter(const string & key){
    	 return parameters.hasKey(key);
    }
	
    template <class T>
    inline
	void setParameter(const string & key,const T & value){
   		parameters[key] = value;
    }
	
	/*
	template <class T>
	Data getParameter(const string & key,T defaultValue){
   		return parameters.get(key,defaultValue);
    }
    */
    template <class T>
    void read(image::Image<T> &dst,const string &parameters);
    
	
	// protect
	Options parameters;
	
	
	
};


template <class T>
void ProductXReader::read(image::Image<T> &dstImage,const string &queryParameters){
		
	/// Open 
	// ??? const string & path = this->parameters.get("productX","productX");
	const string & productX = this->parameters.get("productX","productX");
	
	/// Stage 1: chatting with "productx" application
	/// exec(
	istream istr;
	ostream ostr;
	string response;
	/// execOpen(productX,ostr,istr);
	// Chat
	ostr << "choice1\n";
	istr >> response;
	if (response == ""){
		ostr << "choice2\n";	
	} // else 
	
	istr >> response;
	if (response == ""){
		ostr << "choice2\n";	
	} // else 
	
	const float coeff = this->parameters.get("coeff",1.0);
	const float bias  = this->parameters.get("bias",0);
	const bool  logarithmic = this->parameters.get("log",false);
	
	streambuf b;
	/// Main loop 1: header
	string line;
	while (istr.peek() == '#'){
		//istr.getline(line);
		// consider streambuf
		istr.get(b); // read line;
		istr >> line;
		if (istr.eof()){
			throw runtime_error("ProductXReader: Premature end of header");
		};
	};
	
	
	const unsigned int width  = 0;
	const unsigned int height = 0;
	const unsigned int channels = 0;
	
	dstImage.setGeometry(width,height,channels);
	
	/// Main loop 2: data
	float f;
	unsigned int i = 0;
	while (!istr.eof()){
		
		// beam header
		// -----
		// beam
		while (!istr.eof()){
		  // element
		  // istr.get(jotain)
		  // if ( == "-.--"){
		  
		  // Ehkä hyvä näinkin:
		  istr >> f;
		  if (istr.fail()){
		  	 
		  }
		  dstImage.at(i) = image::Intensity::limit<T>(coeff*f + bias);
		}
	};
	
};
   

}

}

#endif /*PRODUCTXREADER_H_*/
