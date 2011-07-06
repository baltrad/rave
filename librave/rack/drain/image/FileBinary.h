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
#ifndef FILEBINARY_H_
#define FILEBINARY_H_


#include <string>
#include <iostream>
#include <fstream>



#include "Image.h"


namespace drain
{
namespace image
{

using namespace std;

/** Format applied at FMI
 * 
 */
class FileBinary
{
public:
	//((FileBinary(const string &FileBinaryname);

	virtual ~FileBinary();

	/*
	template <class T>
	static void read(BufferedImage<T> &image,const string &path);

	static void read(BufferedImage<unsigned char> &image,const string &path);
	*/

	template <class T>
	static void write(const Image<T> &image,const string &path, const string &header = "");

	/*
	static void write(const BufferedImage<unsigned char> &image,const string &path);
	*/
	
protected:
	//string filename;
	
	
};


// Consider template <class T2> for on-the-fly cast T=>T2
// Consider all permutations WHD dwH  
template <class T>
void FileBinary::write(const Image<T> &image,const string &path, const string &header){
	ofstream ofstr;
	ofstr.open(path.c_str(),ios::out);
	
	ofstr << header;
	
	const vector<T> & v = image.getVector();
	const typename vector<T>::size_type s = v.size();
	// for (int k=0; k<image.getChannelCount(); k++)
	// for (int j=0; j<image.getChannelCount(); j++)
	
	for (typename vector<T>::size_type i=0; i<s;i++){
		ofstr << v[i]; 
	}
	
	
	/*
	string timestamp = image.parameters.get("TIMESTAMP","197001010000");
	timestamp.append(14-timestamp.size(), '0');  // add seconds if needed
	
	string levels = image.parameters.get("LEVELS","197001010000");
	string site = image.parameters.get("SITE","197001010000");

105 300m 300
dBz 123
vantaantutka 1201
567 623
latlon:19.5,57.4,23.7,62.5
4567456
	*/

}

}

}

#endif /*FILEBINARY_H_*/
