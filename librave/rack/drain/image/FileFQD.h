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
#ifndef FILEFQD_H_
#define FILE_H_


#include <string>

#include "Image.h"


namespace drain
{
namespace image
{

using namespace std;

/** Format applied at FMI
 * 
 */
class FileFQD
{
public:
	//((FileFQD(const string &FileFQDname);

	virtual ~FileFQD();

	/*
	template <class T>
	static void read(BufferedImage<T> &image,const string &path);

	static void read(BufferedImage<unsigned char> &image,const string &path);
	*/

	template <class T>
	static void write(const Image<T> &image,const string &path);

	/*
	static void write(const BufferedImage<unsigned char> &image,const string &path);
	*/
	
protected:
	//string filename;
	
	
};


template <class T>
void FileFQD::write(const Image<T> &image,const string &path){
	ofstream ofstr;
	//ofstr.open(path,"w");
	
	// 
	string timestamp = image.parameters.get("TIMESTAMP","197001010000");
	timestamp.append(14-timestamp.size(), '0');  // add seconds if needed
	
	string levels = image.parameters.get("LEVELS","197001010000");
	string site = image.parameters.get("SITE","197001010000");
	/*
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

#endif /*FILEFQD_H_*/
