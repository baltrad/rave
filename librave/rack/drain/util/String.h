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
/*
 * String.h
 *
 *  Created on: Jul 21, 2010
 *      Author: mpeura
 */

#ifndef STRING_H_
#define STRING_H_

#include <string>
#include <iterator>

#include "RegExp.h"


using namespace std;

namespace drain {

class String {

public:

	/// Replaces instances of 'from' to 'to' in src, storing the result in dst.
	/** Safe. Uses temporary string.
	 *
	 */
	static string replace(const string &src,const string &from,const string &to);

	/// Replaces instances of 'from' to 'to' in src, storing the result in dst.
	/** Safe. Uses temporary string.
	 *
	 */
	static string replace_regexp(const string &src,RegExp &reg,const string &to);

	inline
	static string replace_regexp(const string &str,const string &reg,const string &to){
		RegExp r(string("^(.*)(") + reg + string(")(.*)$"));
		return replace_regexp(str,r,to);
	};




	/// Replaces instances of 'from' to 'to' in src, storing the result in dst.
	/*
	inline
	static void replace(const string &src,const string from,const string to,string &dst){
		dst = replace(src,from,to);
	}
	*/
};

}

#endif /* STRING_H_ */
