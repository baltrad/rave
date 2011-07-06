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
#include "String.h"


#include <stdexcept>
#include <iostream>

namespace drain {

string String::replace(const string &src,const string &from,const string &to) { //,string &dst){

	string result;

	string::size_type i = 0;
	string::size_type pos;

	while (true) {
		pos = src.find(from,i);
		if (pos == string::npos){
			result.append(src,i,src.size()-i);
			//std::cerr << result << '\n';
			return result;
			//return;
		}
		result.append(src,i,pos-i);
		//std::cerr << result << " 2\n";
		result.append(to);
		//std::cerr << result << " 3\n";
		i = pos + from.size();
	}

}

string String::replace_regexp(const string &src,RegExp &reg,const string &to) { //,string &dst){

	string result = src;

	while (reg.execute(result) != REG_NOMATCH){
		//cerr << "replace_regexp: regexp "<< reg.regExpString << " found in " << result << '\n';
		if (reg.result.size()==4){

			// Skip infinite loop. ('to' would be matched infinitely)
			if (reg.result[2] == to)
				return result;
			result = reg.result[1] + to + reg.result[3];
		}
		else {
			throw runtime_error(reg.regExpString + " [ERROR] regexp error");
		}
	}

	return result;
}

}
