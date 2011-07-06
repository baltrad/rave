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
#include <stdexcept>

#include "Options.h"

namespace drain
{




const Data Options::empty;
//const bool Options::isLimited = false;

//bool Options::hasKey(const string &key){
//return map<string,Data>::hasKey( getAlias(key) );
//}

void Options::set(const string & s){
	string s2(s);
	
	if (!isLimited()){
		(*this)[""] = s;	
	}
	else {
		vector<string> p;
		Data(s).trim().splitTo(p);
			
		vector<string> keys;
		//getDefaultKeys().splitTo(keys);
		getParameterNames().splitTo(keys);

		if (p.size() > keys.size()){
			throw runtime_error(s + " : too many parameters, keys=" + getParameterNames());
		}
		
	    for (vector<string>::size_type i=0; i<p.size(); i++){
	    	//cerr << " Options::set " << i << ": " << flush;
	    	//cerr << keys[i] << flush;
	    	//cerr << " << " << p[i] << flush;
    		map<string,Data>::operator[](keys[i]) = p[i];	
    		//cerr << " (OK)" << endl;
    	}
	}

}

Data & Options::operator[](const string &key){
	if (isLimited()){
		if (!hasKey(key))
			throw runtime_error(key + ": key does not exist");
	}
	return map<string,Data>::operator[]( getAlias(key) );
}


const Data & Options::get(const string &key) const {
	const_iterator it = find(key);
	if (it == end())
		if (!isLimited()){
			return empty;
		}
		else {
			throw runtime_error(key + ": key does not exist");
		}
	else 
		return it->second;	
}


void Options::setAlias(const  string &key,const char &k){
	aliases[k] = key;
}

const string & Options::getAlias(const string &key) const {
	if (key.size() == 1){
		map<char,string>::const_iterator it = aliases.find(key[0]);
		if (it != aliases.end())
			return it->second;
	}
	return key;
	    
}

void Options::help(ostream &ostr,const string & postfix) const {
	//stringstream sstr;
	map<char,string>::const_iterator ait;
	
	for (map<string,Data>::const_iterator it = begin(); it != end(); it++){
      	ostr << "--" << it->first;  
      	for (ait = aliases.begin(); ait!=aliases.end(); ait++)
      		if (ait->second == it->first)
      			ostr << ",-" << ait->first  ;
      	ostr << ' ';
      	if (!it->second.isFlag()){
      		ostr << it->second.syntax;
      		//ostr << " (" << it->second.getType().name() << ')';
      	}
      	ostr << "\t\t(" << it->second << ")";
      	ostr << postfix <<  it->second.description << '\n';
      	//ostr << " (" << it->second;
	}
   	//target = sstr.str();
}

/*
string Options::str() const {
	stringstream sstr;
	sstr << *this;
	return sstr.str();
}
*/

//ostream &Options::operator<<(ostream &ostr){
ostream &operator<<(ostream &ostr,const Options &opt){
    for (map<string,Data>::const_iterator it = opt.begin(); it != opt.end(); it++)
      	ostr << it->first << '=' << it->second << '\n';
   	return ostr;
}

}

