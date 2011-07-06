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
#include "Data.h"

#include <iostream>


using namespace std;


namespace drain {


Data::Data() {
	syntax=""; //"<value>"; => FLAG
	separators = ",";
	trimChars = " \t\n";
	type = &typeid(void);
};


Data::Data(const Data &d) : string(d) { // type("jurpo") {
	//isFlag = d.isFlag;
	syntax = d.syntax;
	separators = d.separators;
	trimChars = d.trimChars;
	type = d.type;
	//type = &typeid(string);
};

Data &Data::operator =(const Data &d){
    clear();
    (*this) << d;
    if (!typeIsSet())
    	type = &d.getType();
	return (*this);
}

Data &Data::operator =(const char *str){ 
    this->assign(str);
    if (!typeIsSet())
    	type = &typeid(string);
    return (*this);  
}

Data &Data::operator =(const string &str){ 
    this->assign(str);
    if (!typeIsSet())
    	type = &typeid(string);
    return (*this);  
}


Data &Data::operator<<(const Data &d){

	if (!typeIsSet())
		if (empty())
			type = &d.getType();

	list<string> l;
	d.splitTo(l);
	for (list<string>::iterator it = l.begin(); it != l.end(); it++)
		(*this) << *it;
	return (*this);
}

/*
void Data::split(vector<Data> &v)  {
	v.resize(size());
	for (int i = 0; i < size(); ++i) {
		v[i] = (*this)(i);
	}
}
*/
void Data::trim(string &s) const{

	const string::size_type pos     = s.find_first_not_of(trimChars);
	
	// Only trim characters found, so trim all.
	if (pos == string::npos){
		s.clear();
	}
	else {
		const string::size_type lastPos = s.find_last_not_of(trimChars);
		s = s.substr(pos, lastPos-pos+1);
	}
	
}


/*
Data Data::operator()(int i) const {
	
	
	const string::size_type n  = size();
	string::size_type i1 = 0;
	string::size_type i2 = 0;
	   
	while (true){ 
	   i2 = find_first_of(separators,i1);
	   //cout << i << ':' << i1 << ',' << i2 << '\n';

	   // Found i'th segment 	   
	   if (i == 0){
		  Data x;
	  	  if (i2 == string::npos)
        	 i2 = n;
		  x = substr(i1,i2-i1);
	  	  i1 = x.find_first_not_of(trimChars);
		  i2 = x.find_last_not_of(trimChars);
		  x = x.substr(i1,i2-i1+1);
	  	  return x; 	
	   }
	   
	   // Not found, end reached.
	   if (i2 == string::npos)
         return Data();
	     
	   i1 = i2+1;
       i--; 
    }
    
}
*/

}
