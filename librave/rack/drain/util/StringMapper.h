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

    Author: mpeura
*/

#ifndef STRINGMAPPER_H_
#define STRINGMAPPER_H_

#include <map>
#include <list>
#include <iterator>
#include "RegExp.h"

namespace drain {

/// A helper class for StringMapper.
/**  A Stringlet behaves either as a literal (simple std::string) or variable (dynamically expanded).
 *   When a StringMapper parses a given string, it splits the string to segments containing
 *   literals and variables, in turn.
 *   The result is stored in a list.
 */
template<class T>
class Stringlet: public string {
public:

        Stringlet( const map<string, T> & m, const string & s = "", bool isVariable = false)
			:  variable(isVariable), _map(m) {
                assign(s);
        };


        // oldish
        const T & expand() const {
               if (isVariable) {
                        typename map<string, T >::iterator it = _map.find(*this);
                        if (it != _map.end()) {
                                return (const string &) it->second;
                        }
                }
               return *this;
        };

        ostream & expand(ostream &ostr) const {
        	if (isVariable()) {
        		typename map<string, T >::const_iterator it = _map.find(*this);
        		if (it != _map.end()) {
        			//ostr << (const string &)*this << '='
        			ostr <<  it->second;
        			return ostr;
        		}
        	}
        	ostr << (const string &)*this;
        	return ostr;
        };

        inline
        bool isVariable() const { return variable; };

protected:
        bool variable;
        const map<string, T> & _map;

};

/*
template <class T>
ostream &  operator<<(ostream & ostr, Stringlet<T> & strl){

	ostr << strl.expand();
	return ostr;
};
*/

/// A tool for expanding variables embedded in a string to literals.
/** The input variables are provided as a map, which is allowed to change dynamically.
 *  The input string, issued with parse(), contains the variables as "$key" or "${key}", a
 *  format familiar in Unix shell.
 *
 */
template<class T>
class StringMapper {
public:

	/**
	 *  \par m - the map containing variables as (key,value) pairs.
	 *  \par valid - string defining the allowed variable key syntax.
	 *
	 *
	 */
	StringMapper(const map<string, T> & m, const string &valid = "a-zA-Z0-9_") : valid(valid), _map(m) {
	//	string valid = "a-zA-Z0-9_";
		string e;
		e = e + "^([^\\$]*)" + "\\$\\{([" + valid + "]*)\\}" + "([^" + valid
				+ "].*)?$";
		regExp.setExpression(e);
	};

	virtual ~StringMapper() {};

	//void setMap(const map<string,T> & m);

	void parse(const string &s) {
		l.clear();

		//string valid = "a-zA-Z0-9_";
		RegExp r("([^\\$]*)\\$\\{([a-zA-Z0-9_]+)\\}(.*)");
		//string e;
		//e = e + "^([^\\$]*)" + "\\$\\{([" + valid + "]*)\\}" + "([^" + valid + "].*)?$";
		//r.setExpression(e);
		//r.setExpression("(.*)\\$\\{(a-zA-Z0-9_\-)\\}(.*)");
		r.execute(s);

		cout << " StringMapper size=" << r.result.size() << "\n";
		// The size of the result is 4, if ${variable} found, else 0, and result is literal.
		switch (r.result.size()) {
		case 4:
			parse(r.result[3]);
		case 3:
			l.push_front(Stringlet<T> (_map, r.result[2], true));
		case 2:
			l.push_front(Stringlet<T> (_map, r.result[1], false));
			break;
		case 0:
			l.push_front(Stringlet<T> (_map, s, false));
			break;
		default:
			cerr << " StringMapper warning: parsing oddity: " << s << "\n";
			break;
		}
	};

	string str() const {
		stringstream sstr;
		sstr << *this;
		return sstr.str();
	}

	inline
	typename list< Stringlet<T> >::const_iterator begin() const {
		return l.begin();
	}

	inline
	typename list< Stringlet<T> >::const_iterator end() const {
		return l.end();
	}

protected:

	RegExp regExp;
	list<Stringlet<T> > l;
	const string valid;
	const map<string, T> & _map;
};

template <class T>
ostream &  operator<<(ostream & ostr, const StringMapper<T> & strm){
	typename list< Stringlet<T> >::const_iterator it;
	for (it = strm.begin(); it != strm.end(); it++){
		//ostr << *it;
		it->expand(ostr);
	};
	return ostr;
};


} // NAMESPACE

#endif /* STRINGMAPPER_H_ */
