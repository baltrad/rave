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
#ifndef OPTIONS_H_
#define OPTIONS_H_

#include <ostream>
#include <string>

//#include "OptionData.h"
#include "Data.h"
//#include "MapWrapper.h"
#include "MapReader.h"

using namespace std;	



namespace drain
{

/// drain::Options is a utility for handling command line Options, configuration files.
/*! Utility for handling variable data inside an application.
 *  Supports:
 *  # automatic casts
 *  # command line options
 *  # config file read and write
 * 
 *  Examples:
 *  
 * 
 */
class Options :  private map<string,Data>
{
public:

	Options() : reader(*this) {};

	virtual ~Options(){};

	//inline Data &operator[](const string &key){return map<string,Data>::operator[](key);};
	Data &operator[](const string &key); //{return map<string,Data>::operator[](key);};
	inline const map<string,Data>::const_iterator begin() const {return map<string,Data>::begin();};
	inline const map<string,Data>::const_iterator end()   const {return map<string,Data>::end();};

	void clear(){ map<string,Data>::clear(); };

	void erase(const string & key){ map<string,Data>::erase(key); };

	inline
	bool hasKey(const string &key) const {return (find(getAlias(key)) != end());}

	inline
	size_t size() const {return map<string,Data>::size();}

	/// Returns empty, if key not found. Unlike operator[key], does not add an entry.
	const Data &get(const string &key) const;

	/// Returns the underlying map.
	const map<string,Data> & getMap() const { return *this; };

	/// Associate a long key with a single-character alias
	void setAlias(const string &key,const char &k);

	/// Given the single-character alias, returns the primary, long key.
	/*! Given the long key, returns it as such.
	 */
	const string &getAlias(const string &key) const;

	/// Returns value corresponding to key, if defined, else
	/**
	 * @param key Name associated with a value
	 * @param defaultValue is returned, if key is not defined
	 * @return value or default value.
	 */
	template <class T>
	T get(const string &key,T defaultValue) const {
		const_iterator it = find(key);
		if (it == end())
			return defaultValue;
		else 
			return static_cast<T>(it->second);
	}


	string get(const string &key,const char *defaultValue) const {
		const_iterator it = find(key);
		if (it == end())
			return defaultValue;
		else 
			return string(it->second);
	}

	/*! Assigns comma-separated values to keys in the order determined by defaultKeys.
	 *  @return 
	 *  @throws runtime_error, if number of values exceeds that of defaultKeys.   
	 */
	void set(const string & s);	

	/// Copies a map
	template <class T,class T2>
	void copy(const map<T,T2> &m);

	// Dumps current entries and their syntax.
	void help(ostream &ostr = cout,const string & postfix = "\n\t") const;

	/// The set of keys can be limited by defining a the keys in order.	
	/// Todo: change to locked and unlocked.
	/*
    inline void setAllowedKeys(const string & s){
       allowedKeys = s;
    }
	 */

	/*! The set of keys can be limited by defining the keys in order.
	 *  Initializes with empty strings, "".
	 *  This allows writing 
	 *
	 *  \code
	 *  options.setDefaultKeys("width,height,name");
	 *  //...
	 *  options.set("210,297,A4");
	 *  // or
	 *  options.set("210,297");
	 *  // or
	 *  options.set("210");
	 *  \endcode
	 * 
	 *  instead of
	 *  \code
	 *  options["width"]  = 210;
	 *  // etc.
	 *  options["height"] = 297;
	 *  options["name"]   = "A4";
	 *  \endcode
	 *  
	 */	
	inline
	void setParameterNames(const string & names){
		info = names;
		list<string> l;
		info.splitTo(l);
		for (list<string>::const_iterator it = l.begin(); it != l.end(); it++)
			map<string,Data>::operator[](*it);  // yes, initialize

		//set(names);  // kludge, KEY=KEY
	};


	inline
	const Data & getParameterNames() const {
		return info;
	};



	inline
	const string & getParameterValues() const {
		list<string> l;
		info.splitTo(l);
		values.clear();
		for (list<string>::const_iterator it = l.begin(); it != l.end(); it++)
			values << get(*it,"");
		return values;
	};


	/*
	inline void setDefaultKeys(const string & s){
       info = s;
    }

    ///

    inline const Data & getDefaultKeys() const {
        return info;
    }
	 */


	inline
	void setDescription(const string & description){
		info.description = description;
	};

	inline
	const string & getDescription() const {
		return info.description;
	};


	//string str() const;

	// will be changed to "return limited"
	bool isLimited() const { return !info.empty(); };

	// TODO PROTECT
	MapReader<string,Data> reader;

protected:

	/// Stores parameter names
	Data info;

	/// Stores parameter values
	mutable Data values;

	/// Dummy object which is returned when something is not defined.
	static const Data empty;
	map<char,string> aliases;


};

ostream &operator<<(ostream &ostr,const Options &opt);


template <class T,class T2>
void Options::copy(const map<T,T2> &m){
	for (typename map<T,T2>::const_iterator it = m.begin(); it != m.end(); it++){
		(*this)[Data(it->first)] = it->second;
	}
	//ostr << it->first << '=' << it->second << '\n';

}


}

#endif /*Options_H_*/
