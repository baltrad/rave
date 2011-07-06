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
#ifndef REGEXP_H__
#define REGEXP_H__ "deer::RegExp v0.1 Dec 2004 Markus.Peura@fmi.fi"

#include <iostream>
#include <string>
#include <vector>

#include <regex.h> // wants malloc?

// g++ deer_regexp.cpp -o deer_regexp
// typedef size_t int;
using namespace std;

namespace drain {

  /*!
   *  Based on POSIX regex functions <regex.h>, see man regex.
   *
   *  Examples: 
   *  Regexp r='ksooskks';
   *  r.regexp('dkksks');
   *  //r.exec('Test (s)tribng');
   *  r == 'test_string';
   *
   */
  class RegExp { //: public vector<string> {
  public:

	/// Constructor
    //RegExp(const char *str = "");

    /// Constructor
    //RegExp(const char *str = "",int flags = REG_EXTENDED);
    RegExp(const string &str = "",int flags = REG_EXTENDED);

    /// Copy constructor.
    // Lack of this caused memory leakages.
    RegExp(const RegExp &r);

    /// Destructor.
    virtual
    ~RegExp();
 
 	//RegExp &operator=(const char *str);
 	RegExp &operator=(const string &str);
 	RegExp &operator=(const RegExp &r);
 	
    //    bool exp(const char *str);
    bool setExpression(const string &str);

	//    bool exp(const char *str);
	
	inline 
    void clear(){writableResult.clear();};

	const vector<string> &result;

    //    inline    
    //bool test(const char *str) const;
	bool test(const string & str) const;

    //    bool exec(const char *str);
    // Returns 0 on success, REG_NOMATCH on failure
    int execute(const string &str);

    
    // TODO protect
	string regExpString;

  protected:
    int flags;
    regex_t regExpBinary;  // this is weird
	
	
	vector<string> writableResult;

  private:
    int expectedMatchCount() const;
  };


 
}

#endif
