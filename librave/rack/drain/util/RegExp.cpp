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
#include <stddef.h>  // size_t
//#include <regex.h> // wants mall
#include <iostream>
#include <string>
//#include <vector>

// g++ deer_regexp.cpp -o deer_regexp

#include "RegExp.h"

using namespace std;

namespace drain {

  /// BUGS! Reads a regular expression string and stores it in a preprocessed format.
  /*! 
   *  BUGS! Calling exp() later does not work!
   * 
   *  Based on POSIX regex functions <regex.h>, see man regex.
   *
   *  Examples: 
   *
   *  Regexp r;
   *  r.regexp('\.html');
   *
   *  if (r.test('index.html'))
   *    cerr << "OK\n";
   *
   * const string &str
   */
   /*
	RegExp::RegExp(){
     	flags = REG_EXTENDED
    };
	*/
	//RegExp::RegExp() : result(writableResult) {	};
  //RegExp::RegExp(const char *str,int flags) 
  RegExp::RegExp(const string &str,int flags) :  result(writableResult), flags(flags){
  	regcomp(&regExpBinary,"",0);  // IMPORTANT, because setExpression calls regfree.
    setExpression(str);
  }

  // Lack of this caused memory leakages.
  RegExp::RegExp(const RegExp &r) :  result(writableResult), flags(r.flags){
	  regcomp(&regExpBinary,"",0);  // IMPORTANT, because setExpression calls regfree.
	  setExpression(r.regExpString);
  }

  RegExp::~RegExp(){ 
    // Clear memory;
    //cerr << "~RegExp()" << endl;
    regfree(&regExpBinary); 
    //    regExpBinary = NULL;
    // Clear result variables
    writableResult.clear();
  }
  
  RegExp &RegExp::operator=(const RegExp &r){
  	  setExpression(r.regExpString); 
      return *this; 
  };

  // TODO: skip this?
  //RegExp &RegExp::operator=(const char *str){
  RegExp &RegExp::operator=(const string &str){
    setExpression(str); 
    return *this; 
  }	

  bool RegExp::setExpression(const string &str){

    regfree(&regExpBinary); 
    writableResult.clear(); 

    int errorCode = regcomp(&regExpBinary,str.c_str(),flags);

    if (errorCode){
      cerr << "illegal (" << errorCode << ") regex string: " << str << '\n';
      //regExpBinary = NULL;
      regExpString = "";  // ? should be saved instead? 
      return false;
    }
    else {
      regExpString = str;
      //matches.clear();
      //      this->clear();
      return true;
    }
  }




  /// Attempts to match given string against the (compiled) regexp.
  //  inline
  bool RegExp::test(const string &str) const {
    return (regexec(&regExpBinary,str.c_str(),0,NULL,0) == 0);
  }
  /*
  bool RegExp::test(const char *str) const {  
      return (regexec(&regExpBinary,str,0,NULL,0) == 0);
  }
  */


  /// Like test, but stores the matches.
  /// Attempts to match given string against the (compiled) regexp.
  //  bool RegExp::exec(const char *str){ 
  int RegExp::execute( const string &str ){ 

    /// Allocates space for the matches. 
    size_t n = regExpBinary.re_nsub + 1;
	
	writableResult.clear();
	writableResult.resize(n);
	
	//cout << "resize => " << this->size() << endl;

    /// Allocates temp array for <regex.h> processing.
    regmatch_t *pmatch = new regmatch_t[n];

    /// The essential <regex.h> wrapper. 
    /// (Notice the negation of boolean return values.)
    /// In success, indices (rm_so,rm_eo)
    /// will point to matching segments of str.
    /// Eflags not implemented (yet?).
    //cerr << "binary has subs:" << regExpBinary.re_nsub << endl;
    
    //cerr << "\nTrying " << str.c_str() << endl;
    int resultCode = regexec(&regExpBinary,str.c_str(),n,pmatch,0) ;
    
    //cerr << "result " << result << endl;
    
    if (resultCode == REG_NOMATCH){
      //cerr << "dont like " << str.c_str() << endl;
      writableResult.clear();
    }
    else {
      regoff_t so;
      regoff_t eo;
      //size_t i;
      for (size_t i=0; i < n; i++){
	    so = pmatch[i].rm_so;
	    eo = pmatch[i].rm_eo;
	    //cerr << "match" << so << "..." << eo << endl;
     	if (so != -1) 
	      writableResult[i].assign(str,so,eo - so);
      }
    }
    delete[] pmatch; // valgrind herjasi muodosta: delete pmatch
    return resultCode;
   
  }
}


