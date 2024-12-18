#ifndef ODIM_SOURCE_H
#define ODIM_SOURCE_H
#include <string>

class ODIM_Source {
public:
    ODIM_Source(std::string & src);
    ~ODIM_Source();
    std::string str(void);
    
private:
    //## Initializer
    //@param src string containing a '/what/source' attribute
    void init(std::string & src);
    //## Splits the input string into identifier values        
    void split_source(void);
public:
    std::string source;
    std::string wmo;
    std::string nod;
    std::string rad;
    std::string plc;
    std::string org;
    std::string cty;
    std::string cmt;
    std::string wigos;
};
#endif
