#include "odim_source.h"
#include <vector>
#include <sstream>
#include <algorithm>

/* simplified version of odim_source.py */

ODIM_Source::ODIM_Source(std::string & src) {
    init(src);
};

ODIM_Source::~ODIM_Source() {
};

void ODIM_Source::init(std::string & src) {
   source = src;
   wmo = nod = rad = plc = org = cty = cmt = wigos = "";
   if (source.length()) {
       split_source();
   }
};

void ODIM_Source::split_source() {
    std::vector<std::string> split;
    std::istringstream f(source);
    std::string tmp;    
    while (getline(f, tmp, ',')) {
        split.push_back(tmp);
    }
    for (std::string s : split) {
        std::vector<std::string> tokens;
        std::istringstream f(s);
        std::string token;    
        while (getline(f, token, ':')) {
            tokens.push_back(token);
        }
        if (tokens.size() != 2)
            continue;
        std::string prefix=tokens[0];
        //# safety precaution in case someone changes case in their files
        std::transform(prefix.begin(), prefix.end(), prefix.begin(),[](unsigned char c){ return std::tolower(c); });
        std::string value=tokens[1];
        if (prefix == "wmo")
            wmo = value;  //# Keep this as a string!
        else if(prefix == "rad")
            rad = value;
        else if(prefix == "plc")
            plc = value;
        else if(prefix == "nod")
            nod = value;
        else if(prefix == "org")
            org = value;
        else if(prefix == "cty")
            cty = value;
        else if(prefix == "cmt")
            cmt = value; 
        else if(prefix == "wigos")
            wigos = value; 
    }
};

std::string ODIM_Source::str(void) {
    std::ostringstream odim;
    odim << "ODIM_Source(nod=" << nod << ", wmo=" << wmo << ", rad=" << rad << ", plc=" << plc <<", org=" << org << ", cty=" << cty << ", cmt=" << cmt << ", wigos=" << wigos << std::ends;
    return odim.str();
};

