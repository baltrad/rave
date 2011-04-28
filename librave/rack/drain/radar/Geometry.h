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
#ifndef GEOMETRY_Hfoo_
#define GEOMETRY_Hfoo_

#include <vector>


using namespace std;

namespace drain
{

namespace radar
{

	
    /*! 
     *
     * \image latex radar-geometry-fig.pdf
     * \image html  radar-geometry-fig.png
     *
     *  Distances in meters, angles in radians.
     *
     *   - \f$a\f$: Earth radius, the distance from the Earth's center to the radar 
     *   - \f$b\f$: beam distance; the distance from the radar to the bin
     *   - \f$c\f$: distance from the Earth's center to the bin 
     *
     *   - \f$h = c - a\f$: altitude from ground to beam point
     *   - \f$g\f$: ground distance; surface distance 
     *        from the radar to the ground point
     *
     *  - eta \f$ \eta \f$: beam elevation
     *
     *  - alpha \f$ \alpha = \angle(b,c)\f$: "sky angle", 
     *  - beta  \f$ \beta  = \angle(a,c)\f$: "ground angle"
     *  - gamma \f$ \gamma = \angle(a,b) = \eta + \pi/2 \f$: "radar angle" 
     *
     *  - r = radian
     *  - d = degree
     *
     * Cosine rule:
     *  \f[
     *      c^2 = a^2 + b^2 - 2ab·\cos(\gamma);
     *  \f]
     *
     * Sine rule:
     *  \f[
     *      \frac{\sin\alpha}{a} = \frac{\sin\beta}{b} = \frac{\sin\gamma}{c}
     *  \f]
     *
     *
     */
class Geometry
{
public:
	Geometry();
	virtual ~Geometry();
	
	static double heightFromEtaBeam(float eta,float b);
    static double heightFromEtaBeta(double eta,double beta);
    static double heightFromEtaGround(double eta,double g);

    inline static double betaFromGround(float g){ return g / EARTH_RADIUS_43; };
    static double beamFromBetaH(double beta,double h);
    static double beamFromEtaH(double eta,double h);
    static double beamFromEtaBeta(double eta,double beta);
    static double beamFromEtaGround(float eta,float g);

    //inline unsigned int binFromEtaBeta(double eta,double beta){ 
    //	return static_cast<unsigned int>(binDepth/2 + binDepth*beamFromEtaBeta(beta,eta));};
    
    static double groundFromEtaB(float eta,float b);
    static double etaFromBetaH(double beta,double h);
    
    double bFromGH(double g,double h);
    double etaFromGH(double g,double h);
	
	
	double normalizedBeamPower(double angle);
    
    void findClosestElevations(const float &elevationAngle,
    	int &elevationIndexLower,float &elevationAngleLower,int &elevationIndexUpper, float &elevationAngleUpper);
    
	float beamWidth;
	
	// new
	static int EARTH_RADIUSi;
    static double EARTH_RADIUS_43;
	
	/// Note: radians!
	vector<float> elevationAngles;
	vector<unsigned int> bins;
};

}

}

#endif /*GEOMETRY_H_*/
