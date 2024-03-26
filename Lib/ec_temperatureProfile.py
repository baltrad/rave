'''
Copyright (C) 2020 The Crown (i.e. Her Majesty the Queen in Right of Canada)

This file is an add-on to RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE and this software are distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.

'''
##
# 


## 
# @file
# @author Daniel Michelson, Environment and Climate Change Canada
# @date 2020-11-27

import numpy as np


# Bright band (melting layer) top and bottom temperatures for wet- and dry-bulb
# temperatures, respectively. The Tw values represent 1.0 and 0.0 proportion of
# snow in precipitation based on a Swedish study (SMHI, RMK 75, 1997).
BB = {"Tw" : (-0.2, 2.43),
      "Td" : ( 0.0, 3.0)}


## Reads a height-temperature profile from ASCII file, where the first column
#  is height (metres above sea level) and the second column is wet-bulb
#  temperature (Tw) in C, and third column is dry-bulb temperature (Td) in C.
#  The profile should be ascending by height, such that the first row in the
#  file is the lowest.
# @param string input file string
# @param boolean whether (True) to flip the profile vertically or not (False)
# @param float scaling factor for height, in case heights are provided in e.g. km.
# @return array of doubles in two dimensions (height, temperature)
def readProfile(fstr, flip=False, scale_height=0.0):
  profile = np.loadtxt(fstr, unpack=True, dtype='d')
  if flip: profile = np.fliplr(profile)
  if scale_height: profile[0] *= scale_height
  return profile


## Interpolates a vertical temperature profile to central beam heights given by
#  radar. In cases where a low tilt propagates negatively at first, it becomes
#  necessary to sort the radar beam heights for each bin along the ray so that
#  they are always ascending, and then to match the resulting temperatures back
#  up with the original order.
# @param array containing input profile heights
# @param array containing input profile temperatures
# @param array containing radar-derived heights
# @return array containing interpolated temperatures at the heights given by rheight
def interpolateProfile(pheight, ptempc, rheight):
  tmp_height = np.array(rheight)
  tmp_height.sort()
  rheight_idx = rheight.argsort()

  # At tmp_height, interpolate the profile variable ptempc known at pheight
  mytempc = np.interp(tmp_height, pheight, ptempc)
  return mytempc[rheight_idx]


## Pulls out the heights of bins along the ray of a scan and derives a
#  matching temperture profile
#  @param array (2-D) containing profile heights[0], Tw[1], and Td[2]
#  @param sequence or array containing profile temperatures in Celcius
#  @param boolean whether to use Tw=True or Td=False
#  @return array of temperatures matching the scan's ray
def getTempcProfile(scan, profile, Tw=True):
  heightf = scan.getHeightField()
  heights = heightf.getData()[0]
  if Tw: rtempc = interpolateProfile(profile[0], profile[1], heights)
  else:  rtempc = interpolateProfile(profile[0], profile[2], heights)
#  return heights, rtempc  # For debugging
  return rtempc        


if __name__=="__main__":
    pass
