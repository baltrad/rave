#!/usr/bin/env python
'''
Copyright (C) 2013- Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
'''
## Math functions

## @file
## @author Daniel Michelson, SMHI
## @date 2013-12-03

# Third-party:
from numpy import *


## Derives least-square fit of the "order" order
# @param order int order
# @param xl array of floats
# @param yl array of floats
# @return array
def least_square_nth_degree(order, xl, yl):
    if len(xl) != len(yl):
        raise AttributeError("Inconsistant number of points")

    n = len(xl)

    x = array(xl).astype('d')
    y = array(yl).astype('d')

    x_sums = []

    noofpowers = order * 2 + 1  # +1 just to get it to loop from 0 to order*2 :)
    for i in range(0, noofpowers):
        x_sums.append(sum(power(x, float(i))))

    A = zeros((order + 1, order + 1), 'd')
    for yy in range(0, order + 1):
        for xx in range(0, order + 1):
            A[yy][xx] = x_sums[yy + xx]

    A_minus_1 = linalg.inv(A)

    b = zeros((order + 1, 1), 'd')

    for i in range(0, order + 1):
        b[i] = sum(power(x, float(i)) * y)

    return dot(A_minus_1, b)


## Calculates standard deviation
# @param fgs array of floats
# @return tuple of meanvalue,deviation
def get_std_deviation(fgs):
    meanvalue = sum(fgs) / float(len(fgs))
    deviation = fgs - meanvalue
    deviation = sqrt(sum(power(deviation, 2)) / float(len(deviation)))
    return meanvalue, deviation


if __name__ == "__main__":
    print(str(least_square_nth_degree(1, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])))
    print(str(least_square_nth_degree(2, [0.0, 1.0, 2.0], [1.0, 0.0, 1.0])))
