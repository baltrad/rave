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
## T-tests for whether differences in correlations and means are
#  real or bogus. Elaborated since 1998.

## @file
## @author Daniel Michelson, SMHI
## @date 2013-01-01

from numpy import sqrt, power, minimum
from gadjust.tcritical import getTTABLE

# Percentile values for the Student's t distribution with 'key' degrees 
# of freedom.
# 95% confidence interval:
t95 = {1:6.31,
       2:2.92,
       3:2.35,
       4:2.13,
       5:2.02,
       6:1.94,
       7:1.9,
       8:1.86,
       9:1.83,
       10:1.81,
       11:1.8,
       12:1.78,
       13:1.77,
       14:1.76,
       15:1.75,
       16:1.75,
       17:1.74,
       18:1.73,
       19:1.73,
       20:1.72,
       21:1.72,
       22:1.72,
       23:1.71,
       24:1.71,
       25:1.71,
       26:1.71,
       27:1.70,
       28:1.70,
       29:1.70,
       30:1.70,
       40:1.68,
       60:1.67,
       120:1.66,
       121:1.645}  # infinity


## Student's T-test of correlation coefficient.
# @param r float correlation coefficient
# @param n int sample size (number of points)
# @return string 'T' if the correlation is accepted, else 'F'.
def ttest(r, n):
  f = (n - 2)  # degrees of freedom

  # get the right t distribution for f
  keys = list(t95.keys())
  keys.sort()
  if f in keys: 
    tp = t95[f]
  elif f > 120:
    tp = t95[121]
  else:
    MIN = 1e100
    for k in keys:
      MIN = minimum(abs(f-k), MIN)
    tp = t95[MIN]

  t = (r * sqrt(f)) / sqrt(1 - (r**2))
  #    print tp, t
  if t < tp:
    return 'F'
  else:
    return 'T'

## Student's T-test of means.
# @param m1 float first mean
# @param m2 float second mean
# @param d1 first deviation
# @param d2 second deviation
# @param n1 first sample size
# @param n2 second sample size
# @param ci float confidence interval, 95% = 0.05
# @param tailed int 1 or 2-tailed test
# @return string 'T' if the difference is accepted, else 'F'.
def ttest_means(m1, m2, d1, d2, n1, n2, ci=0.05, tailed=2):
  ttable = getTTABLE()
  
  sab = sqrt(((n1-1)*power(d1,2) + (n2-1)*power(d2,2)) / (n1 + n2 - 2))
  texp = abs(m1 - m2) / (sab * sqrt(1./n1 + 1./n2))
  df = n1 + n2 - 2
  if df > 200: df = 200  # Cap!
  if tailed == 1:
    ttheo = ttable["ONETAIL"][df][str(ci)]
    #ttheo = TTABLE["ONETAIL"][df][str(ci)]
  elif tailed == 2:
    ttheo = ttable["TWOTAIL"][df][str(ci)]
    #ttheo = TTABLE["TWOTAIL"][df][str(ci)]
  else:
    raise AttributeError("Tailed argument must be either 1 or 2.")

  if texp > ttheo:
    return 'T'
  else:
    return 'F'

if __name__ == "__main__":
  m1, m2, d1, d2, n1, n2 = 48.73, 50.43, 1.16, 1.21, 5., 5.
  sig = ttest_means(m1, m2, d1, d2, n1, n2)
  print(sig)
