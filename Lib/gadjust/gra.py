#!/opt/baltrad/third_party/bin/python
'''
Copyright (C) 2012- Swedish Meteorological and Hydrological Institute (SMHI)

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

## Gauge-radar analysis.
#  Most of this functionality is carried over and updated from gra.py from 1999.
#  It doesn't perform gauge adjustment, just determines the coefficients for it.

## @file
## @author Daniel Michelson, SMHI
## @date 2012-02-11

import sys, os
import rave_math
from gadjust import ttest
from numpy import *
from rave_defines import GADJUST_STATFILE

## Generator of gauge-adjustment coefficients
# @param points list of rave_synop.gra points
# @param DATE string date in format YYYYMMDD
# @param TIME string time in format HHmm
# @param LOGFILE file name of the file to which to write statistics
def generate(points, DATE, TIME, LOGFILE=GADJUST_STATFILE):
  g = gra(points)

  r, n, sig = general_correlation(points)

  a, b, c, m, dev, loss = g.get_2nd_order_adjustment()

  fd = open(LOGFILE, 'a')
  sformat = "%s %s %s %i %i %f %s %f %f %f %f %f %f\n"
  fd.write(sformat % (DATE, TIME, g.significant, len(g.points), loss, r, sig, g.corr_coeff, a, b, c, m, dev))
  fd.close()
  
  return g.significant, len(g.points), loss, r, sig, g.corr_coeff, a, b, c, m, dev


## Computes the correlation coefficient between gauge and radar (dBR)
# @param points list of rave_synop.gra points
# @return tuple containing correlation coefficient, sample size, and boolean
# string 'T' or 'F' as to whether the correlation is statistically significant.
def general_correlation(points):
  xarr, yarr = [], []

  for point in points:
    xarr.append(10*log10(point.observation))
    yarr.append(10*log10(point.radarvalue))

  xarr = array(xarr, float64)
  yarr = array(yarr, float64)

  n = len(xarr) - 1
  mx = sum(xarr) / len(xarr)
  my = sum(yarr) / len(yarr)

  Sx = sqrt(sum(power(xarr-mx, 2)) / n)
  Sy = sqrt(sum(power(yarr-my, 2)) / n)

  Sxy = sum((xarr-mx) * (yarr-my))

  r = Sxy / (n * Sx * Sy)
  return r, n, ttest.ttest(r, n)


class gra_coefficient(object):
  def __init__(self, area, date, time, significant, points, loss, r, r_significant, corr_coeff, a, b, c, mean, stddev):
    self.area = area
    self.date = date
    self.time = time
    self.significant = significant
    self.points = points
    self.loss = loss
    self.r = r
    self.r_significant = r_significant
    self.corr_coeff = corr_coeff
    self.a = a
    self.b = b
    self.c = c
    self.mean = mean
    self.stddev = stddev

## Gauge-Radar Analysis class
class gra:
  ## Initializer
  # @param points list of synop points
  def __init__(self, points):
    self.points = points
    self.corr_coeff=1.0   # What is the correlation between range and F(G)?
    self.significant = "False"
    self.qc = 0.0


  ## Derives a second-order statistical relation including quality control
  # @return coefficients a, b, c together with mean and standard deviation
  # of the G-R point pairs (dB)
  def get_2nd_order_adjustment(self):
    n = len(self.points)
    a, b, c = self.least_square_nth_degree(2)
    self.corr_coeff = self.get_correlation()
    m, dev = self.get_std_deviation()
    self.quality_control_2nd_order(a, b, c)

    # Always regenerate a,b,c,m,dev,corr_coeff if QC has removed some obs
    if n > len(self.points):
      a, b, c = self.least_square_nth_degree(2)
      self.corr_coeff = self.get_correlation()
      m, dev = self.get_std_deviation()

    if ttest.ttest(abs(self.corr_coeff),len(self.points))=="T":
      self.significant = "True"
    else:
      self.significant = "False"

    return a, b, c, m, dev, n-len(self.points)


  ## Utility method for deriving least-squares fit of the nth order
  # @param order int representing order of fit
  def least_square_nth_degree(self, order):
    xl, yl = [], []
    for point in self.points:
      xl.append(point.radardistance)
      yl.append(point.gr)

    arr=rave_math.least_square_nth_degree(order,xl,yl)
    
    alist=arr.tolist()
    for i in range(len(alist)):
      alist[i]=alist[i][0]

    return tuple(alist)

  ## Derives correlation coefficient between G-R (dB) and distance
  # @return float correlation coefficient
  def get_correlation(self):
    gr, dist = [], []
    for point in self.points:
      gr.append(point.gr)
      dist.append(point.radardistance)
      
    xarr = array(dist, 'd')
    yarr = array(gr, 'd')
    n = len(xarr) - 1

    mx = sum(xarr) / len(xarr)
    my = sum(yarr) / len(yarr)

    Sx = sqrt(sum(power(xarr-mx, 2)) / n)
    Sy = sqrt(sum(power(yarr-my, 2)) / n)

    Sxy = sum((xarr-mx) * (yarr-my))

    r = Sxy / (n * Sx * Sy)
    return r


  ## Derives mean and standard deviation of the G-R point pairs
  # @return tuple containing mean and standard deviation
  def get_std_deviation(self):
    vals=[]
    for point in self.points:
      vals.append(point.gr)
    valarray=array(vals, 'd')
    m, dev = rave_math.get_std_deviation(valarray)
    return m, dev


  ## Derives mean QUALITY of the point pairs
  # @return float mean quality
  def get_mean_quality(self):
    S = 0.0
    for p in self.points:
      S += p.Fq
    return S / len(self.points)

  ## Derives the standard deviation of G-R point-pair quality
  # @param m float mean
  # @return float standard deviation
  def get_stddev_quality(self, m):
    std = 0.0
    for p in self.points:
      std += (p.Fq - m)**2
    return sqrt(std / len(self.points))

  ## Conducts quality control of the relation between G-R point pairs and
  # surface distance. Point pairs more than 2 standard deviations in error
  # are rejected.
  # @param a float coefficient a
  # @param b float coefficient b
  # @param c float coefficient c
  def quality_control_2nd_order(self, a, b, c):
    for point in self.points:
      fr = a + point.radardistance*b + c*(point.radardistance**2)
      point.Fq = point.gr - fr

    m = self.get_mean_quality()
    std = self.get_stddev_quality(m)

    tokill = []
    for i in range(len(self.points)):
      z = (self.points[i].Fq - m) / std
      if abs(z) <= 1:
        self.points[i].quality_ok = 1.0
      elif 1 < abs(z) <= 2:
        self.points[i].quality_ok = 1.0 - (abs(z) - 1.0)
      elif abs(z) > 2:
        tokill.append(i)
    if len(tokill):
      tokill.reverse()
      for i in tokill:
        del self.points[i]

if __name__ == "__main__":
    """
               2             0.12
              25                1
              50             1.25
              75              1.5
             100                2
             125             2.25
             150                3
             175                4
             200                5
             225                6
             250                8

Results from xmgr:
SSerr =          0.853693

Variable order:
    0    1    2 
Vector Beta      : 
 0: 0.6224 
 1: 0.001098 
 2: 0.0001071 
Vector d         : 
 0: 11 
 1: 6.825e+04 
 2: 3.26e+08 
rbar matrix:
       125.182       21875.4
                      251.17
Vector thetab    : 
 0: 3.102 
 1: 0.02799 
 2: 0.0001071 

Regression of set 0 results to set 1
    """
    pass

A="""
    from rave_synop import point

    gr = [0.12, 1.0, 1.25, 1.5, 2.0, 2.25, 3.0, 4.0, 5.0, 6.0, 8.0]
    dist = [2.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0]
    points = []
#    gr.append(-20.0)
#    dist.append(110.0)

    for i in range(len(gr)):
        p = point()
        p.dist = dist[i]
        p.gr = gr[i]

        points.append(p)

    g = gra(points)

    a, b, c, m, dev, loss = g.get_2nd_order_adjustment()

    print g.significant, len(g.points), loss, g.corr_coeff, a, b, c, m, dev
"""    