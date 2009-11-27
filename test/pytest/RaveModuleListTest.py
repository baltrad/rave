'''
Created on Nov 26, 2009

@author: anders
'''
import unittest
import os
import _rave

class RaveModuleListTest(unittest.TestCase):
  
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testNewRaveIO(self):
    obj = _rave.list()
    isravelist = string.find(`type(obj)`, "RaveListCore")
    self.assertNotEqual(-1, isravelist)

  def testAdd(self):
    obj = _rave.list()
    scan = _rave.scan()
    scan.elangle = 1.0
    obj.add(scan)
    self.assertEquals(1, obj.size())
    
