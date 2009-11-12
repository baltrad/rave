'''
Created on Nov 12, 2009
@author: Anders Henja
'''
import unittest
import os
import _rave
import string
import numpy

class RaveModuleRaveIOTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def testNewRaveIO(self):
    obj = _rave.io()
    israveio = string.find(`type(obj)`, "RaveIOCore")
    self.assertNotEqual(-1, israveio)
    
