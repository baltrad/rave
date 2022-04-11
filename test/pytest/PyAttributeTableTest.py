'''
Created on Mar 20, 2022

@author: anders
'''
import unittest
import _attributetable, _rave
import string
import numpy

class PyAttributeTableTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _attributetable.new()
    self.assertNotEqual(-1, str(type(obj)).find("AttributeTableCore"))
  
  def test_attribute_visibility(self):
    attrs = ['version']
    obj = _attributetable.new()
    alist = dir(obj)
    for a in attrs:
      self.assertEqual(True, a in alist)

  def test_addAttribute(self):
    obj = _attributetable.new()
    r1 = obj.addAttribute("what/is", 10.0)
    r2 = obj.addAttribute("where/is", "that")
    r3 = obj.addAttribute("how/are", 5)

    self.assertEqual("what/is", r1[0])
    self.assertAlmostEqual(10.0, r1[1], 4)
    self.assertEqual("where/is", r2[0])
    self.assertEqual("that", r2[1])
    self.assertEqual("how/are", r3[0])
    self.assertEqual(5, r3[1])

  def test_getAttribute(self):
    obj = _attributetable.new()
    obj.addAttribute("what/is", 10.0)
    obj.addAttribute("where/is", "that")
    obj.addAttribute("how/are", 5)

    self.assertAlmostEqual(10.0, obj.getAttribute("what/is"), 4)
    self.assertEqual("that", obj.getAttribute("where/is"))
    self.assertEqual(5, obj.getAttribute("how/are"))

  def test_size(self):
    obj = _attributetable.new()
    obj.addAttribute("what/is", 10.0)
    obj.addAttribute("where/is", "that")
    obj.addAttribute("how/are", 5)

    self.assertEqual(3, obj.size())
    
  def test_hasAttribute(self):
    obj = _attributetable.new()
    obj.addAttribute("what/is", 10.0)
    obj.addAttribute("where/is", "that")
    obj.addAttribute("how/are", 5)

    self.assertTrue(obj.hasAttribute("what/is"))
    self.assertTrue(obj.hasAttribute("where/is"))
    self.assertTrue(obj.hasAttribute("how/are"))
    
    self.assertFalse(obj.hasAttribute("whatis"))
    self.assertFalse(obj.hasAttribute("whe/is"))
    self.assertFalse(obj.hasAttribute("ho/are"))
    
  def test_removeAttribute(self):
    obj = _attributetable.new()
    obj.addAttribute("what/is", 10.0)
    obj.addAttribute("where/is", "that")
    obj.addAttribute("how/are", 5)

    obj.removeAttribute("what/is")

    self.assertFalse(obj.hasAttribute("what/is"))
    self.assertTrue(obj.hasAttribute("where/is"))
    self.assertTrue(obj.hasAttribute("how/are"))

  def test_clear(self):
    obj = _attributetable.new()
    obj.addAttribute("what/is", 10.0)
    obj.addAttribute("where/is", "that")
    obj.addAttribute("how/are", 5)

    obj.clear()

    self.assertFalse(obj.hasAttribute("what/is"))
    self.assertFalse(obj.hasAttribute("where/is"))
    self.assertFalse(obj.hasAttribute("how/are"))

  def test_getAttributeNames(self):
    obj = _attributetable.new()
    obj.addAttribute("what/is", 10.0)
    obj.addAttribute("where/is", "that")
    obj.addAttribute("how/are", 5)

    result = obj.getAttributeNames()
    self.assertEqual(3, len(result))
    self.assertTrue("what/is" in result)
    self.assertTrue("where/is" in result)
    self.assertTrue("how/are" in result)

  def test_addHowRpm(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/rpm", 10.0)
    self.assertEqual("how/rpm", result[0])
    self.assertAlmostEqual(10.0, result[1], 4)
    self.assertAlmostEqual(10.0, obj.getAttribute("how/rpm"), 4)
    self.assertAlmostEqual(60.0, obj.getAttribute("how/antspeed"), 4)
    self.assertTrue(obj.hasAttribute("how/rpm"))

  def test_addHowAntspeed(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/antspeed", 60.0)
    self.assertEqual("how/rpm", result[0])
    self.assertAlmostEqual(10.0, result[1], 4)
    self.assertAlmostEqual(10.0, obj.getAttribute("how/rpm"), 4)
    self.assertAlmostEqual(60.0, obj.getAttribute("how/antspeed"), 4)
    self.assertTrue(obj.hasAttribute("how/rpm"))

  
  def test_addHowS2N(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/S2N", 10.0)
    self.assertEqual("how/S2N", result[0])
    self.assertAlmostEqual(10.0, result[1], 4)
    self.assertAlmostEqual(10.0, obj.getAttribute("how/SNR_threshold"), 4)
    self.assertAlmostEqual(10.0, obj.getAttribute("how/S2N"), 4)

  def test_addHowSNR_threshold(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/SNR_threshold", 10.0)
    self.assertEqual("how/S2N", result[0])
    self.assertAlmostEqual(10.0, result[1], 4)
    self.assertAlmostEqual(10.0, obj.getAttribute("how/SNR_threshold"), 4)
    self.assertAlmostEqual(10.0, obj.getAttribute("how/S2N"), 4)

  def test_addHowStartazT(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/startazT", numpy.array([1.0,2.0,3.0]))
    self.assertEqual("how/startazT", result[0])
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), result[1], 4))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), obj.getAttribute("how/startT"), 4))    
    self.assertTrue(obj.hasAttribute("how/startazT"))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), obj.getAttribute("how/startazT"), 4))    

  def test_addHowStartT(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/startT", numpy.array([1.0,2.0,3.0]))
    self.assertEqual("how/startazT", result[0])
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), result[1], 4))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), obj.getAttribute("how/startT"), 4))    
    self.assertTrue(obj.hasAttribute("how/startazT"))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), obj.getAttribute("how/startazT"), 4))    

  def test_addHowStopazT(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/stopazT", numpy.array([1.0,2.0,3.0]))
    self.assertEqual("how/stopazT", result[0])
    self.assertTrue(obj.hasAttribute("how/stopazT"))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), obj.getAttribute("how/stopazT"), 4))    
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), obj.getAttribute("how/stopT"), 4))    

  def test_addHowGasattn_2_3(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/gasattn", 1.0, _rave.RaveIO_ODIM_Version_2_3) # dB/km => dB/m
    self.assertEqual("how/gasattn", result[0])
    self.assertAlmostEqual(1.0, result[1], 4)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(0.001, obj.getAttribute("how/gasattn", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowGasattn_2_4(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/gasattn", 1.0, _rave.RaveIO_ODIM_Version_2_4)
    self.assertEqual("how/gasattn", result[0])
    self.assertAlmostEqual(1000.0, result[1], 4)
    self.assertAlmostEqual(1000.0, obj.getAttribute("how/gasattn"), 4)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/gasattn", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowGasattn_default(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/gasattn", 1.0)
    self.assertEqual("how/gasattn", result[0])
    self.assertAlmostEqual(1.0, result[1], 4)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/gasattn"), 4)

  def test_addHowMinrange_2_3(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/minrange", 1.0, _rave.RaveIO_ODIM_Version_2_3) # km => m
    self.assertEqual("how/minrange", result[0])
    self.assertAlmostEqual(1.0, result[1], 4)
    self.assertAlmostEqual(1000.0, obj.getAttribute("how/minrange", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowMinrange_2_4(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/minrange", 1000.0, _rave.RaveIO_ODIM_Version_2_4)
    self.assertEqual("how/minrange", result[0])
    self.assertAlmostEqual(1.0, result[1], 4)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/minrange"), 4)
    self.assertAlmostEqual(1000.0, obj.getAttribute("how/minrange", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowMinrange_default(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/minrange", 1.0)
    self.assertEqual("how/minrange", result[0])
    self.assertAlmostEqual(1.0, result[1], 4)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/minrange"), 4)

  def test_addHowMaxrange_2_3(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/maxrange", 1.0, _rave.RaveIO_ODIM_Version_2_3) # km => m
    self.assertEqual("how/maxrange", result[0])
    self.assertAlmostEqual(1.0, result[1], 4)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/maxrange", _rave.RaveIO_ODIM_Version_2_3), 4)
    self.assertAlmostEqual(1000.0, obj.getAttribute("how/maxrange", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowMaxrange_2_4(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/maxrange", 1000.0, _rave.RaveIO_ODIM_Version_2_4)
    self.assertEqual("how/maxrange", result[0])
    self.assertAlmostEqual(1.0, result[1], 4)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/maxrange"), 4)
    self.assertAlmostEqual(1000.0, obj.getAttribute("how/maxrange", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowMaxrange_default(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/maxrange", 1.0)
    self.assertEqual("how/maxrange", result[0])
    self.assertAlmostEqual(1.0, result[1], 4)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/maxrange"), 4)

  def test_addHowRadhoriz_2_3(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/radhoriz", 1.0, _rave.RaveIO_ODIM_Version_2_3) # km => m
    self.assertEqual("how/radhoriz", result[0])
    self.assertAlmostEqual(1.0, result[1], 4)
    self.assertAlmostEqual(1, obj.getAttribute("how/radhoriz", _rave.RaveIO_ODIM_Version_2_3), 4)
    self.assertAlmostEqual(1000.0, obj.getAttribute("how/radhoriz", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowRadhoriz_2_4(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/radhoriz", 1000.0, _rave.RaveIO_ODIM_Version_2_4)
    self.assertEqual("how/radhoriz", result[0])
    self.assertAlmostEqual(1.0, result[1], 4)
    self.assertAlmostEqual(1, obj.getAttribute("how/radhoriz", _rave.RaveIO_ODIM_Version_2_3), 4)
    self.assertAlmostEqual(1000.0, obj.getAttribute("how/radhoriz", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowRadhoriz_default(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/radhoriz", 1.0)
    self.assertEqual("how/radhoriz", result[0])
    self.assertAlmostEqual(1.0, result[1], 4)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/radhoriz"), 4)

  def test_addHowNomTXpower_2_3(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/nomTXpower", 370.1008, _rave.RaveIO_ODIM_Version_2_3) # kw => dBm
    self.assertEqual("how/nomTXpower", result[0])
    self.assertAlmostEqual(370.1008, result[1], 4)
    self.assertAlmostEqual(370.1008, obj.getAttribute("how/nomTXpower", _rave.RaveIO_ODIM_Version_2_3), 4)
    self.assertAlmostEqual(85.6832, obj.getAttribute("how/nomTXpower", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowNomTXpower_2_4(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/nomTXpower", 85.6832, _rave.RaveIO_ODIM_Version_2_4)
    self.assertEqual("how/nomTXpower", result[0])
    self.assertAlmostEqual(370.1008, result[1], 4)
    self.assertAlmostEqual(370.1008, obj.getAttribute("how/nomTXpower", _rave.RaveIO_ODIM_Version_2_3), 4)
    self.assertAlmostEqual(85.6832, obj.getAttribute("how/nomTXpower", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowNomTXpower_default(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/nomTXpower", 370.1008)
    self.assertEqual("how/nomTXpower", result[0])
    self.assertAlmostEqual(370.1008, result[1], 4)
    self.assertAlmostEqual(370.1008, obj.getAttribute("how/nomTXpower"), 4)

  def test_addHowPeakpwr_2_3(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/peakpwr", 370.1008, _rave.RaveIO_ODIM_Version_2_3) # kw => dBm
    self.assertEqual("how/peakpwr", result[0])
    self.assertAlmostEqual(370.1008, result[1], 4)
    self.assertAlmostEqual(370.1008, obj.getAttribute("how/peakpwr", _rave.RaveIO_ODIM_Version_2_3), 4)
    self.assertAlmostEqual(85.6832, obj.getAttribute("how/peakpwr", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowPeakpwr_2_4(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/peakpwr", 85.6832, _rave.RaveIO_ODIM_Version_2_4)
    self.assertEqual("how/peakpwr", result[0])
    self.assertAlmostEqual(370.1008, result[1], 4)
    self.assertAlmostEqual(370.1008, obj.getAttribute("how/peakpwr", _rave.RaveIO_ODIM_Version_2_3), 4)
    self.assertAlmostEqual(85.6832, obj.getAttribute("how/peakpwr", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowPeakpwr_default(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/peakpwr", 370.1008)
    self.assertEqual("how/peakpwr", result[0])
    self.assertAlmostEqual(370.1008, result[1], 4)
    self.assertAlmostEqual(370.1008, obj.getAttribute("how/peakpwr"), 4)

  def test_addHowAvgpwr_2_3(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/avgpwr", 370.1008, _rave.RaveIO_ODIM_Version_2_3) # kw => dBm
    self.assertEqual("how/avgpwr", result[0])
    self.assertAlmostEqual(370.1008, result[1], 4)
    self.assertAlmostEqual(370.1008, obj.getAttribute("how/avgpwr", _rave.RaveIO_ODIM_Version_2_3), 4)
    self.assertAlmostEqual(85.6832, obj.getAttribute("how/avgpwr", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowAvgpwr_2_4(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/avgpwr", 85.6832, _rave.RaveIO_ODIM_Version_2_4)
    self.assertEqual("how/avgpwr", result[0])
    self.assertAlmostEqual(370.1008, result[1], 4)
    self.assertAlmostEqual(370.1008, obj.getAttribute("how/avgpwr", _rave.RaveIO_ODIM_Version_2_3), 4)
    self.assertAlmostEqual(85.6832, obj.getAttribute("how/avgpwr", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowAvgpwr_default(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/avgpwr", 370.1008)
    self.assertEqual("how/avgpwr", result[0])
    self.assertAlmostEqual(370.1008, result[1], 4)
    self.assertAlmostEqual(370.1008, obj.getAttribute("how/avgpwr"), 4)

  def test_addHowPulsewidth_2_3(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/pulsewidth", 1.0, _rave.RaveIO_ODIM_Version_2_3) # ms => s
    self.assertEqual("how/pulsewidth", result[0])
    self.assertAlmostEqual(1.0, result[1], 4)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/pulsewidth", _rave.RaveIO_ODIM_Version_2_3), 4)
    self.assertAlmostEqual(0.0000001, obj.getAttribute("how/pulsewidth", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowPulsewidth_2_4(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/pulsewidth", 1.0, _rave.RaveIO_ODIM_Version_2_4)
    self.assertEqual("how/pulsewidth", result[0])
    self.assertAlmostEqual(1000000.0, result[1], 4)
    self.assertAlmostEqual(1000000.0, obj.getAttribute("how/pulsewidth", _rave.RaveIO_ODIM_Version_2_3), 4)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/pulsewidth", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowPulsewidth_default(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/pulsewidth", 1.0)
    self.assertEqual("how/pulsewidth", result[0])
    self.assertAlmostEqual(1.0, result[1], 4)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/pulsewidth"), 4)

  def test_addHowRXbandwidth_2_3(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/RXbandwidth", 1.0, _rave.RaveIO_ODIM_Version_2_3) # MHz => MHz
    self.assertEqual("how/RXbandwidth", result[0])
    self.assertAlmostEqual(1.0, result[1], 4)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/RXbandwidth", _rave.RaveIO_ODIM_Version_2_3), 4)
    self.assertAlmostEqual(1000000.0, obj.getAttribute("how/RXbandwidth", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowRXbandwidth_2_4(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/RXbandwidth", 1.0, _rave.RaveIO_ODIM_Version_2_4) # Hz => MHz
    self.assertEqual("how/RXbandwidth", result[0])
    self.assertAlmostEqual(0.000001, result[1], 4)
    self.assertAlmostEqual(0.000001, obj.getAttribute("how/RXbandwidth", _rave.RaveIO_ODIM_Version_2_3), 4)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/RXbandwidth", _rave.RaveIO_ODIM_Version_2_4), 4)

  def test_addHowRXbandwidth_default(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/RXbandwidth", 1.0)
    self.assertEqual("how/RXbandwidth", result[0])
    self.assertAlmostEqual(1.0, result[1], 4)
    self.assertAlmostEqual(1.0, obj.getAttribute("how/RXbandwidth"), 4)

  def test_addHowTXPower_2_3(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/TXpower", numpy.array([370.1008,371.1008,372.1008]), _rave.RaveIO_ODIM_Version_2_3) # kW=>kW
    self.assertEqual("how/TXpower", result[0])
    self.assertTrue(numpy.allclose(numpy.array([370.1008,371.1008,372.1008]), result[1], 1e-4))
    self.assertTrue(numpy.allclose(numpy.array([370.1008,371.1008,372.1008]), obj.getAttribute("how/TXpower", _rave.RaveIO_ODIM_Version_2_3), 1e-4))    
    self.assertTrue(numpy.allclose(numpy.array([85.6832, 85.6949, 85.7066]), obj.getAttribute("how/TXpower", _rave.RaveIO_ODIM_Version_2_4), 1e-4))    

  def test_addHowTXPower_2_4(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/TXpower", numpy.array([85.6832, 85.6949, 85.7066]), _rave.RaveIO_ODIM_Version_2_4)
    self.assertEqual("how/TXpower", result[0])
    self.assertTrue(numpy.allclose(numpy.array([370.1008,371.1008,372.1008]), result[1], 1e-4))
    self.assertTrue(numpy.allclose(numpy.array([370.1008,371.1008,372.1008]), obj.getAttribute("how/TXpower", _rave.RaveIO_ODIM_Version_2_3), 1e-4))    
    self.assertTrue(numpy.allclose(numpy.array([85.6832, 85.6949, 85.7066]), obj.getAttribute("how/TXpower", _rave.RaveIO_ODIM_Version_2_4), 1e-4))    

  def test_addHowTXPower_default(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/TXpower", numpy.array([370.1008,371.1008,372.1008]))
    self.assertEqual("how/TXpower", result[0])
    self.assertTrue(numpy.allclose(numpy.array([370.1008,371.1008,372.1008]), result[1], 1e-4))
    self.assertTrue(numpy.allclose(numpy.array([370.1008,371.1008,372.1008]), obj.getAttribute("how/TXpower"), 1e-4))    

  def test_addHowWavelength_2_3(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/wavelength", 5.34866, _rave.RaveIO_ODIM_Version_2_3)
    self.assertEqual("how/wavelength", result[0])
    self.assertAlmostEqual(5.34866, result[1], 4)
    self.assertAlmostEqual(5.34866, obj.getAttribute("how/wavelength", _rave.RaveIO_ODIM_Version_2_3), 4)    
    self.assertAlmostEqual(5605001215.2577, obj.getAttribute("how/frequency", _rave.RaveIO_ODIM_Version_2_4), 4)    

  def test_addHowWavelength_2_4(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/frequency", 5605001215.2577, _rave.RaveIO_ODIM_Version_2_4)
    self.assertEqual("how/wavelength", result[0])
    self.assertAlmostEqual(5.34866, result[1], 4)
    self.assertAlmostEqual(5.34866, obj.getAttribute("how/wavelength", _rave.RaveIO_ODIM_Version_2_3), 4)    
    self.assertAlmostEqual(5.34866, obj.getAttribute("how/wavelength", _rave.RaveIO_ODIM_Version_2_4), 4)    
    self.assertAlmostEqual(5605001215.2577, obj.getAttribute("how/frequency", _rave.RaveIO_ODIM_Version_2_4), 4)    

  def test_addHowWavelength_default(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/wavelength", 5.34866)
    self.assertEqual("how/wavelength", result[0])
    self.assertAlmostEqual(5.34866, result[1], 4)
    self.assertAlmostEqual(5.34866, obj.getAttribute("how/wavelength"), 4)    
 
  def test_addHowMeltingLayerTop_2_3(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/melting_layer_top", numpy.array([1.0, 2.0, 3.0]), _rave.RaveIO_ODIM_Version_2_3)
    self.assertEqual("how/melting_layer_top", result[0])
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), result[1], 1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), obj.getAttribute("how/melting_layer_top", _rave.RaveIO_ODIM_Version_2_3), 1e-1))    
    self.assertTrue(numpy.allclose(numpy.array([1000.0, 2000.0, 3000.0]), obj.getAttribute("how/melting_layer_top_A", _rave.RaveIO_ODIM_Version_2_4), 1e-1))    

  def test_addHowMeltingLayerTop_2_4(self):
    obj = _attributetable.new()
    mlt = obj.addAttribute("how/melting_layer_top", 1.0, _rave.RaveIO_ODIM_Version_2_4)  # This is a bad one.. how/melting_layer_top exists as double in 2.3 and array in 2.4...
    self.assertEqual("how/_melting_layer_top", mlt[0])
    self.assertAlmostEqual(1.0, mlt[1], 4)
    result = obj.addAttribute("how/melting_layer_top_A", numpy.array([1000.0, 2000.0, 3000.0]), _rave.RaveIO_ODIM_Version_2_4)
    self.assertEqual("how/melting_layer_top", result[0])
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), result[1], 1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1000.0, 2000.0, 3000.0]), obj.getAttribute("how/melting_layer_top_A", _rave.RaveIO_ODIM_Version_2_4), 1e-1))    
    self.assertAlmostEqual(1.0, obj.getAttribute("how/melting_layer_top", _rave.RaveIO_ODIM_Version_2_4), 1)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), obj.getAttribute("how/melting_layer_top", _rave.RaveIO_ODIM_Version_2_3), 1e-1))    

  def test_addHowMeltingLayerBottom_2_3(self):
    obj = _attributetable.new()
    result = obj.addAttribute("how/melting_layer_bottom", numpy.array([1.0, 2.0, 3.0]), _rave.RaveIO_ODIM_Version_2_3)
    self.assertEqual("how/melting_layer_bottom", result[0])
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), result[1], 1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), obj.getAttribute("how/melting_layer_bottom", _rave.RaveIO_ODIM_Version_2_3), 1e-1))    
    self.assertTrue(numpy.allclose(numpy.array([1000.0, 2000.0, 3000.0]), obj.getAttribute("how/melting_layer_bottom_A", _rave.RaveIO_ODIM_Version_2_4), 1e-1))    

  def test_addHowMeltingLayerBottom_2_4(self):
    obj = _attributetable.new()
    mlt = obj.addAttribute("how/melting_layer_bottom", 1.0, _rave.RaveIO_ODIM_Version_2_4)  # This is a bad one.. how/melting_layer_top exists as double in 2.3 and array in 2.4...
    self.assertEqual("how/_melting_layer_bottom", mlt[0])
    self.assertAlmostEqual(1.0, mlt[1], 4)
    result = obj.addAttribute("how/melting_layer_bottom_A", numpy.array([1000.0, 2000.0, 3000.0]), _rave.RaveIO_ODIM_Version_2_4)
    self.assertEqual("how/melting_layer_bottom", result[0])
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), result[1], 1e-1))
    self.assertTrue(numpy.allclose(numpy.array([1000.0, 2000.0, 3000.0]), obj.getAttribute("how/melting_layer_bottom_A", _rave.RaveIO_ODIM_Version_2_4), 1e-1))    
    self.assertAlmostEqual(1.0, obj.getAttribute("how/melting_layer_bottom", _rave.RaveIO_ODIM_Version_2_4), 1)
    self.assertTrue(numpy.allclose(numpy.array([1.0, 2.0, 3.0]), obj.getAttribute("how/melting_layer_bottom", _rave.RaveIO_ODIM_Version_2_3), 1e-1))    

  def getValueFromList(self, thelist, name):
      for v in thelist:
        if v[0] == name:
          return v
      return None
  
  def test_getValues_2_3(self):
    obj = _attributetable.new()
    obj.addAttribute("how/wavelength", 5.34866)
    obj.addAttribute("how/pulsewidth", 1.0)
    obj.addAttribute("how/something", 2.0)
    
    result = obj.getValues()
    self.assertEqual(3, len(result))
    v1 = self.getValueFromList(result, "how/wavelength")
    v2 = self.getValueFromList(result, "how/pulsewidth")
    v3 = self.getValueFromList(result, "how/something")
    self.assertAlmostEqual(5.34866, v1[1], 4)
    self.assertAlmostEqual(1.0, v2[1], 4)
    self.assertAlmostEqual(2.0, v3[1], 4)
    
  def test_getValues_2_3_to_2_4(self):
    obj = _attributetable.new()
    obj.addAttribute("how/wavelength", 5.34866)
    obj.addAttribute("how/pulsewidth", 1.0)
    obj.addAttribute("how/something", 2.0)
    
    result = obj.getValues(_rave.RaveIO_ODIM_Version_2_4)
    self.assertEqual(3, len(result))
    v1 = self.getValueFromList(result, "how/frequency")
    v2 = self.getValueFromList(result, "how/pulsewidth")
    v3 = self.getValueFromList(result, "how/something")
    self.assertAlmostEqual(5605001215.2577, v1[1], 4)
    self.assertAlmostEqual(0.000001, v2[1], 4)
    self.assertAlmostEqual(2.0, v3[1], 4)
  
  def test_getAttributeNames_2_3(self):
    obj = _attributetable.new()
    obj.addAttribute("how/wavelength", 5.34866)
    obj.addAttribute("how/pulsewidth", 1.0)
    obj.addAttribute("how/something", 2.0)
    
    result = obj.getAttributeNames()
    self.assertEqual(3, len(result))
    self.assertTrue("how/wavelength" in result)
    self.assertTrue("how/pulsewidth" in result)
    self.assertTrue("how/something" in result)

  def test_getAttributeNames_2_3_to_2_4(self):
    obj = _attributetable.new()
    obj.addAttribute("how/wavelength", 5.34866)
    obj.addAttribute("how/pulsewidth", 1.0)
    obj.addAttribute("how/something", 2.0)
    
    result = obj.getAttributeNames(_rave.RaveIO_ODIM_Version_2_4)
    self.assertEqual(3, len(result))
    self.assertTrue("how/frequency" in result)
    self.assertTrue("how/pulsewidth" in result)
    self.assertTrue("how/something" in result)
