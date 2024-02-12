'''
Created on Feb 10, 2024

@author: anders
'''
import unittest
import _ravelegend
import string
import numpy

class PyRaveLegendTest(unittest.TestCase):
  def setUp(self):
    pass

  def tearDown(self):
    pass

  def test_new(self):
    obj = _ravelegend.new()
    self.assertNotEqual(-1, str(type(obj)).find("RaveLegendCore"))

  def test_legend(self):
    obj = _ravelegend.new()
    obj.legend = [
        ("NONE", "0"),
        ("GROUNDCLUTTER", "1"),
        ("SEACLUTTER", "2")
    ]

    self.assertEqual(("NONE", "0"), obj.legend[0])
    self.assertEqual(("GROUNDCLUTTER", "1"), obj.legend[1])
    self.assertEqual(("SEACLUTTER", "2"), obj.legend[2])

  def test_addValue(self):
    obj = _ravelegend.new()
    obj.addValue("NONE", "0")
    obj.addValue("GROUNDCLUTTER", "1")
    obj.addValue("SEACLUTTER", "2")

    self.assertEqual(("NONE", "0"), obj.legend[0])
    self.assertEqual(("GROUNDCLUTTER", "1"), obj.legend[1])
    self.assertEqual(("SEACLUTTER", "2"), obj.legend[2])

  def test_size(self):
    obj = _ravelegend.new()
    self.assertEqual(0, obj.size())

    obj.addValue("NONE", "0")
    self.assertEqual(1, obj.size())

    obj.addValue("GROUNDCLUTTER", "1")
    self.assertEqual(2, obj.size())

  def test_exists_1(self):
    obj = _ravelegend.new()
    self.assertFalse(obj.exists("NONE"))
    self.assertFalse(obj.exists(""))

  def test_exists_2(self):
    obj = _ravelegend.new()
    obj.legend = [
        ("NONE", "0"),
        ("GROUNDCLUTTER", "1"),
        ("SEACLUTTER", "2")
    ]

    self.assertTrue(obj.exists("NONE"))
    self.assertTrue(obj.exists("GROUNDCLUTTER"))
    self.assertTrue(obj.exists("SEACLUTTER"))

    self.assertFalse(obj.exists(" NONE"))
    self.assertFalse(obj.exists("GROUND CLUTTER"))
    self.assertFalse(obj.exists("SEACLUTTER "))

  def test_getValue(self):
    obj = _ravelegend.new()
    obj.legend = [
        ("NONE", "0"),
        ("GROUNDCLUTTER", "1"),
        ("SEACLUTTER", "2")
    ]

    self.assertEqual("0", obj.getValue("NONE"))
    self.assertEqual("1", obj.getValue("GROUNDCLUTTER"))
    self.assertEqual("2", obj.getValue("SEACLUTTER"))

    try:
        obj.getValue("NONEXISTING")
        self.fail("NO")
    except KeyError:
        pass

  def test_getValueAt(self):
    obj = _ravelegend.new()
    obj.legend = [
        ("NONE", "0"),
        ("GROUNDCLUTTER", "1"),
        ("SEACLUTTER", "2")
    ]

    self.assertEqual("0", obj.getValueAt(0))
    self.assertEqual("1", obj.getValueAt(1))
    self.assertEqual("2", obj.getValueAt(2))

    try:
        obj.getValueAt(3)
        self.fail("NO")
    except IndexError:
        pass

  def test_getNameAt(self):
    obj = _ravelegend.new()
    obj.legend = [
        ("NONE", "0"),
        ("GROUNDCLUTTER", "1"),
        ("SEACLUTTER", "2")
    ]

    self.assertEqual("NONE", obj.getNameAt(0))
    self.assertEqual("GROUNDCLUTTER", obj.getNameAt(1))
    self.assertEqual("SEACLUTTER", obj.getNameAt(2))

    try:
        obj.getNameAt(3)
        self.fail("NO")
    except IndexError:
        pass

  def test_clear(self):
    obj = _ravelegend.new()
    obj.legend = [
        ("NONE", "0"),
        ("GROUNDCLUTTER", "1"),
        ("SEACLUTTER", "2")
    ]
    obj.clear()
    self.assertEqual(0, obj.size())

  def test_remove(self):
    obj = _ravelegend.new()
    obj.legend = [
        ("NONE", "0"),
        ("GROUNDCLUTTER", "1"),
        ("SEACLUTTER", "2")
    ]
    obj.remove("GROUNDCLUTTER")

    self.assertEqual(2, obj.size())
    self.assertEqual("NONE", obj.getNameAt(0))
    self.assertEqual("SEACLUTTER", obj.getNameAt(1))

    obj.remove("SEACLUTTER")
    self.assertEqual(1, obj.size())
    self.assertEqual("NONE", obj.getNameAt(0))

    obj.remove("NONE")
    self.assertEqual(0, obj.size())

  def test_removeAt(self):
    obj = _ravelegend.new()
    obj.legend = [
        ("NONE", "0"),
        ("GROUNDCLUTTER", "1"),
        ("SEACLUTTER", "2")
    ]
    obj.removeAt(1)

    self.assertEqual(2, obj.size())
    self.assertEqual("NONE", obj.getNameAt(0))
    self.assertEqual("SEACLUTTER", obj.getNameAt(1))

    obj.removeAt(1)
    self.assertEqual(1, obj.size())
    self.assertEqual("NONE", obj.getNameAt(0))

    obj.removeAt(0)
    self.assertEqual(0, obj.size())
