'''
Copyright (C) 2010- Swedish Meteorological and Hydrological Institute (SMHI)

This file is part of RAVE.

RAVE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RAVE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with RAVE.  If not, see <http://www.gnu.org/licenses/>.
'''
# """Regression tests for temporary-directory initialization (issue #106)."""

## @file
## @author Isac Einarsson, SMHI
## @date 2026-11-16

import importlib.util
import os
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock


class rave_tempfile_test(unittest.TestCase):
  def setUp(self):
    self.workspace = tempfile.TemporaryDirectory(prefix="rave-tempfile-test-", dir="/tmp")
    self.addCleanup(self.workspace.cleanup)
    self.tempdir = os.path.join(self.workspace.name, "tmp")
    self.libdir = os.path.join(self.workspace.name, "Lib")
    os.mkdir(self.libdir)
    self.source = os.path.abspath(os.path.join(
      os.path.dirname(__file__), "..", "..", "Lib", "rave_tempfile.py"))

  def import_tempfile(self):
    rave = types.ModuleType("rave")
    rave.__file__ = os.path.join(self.libdir, "rave.py")
    defines = types.ModuleType("rave_defines")
    real_isdir = os.path.isdir

    def isdir(path):
      if path == "/dev/shm":
        return False
      return real_isdir(path)

    spec = importlib.util.spec_from_file_location("_rave_tempfile_test_subject", self.source)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"rave": rave, "rave_defines": defines}), \
         mock.patch.object(os.path, "isdir", side_effect=isdir), \
         mock.patch.object(tempfile, "tempdir", tempfile.tempdir):
      spec.loader.exec_module(module)

      fd, filename = module.mktemp(suffix=".h5", close="True")
      self.assertTrue(os.path.isfile(filename))
      self.assertEqual(self.tempdir, os.path.realpath(os.path.dirname(filename)))
      os.unlink(filename)
    return module

  def test_import_creates_missing_directory(self):
    module = self.import_tempfile()
    self.assertEqual(self.tempdir, os.path.realpath(module.RAVETEMP))
    self.assertEqual(0o777, stat.S_IMODE(os.stat(self.tempdir).st_mode))

  def test_import_accepts_existing_directory(self):
    os.mkdir(self.tempdir)
    self.import_tempfile()

  def test_import_directory_created_between_check_and_mkdir(self):
    real_mkdir = os.mkdir
    competing_creations = []

    def mkdir(path, mode=0o777, *, dir_fd=None):
      if os.path.abspath(path) == self.tempdir and not competing_creations:
        real_mkdir(path, mode, dir_fd=dir_fd)
        competing_creations.append(path)
      return real_mkdir(path, mode, dir_fd=dir_fd)

    with mock.patch.object(os, "mkdir", side_effect=mkdir):
      self.import_tempfile()
    self.assertEqual(1, len(competing_creations))
    self.assertEqual(0o777, stat.S_IMODE(os.stat(self.tempdir).st_mode))

  def test_import_rejects_existing_regular_file(self):
    with open(self.tempdir, "w") as output:
      output.write("not a directory")
    with self.assertRaises(FileExistsError):
      self.import_tempfile()
    with open(self.tempdir) as source:
      self.assertEqual("not a directory", source.read())


if __name__ == "__main__":
  unittest.main()
