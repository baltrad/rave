/* --------------------------------------------------------------------
Copyright (C) 2012 Institute of Meteorology and Water Management -
National Research Institute, IMGW-PIB

This file is part of Radvol-QC package.

Radvol-QC is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Radvol-QC is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Radvol-QC.  If not, see <http://www.gnu.org/licenses/>.
------------------------------------------------------------------------*/
/**
 * Tests for radvol.
 * @file testRadvol.c
 * @author Katarzyna Osrodka (Institute of Meteorology and Water Management, IMGW-PIB)
 * @date 2012-10-15
 */

#include <stdio.h>
#include <stdlib.h>
#include "CUnit/Basic.h"
#include "radvol.h"
#include "rave_io.h"
#include "rave_alloc.h"
#include "rave_attribute.h"
#include "polarvolume.h"
#include "polarscan.h"
#include "polarscanparam.h"

/*
 * CUnit Test Suite
 */

char* XML_FILE;
char* H5_FILE;
static FILE* temp_file = NULL;

int init_suite_testRadvol(void) {

  XML_FILE = "fixtures/radvol_params.xml";
  if (NULL == (temp_file = fopen(XML_FILE, "r"))) {
    return -1;
  } else {
    fclose(temp_file);
  }
  H5_FILE = "fixtures/fake_att.h5";
  if (NULL == (temp_file = fopen(H5_FILE, "r"))) {
    return -1;
  } else {
    fclose(temp_file);
  }
  return 0;
}

int clean_suite_testRadvol(void) {
  return 0;
}

void testRadvol_getCurvature(void) {
  Radvol_t* self = NULL;
  PolarVolume_t* pvol = NULL;
  RaveIO_t* raveio = NULL;
  double result;

  raveio = RaveIO_open(H5_FILE);
  CU_ASSERT_PTR_NOT_NULL_FATAL(raveio);

  pvol = (PolarVolume_t*) RaveIO_getObject(raveio);
  CU_ASSERT_PTR_NOT_NULL_FATAL(pvol);

  self = RAVE_OBJECT_NEW(&Radvol_TYPE);
  result = Radvol_load_pvol(self, pvol);
  CU_ASSERT_FATAL(result);
  CU_ASSERT_PTR_NOT_NULL_FATAL(self);
  self->TabElev[0].rscale = 0.0;
  result = Radvol_getCurvature(self, 0, -10);
  CU_ASSERT_EQUAL(result, 0);
  self->TabElev[0].rscale = 1.0;
  result = Radvol_getCurvature(self, 0, 20);
  CU_ASSERT_DOUBLE_EQUAL(result, 0.025963, 0.00001);
  self->TabElev[0].elangle = 0.017453;
  result = Radvol_getCurvature(self, 0, 20);
  CU_ASSERT_DOUBLE_EQUAL(result, 0.392448, 0.00001);
  CU_ASSERT_DOUBLE_EQUAL(self->Eer, 8493.0, 0.001);
  Radvol_setEquivalentEarthRadius(self, PolarVolume_getLatitude(pvol));
  CU_ASSERT_DOUBLE_EQUAL(self->Eer, 8452.203, 0.001);
  result = Radvol_getCurvature(self, 0, 20);
  CU_ASSERT_DOUBLE_EQUAL(result, 0.392573, 0.00001);

  RAVE_OBJECT_RELEASE(self);
  RAVE_OBJECT_RELEASE(pvol);
  RAVE_OBJECT_RELEASE(raveio);
}

void testRadvol_getFactorChild(void) {
  SimpleXmlNode_t* node = NULL;
  int IsDefault;
  Radvol_t* self = NULL;
  self = RAVE_OBJECT_NEW(&Radvol_TYPE);

  node = Radvol_getFactorChild(NULL, "nonexisting.xml", "nothing", &IsDefault);
  CU_ASSERT_PTR_NULL(node);
  node = Radvol_getFactorChild(NULL, XML_FILE, "nothing", &IsDefault);
  CU_ASSERT_PTR_NOT_NULL(node);
  CU_ASSERT_EQUAL(IsDefault, 1);
  RAVE_OBJECT_RELEASE(node);
  Radvol_getName(self,"NOD:plram");
  node = Radvol_getFactorChild(self, XML_FILE, "nothing", &IsDefault);
  CU_ASSERT_PTR_NOT_NULL(node);
  CU_ASSERT_EQUAL(IsDefault, 1);
  RAVE_OBJECT_RELEASE(node);
  node = Radvol_getFactorChild(self, XML_FILE, "ATT_QIOn", &IsDefault);
  CU_ASSERT_PTR_NOT_NULL(node);
  CU_ASSERT_EQUAL(IsDefault, 0);

  RAVE_OBJECT_RELEASE(node);
  RAVE_OBJECT_RELEASE(self);
}

void testRadvol_getLinearQuality(void) {
  double x, a, b;
  double result;

  x = 0.5;
  a = 0.0;
  b = 1.0;
  result = Radvol_getLinearQuality(x, a, b);
  CU_ASSERT_DOUBLE_EQUAL(result, 0.5, 0.001);
  x = -0.5;
  result = Radvol_getLinearQuality(x, a, b);
  CU_ASSERT_DOUBLE_EQUAL(result, QI_GOOD, 0.001);
  x = 0.0;
  result = Radvol_getLinearQuality(x, a, b);
  CU_ASSERT_DOUBLE_EQUAL(result, QI_GOOD, 0.001);
  x = 0.01;
  result = Radvol_getLinearQuality(x, a, b);
  CU_ASSERT_DOUBLE_NOT_EQUAL(result, QI_GOOD, 0.001);
  x = 1.5;
  result = Radvol_getLinearQuality(x, a, b);
  CU_ASSERT_DOUBLE_EQUAL(result, QI_BAD, 0.001);
  x = 1.0;
  result = Radvol_getLinearQuality(x, a, b);
  CU_ASSERT_DOUBLE_EQUAL(result, QI_BAD, 0.001);
  x = 0.99;
  result = Radvol_getLinearQuality(x, a, b);
  CU_ASSERT_DOUBLE_NOT_EQUAL(result, QI_BAD, 0.001);
  x = 0.0;
  a = 0.0;
  b = 0.0;
  result = Radvol_getLinearQuality(x, a, b);
  CU_ASSERT_DOUBLE_EQUAL(result, QI_GOOD, 0.001);
}

void testRadvol_getParValueDouble(void) {
  SimpleXmlNode_t* node = NULL;
  double value;
  int IsDefault;
  int result;

  node = Radvol_getFactorChild(NULL, XML_FILE, "ATT", &IsDefault);
  CU_ASSERT_PTR_NOT_NULL_FATAL(node);

  result = Radvol_getParValueDouble(node, "ATT_ZRb1", &value);
  CU_ASSERT_FALSE(result);

  result = Radvol_getParValueDouble(node, "ATT_ZRb", &value);
  CU_ASSERT_TRUE_FATAL(result);
  CU_ASSERT_DOUBLE_EQUAL(value, 1.6, 0.001);
  RAVE_OBJECT_RELEASE(node);
}

void testRadvol_getParValueInt(void) {
  SimpleXmlNode_t* node = NULL;
  int value;
  int IsDefault;
  int result;

  node = Radvol_getFactorChild(NULL, XML_FILE, "ATT", &IsDefault);
  CU_ASSERT_PTR_NOT_NULL_FATAL(node);
  result = Radvol_getParValueInt(node, "ATT_QIOn1", &value);
  CU_ASSERT_FALSE(result);

  result = Radvol_getParValueInt(node, "ATT_QIOn", &value);
  CU_ASSERT_TRUE_FATAL(result);
  CU_ASSERT_EQUAL(value, 1);
  RAVE_OBJECT_RELEASE(node);
}

void testRadvol_load_pvol(void) {
  Radvol_t* self = NULL;
  PolarVolume_t* pvol = NULL;
  RaveIO_t* raveio = NULL;
  int result;

  raveio = RaveIO_open(H5_FILE);
  CU_ASSERT_PTR_NOT_NULL_FATAL(raveio);

  pvol = (PolarVolume_t*) RaveIO_getObject(raveio);
  CU_ASSERT_PTR_NOT_NULL_FATAL(pvol);

  self = RAVE_OBJECT_NEW(&Radvol_TYPE);
  result = Radvol_load_pvol(self, pvol);
  CU_ASSERT_FATAL(result);
  CU_ASSERT_PTR_NOT_NULL_FATAL(self);
  CU_ASSERT_EQUAL(self->nele, 1);
  CU_ASSERT_PTR_NOT_NULL_FATAL(self->TabElev);
  CU_ASSERT_PTR_NOT_NULL_FATAL(self->TabElev[0].ReflElev);
  CU_ASSERT_DOUBLE_EQUAL(self->TabElev[0].ReflElev[50], 20.5, 0.001);

  RAVE_OBJECT_RELEASE(self);
  RAVE_OBJECT_RELEASE(pvol);
  RAVE_OBJECT_RELEASE(raveio);
}

void testRadvol_save_pvol(void) {
  Radvol_t* self = NULL;
  PolarVolume_t* pvol = NULL;
  PolarScan_t* scan = NULL;
  RaveIO_t* raveio = NULL;
  int result;

  raveio = RaveIO_open(H5_FILE);
  CU_ASSERT_PTR_NOT_NULL_FATAL(raveio);

  pvol = (PolarVolume_t*) RaveIO_getObject(raveio);
  CU_ASSERT_PTR_NOT_NULL_FATAL(pvol);

  scan = PolarVolume_getScan(pvol, 0);
  CU_ASSERT_PTR_NOT_NULL_FATAL(scan);
  CU_ASSERT_FALSE_FATAL(PolarScan_hasParameter(scan, "TH"));
  RAVE_OBJECT_RELEASE(scan);

  self = RAVE_OBJECT_NEW(&Radvol_TYPE);
  result = Radvol_load_pvol(self, pvol);
  CU_ASSERT_FATAL(result);
  CU_ASSERT_PTR_NOT_NULL_FATAL(self->TabElev);
  CU_ASSERT_PTR_NOT_NULL_FATAL(self->TabElev[0].ReflElev);

  result = Radvol_save_pvol(self, pvol);
  CU_ASSERT_FATAL(result);
  CU_ASSERT_EQUAL_FATAL(PolarVolume_getNumberOfScans(pvol), 1);
  scan = PolarVolume_getScan(pvol, 0);
  CU_ASSERT_PTR_NOT_NULL_FATAL(scan);
  CU_ASSERT(PolarScan_hasParameter(scan, "TH"));
  CU_ASSERT_EQUAL(PolarScan_getNumberOfQualityFields(scan), 0);
  RAVE_OBJECT_RELEASE(scan);

  Radvol_setTaskArgs(self, NULL);
  Radvol_setTaskName(self, "task name");
  result = Radvol_save_pvol(self, pvol);
  CU_ASSERT_FATAL(result);
  CU_ASSERT_EQUAL_FATAL(PolarVolume_getNumberOfScans(pvol), 1);
  scan = PolarVolume_getScan(pvol, 0);
  CU_ASSERT_PTR_NOT_NULL_FATAL(scan);
  CU_ASSERT_EQUAL(PolarScan_getNumberOfQualityFields(scan), 0);
  RAVE_OBJECT_RELEASE(scan);

  Radvol_setTaskArgs(self, "test args");
  Radvol_setTaskName(self, NULL);
  result = Radvol_save_pvol(self, pvol);
  CU_ASSERT_FATAL(result);
  CU_ASSERT_EQUAL_FATAL(PolarVolume_getNumberOfScans(pvol), 1);
  scan = PolarVolume_getScan(pvol, 0);
  CU_ASSERT_PTR_NOT_NULL_FATAL(scan);
  CU_ASSERT_EQUAL(PolarScan_getNumberOfQualityFields(scan), 0);
  RAVE_OBJECT_RELEASE(scan);

  Radvol_setTaskArgs(self, "test args");
  Radvol_setTaskName(self, "task name");
  result = Radvol_save_pvol(self, pvol);
  CU_ASSERT_FATAL(result);
  CU_ASSERT_EQUAL_FATAL(PolarVolume_getNumberOfScans(pvol), 1);
  scan = PolarVolume_getScan(pvol, 0);
  CU_ASSERT_PTR_NOT_NULL_FATAL(scan);
  CU_ASSERT_EQUAL(PolarScan_getNumberOfQualityFields(scan), 1);

  RAVE_OBJECT_RELEASE(pvol);
  RAVE_OBJECT_RELEASE(scan);
  RAVE_OBJECT_RELEASE(self);
  RAVE_OBJECT_RELEASE(raveio);
}

void testRadvol_setTaskArgs(void) {
  Radvol_t* self = NULL;
  PolarVolume_t* pvol = NULL;
  PolarScan_t* scan = NULL;
  RaveIO_t* raveio = NULL;
  PolarScanParam_t* parameter = NULL;
  RaveAttribute_t* attribute = NULL;
  int result;
  char* value = NULL;

  raveio = RaveIO_open(H5_FILE);
  CU_ASSERT_PTR_NOT_NULL_FATAL(raveio);
  pvol = (PolarVolume_t*) RaveIO_getObject(raveio);
  CU_ASSERT_PTR_NOT_NULL_FATAL(pvol);

  self = RAVE_OBJECT_NEW(&Radvol_TYPE);
  result = Radvol_load_pvol(self, pvol);
  CU_ASSERT_FATAL(result);

  Radvol_setTaskName(self, "task name");
  Radvol_setTaskArgs(self, "test args");
  result = Radvol_save_pvol(self, pvol);
  CU_ASSERT_FATAL(result);
  CU_ASSERT_EQUAL_FATAL(PolarVolume_getNumberOfScans(pvol), 1);
  scan = PolarVolume_getScan(pvol, 0);
  CU_ASSERT_PTR_NOT_NULL_FATAL(scan);
  parameter = PolarScan_getParameter(scan, "DBZH");
  CU_ASSERT_PTR_NOT_NULL_FATAL(parameter);
  attribute = PolarScanParam_getAttribute(parameter, "how/task");
  CU_ASSERT_PTR_NOT_NULL_FATAL(attribute);
  RaveAttribute_getString(attribute, &value);
  CU_ASSERT_STRING_EQUAL(value, "task name");

  RAVE_OBJECT_RELEASE(pvol);
  RAVE_OBJECT_RELEASE(scan);
  RAVE_OBJECT_RELEASE(parameter);
  RAVE_OBJECT_RELEASE(attribute);
  RAVE_OBJECT_RELEASE(self);
  RAVE_OBJECT_RELEASE(raveio);
}

void testRadvol_setTaskName(void) {
  Radvol_t* self = NULL;
  PolarVolume_t* pvol = NULL;
  PolarScan_t* scan = NULL;
  RaveIO_t* raveio = NULL;
  PolarScanParam_t* parameter = NULL;
  RaveAttribute_t* attribute = NULL;
  int result;
  char* value = NULL;

  raveio = RaveIO_open(H5_FILE);
  CU_ASSERT_PTR_NOT_NULL_FATAL(raveio);
  pvol = (PolarVolume_t*) RaveIO_getObject(raveio);
  CU_ASSERT_PTR_NOT_NULL_FATAL(pvol);

  self = RAVE_OBJECT_NEW(&Radvol_TYPE);
  result = Radvol_load_pvol(self, pvol);
  CU_ASSERT_FATAL(result);

  Radvol_setTaskName(self, "task name");
  Radvol_setTaskArgs(self, "test args");
  result = Radvol_save_pvol(self, pvol);
  CU_ASSERT_FATAL(result);
  CU_ASSERT_EQUAL_FATAL(PolarVolume_getNumberOfScans(pvol), 1);
  scan = PolarVolume_getScan(pvol, 0);
  CU_ASSERT_PTR_NOT_NULL_FATAL(scan);
  parameter = PolarScan_getParameter(scan, "DBZH");
  CU_ASSERT_PTR_NOT_NULL_FATAL(parameter);
  attribute = PolarScanParam_getAttribute(parameter, "how/task_args");
  CU_ASSERT_PTR_NOT_NULL_FATAL(attribute);
  RaveAttribute_getString(attribute, &value);
  CU_ASSERT_STRING_EQUAL(value, "test args");

  RAVE_OBJECT_RELEASE(pvol);
  RAVE_OBJECT_RELEASE(scan);
  RAVE_OBJECT_RELEASE(parameter);
  RAVE_OBJECT_RELEASE(attribute);
  RAVE_OBJECT_RELEASE(self);
  RAVE_OBJECT_RELEASE(raveio);
}

int testRadvol_main(void) {
  CU_pSuite pSuite = NULL;


  /* Add a suite to the registry */
  pSuite = CU_add_suite("testRadvol", init_suite_testRadvol, clean_suite_testRadvol);
  if (NULL == pSuite) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  /* Add the tests to the suite */
  if ((NULL == CU_add_test(pSuite, "testRadvol_getCurvature", testRadvol_getCurvature)) ||
          (NULL == CU_add_test(pSuite, "testRadvol_getFactorChild", testRadvol_getFactorChild)) ||
          (NULL == CU_add_test(pSuite, "testRadvol_getLinearQuality", testRadvol_getLinearQuality)) ||
          (NULL == CU_add_test(pSuite, "testRadvol_getParValueDouble", testRadvol_getParValueDouble)) ||
          (NULL == CU_add_test(pSuite, "testRadvol_getParValueInt", testRadvol_getParValueInt)) ||
          (NULL == CU_add_test(pSuite, "testRadvol_load_pvol", testRadvol_load_pvol)) ||
          (NULL == CU_add_test(pSuite, "testRadvol_save_pvol", testRadvol_save_pvol)) ||
          (NULL == CU_add_test(pSuite, "testRadvol_setTaskArgs", testRadvol_setTaskArgs)) ||
          (NULL == CU_add_test(pSuite, "testRadvol_setTaskName", testRadvol_setTaskName))) {
    CU_cleanup_registry();
    return CU_get_error();
  }
  return 0;
}