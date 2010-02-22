#!/usr/bin/env python
# -*- coding: latin-1 -*-
# $Id$
# Author: Günther Haase
# Copyright: SMHI, 2010

import _projection, _area
import _rave, _raveio, _polarvolume, _polarscan, _cartesian, _pycomposite
import _pyhl
from numpy import *
from pylab import *
import numpy

def deg2rad(coord):
	return (coord[0]*pi/180.0, coord[1]*pi/180.0)
	
def read_topo():
	for tile in ["W020N90","E020N90"]:
		# extract meta-data
		fnm = "../db/topo/"+tile+"/"+tile+".HDR"
		fid = open(fnm,"r")
		byteorder = fid.readline().split()[1]
		layout = fid.readline().split()[1]
		nrows = int(fid.readline().split()[1])
		ncols = int(fid.readline().split()[1])
		nbands = int(fid.readline().split()[1])
		nbits = int(fid.readline().split()[1])
		bandrowbytes = int(fid.readline().split()[1])
		totalrowbytes = int(fid.readline().split()[1])
		bandgapbytes = int(fid.readline().split()[1])
		nodata = int(fid.readline().split()[1])
		# longitude of the center of the upper-left pixel (decimal degrees)
		ulxmap = float(fid.readline().split()[1])
		# latitude of the center of the upper-left pixel (decimal degrees)
		ulymap = float(fid.readline().split()[1])
		# x dimension of a pixel in geographic units (decimal degrees)
		xdim = float(fid.readline().split()[1])
		# y dimension of a pixel in geographic units (decimal degrees)
		ydim = float(fid.readline().split()[1])
		fid.close()

		# extract topography
		fnm = "../db/topo/"+tile+"/"+tile+".DEM"
		fid = open(fnm,"r")
		data = fid.read()
		data = fromstring(data,short).byteswap()
		data = data.reshape(nrows,ncols)
		print "data type and shape: ",data.dtype,data.shape
		fid.close()

		# longitude of the lower-left corner of the lower-left pixel (decimal degrees)
		llxmap = ulxmap-xdim/2
		# latitude of the lower-left corner of the lower-left pixel (decimal degrees)
		llymap = (ulymap-ydim/2) - ((nrows-1)*ydim) 
		# longitude of the lower-left corner of the upper-right pixel (decimal degrees)
		urxmap = llxmap + ((ncols-1)*xdim)
		# latitude of the lower-left corner of the upper-right pixel (decimal degrees)
		urymap = llymap + ((nrows-1)*ydim)
		
		a = _area.new()
		a.id = tile
		a.projection = _projection.new("longlat_wgs84", "Plate Caree WGS84", "+proj=eqc +ellps=WGS84 +datum=WGS84")
		(llx,lly) = a.projection.fwd(deg2rad((llxmap,llymap)))
		(urx,ury) = a.projection.fwd(deg2rad((urxmap,urymap)))
		a.extent = (llx,lly,urx,ury)
		a.xsize = ncols
		a.ysize = nrows
		a.xscale = (urx-llx)/(ncols-1)
		a.yscale = (ury-lly)/(nrows-1)
		print (llxmap,llymap,urxmap,urymap)
		print a.extent
		print a.xscale,a.yscale
		
		src = _cartesian.new()
		src.init(a,_rave.RaveDataType_SHORT)
		src.time = "000000"
		src.date = "20100101"
		src.objectType = _rave.Rave_ObjectType_IMAGE
		src.product = _rave.Rave_ProductType_COMP
		src.source = "GTOPO30 topography"
		src.quantity = "TOPO"
		src.gain = 1.0
		src.offset = 0.0
		src.nodata = -9999.0
		src.setData(data)
		
		ios = _raveio.new()
		ios.object = src
		ios.filename = "./"+tile+".h5"
		ios.save()
	
def concatenate_topo():
	ios1 = _raveio.open("./W020N90.h5")
	src1 = ios1.object
	data1 = src1.getData()
	ios2 = _raveio.open("./E020N90.h5")
	src2 = ios2.object
	data2 = src2.getData()
	#data2 = numpy.zeros((6000,4800), numpy.int16)
	a1 = array([[1,2,3],[4,5,6],[7,8,9]],numpy.int16)
	data = concatenate((a1, a1), 1)
	print concatenate((a1, a1), 1)
	print data
	
	a = _area.new()
	a.id = "BALTRAD"
	a.projection = src1.projection
	a.extent = (src1.areaextent[0],src1.areaextent[1],src2.areaextent[2],src2.areaextent[3])
	a.xsize = src1.xsize+src2.xsize
	a.ysize = src1.ysize
	a.xscale = (a.extent[2]-a.extent[0])/(a.xsize-1)
	a.yscale = (a.extent[3]-a.extent[1])/(a.ysize-1)
	print a.id
	print a.projection.definition	
	print a.extent
	print a.xsize,a.ysize
	print a.xscale,a.yscale

	src = _cartesian.new()
	src.init(a,_rave.RaveDataType_SHORT)
	src.time = src1.time
	src.date = src1.date
	src.objectType = src1.objectType
	src.product = src1.product
	src.source = src1.source
	src.quantity = src1.quantity
	src.gain = src1.gain
	src.offset = src1.offset
	src.nodata = src1.nodata
	src.setData(data)
	
	ios = _raveio.new()
	ios.object = src
	ios.filename = "./BALTRAD_topo.h5"
	ios.save()
	
	#el = 1.0
	#that = to_pvol(src,el)
	
	#volume = _rave.open("ODIM_H5_pvol_ang_20090501T1200Z.h5_0.h5")
	#volume1 = _raveio.open("ODIM_H5_pvol_ang_20090501T1200Z.h5_0.h5")
	#nodelist = _pyhl.read_nodelist("../test/pytest/fixture_ODIM_cvol_cappi.h5")
	#nodelist.selectAll()
	#nodelist.fetch()
	#print nodelist.getNode("/dataset1/data1/data/CLASS").data()

def to_pvol(self,el):
	import _ctop
	dest = _polarvolume.new()
	scan1 = _polarscan.new()
	scan1.elangle = el
	_ctop.transform(self, dest)
	return dest
	
    # Cartesian to polar transformation.
    #def to_pvol(self, p_areaid, source_elev, method=NEAREST,
	#	radius=0.707106781187):
	#import _ctop # cartesian to polar C module
	#dest = pvol(p_areaid, source_elev)
	#dest.info["place"] = dest.info["name"]
	#if dest.info["id"][-1:]=="n":
	#    dest.info["doppler"]="F"
	#else:
	#    dest.info["doppler"]="T"
	#if self.info["nodata"]: dest.info["nodata"] = self.info["nodata"]
	#dest.info["cressman_xy"] = radius
	#dest.info["i_method"] = method
	#src_pcs = pcs.pcs(self.info["pcs"])
	#if self.data.typecode() != "b":
	#    for scan in range(len(dest.data)):
	#	dest.data[scan] = dest.data[scan].astype(self.data.typecode())
	#_ctop.transform(self, dest)
	#return dest 




#	for line in fid.readline():
#		print fid.readline()
#		byteorder = line.split()[1]
#		print byteorder
#		stn.append(line.split()[0])
#		xxrad.append(float(line.split()[1]))
#		yyrad.append(float(line.split()[2]))
#		hrad.append(float(line.split()[3]))
#		mcrad.append(float(line.split()[4]))

	


#    if (tile=="W020N90_E020N90"):
#        fd = open("../"+tile[:7]+"/"+tile[:7]+".DEM")
#        data1 = fd.read()
#        fd.close()
#        data1 = fromstring(data1, "s").byteswapped()
#        data1 = reshape(data1, (NROWS, NCOLS/2))

#        fd = open("../"+tile[-7:]+"/"+tile[-7:]+".DEM")
#        data2 = fd.read()
#        fd.close()
#        data2 = fromstring(data2, "s").byteswapped()
#        data2 = reshape(data2, (NROWS, NCOLS/2))

#        data = concatenate((data1, data2), 1)

		#pcs.define("longlat_wgs84", "Plate Caree WGS84", ["proj=eqc","ellps=WGS84"])
		#[(llx,lly),(urx,ury)] = Proj.c2s([(llxmap,llymap),(urxmap,urymap)],"longlat_wgs84")
		#a = area.AREA()
		#a.Id = tile
		#a.name = tile
		#a.pcs = "longlat_wgs84"
		#a.extent = (llx,lly,urx,ury)
		#a.xsize = ncols
		#a.ysize = nrows
		#a.xscale = xdim
		#a.yscale = ydim
		#area.register(a)		
		#print (llxmap,llymap,urxmap,urymap)
		#print (llx,lly,urx,ury)
		#this = rave.RAVE()
		#this.data = data
		##this.info["nodata"] = -9999

		#fd = open("gtopo30_W020N90.dat","w")
		#for i in range(nrows):
		#	for j in range(ncols):
		#		value = data[i][j]
		#		fd.write(`value`)
		#		fd.write(" ")
		#	fd.write("\n")
		#fd.close()
		#data = where(equal(data,nodata),0,data)
		#pcolor(data[5900:6000,1:4800])
		#colorbar()
		#show()

if __name__ == "__main__":
	read_topo()
	concatenate_topo()
