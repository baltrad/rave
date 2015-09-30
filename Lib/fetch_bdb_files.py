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
## Utility class for reading files from the bdb database
##
## @file
## @author Anders Henja
## @date 2014-10-08
from baltrad.bdbcommon import expr, oh5
from baltrad.bdbclient import db
import contextlib, shutil
import rave_bdb
import datetime, os

class rave_bdb_file_fetcher(object):
  def __init__(self, bdb):
    self.bdb = bdb
    self.outdir = "."
    self.object = "PVOL"
    self.quantity = "DBZH"
    self.sources = []
    self.fromdt = None
    self.todt = None
    
  def create_query(self):
    qry = expr.eq(expr.attribute("what/object", "string"),expr.literal(self.object))
    qry = expr.and_(qry, expr.eq(expr.attribute("what/quantity", "string"), expr.literal(self.quantity)))
    if self.object in ["COMP", "IMAGE", "CVOL"]:
      if len(self.sources) > 0:
        qry = expr.and_(qry, expr.in_(expr.attribute("what/source:CMT", "string"), self.sources))
    else:
      # _bdb/source_name
      if len(self.sources) > 0:
        qry = expr.and_(qry, expr.in_(expr.attribute("_bdb/source_name", "string"), self.sources))
    
    if not self.fromdt is None:
      qry = expr.and_(qry, expr.le(expr.literal(self.fromdt),
                                   expr.add(expr.attribute("what/date", "date"),
                                            expr.attribute("what/time", "time"))))
    if not self.todt is None:
      qry = expr.and_(qry, expr.le(expr.add(expr.attribute("what/date", "date"),
                                            expr.attribute("what/time", "time")), expr.literal(self.todt)))
    dbqry = db.FileQuery()
    dbqry.filter = qry
    return dbqry
  
  def fetch_files(self, namer):
    r = self.bdb.execute_file_query(self.create_query())
    while r.next():
      uuid = r.get_uuid()
      print `dir(r.get_file_entry().metadata)`
      filename = namer.create(r.get_file_entry().metadata)
      content = self.bdb.get_file_content(uuid)
      try:
        with contextlib.closing(content):
          with open("%s/%s"%(self.outdir, filename), "w") as outf:
            shutil.copyfileobj(content, outf)
            outf.close()
            print "Fetched %s/%s"%(self.outdir, filename)
      except Exception, e:
        import traceback
        traceback.print_exc(e)
    print "%d files fetched"%r.size()
    
class basenamer(object):
  def __init__(self):
    pass
  def create(self, meta):
    raise Exception, "Namer must implement create(self, meta)"

class stdnamer(basenamer):
  def __init__(self):
    super(stdnamer, self).__init__()

  def create(self, meta):
    return "%s_%s_%s_%s.h5"%(meta.what_object.lower(), meta.bdb_source_name, meta.what_date.strftime("%Y%m%d"), meta.what_time.strftime("%H%M%S"))


def main(options):
  a = rave_bdb.rave_bdb()
  fetcher = rave_bdb_file_fetcher(a.get_database())
  fetcher.outdir=options.outdir
  if not options.fromdt is None:
    fetcher.fromdt = datetime.datetime.strptime(options.fromdt, "%Y%m%d%H%M%S")
  if not options.todt is None:
    fetcher.todt = datetime.datetime.strptime(options.todt, "%Y%m%d%H%M%S")
  fetcher.object = options.object
  fetcher.quantity = options.quantity
  if not options.sources is None:
    fetcher.sources = options.sources.split(",")
  fetcher.fetch_files(stdnamer())
  
if __name__=="__main__":
  import rave_bdb
  import sys
  from optparse import OptionParser
  
  description = "Fetches a number of files from the baltrad db"
  usage = 'usage: %prog --output <output directory> [args] [h]'
  usage += "\nFetches a number of files from the baltrad database"
  parser = OptionParser(usage=usage, description=description)

  parser.add_option("--output", dest="outdir", default=".",
                    help="Name of the output directory, default is current directory (.).")
  parser.add_option("--fromdt", dest="fromdt", 
                    help="From datetime (YYYYmmddHHMMSS) that the files should be fetched.")
  parser.add_option("--todt", dest="todt", 
                    help="To datetime (YYYYmmddHHMMSS) that the files should be fetched.")
  parser.add_option("--object", dest="object", default="PVOL",
                    help="The object type of the data to be fetched, default is PVOL.")
  parser.add_option("--quantity", dest="quantity", default="DBZH",
                    help="The quantity we are looking for in the data, default is DBZH.")
  parser.add_option("--sources", dest="sources", default=None,
                    help="The sources we are looking for, default is all.")
  parser.add_option("--namer", dest="namer", default="std",
                    help="How the fetched files should be named. Default is std which is <object>_<source>_<date>_<time>.h5.")
  (options, args) = parser.parse_args()

  if options.outdir != None:
    main(options)
  else:
    parser.print_help()

  
  