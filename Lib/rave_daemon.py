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
## Generic daemon functionality. Doxygenified from
# http://www.jejik.com/articles/2007/02/a_simple_unix_linux_daemon_in_python/

## @file
## @author Sander Marechal (jejik.com) and Daniel Michelson (SMHI)
## @date 2010-07-15

import sys, os, time, atexit
from signal import SIGTERM


## A generic daemon class.
# Usage: subclass the Daemon class and override the run() method
class Daemon(object):
	## Constructor.
	# @param pidfile string to the file containing the PID
	# @param stdin string path to where to direct stdin
	# @param stdout string path to where to direct stdout
	# @param stderr string path to where to direct stderr
	def __init__(self, pidfile, stdin='/dev/null', stdout='/dev/null', stderr='/dev/null'):
		self.stdin = stdin
		self.stdout = stdout
		self.stderr = stderr
		self.pidfile = pidfile
	
	## Do the UNIX double-fork magic, see Stevens' "Advanced
	# Programming in the UNIX Environment" for details (ISBN 0201563177)
	# http://www.erlenstar.demon.co.uk/unix/faq_2.html#SEC16
	def daemonize(self):
		try: 
			pid = os.fork() 
			if pid > 0:
				# exit first parent
				sys.exit(0) 
		except OSError, e: 
			sys.stderr.write("fork #1 failed: %d (%s)\n" % (e.errno, e.strerror))
			sys.exit(1)
	
		# decouple from parent environment
		os.chdir("/") 
		os.setsid() 
		os.umask(0) 
	
		# do second fork
		try: 
			pid = os.fork() 
			if pid > 0:
				# exit from second parent
				sys.exit(0) 
		except OSError, e: 
			sys.stderr.write("fork #2 failed: %d (%s)\n" % (e.errno, e.strerror))
			sys.exit(1) 
	
		# redirect standard file descriptors
		sys.stdout.flush()
		sys.stderr.flush()
		si = file(self.stdin, 'r')
		so = file(self.stdout, 'a+')
		se = file(self.stderr, 'a+', 0)
		os.dup2(si.fileno(), sys.stdin.fileno())
		os.dup2(so.fileno(), sys.stdout.fileno())
		os.dup2(se.fileno(), sys.stderr.fileno())
	
		# write pidfile
		atexit.register(self.delpid)
		pid = str(os.getpid())
		file(self.pidfile,'w+').write("%s\n" % pid)
	
	## Deletes the PID file.
	def delpid(self):
		os.remove(self.pidfile)

	## Checks if the process with provided pid is running
	# by checking the /proc directory.
	# @param pid - the pid to check for
	# @return True if a process with provided pid is running, otherwise False
	def _isprocessrunning(self, pid):
		return os.path.exists("/proc/%d"%pid)

	## Start the daemon.
	def start(self):
		# Check for a pidfile to see if the daemon already runs
		try:
			pf = file(self.pidfile,'r')
			pid = int(pf.read().strip())
			pf.close()
		except IOError:
			pid = None
	
		if pid:
			if self._isprocessrunning(pid):
				message = "pidfile %s already exists with a process with pid=%d is running. Daemon already running?\n"
				sys.stderr.write(message % (self.pidfile, pid))
				sys.exit(1)
			else:
				message = "pidfile exists but it seems like process is not running, probably due to an uncontrolled shutdown. Resetting.\n"
				sys.stderr.write(message)
				self.delpid()
		
		# Start the daemon.
		self.daemonize()
		self.run()

	## Stop the daemon.
	def stop(self):
		# Get the pid from the pidfile
		try:
			pf = file(self.pidfile,'r')
			pid = int(pf.read().strip())
			pf.close()
		except IOError:
			pid = None
	
		if not pid:
			message = "pidfile %s does not exist. Daemon not running?\n"
			sys.stderr.write(message % self.pidfile)
			return # not an error in a restart

		# Try killing the daemon process. Use the equivalent of a
		# KeyboardInterrupt because this will trigger functionality
		# registered to atexit. This is a soft exit, and may require
		# toughening up in the future.
		try:
			ctr = 0   # We don't want to hang indefenetly trying to kill a process..
			sys.stderr.write("Waiting for server to shutdown ")
			sys.stderr.flush()
			while ctr <= 30:
				os.kill(pid, SIGTERM)
				time.sleep(0.1)
				sys.stderr.write(".")
				sys.stderr.flush()
				ctr = ctr + 1
		except OSError, err:
			err = str(err)
			if err.find("No such process") > 0 or err.find("Operation not permitted") > 0:
				if os.path.exists(self.pidfile):
					os.remove(self.pidfile)
			else:
				print str(err)
				sys.exit(1)

		sys.stderr.write("\n")
		sys.stderr.flush()

	## Restart the daemon.
	def restart(self):
		self.stop()
		self.start()

	## You should override this method when you subclass Daemon.
	# It will be called after the process has been daemonized by
	# start() or restart().
	def run(self):
		"""
		"""
