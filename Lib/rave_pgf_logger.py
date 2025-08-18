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
## Management utilities for the logging system.
# Server and receiver functionality are taken almost straight from
# the Python documentation, with slightly better security.

## @file
## @author Daniel Michelson, SMHI
## @date 2010-07-24

import sys, os, struct
import logging.handlers
import multiprocessing
import tempfile
from rave_defines import (
    PGF_HOST,
    LOGID,
    SYSLOG,
    LOGFACILITY,
    LOGFILE,
    LOGFILESIZE,
    LOGFILES,
    LOGPORT,
    LOGPIDFILE,
    LOGLEVEL,
    SYSLOG_FORMAT,
    LOGFILE_FORMAT,
    STDOE,
)
from rave_daemon import Daemon

if sys.version_info < (3,):
    import SocketServer as socketserver
else:
    import socketserver

import pickle


LOGLEVELS = {
    "notset": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "fatal": logging.FATAL,
}

tempfile.tempdir = ''


## Initializes the system logger.
# @param logger an instance returned by \ref logging.getLogger()
# @param level int log level
def init_logger(logger, level=LOGLEVEL, logfile=LOGFILE):
    logger.setLevel(LOGLEVELS[level])
    if not len(logger.handlers):
        handler = logging.handlers.RotatingFileHandler(logfile, maxBytes=LOGFILESIZE, backupCount=LOGFILES)
        # This formatter removes the fractions of a second in the time.
        #       formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s',
        #                                     '%Y-%m-%d %H:%M:%S %Z')
        # The default formatter contains fractions of a second in the time.
        formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        if sys.version_info.major > 3 or (sys.version_info.major == 3 and sys.version_info.minor < 9):
            handler.lock = multiprocessing.RLock()
        logger.addHandler(handler)


## Locks, logs, and unlocks, with rudimentary level filtering.
# @param logger the logger object initialized with \ref init_logger()
# @param level string log level
# @param msg string log message
def log(logger, level, msg):
    if LOGLEVELS[level] >= logger.level:
        logger.handlers[0].acquire()
        name = multiprocessing.current_process().name
        logger.log(LOGLEVELS[level], "%s: %s" % (name, msg))
        logger.handlers[0].release()


# Stuff below is for creating a logger server and simple client.


## Handler for a streaming logging request.
#  This basically logs the record using whatever logging policy is
#  configured locally.
class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    ## Handle multiple requests - each expected to be a 4-byte length,
    # followed by the LogRecord in pickle format. Logs the record
    # according to whatever policy is configured locally.
    def handle(self):
        while 1:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack(">L", chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = self.unPickle(chunk)
            # Rudimentary security, to overcome the vulnerability of pickling
            if type(obj['msg']) in (StringType, UnicodeType):
                record = logging.makeLogRecord(obj)
                self.handleLogRecord(record)

    ## Unpacks whatever comes in over the wire.
    # @param data payload in pickle format
    # @return unpacked payload in native Python format
    def unPickle(self, data):
        return pickle.loads(data)

    ## Handles a log record
    # @param record log record
    def handleLogRecord(self, record):
        # if a name is specified, we use the named logger rather than the one
        # implied by the record.
        if self.server.logname is not None:
            name = self.server.logname
        else:
            name = record.name
        logger = logging.getLogger(name)
        init_logger(logger)

        # N.B. EVERY record gets logged. This is because Logger.handle
        # is normally called AFTER logger-level filtering. If you want
        # to do filtering, do it at the client end to save wasting
        # cycles and network bandwidth!
        logger.handle(record)


## Simple TCP socket-based logging receiver suitable for testing, and maybe a little more.
class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    allow_reuse_address = 1

    ## Initializer
    # @param host string host name
    # @param port int port number
    # @param handler server instance
    def __init__(self, host=PGF_HOST, port=LOGPORT, handler=LogRecordStreamHandler):
        socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.abort = 0
        self.timeout = 1
        self.logname = None

    ## Serve until stopped
    def serve_until_stopped(self):
        import select

        abort = 0
        while not abort:
            rd, wr, ex = select.select([self.socket.fileno()], [], [], self.timeout)
            if rd:
                self.handle_request()
            abort = self.abort


class rave_pgf_logger_server(Daemon):
    ## Constructor
    # @param host URI to the host for this server.
    # @param port int port number to the host for this server.
    # @param pidfile string file name to which to write the server's PID
    # @param stdin string path to where to direct stdin
    # @param stdout string path to where to direct stdout
    # @param stderr string path to where to direct stderr
    def __init__(
        self,
        host=PGF_HOST,
        port=LOGPORT,
        pidfile=LOGPIDFILE,
        #               stdin='/dev/null', stdout=STDOE, stderr=STDOE):
        stdin='/dev/null',
        stdout='/dev/null',
        stderr='/dev/null',
    ):
        self.pidfile = pidfile
        self.stdin = stdin
        self.stderr = stdout
        self.stdout = stderr
        self.host = host
        self.port = port

    ## Determines whether the server is running or not.
    def status(self):
        if os.path.isfile(self.pidfile):
            fd = open(self.pidfile)
            c = fd.read()
            fd.close()
            try:
                pgid = os.getpgid(int(c))
                return "running with PID %s and GID %i" % (c[:-1], pgid)
            except:
                return "not running"
        else:
            return "not running"

    ## Runs the server.
    # Creates an instance of a LogRecordSocketReceiver and serves it.
    # Note that the start(), stop(), and restart() methods are inherited from
    # \ref Daemon , but you can call fg() to run the server in the
    # foreground, ie. not daemonize, which is useful for debugging.
    def run(self):
        logging.basicConfig(format=LOGFILE_FORMAT)
        self.server = LogRecordSocketReceiver(host=self.host, port=self.port)
        self.server.serve_until_stopped()


## Client logger.
# Clients can be created even when the server isn't running. They will succeed in connecting
# and logging when the server starts, even though messages will be lost before this is done.
# The only time this could happen is when the PGF and logger servers are initializing, and the
# client loggers in the PGF server send messages before the logger server is ready, which is
# highly unlikely.
# @param host URI to the host for this server.
# @param port int port number to the host for this server.
# @param level string log level
# @returns logging.getLogger client
def rave_pgf_logger_client(host=PGF_HOST, port=LOGPORT, level=LOGLEVEL):
    myName = 'PGF-' + tempfile.mktemp()  # Needs a unique name to avoid confusion causing replicate log entries
    myLogger = logging.getLogger(myName)
    myLogger.setLevel(LOGLEVELS[level])
    socketHandler = logging.handlers.SocketHandler(host, port)
    myLogger.addHandler(socketHandler)
    return myLogger


## SysLog client.
# It is up to the user to figure out how to sort/filter syslog messages.
# The level of the messages actually appearing in syslog will be determined by the level set
# by your host, and changing this may require root access.
# @param name string logger identifier. This will look up a logger that is already initialized
# after it has been initialized the first time, normally when the PGF server starts.
# @param address tuple containing (string, int) for (host, port) or the device file containing
# the socket used for syslog.
# @param facility string representing the syslog facility
# @param level string log level
# @returns logging.getLogger client that your application will use to send messages to syslog
def rave_pgf_syslog_client(name=LOGID, address=SYSLOG, facility=LOGFACILITY, level=LOGLEVEL):
    myLogger = logging.getLogger(name)
    if not len(myLogger.handlers):
        myLogger.setLevel(LOGLEVELS[level])
        handler = logging.handlers.SysLogHandler(address, facility)
        formatter = logging.Formatter(SYSLOG_FORMAT)
        handler.setFormatter(formatter)
        myLogger.addHandler(handler)
    return myLogger


## stdout client.
# @param level string log level
# @returns logging.getLogger client that your application will use to send messages to stdout
def rave_pgf_stdout_client(name="RAVE-STDOUT", level=LOGLEVEL):
    myLogger = logging.getLogger(name)
    if not len(myLogger.handlers):
        myLogger.setLevel(LOGLEVELS[level])
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(LOGLEVELS[level])
        formatter = logging.Formatter(SYSLOG_FORMAT)
        handler.setFormatter(formatter)
        myLogger.addHandler(handler)
    return myLogger


def rave_pgf_logfile_client(
    name="RAVE-LOGFILE", level=LOGLEVEL, logfile=LOGFILE, logfilesize=LOGFILESIZE, nrlogfiles=LOGFILES
):
    myLogger = logging.getLogger(name)
    if not len(myLogger.handlers):
        myLogger.setLevel(LOGLEVELS[level])
        handler = logging.handlers.RotatingFileHandler(logfile, maxBytes=logfilesize, backupCount=nrlogfiles)
        # The default formatter contains fractions of a second in the time.
        formatter = logging.Formatter(LOGFILE_FORMAT)
        handler.setFormatter(formatter)
        handler.lock = multiprocessing.RLock()
        myLogger.addHandler(handler)
    return myLogger


def create_logger(level=LOGLEVEL, name=None):
    from rave_defines import LOGGER_TYPE

    if LOGGER_TYPE == "stdout":
        if name != None:
            return rave_pgf_stdout_client(name, level)
        else:
            return rave_pgf_stdout_client(level)
    elif LOGGER_TYPE == "logfile":
        if name != None:
            return rave_pgf_logfile_client(name=name, level=level)
        else:
            return rave_pgf_logfile_client(level=level)
    else:
        if name != None:
            return rave_pgf_syslog_client(name=name, level=level)
        else:
            return rave_pgf_syslog_client(level=level)


if __name__ == "__main__":
    # Functionality below for testing. Otherwise use command-line binary.
    prog = "rave_pgf_logger_server"
    usage = "usage: %s start|stop|status|restart|fg" % prog

    if len(sys.argv) != 2:
        print(usage)
        sys.exit()

    ARG = sys.argv[1].lower()

    if ARG not in ('start', 'stop', 'status', 'restart', 'fg'):
        print(usage)
        sys.exit()

    this = rave_pgf_logger_server()

    if ARG == 'stop':
        myLogger = rave_pgf_syslog_client()
        myLogger.info("Shutting down log TCP server on %s:%i" % (this.host, this.port))
        this.stop()

    if ARG == 'start':
        if this.status() == "not running":
            print("Starting log TCP server on %s:%i" % (this.host, this.port))
            this.start()
        else:
            print("Log TCP server already running on %s:%i" % (this.host, this.port))

    if ARG == 'restart':
        this.restart()

    if ARG == 'status':
        print("%s is %s" % (prog, this.status()))

    if ARG == 'fg':
        this.run()
