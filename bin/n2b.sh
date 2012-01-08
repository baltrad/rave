#!/bin/sh
# Daniel Michelson, SMHI
# 2010-05-27

# chkconfig: 2345 90 10
# description: Init script for n2b


export N2B_HOME="/opt/baltrad/n2b"

start() {
  echo $"Starting n2b... "
  su - baltrad --shell=/bin/bash -c "${N2B_HOME}/n2b --catchup"
}

stop() {
  echo $"Stopping n2b... "
  su - baltrad --shell=/bin/bash -c "${N2B_HOME}/n2b --kill"
}

# See how we were called.
case "$1" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  restart)
    stop
    start
    ;;
  *)
    echo $"Usage: ./n2b {start|stop|restart}"
    exit 1
esac
