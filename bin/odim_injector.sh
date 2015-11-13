#!/bin/sh
# Daniel Michelson, SMHI
# 2010-05-27

# chkconfig: 2345 90 10
# description: Init script for n2b


export ODIM_INJECTOR_HOME="/opt/baltrad/rave/bin"

start() {
  echo $"Starting odim_injector... "
  su - baltrad --shell=/bin/bash -c "${ODIM_INJECTOR_HOME}/odim_injector --catchup"
}

stop() {
  echo $"Stopping odim_injector... "
  su - baltrad --shell=/bin/bash -c "${ODIM_INJECTOR_HOME}/odim_injector --kill"
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
    echo $"Usage: ./odim_injector.sh {start|stop|restart}"
    exit 1
esac
