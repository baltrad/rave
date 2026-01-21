#
# SYNOPSIS
#
#   CHECK_PYTHON()
#
#   Optional argument 'keepvar' that will indicate to the script that CPPFLAGS & LDFLAGS should be restored after
#   macro has been run.
#
# DESCRIPTION
#
#   Macro for identifying python config
#
#   The macro provides two options.
#
#   --with-python=[yes | no | path] 
#                        where path should be name of python binary, either absolute path or just the binary, like python3.
#                        If path is specified, then this path will be used for an atempt to compile the code.
#                        Default is yes and to use python3.
#
#
#   This macro will atempt to identify and extract all variables that are necessary to compile and build a python c module.
#   It has only been verified for python3. It assumes that python3 sysconfig exists.
#
#   The following variables will be set.
#   - PYTHON_SUPPRESSED    If python should not be checked for (--with-python=no)
#   PYTHON_FOUND=yes|no
#   PYTHON_CHECK_OK=yes|no
#   PYTHON_BINARY=
#   PYTHON_VERSION=
#   PYTHON_CC=
#   PYTHON_INCLUDE=
#   PYTHON_SITEPACK=
#   PYTHON_LIBDIR=
#   PYTHON_OPTS=
#   PYTHON_LDFLAGS=
#   PYTHON_LDSHARED=
#   PYTHON_CCSHARED=
#
# LICENSE
#   Copyright (c) 2024 Anders Henja (anders@henjab.se)
#
#   Copying and distribution of this file, with or without modification, are
#   permitted in any medium without royalty provided the copyright notice
#   and this notice are preserved. This file is offered as-is, without any
#   warranty.
AC_DEFUN([CHECK_PYTHON], [

PYTHON_FOUND=no
PYTHON_CHECK_OK=yes
PYTHON_BINARY=
PYTHON_VERSION=
PYTHON_CC=
PYTHON_INCLUDE=
PYTHON_SITEPACK=
PYTHON_LIBDIR=
PYTHON_OPTS=
PYTHON_LDFLAGS=
PYTHON_LDSHARED=
PYTHON_CCSHARED=
PYTHON_SUPPRESSED=no

check_python_with_python=
check_python_python_path=python3

# Add a default --with-python configuration option.
AC_ARG_WITH([python],
  AS_HELP_STRING(
    [--with-python=[yes|no|<python binary>]],
            [name of python binary or full path to python binary]
  ),
  [ if test "$withval" = "no" -o "$withval" = "yes"; then
      check_python_with_python="$withval"
    else
      check_python_with_python="yes"
      check_python_python_path="$withval"
    fi
  ],
  [check_python_with_python="yes"]
)

if [[ "$check_python_with_python" != "no" ]]; then
  PYTHON_FOUND=yes

  AC_MSG_CHECKING([if we can identify a python executable])
  which "$check_python_python_path" >> /dev/null 2>&1
  if [[ $? -eq 0 ]]; then
    X=`"$check_python_python_path" -c "import sys" >> /dev/null 2>&1`
    if [[ $? -eq 0 ]]; then
      AC_MSG_RESULT([Yes])
    else
      AC_MSG_RESULT([No])
      PYTHON_FOUND=no
    fi
  else
    AC_MSG_RESULT([No])
    PYTHON_FOUND=no
  fi
  
  if [[ "$PYTHON_FOUND" = "yes" ]]; then
    AC_MSG_CHECKING([python binary path])
    PYTHON_BINARY="$check_python_python_path"
    AC_MSG_RESULT([$PYTHON_BINARY])
  

    AC_MSG_CHECKING([Python version])
    PYTHON_VERSION=`"$check_python_python_path" -c "import sys;v=sys.version_info;print('%d.%d.%d'%(v.major, v.minor, v.micro))"`
    if [[ $? -eq 0 ]]; then
      AC_MSG_RESULT([$PYTHON_VERSION])
    else
      AC_MSG_WARN([failed])
      PYTHON_CHECK_OK=no
    fi

    AC_MSG_CHECKING([for python compiler])
    PYTHON_CC=`"$PYTHON_BINARY" -c "import sysconfig; print(sysconfig.get_config_var('CC') or '')"`
    if [[ $? -eq 0 ]]; then
      AC_MSG_RESULT([$PYTHON_CC])
    else
      AC_MSG_WARN([failed])
      PYTHON_CHECK_OK=no
    fi

    AC_MSG_CHECKING([for python include path])
    P_INCLUDE=`"$PYTHON_BINARY" -c "import sysconfig; print(sysconfig.get_path('include') or '')"`
    if [[ $? -eq 0 ]]; then
      P_INCLUDE="-I$P_INCLUDE"
      AC_MSG_RESULT([$P_INCLUDE])
    else
      AC_MSG_WARN([failed])
      PYTHON_CHECK_OK=no
    fi

    AC_MSG_CHECKING([for python plat include path])
    PLAT_INCLUDE=`"$PYTHON_BINARY" -c "import sysconfig; print(sysconfig.get_path('platinclude') or '')"`
    if [[ $? -eq 0 ]]; then
      if [[ "$PLAT_INCLUDE" != "" ]]; then
        PLAT_INCLUDE="-I$PLAT_INCLUDE"
        if [[ "$PLAT_INCLUDE" = "$P_INCLUDE" ]]; then
          PLAT_INCLUDE=""
        fi
      fi
      AC_MSG_RESULT([$PLAT_INCLUDE])
    else
      AC_MSG_WARN([failed])
      PYTHON_CHECK_OK=no
    fi

    PYTHON_INCLUDE="$P_INCLUDE $PLAT_INCLUDE"

    AC_MSG_CHECKING([for python site package path])
    PYTHON_SITEPACK=`"$PYTHON_BINARY" -c "import sysconfig; print(sysconfig.get_path('platlib') or '')"`
    if [[ $? -eq 0 ]]; then
      AC_MSG_RESULT([$PYTHON_SITEPACK])
    else
      AC_MSG_WARN([failed])
      PYTHON_CHECK_OK=no
    fi

    AC_MSG_CHECKING([for python library dir])
    PYTHON_LIBDIR=`"$PYTHON_BINARY" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')"`
    if [[ $? -eq 0 ]]; then
      AC_MSG_RESULT([$PYTHON_LIBDIR])
    else
      AC_MSG_WARN([failed])
      PYTHON_CHECK_OK=no
    fi

    AC_MSG_CHECKING([for python options])
    PYTHON_OPTS=`"$PYTHON_BINARY" -c "import sysconfig; print(sysconfig.get_config_var('OPT') or '')"`
    if [[ $? -eq 0 ]]; then
      AC_MSG_RESULT([$PYTHON_OPTS])
    else
      AC_MSG_WARN([failed])
      PYTHON_CHECK_OK=no
    fi

    AC_MSG_CHECKING([for python ldflags])
    PYTHON_LDFLAGS=`"$PYTHON_BINARY" -c "import sysconfig; print(sysconfig.get_config_var('LDFLAGS') or '')"`
    if [[ $? -eq 0 ]]; then
      AC_MSG_RESULT([$PYTHON_LDFLAGS])
    else
      AC_MSG_WARN([failed])
      PYTHON_CHECK_OK=no
    fi

    AC_MSG_CHECKING([for python ldshared])
    PYTHON_LDSHARED=`"$PYTHON_BINARY" -c "import sysconfig; print(sysconfig.get_config_var('LDSHARED') or '')"`
    if [[ $? -eq 0 ]]; then
    # Special hack for mac osx.
    check_python_ismacos=no
    case `uname -s` in
      Darwin*)
        check_python_ismacos=yes
        ;;
      darwin*)
        check_python_ismacos=yes
        ;;
      esac
      if [[ "$check_python_ismacos" = "yes" ]]; then
        PYTHON_LDSHARED=`echo $PYTHON_LDSHARED | sed -e "s/[[ \t]]-bundle[[ \t]]/ -dynamiclib /g"`
      fi
      AC_MSG_RESULT([$PYTHON_LDSHARED])
    else
      AC_MSG_WARN([failed])
      PYTHON_CHECK_OK=no
    fi

    AC_MSG_CHECKING([for python ccshared])
    PYTHON_CCSHARED=`"$PYTHON_BINARY" -c "import sysconfig; print(sysconfig.get_config_var('CCSHARED') or '')"`
    if [[ $? -eq 0 ]]; then
      AC_MSG_RESULT([$PYTHON_CCSHARED])
    else
      AC_MSG_WARN([failed])
      PYTHON_CHECK_OK=no
    fi

  fi
else
  AC_MSG_NOTICE([Python check suppressed])
  PYTHON_SUPPRESSED=yes
fi

])
