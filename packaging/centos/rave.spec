%{!?python_sitearch: %global python_sitearch %(%{__python} -c "from distutils.sysconfig import get_python_lib; print get_python_lib(1)")}
%define _prefix /opt/baltrad/%{name}

Name: rave
Version: %{version}
Release: %{snapshot}%{?dist}
Summary: RAVE - Product generation framework and toolbox. Injector using ODIM_H5 files
License: GPL-3 and LGPL-3
URL: http://www.baltrad.eu/
Source0: %{name}-%{version}.tar.gz
Source1: rave.conf
BuildRequires: hlhdf-devel
BuildRequires: hlhdf-python
BuildRequires: hdf5-devel
BuildRequires: zlib-devel
BuildRequires: python2-devel
# Workaround for centos6
BuildRequires: atlas
BuildRequires: numpy
BuildRequires: proj-devel
#expat requires
Requires: expat
BuildRequires: expat-devel
# Don't see any actual imports, just mentioned in README
#BuildRequires: python-pycurl

%description
Product generation framework and toolbox. Injector using ODIM_H5 files

%package devel
Summary: RAVE development files
Group: Development/Libraries
Requires: %{name} = %{version}-%{release}
# rave development headers include headers from proj
Requires: proj-devel
# arrayobject.h and other needs
Requires: numpy
# Workaround for centos6
Requires: atlas

%description devel
RAVE development headers and libraries.

%prep
%setup -q

%build

%configure --with-hlhdf=/opt/baltrad/hlhdf --with-expat
# --with-bufr=/opt/baltrad/bbufr
# --with-netcdf=yes
make

%install

# FIXME: Why is this mkdir necessary?
# With full _prefix the custom installscripts think there was already an old version
# present and does some special things we may not want (migration to newer version)
rm -rf $RPM_BUILD_ROOT
mkdir -p %{buildroot}/opt/baltrad
make install DESTDIR=%{buildroot}
install -p -D -m 0644 %{SOURCE1} %{buildroot}%{_sysconfdir}/ld.so.conf.d/rave.conf

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%files
%doc %{_prefix}/COPYING
%doc %{_prefix}/COPYING.LESSER
%doc %{_prefix}/LICENSE

# Move to a python module? But the subdir name is very bad for site-packages
%{_prefix}/Lib/*.py
%{_prefix}/Lib/*.pyc
%{_prefix}/Lib/*.pyo
%{_prefix}/Lib/_*.so
%{_prefix}/Lib/gadjust
#%{_prefix}/Lib/gadjust/*.py
#%{_prefix}/Lib/gadjust/*.pyc
#%{_prefix}/Lib/gadjust/*.pyo
%{_prefix}/Lib/ravemigrate
#%{_prefix}/Lib/ravemigrate/*.py
#%{_prefix}/Lib/ravemigrate/*.pyc
#%{_prefix}/Lib/ravemigrate/*.pyo
#%{_prefix}/Lib/ravemigrate/*.cfg
#%{_prefix}/Lib/ravemigrate/versions
#%{_prefix}/Lib/ravemigrate/versions/*.py
#%{_prefix}/Lib/ravemigrate/versions/*.pyc
#%{_prefix}/Lib/ravemigrate/versions/*.pyo
%{_prefix}/lib/*.so
%{_prefix}/bin/*
%{_prefix}/config/*.xml
%{_prefix}/mkf/def.mk
%{_prefix}/rave.xbm
%{_prefix}/etc/rave_pgf
%{_prefix}/etc/rave_pgf_*.xml
%{_prefix}/etc/rave_tile_registry.xml
%{_prefix}/etc/rave.pth

%config(noreplace) %{python_sitelib}/rave.pth
%config(noreplace) %{_sysconfdir}/ld.so.conf.d/rave.conf

%files devel
%{_prefix}/include/python/*.h
%{_prefix}/include/*.h
