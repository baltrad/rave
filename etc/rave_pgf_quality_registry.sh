#!/usr/bin/env bash
set -x

grep -l ec_drqc_quality_plugin ${RAVEROOT}/rave/etc/rave_pgf_quality_registry.xml
if [ $? -eq 1 ]
then
sed -i 's/<\/rave-pgf-quality-registry>/  <quality-plugin name="drqc" module="ec_drqc_quality_plugin" class="ec_drqc_quality_plugin"\/>\n<\/rave-pgf-quality-registry>/g' ${RAVEROOT}/rave/etc/rave_pgf_quality_registry.xml
fi
