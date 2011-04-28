################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../util/Data.cpp \
../util/Debug.cpp \
../util/Options.cpp \
../util/Proj4.cpp \
../util/ProjectionFrame.cpp \
../util/RegExp.cpp \
../util/String.cpp 

OBJS += \
./util/Data.o \
./util/Debug.o \
./util/Options.o \
./util/Proj4.o \
./util/ProjectionFrame.o \
./util/RegExp.o \
./util/String.o 

CPP_DEPS += \
./util/Data.d \
./util/Debug.d \
./util/Options.d \
./util/Proj4.d \
./util/ProjectionFrame.d \
./util/RegExp.d \
./util/String.d 


# Each subdirectory must supply rules for building sources it contributes
util/%.o: ../util/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"$$hdfroot/include" -I"$$projroot/include" -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


