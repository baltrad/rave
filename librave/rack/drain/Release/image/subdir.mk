################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../image/CoordinateHandler.cpp \
../image/FileBinary.cpp \
../image/FileFQD.cpp \
../image/Geometry.cpp 

OBJS += \
./image/CoordinateHandler.o \
./image/FileBinary.o \
./image/FileFQD.o \
./image/Geometry.o 

CPP_DEPS += \
./image/CoordinateHandler.d \
./image/FileBinary.d \
./image/FileFQD.d \
./image/Geometry.d 


# Each subdirectory must supply rules for building sources it contributes
image/%.o: ../image/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"$$hdfroot/include" -I"$$projroot/include" -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


