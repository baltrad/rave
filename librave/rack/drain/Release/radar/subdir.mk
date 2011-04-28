################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../radar/Composite.cpp \
../radar/Coordinates.cpp \
../radar/Geometry.cpp \
../radar/PolarCoordinateHandler.cpp \
../radar/ProductXReader.cpp \
../radar/SubComposite.cpp 

OBJS += \
./radar/Composite.o \
./radar/Coordinates.o \
./radar/Geometry.o \
./radar/PolarCoordinateHandler.o \
./radar/ProductXReader.o \
./radar/SubComposite.o 

CPP_DEPS += \
./radar/Composite.d \
./radar/Coordinates.d \
./radar/Geometry.d \
./radar/PolarCoordinateHandler.d \
./radar/ProductXReader.d \
./radar/SubComposite.d 


# Each subdirectory must supply rules for building sources it contributes
radar/%.o: ../radar/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"$$hdfroot/include" -I"$$projroot/include" -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

radar/SubComposite.o: ../radar/SubComposite.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ `$$DRAIN_MAGICK` -I"$$hdfroot/include" -I"$$projroot/include" -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"radar/SubComposite.d" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


