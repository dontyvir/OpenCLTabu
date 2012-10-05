# This is a simple makefile that compiles multiple C++ source files

# set the names here to be the names of your source files with the
# .cxx or .cpp replaced by .o
# Be *** SURE *** to put the .o files here rather than the source files

ProjectObjects =  InstanceParser.o Main.o Solution.o SolutionBuilder.o Solver.o Matrix.o TabuList.o ParallelSolver.o

#------------ no need to change between these lines -------------------
LPATH = -L/opt/AMDAPP/TempSDKUtil/lib/x86_64 -L/opt/AMDAPP/lib/x86_64 -L/usr/X11R6/lib
LFLAGS= -lpthread -ldl -lSDKUtil -lOpenCL
INCLUDES = -I/opt/AMDAPP/include -I/home/donty/tarballs/amd-app-sdk/AMD-APP-SDK-v2.7-RC-lnx64/samples/opencl/SDKUtil/include
CFLAGS = -g3 -Wall -Wpointer-arith -Wfloat-equal -ffor-scope
BUILD_DIR = build/debug/x86_64
.SUFFIXES: .cpp
.cpp.o:
	g++ $(CFLAGS) $(INCLUDES) -c $<

#------------ targets --------------------------------------------
# describe how to create the targets - often there will be only one target

all: TabuSolver

TabuSolver: $(ProjectObjects)
	g++ -o TabuSolver $(ProjectObjects) $(LFLAGS) $(LPATH)
	
clean:
	rm -f $(ProjectObjects) $(OtherProjectObjects) TabuSolver

#------------ dependencies --------------------------------------------
# put the .o that depends on a .h, then colon, then TAB, then the .h
