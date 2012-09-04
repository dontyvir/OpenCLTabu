# This is a simple makefile that compiles multiple C++ source files

# set the names here to be the names of your source files with the
# .cxx or .cpp replaced by .o
# Be *** SURE *** to put the .o files here rather than the source files

ProjectObjects =  InstanceParser.o Main.o Solution.o SolutionBuilder.o Solver.o Matrix.o TabuList.o

#------------ no need to change between these lines -------------------
LIBS = -L/home/donty/tarballs/amd-app-sdk/AMD-APP-SDK-v2.7-RC-lnx64/TempSDKUtil/lib/x86_64 -L/opt/AMDAPP/lib/x86_64
INCLUDES = -I/home/donty/tarballs/amd-app-sdk/AMD-APP-SDK-v2.7-RC-lnx64/samples/opencl/SDKUtil/include -I/opt/AMDAPP/include
CFLAGS = -g -Wall $(INCLUDES)
BUILD_DIR = build/debug/x86_64/HelloCL build/debug/x86_64
.SUFFIXES: .cpp
.cpp.o:
	g++ $(CFLAGS) -c $<
#------------ no need to change between these lines -------------------


#------------ targets --------------------------------------------
# describe how to create the targets - often there will be only one target

all: TabuSolver

TabuSolver: $(ProjectObjects)
	g++ $(CFLAGS) $(LIBS) $(ProjectObjects) -o TabuSolver 

clean:
	rm -f $(ProjectObjects) $(OtherProjectObjects) TabuSolver

#------------ dependencies --------------------------------------------
# put the .o that depends on a .h, then colon, then TAB, then the .h

Matrix.o :	Matrix.h
InstanceParser.o:	InstanceParser.h	Matrix.h
Solution.o:	Solution.h
SolutionBuilder.o:	SolutionBuilder.h
Solver.o:	Solver.h
TabuList.o : TabuList.h
Main.o:	InstanceParser.h Solution.h Solver.h 