# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /tmp/tmp.ZO2yVo1pOb

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /tmp/tmp.ZO2yVo1pOb/cmake-build-debug

# Include any dependencies generated for this target.
include src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/compiler_depend.make

# Include the progress variables for this target.
include src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/progress.make

# Include the compile flags for this target's objects.
include src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/flags.make

src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/uc_circuit_test.cpp.o: src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/flags.make
src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/uc_circuit_test.cpp.o: ../src/examples/uc_circuit/uc_circuit_test.cpp
src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/uc_circuit_test.cpp.o: src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/uc_circuit_test.cpp.o"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/uc_circuit && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/uc_circuit_test.cpp.o -MF CMakeFiles/uc_circuit_test.dir/uc_circuit_test.cpp.o.d -o CMakeFiles/uc_circuit_test.dir/uc_circuit_test.cpp.o -c /tmp/tmp.ZO2yVo1pOb/src/examples/uc_circuit/uc_circuit_test.cpp

src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/uc_circuit_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/uc_circuit_test.dir/uc_circuit_test.cpp.i"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/uc_circuit && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/tmp.ZO2yVo1pOb/src/examples/uc_circuit/uc_circuit_test.cpp > CMakeFiles/uc_circuit_test.dir/uc_circuit_test.cpp.i

src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/uc_circuit_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/uc_circuit_test.dir/uc_circuit_test.cpp.s"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/uc_circuit && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/tmp.ZO2yVo1pOb/src/examples/uc_circuit/uc_circuit_test.cpp -o CMakeFiles/uc_circuit_test.dir/uc_circuit_test.cpp.s

src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/common/uc_circuit.cpp.o: src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/flags.make
src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/common/uc_circuit.cpp.o: ../src/examples/uc_circuit/common/uc_circuit.cpp
src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/common/uc_circuit.cpp.o: src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/common/uc_circuit.cpp.o"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/uc_circuit && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/common/uc_circuit.cpp.o -MF CMakeFiles/uc_circuit_test.dir/common/uc_circuit.cpp.o.d -o CMakeFiles/uc_circuit_test.dir/common/uc_circuit.cpp.o -c /tmp/tmp.ZO2yVo1pOb/src/examples/uc_circuit/common/uc_circuit.cpp

src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/common/uc_circuit.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/uc_circuit_test.dir/common/uc_circuit.cpp.i"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/uc_circuit && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/tmp.ZO2yVo1pOb/src/examples/uc_circuit/common/uc_circuit.cpp > CMakeFiles/uc_circuit_test.dir/common/uc_circuit.cpp.i

src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/common/uc_circuit.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/uc_circuit_test.dir/common/uc_circuit.cpp.s"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/uc_circuit && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/tmp.ZO2yVo1pOb/src/examples/uc_circuit/common/uc_circuit.cpp -o CMakeFiles/uc_circuit_test.dir/common/uc_circuit.cpp.s

# Object files for target uc_circuit_test
uc_circuit_test_OBJECTS = \
"CMakeFiles/uc_circuit_test.dir/uc_circuit_test.cpp.o" \
"CMakeFiles/uc_circuit_test.dir/common/uc_circuit.cpp.o"

# External object files for target uc_circuit_test
uc_circuit_test_EXTERNAL_OBJECTS =

bin/uc_circuit_test: src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/uc_circuit_test.cpp.o
bin/uc_circuit_test: src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/common/uc_circuit.cpp.o
bin/uc_circuit_test: src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/build.make
bin/uc_circuit_test: lib/libaby.a
bin/uc_circuit_test: lib/libencrypto_utils.a
bin/uc_circuit_test: lib/libotextension.a
bin/uc_circuit_test: lib/libencrypto_utils.a
bin/uc_circuit_test: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.74.0
bin/uc_circuit_test: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.74.0
bin/uc_circuit_test: /usr/lib/x86_64-linux-gnu/libboost_atomic.so.1.74.0
bin/uc_circuit_test: /usr/lib/x86_64-linux-gnu/libgmp.so
bin/uc_circuit_test: /usr/lib/x86_64-linux-gnu/libgmpxx.so
bin/uc_circuit_test: /usr/lib/x86_64-linux-gnu/libcrypto.so
bin/uc_circuit_test: lib/librelic_s.a
bin/uc_circuit_test: lib/librelic_s.a
bin/uc_circuit_test: /usr/lib/x86_64-linux-gnu/libgmp.so
bin/uc_circuit_test: /usr/lib/x86_64-linux-gnu/libgmp.so
bin/uc_circuit_test: src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../../bin/uc_circuit_test"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/uc_circuit && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/uc_circuit_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/build: bin/uc_circuit_test
.PHONY : src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/build

src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/clean:
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/uc_circuit && $(CMAKE_COMMAND) -P CMakeFiles/uc_circuit_test.dir/cmake_clean.cmake
.PHONY : src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/clean

src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/depend:
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /tmp/tmp.ZO2yVo1pOb /tmp/tmp.ZO2yVo1pOb/src/examples/uc_circuit /tmp/tmp.ZO2yVo1pOb/cmake-build-debug /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/uc_circuit /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/examples/uc_circuit/CMakeFiles/uc_circuit_test.dir/depend

