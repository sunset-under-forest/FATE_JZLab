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
include src/examples/bench_operations/CMakeFiles/bench_operations.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/examples/bench_operations/CMakeFiles/bench_operations.dir/compiler_depend.make

# Include the progress variables for this target.
include src/examples/bench_operations/CMakeFiles/bench_operations.dir/progress.make

# Include the compile flags for this target's objects.
include src/examples/bench_operations/CMakeFiles/bench_operations.dir/flags.make

src/examples/bench_operations/CMakeFiles/bench_operations.dir/bench_operations.cpp.o: src/examples/bench_operations/CMakeFiles/bench_operations.dir/flags.make
src/examples/bench_operations/CMakeFiles/bench_operations.dir/bench_operations.cpp.o: ../src/examples/bench_operations/bench_operations.cpp
src/examples/bench_operations/CMakeFiles/bench_operations.dir/bench_operations.cpp.o: src/examples/bench_operations/CMakeFiles/bench_operations.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/examples/bench_operations/CMakeFiles/bench_operations.dir/bench_operations.cpp.o"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/bench_operations && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/examples/bench_operations/CMakeFiles/bench_operations.dir/bench_operations.cpp.o -MF CMakeFiles/bench_operations.dir/bench_operations.cpp.o.d -o CMakeFiles/bench_operations.dir/bench_operations.cpp.o -c /tmp/tmp.ZO2yVo1pOb/src/examples/bench_operations/bench_operations.cpp

src/examples/bench_operations/CMakeFiles/bench_operations.dir/bench_operations.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bench_operations.dir/bench_operations.cpp.i"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/bench_operations && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/tmp.ZO2yVo1pOb/src/examples/bench_operations/bench_operations.cpp > CMakeFiles/bench_operations.dir/bench_operations.cpp.i

src/examples/bench_operations/CMakeFiles/bench_operations.dir/bench_operations.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bench_operations.dir/bench_operations.cpp.s"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/bench_operations && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/tmp.ZO2yVo1pOb/src/examples/bench_operations/bench_operations.cpp -o CMakeFiles/bench_operations.dir/bench_operations.cpp.s

src/examples/bench_operations/CMakeFiles/bench_operations.dir/__/aes/common/aescircuit.cpp.o: src/examples/bench_operations/CMakeFiles/bench_operations.dir/flags.make
src/examples/bench_operations/CMakeFiles/bench_operations.dir/__/aes/common/aescircuit.cpp.o: ../src/examples/aes/common/aescircuit.cpp
src/examples/bench_operations/CMakeFiles/bench_operations.dir/__/aes/common/aescircuit.cpp.o: src/examples/bench_operations/CMakeFiles/bench_operations.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/examples/bench_operations/CMakeFiles/bench_operations.dir/__/aes/common/aescircuit.cpp.o"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/bench_operations && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/examples/bench_operations/CMakeFiles/bench_operations.dir/__/aes/common/aescircuit.cpp.o -MF CMakeFiles/bench_operations.dir/__/aes/common/aescircuit.cpp.o.d -o CMakeFiles/bench_operations.dir/__/aes/common/aescircuit.cpp.o -c /tmp/tmp.ZO2yVo1pOb/src/examples/aes/common/aescircuit.cpp

src/examples/bench_operations/CMakeFiles/bench_operations.dir/__/aes/common/aescircuit.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bench_operations.dir/__/aes/common/aescircuit.cpp.i"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/bench_operations && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/tmp.ZO2yVo1pOb/src/examples/aes/common/aescircuit.cpp > CMakeFiles/bench_operations.dir/__/aes/common/aescircuit.cpp.i

src/examples/bench_operations/CMakeFiles/bench_operations.dir/__/aes/common/aescircuit.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bench_operations.dir/__/aes/common/aescircuit.cpp.s"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/bench_operations && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/tmp.ZO2yVo1pOb/src/examples/aes/common/aescircuit.cpp -o CMakeFiles/bench_operations.dir/__/aes/common/aescircuit.cpp.s

# Object files for target bench_operations
bench_operations_OBJECTS = \
"CMakeFiles/bench_operations.dir/bench_operations.cpp.o" \
"CMakeFiles/bench_operations.dir/__/aes/common/aescircuit.cpp.o"

# External object files for target bench_operations
bench_operations_EXTERNAL_OBJECTS =

bin/bench_operations: src/examples/bench_operations/CMakeFiles/bench_operations.dir/bench_operations.cpp.o
bin/bench_operations: src/examples/bench_operations/CMakeFiles/bench_operations.dir/__/aes/common/aescircuit.cpp.o
bin/bench_operations: src/examples/bench_operations/CMakeFiles/bench_operations.dir/build.make
bin/bench_operations: lib/libaby.a
bin/bench_operations: lib/libencrypto_utils.a
bin/bench_operations: lib/libotextension.a
bin/bench_operations: lib/libencrypto_utils.a
bin/bench_operations: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.74.0
bin/bench_operations: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.74.0
bin/bench_operations: /usr/lib/x86_64-linux-gnu/libboost_atomic.so.1.74.0
bin/bench_operations: /usr/lib/x86_64-linux-gnu/libgmp.so
bin/bench_operations: /usr/lib/x86_64-linux-gnu/libgmpxx.so
bin/bench_operations: /usr/lib/x86_64-linux-gnu/libcrypto.so
bin/bench_operations: lib/librelic_s.a
bin/bench_operations: lib/librelic_s.a
bin/bench_operations: /usr/lib/x86_64-linux-gnu/libgmp.so
bin/bench_operations: /usr/lib/x86_64-linux-gnu/libgmp.so
bin/bench_operations: src/examples/bench_operations/CMakeFiles/bench_operations.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../../bin/bench_operations"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/bench_operations && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bench_operations.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/examples/bench_operations/CMakeFiles/bench_operations.dir/build: bin/bench_operations
.PHONY : src/examples/bench_operations/CMakeFiles/bench_operations.dir/build

src/examples/bench_operations/CMakeFiles/bench_operations.dir/clean:
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/bench_operations && $(CMAKE_COMMAND) -P CMakeFiles/bench_operations.dir/cmake_clean.cmake
.PHONY : src/examples/bench_operations/CMakeFiles/bench_operations.dir/clean

src/examples/bench_operations/CMakeFiles/bench_operations.dir/depend:
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /tmp/tmp.ZO2yVo1pOb /tmp/tmp.ZO2yVo1pOb/src/examples/bench_operations /tmp/tmp.ZO2yVo1pOb/cmake-build-debug /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/bench_operations /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/bench_operations/CMakeFiles/bench_operations.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/examples/bench_operations/CMakeFiles/bench_operations.dir/depend

