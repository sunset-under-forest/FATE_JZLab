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
include src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/compiler_depend.make

# Include the progress variables for this target.
include src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/progress.make

# Include the compile flags for this target's objects.
include src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/flags.make

src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/threshold-euclidean-dist.cpp.o: src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/flags.make
src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/threshold-euclidean-dist.cpp.o: ../src/examples/threshold_euclidean_dist_2d_simd/threshold-euclidean-dist.cpp
src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/threshold-euclidean-dist.cpp.o: src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/threshold-euclidean-dist.cpp.o"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/threshold_euclidean_dist_2d_simd && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/threshold-euclidean-dist.cpp.o -MF CMakeFiles/threshold-euclidean-dist_test.dir/threshold-euclidean-dist.cpp.o.d -o CMakeFiles/threshold-euclidean-dist_test.dir/threshold-euclidean-dist.cpp.o -c /tmp/tmp.ZO2yVo1pOb/src/examples/threshold_euclidean_dist_2d_simd/threshold-euclidean-dist.cpp

src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/threshold-euclidean-dist.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/threshold-euclidean-dist_test.dir/threshold-euclidean-dist.cpp.i"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/threshold_euclidean_dist_2d_simd && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/tmp.ZO2yVo1pOb/src/examples/threshold_euclidean_dist_2d_simd/threshold-euclidean-dist.cpp > CMakeFiles/threshold-euclidean-dist_test.dir/threshold-euclidean-dist.cpp.i

src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/threshold-euclidean-dist.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/threshold-euclidean-dist_test.dir/threshold-euclidean-dist.cpp.s"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/threshold_euclidean_dist_2d_simd && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/tmp.ZO2yVo1pOb/src/examples/threshold_euclidean_dist_2d_simd/threshold-euclidean-dist.cpp -o CMakeFiles/threshold-euclidean-dist_test.dir/threshold-euclidean-dist.cpp.s

src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/common/threshold-euclidean-dist.cpp.o: src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/flags.make
src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/common/threshold-euclidean-dist.cpp.o: ../src/examples/threshold_euclidean_dist_2d_simd/common/threshold-euclidean-dist.cpp
src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/common/threshold-euclidean-dist.cpp.o: src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/common/threshold-euclidean-dist.cpp.o"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/threshold_euclidean_dist_2d_simd && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/common/threshold-euclidean-dist.cpp.o -MF CMakeFiles/threshold-euclidean-dist_test.dir/common/threshold-euclidean-dist.cpp.o.d -o CMakeFiles/threshold-euclidean-dist_test.dir/common/threshold-euclidean-dist.cpp.o -c /tmp/tmp.ZO2yVo1pOb/src/examples/threshold_euclidean_dist_2d_simd/common/threshold-euclidean-dist.cpp

src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/common/threshold-euclidean-dist.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/threshold-euclidean-dist_test.dir/common/threshold-euclidean-dist.cpp.i"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/threshold_euclidean_dist_2d_simd && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/tmp.ZO2yVo1pOb/src/examples/threshold_euclidean_dist_2d_simd/common/threshold-euclidean-dist.cpp > CMakeFiles/threshold-euclidean-dist_test.dir/common/threshold-euclidean-dist.cpp.i

src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/common/threshold-euclidean-dist.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/threshold-euclidean-dist_test.dir/common/threshold-euclidean-dist.cpp.s"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/threshold_euclidean_dist_2d_simd && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/tmp.ZO2yVo1pOb/src/examples/threshold_euclidean_dist_2d_simd/common/threshold-euclidean-dist.cpp -o CMakeFiles/threshold-euclidean-dist_test.dir/common/threshold-euclidean-dist.cpp.s

# Object files for target threshold-euclidean-dist_test
threshold__euclidean__dist_test_OBJECTS = \
"CMakeFiles/threshold-euclidean-dist_test.dir/threshold-euclidean-dist.cpp.o" \
"CMakeFiles/threshold-euclidean-dist_test.dir/common/threshold-euclidean-dist.cpp.o"

# External object files for target threshold-euclidean-dist_test
threshold__euclidean__dist_test_EXTERNAL_OBJECTS =

bin/threshold-euclidean-dist_test: src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/threshold-euclidean-dist.cpp.o
bin/threshold-euclidean-dist_test: src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/common/threshold-euclidean-dist.cpp.o
bin/threshold-euclidean-dist_test: src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/build.make
bin/threshold-euclidean-dist_test: lib/libaby.a
bin/threshold-euclidean-dist_test: lib/libencrypto_utils.a
bin/threshold-euclidean-dist_test: lib/libotextension.a
bin/threshold-euclidean-dist_test: lib/libencrypto_utils.a
bin/threshold-euclidean-dist_test: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.74.0
bin/threshold-euclidean-dist_test: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.74.0
bin/threshold-euclidean-dist_test: /usr/lib/x86_64-linux-gnu/libboost_atomic.so.1.74.0
bin/threshold-euclidean-dist_test: /usr/lib/x86_64-linux-gnu/libgmp.so
bin/threshold-euclidean-dist_test: /usr/lib/x86_64-linux-gnu/libgmpxx.so
bin/threshold-euclidean-dist_test: /usr/lib/x86_64-linux-gnu/libcrypto.so
bin/threshold-euclidean-dist_test: lib/librelic_s.a
bin/threshold-euclidean-dist_test: lib/librelic_s.a
bin/threshold-euclidean-dist_test: /usr/lib/x86_64-linux-gnu/libgmp.so
bin/threshold-euclidean-dist_test: /usr/lib/x86_64-linux-gnu/libgmp.so
bin/threshold-euclidean-dist_test: src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../../bin/threshold-euclidean-dist_test"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/threshold_euclidean_dist_2d_simd && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/threshold-euclidean-dist_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/build: bin/threshold-euclidean-dist_test
.PHONY : src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/build

src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/clean:
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/threshold_euclidean_dist_2d_simd && $(CMAKE_COMMAND) -P CMakeFiles/threshold-euclidean-dist_test.dir/cmake_clean.cmake
.PHONY : src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/clean

src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/depend:
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /tmp/tmp.ZO2yVo1pOb /tmp/tmp.ZO2yVo1pOb/src/examples/threshold_euclidean_dist_2d_simd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/threshold_euclidean_dist_2d_simd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/examples/threshold_euclidean_dist_2d_simd/CMakeFiles/threshold-euclidean-dist_test.dir/depend

