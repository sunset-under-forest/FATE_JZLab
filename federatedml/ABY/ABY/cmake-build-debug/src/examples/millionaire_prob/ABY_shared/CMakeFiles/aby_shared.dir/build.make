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
include src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/compiler_depend.make

# Include the progress variables for this target.
include src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/progress.make

# Include the compile flags for this target's objects.
include src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/flags.make

src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/aby_shared.cpp.o: src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/flags.make
src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/aby_shared.cpp.o: ../src/examples/millionaire_prob/ABY_shared/aby_shared.cpp
src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/aby_shared.cpp.o: src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/aby_shared.cpp.o"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/millionaire_prob/ABY_shared && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/aby_shared.cpp.o -MF CMakeFiles/aby_shared.dir/aby_shared.cpp.o.d -o CMakeFiles/aby_shared.dir/aby_shared.cpp.o -c /tmp/tmp.ZO2yVo1pOb/src/examples/millionaire_prob/ABY_shared/aby_shared.cpp

src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/aby_shared.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aby_shared.dir/aby_shared.cpp.i"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/millionaire_prob/ABY_shared && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/tmp.ZO2yVo1pOb/src/examples/millionaire_prob/ABY_shared/aby_shared.cpp > CMakeFiles/aby_shared.dir/aby_shared.cpp.i

src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/aby_shared.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aby_shared.dir/aby_shared.cpp.s"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/millionaire_prob/ABY_shared && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/tmp.ZO2yVo1pOb/src/examples/millionaire_prob/ABY_shared/aby_shared.cpp -o CMakeFiles/aby_shared.dir/aby_shared.cpp.s

# Object files for target aby_shared
aby_shared_OBJECTS = \
"CMakeFiles/aby_shared.dir/aby_shared.cpp.o"

# External object files for target aby_shared
aby_shared_EXTERNAL_OBJECTS =

lib/libaby_shared.so: src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/aby_shared.cpp.o
lib/libaby_shared.so: src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/build.make
lib/libaby_shared.so: src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ../../../../lib/libaby_shared.so"
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/millionaire_prob/ABY_shared && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/aby_shared.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/build: lib/libaby_shared.so
.PHONY : src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/build

src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/clean:
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/millionaire_prob/ABY_shared && $(CMAKE_COMMAND) -P CMakeFiles/aby_shared.dir/cmake_clean.cmake
.PHONY : src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/clean

src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/depend:
	cd /tmp/tmp.ZO2yVo1pOb/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /tmp/tmp.ZO2yVo1pOb /tmp/tmp.ZO2yVo1pOb/src/examples/millionaire_prob/ABY_shared /tmp/tmp.ZO2yVo1pOb/cmake-build-debug /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/millionaire_prob/ABY_shared /tmp/tmp.ZO2yVo1pOb/cmake-build-debug/src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/examples/millionaire_prob/ABY_shared/CMakeFiles/aby_shared.dir/depend

