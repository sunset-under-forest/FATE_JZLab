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
CMAKE_SOURCE_DIR = /tmp/tmp.89xZxGpgdR

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /tmp/tmp.89xZxGpgdR/cmake-build-debug

# Include any dependencies generated for this target.
include src/test/CMakeFiles/client_test_lib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/test/CMakeFiles/client_test_lib.dir/compiler_depend.make

# Include the progress variables for this target.
include src/test/CMakeFiles/client_test_lib.dir/progress.make

# Include the compile flags for this target's objects.
include src/test/CMakeFiles/client_test_lib.dir/flags.make

src/test/CMakeFiles/client_test_lib.dir/client_test.cpp.o: src/test/CMakeFiles/client_test_lib.dir/flags.make
src/test/CMakeFiles/client_test_lib.dir/client_test.cpp.o: ../src/test/client_test.cpp
src/test/CMakeFiles/client_test_lib.dir/client_test.cpp.o: src/test/CMakeFiles/client_test_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/tmp/tmp.89xZxGpgdR/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/test/CMakeFiles/client_test_lib.dir/client_test.cpp.o"
	cd /tmp/tmp.89xZxGpgdR/cmake-build-debug/src/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/test/CMakeFiles/client_test_lib.dir/client_test.cpp.o -MF CMakeFiles/client_test_lib.dir/client_test.cpp.o.d -o CMakeFiles/client_test_lib.dir/client_test.cpp.o -c /tmp/tmp.89xZxGpgdR/src/test/client_test.cpp

src/test/CMakeFiles/client_test_lib.dir/client_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/client_test_lib.dir/client_test.cpp.i"
	cd /tmp/tmp.89xZxGpgdR/cmake-build-debug/src/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /tmp/tmp.89xZxGpgdR/src/test/client_test.cpp > CMakeFiles/client_test_lib.dir/client_test.cpp.i

src/test/CMakeFiles/client_test_lib.dir/client_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/client_test_lib.dir/client_test.cpp.s"
	cd /tmp/tmp.89xZxGpgdR/cmake-build-debug/src/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /tmp/tmp.89xZxGpgdR/src/test/client_test.cpp -o CMakeFiles/client_test_lib.dir/client_test.cpp.s

# Object files for target client_test_lib
client_test_lib_OBJECTS = \
"CMakeFiles/client_test_lib.dir/client_test.cpp.o"

# External object files for target client_test_lib
client_test_lib_EXTERNAL_OBJECTS =

src/test/libclient_test_lib.so: src/test/CMakeFiles/client_test_lib.dir/client_test.cpp.o
src/test/libclient_test_lib.so: src/test/CMakeFiles/client_test_lib.dir/build.make
src/test/libclient_test_lib.so: extern/ABY/lib/libaby.a
src/test/libclient_test_lib.so: extern/ABY/lib/libencrypto_utils.a
src/test/libclient_test_lib.so: extern/ABY/lib/libotextension.a
src/test/libclient_test_lib.so: extern/ABY/lib/libencrypto_utils.a
src/test/libclient_test_lib.so: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.74.0
src/test/libclient_test_lib.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.74.0
src/test/libclient_test_lib.so: /usr/lib/x86_64-linux-gnu/libboost_atomic.so.1.74.0
src/test/libclient_test_lib.so: /usr/lib/x86_64-linux-gnu/libgmp.so
src/test/libclient_test_lib.so: /usr/lib/x86_64-linux-gnu/libgmpxx.so
src/test/libclient_test_lib.so: /usr/lib/x86_64-linux-gnu/libcrypto.so
src/test/libclient_test_lib.so: extern/ABY/lib/librelic_s.a
src/test/libclient_test_lib.so: extern/ABY/lib/librelic_s.a
src/test/libclient_test_lib.so: /usr/lib/x86_64-linux-gnu/libgmp.so
src/test/libclient_test_lib.so: /usr/lib/x86_64-linux-gnu/libgmp.so
src/test/libclient_test_lib.so: src/test/CMakeFiles/client_test_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/tmp/tmp.89xZxGpgdR/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libclient_test_lib.so"
	cd /tmp/tmp.89xZxGpgdR/cmake-build-debug/src/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/client_test_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/test/CMakeFiles/client_test_lib.dir/build: src/test/libclient_test_lib.so
.PHONY : src/test/CMakeFiles/client_test_lib.dir/build

src/test/CMakeFiles/client_test_lib.dir/clean:
	cd /tmp/tmp.89xZxGpgdR/cmake-build-debug/src/test && $(CMAKE_COMMAND) -P CMakeFiles/client_test_lib.dir/cmake_clean.cmake
.PHONY : src/test/CMakeFiles/client_test_lib.dir/clean

src/test/CMakeFiles/client_test_lib.dir/depend:
	cd /tmp/tmp.89xZxGpgdR/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /tmp/tmp.89xZxGpgdR /tmp/tmp.89xZxGpgdR/src/test /tmp/tmp.89xZxGpgdR/cmake-build-debug /tmp/tmp.89xZxGpgdR/cmake-build-debug/src/test /tmp/tmp.89xZxGpgdR/cmake-build-debug/src/test/CMakeFiles/client_test_lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/test/CMakeFiles/client_test_lib.dir/depend

