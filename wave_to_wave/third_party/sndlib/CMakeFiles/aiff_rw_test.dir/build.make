# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/rodriguez/cmake-3.10.0/bin/cmake

# The command to remove a file.
RM = /home/rodriguez/cmake-3.10.0/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rodriguez/git-work/libsndfile

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rodriguez/git-work/libsndfile/build

# Include any dependencies generated for this target.
include CMakeFiles/aiff_rw_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/aiff_rw_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/aiff_rw_test.dir/flags.make

CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.o: CMakeFiles/aiff_rw_test.dir/flags.make
CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.o: ../tests/aiff_rw_test.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.o   -c /home/rodriguez/git-work/libsndfile/tests/aiff_rw_test.c

CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/rodriguez/git-work/libsndfile/tests/aiff_rw_test.c > CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.i

CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/rodriguez/git-work/libsndfile/tests/aiff_rw_test.c -o CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.s

CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.o.requires:

.PHONY : CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.o.requires

CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.o.provides: CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.o.requires
	$(MAKE) -f CMakeFiles/aiff_rw_test.dir/build.make CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.o.provides.build
.PHONY : CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.o.provides

CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.o.provides.build: CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.o


# Object files for target aiff_rw_test
aiff_rw_test_OBJECTS = \
"CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.o"

# External object files for target aiff_rw_test
aiff_rw_test_EXTERNAL_OBJECTS =

aiff_rw_test: CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.o
aiff_rw_test: CMakeFiles/aiff_rw_test.dir/build.make
aiff_rw_test: libsndfile.a
aiff_rw_test: libtest_utils.a
aiff_rw_test: /usr/lib/x86_64-linux-gnu/libm.so
aiff_rw_test: libsndfile.a
aiff_rw_test: /usr/lib/x86_64-linux-gnu/libm.so
aiff_rw_test: /usr/lib/x86_64-linux-gnu/libogg.so
aiff_rw_test: /usr/lib/x86_64-linux-gnu/libvorbisfile.so
aiff_rw_test: /usr/lib/x86_64-linux-gnu/libvorbis.so
aiff_rw_test: /usr/lib/x86_64-linux-gnu/libvorbisenc.so
aiff_rw_test: /usr/lib/x86_64-linux-gnu/libFLAC.so
aiff_rw_test: /usr/lib/x86_64-linux-gnu/libogg.so
aiff_rw_test: /usr/lib/x86_64-linux-gnu/libvorbisfile.so
aiff_rw_test: /usr/lib/x86_64-linux-gnu/libvorbis.so
aiff_rw_test: /usr/lib/x86_64-linux-gnu/libvorbisenc.so
aiff_rw_test: /usr/lib/x86_64-linux-gnu/libFLAC.so
aiff_rw_test: CMakeFiles/aiff_rw_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable aiff_rw_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/aiff_rw_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/aiff_rw_test.dir/build: aiff_rw_test

.PHONY : CMakeFiles/aiff_rw_test.dir/build

CMakeFiles/aiff_rw_test.dir/requires: CMakeFiles/aiff_rw_test.dir/tests/aiff_rw_test.c.o.requires

.PHONY : CMakeFiles/aiff_rw_test.dir/requires

CMakeFiles/aiff_rw_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/aiff_rw_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/aiff_rw_test.dir/clean

CMakeFiles/aiff_rw_test.dir/depend:
	cd /home/rodriguez/git-work/libsndfile/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build/CMakeFiles/aiff_rw_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/aiff_rw_test.dir/depend
