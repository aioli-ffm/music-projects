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
include CMakeFiles/checksum_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/checksum_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/checksum_test.dir/flags.make

CMakeFiles/checksum_test.dir/tests/checksum_test.c.o: CMakeFiles/checksum_test.dir/flags.make
CMakeFiles/checksum_test.dir/tests/checksum_test.c.o: ../tests/checksum_test.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/checksum_test.dir/tests/checksum_test.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/checksum_test.dir/tests/checksum_test.c.o   -c /home/rodriguez/git-work/libsndfile/tests/checksum_test.c

CMakeFiles/checksum_test.dir/tests/checksum_test.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/checksum_test.dir/tests/checksum_test.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/rodriguez/git-work/libsndfile/tests/checksum_test.c > CMakeFiles/checksum_test.dir/tests/checksum_test.c.i

CMakeFiles/checksum_test.dir/tests/checksum_test.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/checksum_test.dir/tests/checksum_test.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/rodriguez/git-work/libsndfile/tests/checksum_test.c -o CMakeFiles/checksum_test.dir/tests/checksum_test.c.s

CMakeFiles/checksum_test.dir/tests/checksum_test.c.o.requires:

.PHONY : CMakeFiles/checksum_test.dir/tests/checksum_test.c.o.requires

CMakeFiles/checksum_test.dir/tests/checksum_test.c.o.provides: CMakeFiles/checksum_test.dir/tests/checksum_test.c.o.requires
	$(MAKE) -f CMakeFiles/checksum_test.dir/build.make CMakeFiles/checksum_test.dir/tests/checksum_test.c.o.provides.build
.PHONY : CMakeFiles/checksum_test.dir/tests/checksum_test.c.o.provides

CMakeFiles/checksum_test.dir/tests/checksum_test.c.o.provides.build: CMakeFiles/checksum_test.dir/tests/checksum_test.c.o


# Object files for target checksum_test
checksum_test_OBJECTS = \
"CMakeFiles/checksum_test.dir/tests/checksum_test.c.o"

# External object files for target checksum_test
checksum_test_EXTERNAL_OBJECTS =

checksum_test: CMakeFiles/checksum_test.dir/tests/checksum_test.c.o
checksum_test: CMakeFiles/checksum_test.dir/build.make
checksum_test: libsndfile.a
checksum_test: libtest_utils.a
checksum_test: /usr/lib/x86_64-linux-gnu/libm.so
checksum_test: libsndfile.a
checksum_test: /usr/lib/x86_64-linux-gnu/libm.so
checksum_test: /usr/lib/x86_64-linux-gnu/libogg.so
checksum_test: /usr/lib/x86_64-linux-gnu/libvorbisfile.so
checksum_test: /usr/lib/x86_64-linux-gnu/libvorbis.so
checksum_test: /usr/lib/x86_64-linux-gnu/libvorbisenc.so
checksum_test: /usr/lib/x86_64-linux-gnu/libFLAC.so
checksum_test: /usr/lib/x86_64-linux-gnu/libogg.so
checksum_test: /usr/lib/x86_64-linux-gnu/libvorbisfile.so
checksum_test: /usr/lib/x86_64-linux-gnu/libvorbis.so
checksum_test: /usr/lib/x86_64-linux-gnu/libvorbisenc.so
checksum_test: /usr/lib/x86_64-linux-gnu/libFLAC.so
checksum_test: CMakeFiles/checksum_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable checksum_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/checksum_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/checksum_test.dir/build: checksum_test

.PHONY : CMakeFiles/checksum_test.dir/build

CMakeFiles/checksum_test.dir/requires: CMakeFiles/checksum_test.dir/tests/checksum_test.c.o.requires

.PHONY : CMakeFiles/checksum_test.dir/requires

CMakeFiles/checksum_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/checksum_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/checksum_test.dir/clean

CMakeFiles/checksum_test.dir/depend:
	cd /home/rodriguez/git-work/libsndfile/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build/CMakeFiles/checksum_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/checksum_test.dir/depend

