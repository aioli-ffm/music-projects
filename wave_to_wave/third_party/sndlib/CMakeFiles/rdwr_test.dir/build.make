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
include CMakeFiles/rdwr_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rdwr_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rdwr_test.dir/flags.make

tests/rdwr_test.c: ../cmake/CMakeAutoGenScript.cmake
tests/rdwr_test.c: ../tests/rdwr_test.def
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "CMakeAutoGen: generating rdwr_test.c"
	/home/rodriguez/cmake-3.10.0/bin/cmake -DDEFINITION=/home/rodriguez/git-work/libsndfile/tests/rdwr_test.def -DOUTPUTDIR=/home/rodriguez/git-work/libsndfile/build/tests -P /home/rodriguez/git-work/libsndfile/cmake//CMakeAutoGenScript.cmake

CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.o: CMakeFiles/rdwr_test.dir/flags.make
CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.o: tests/rdwr_test.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.o   -c /home/rodriguez/git-work/libsndfile/build/tests/rdwr_test.c

CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/rodriguez/git-work/libsndfile/build/tests/rdwr_test.c > CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.i

CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/rodriguez/git-work/libsndfile/build/tests/rdwr_test.c -o CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.s

CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.o.requires:

.PHONY : CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.o.requires

CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.o.provides: CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.o.requires
	$(MAKE) -f CMakeFiles/rdwr_test.dir/build.make CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.o.provides.build
.PHONY : CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.o.provides

CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.o.provides.build: CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.o


# Object files for target rdwr_test
rdwr_test_OBJECTS = \
"CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.o"

# External object files for target rdwr_test
rdwr_test_EXTERNAL_OBJECTS =

rdwr_test: CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.o
rdwr_test: CMakeFiles/rdwr_test.dir/build.make
rdwr_test: libsndfile.a
rdwr_test: libtest_utils.a
rdwr_test: /usr/lib/x86_64-linux-gnu/libm.so
rdwr_test: libsndfile.a
rdwr_test: /usr/lib/x86_64-linux-gnu/libm.so
rdwr_test: /usr/lib/x86_64-linux-gnu/libogg.so
rdwr_test: /usr/lib/x86_64-linux-gnu/libvorbisfile.so
rdwr_test: /usr/lib/x86_64-linux-gnu/libvorbis.so
rdwr_test: /usr/lib/x86_64-linux-gnu/libvorbisenc.so
rdwr_test: /usr/lib/x86_64-linux-gnu/libFLAC.so
rdwr_test: /usr/lib/x86_64-linux-gnu/libogg.so
rdwr_test: /usr/lib/x86_64-linux-gnu/libvorbisfile.so
rdwr_test: /usr/lib/x86_64-linux-gnu/libvorbis.so
rdwr_test: /usr/lib/x86_64-linux-gnu/libvorbisenc.so
rdwr_test: /usr/lib/x86_64-linux-gnu/libFLAC.so
rdwr_test: CMakeFiles/rdwr_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable rdwr_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rdwr_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rdwr_test.dir/build: rdwr_test

.PHONY : CMakeFiles/rdwr_test.dir/build

CMakeFiles/rdwr_test.dir/requires: CMakeFiles/rdwr_test.dir/tests/rdwr_test.c.o.requires

.PHONY : CMakeFiles/rdwr_test.dir/requires

CMakeFiles/rdwr_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rdwr_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rdwr_test.dir/clean

CMakeFiles/rdwr_test.dir/depend: tests/rdwr_test.c
	cd /home/rodriguez/git-work/libsndfile/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build/CMakeFiles/rdwr_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rdwr_test.dir/depend

