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
include CMakeFiles/header_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/header_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/header_test.dir/flags.make

tests/header_test.c: ../cmake/CMakeAutoGenScript.cmake
tests/header_test.c: ../tests/header_test.def
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "CMakeAutoGen: generating header_test.c"
	/home/rodriguez/cmake-3.10.0/bin/cmake -DDEFINITION=/home/rodriguez/git-work/libsndfile/tests/header_test.def -DOUTPUTDIR=/home/rodriguez/git-work/libsndfile/build/tests -P /home/rodriguez/git-work/libsndfile/cmake//CMakeAutoGenScript.cmake

CMakeFiles/header_test.dir/tests/header_test.c.o: CMakeFiles/header_test.dir/flags.make
CMakeFiles/header_test.dir/tests/header_test.c.o: tests/header_test.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/header_test.dir/tests/header_test.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/header_test.dir/tests/header_test.c.o   -c /home/rodriguez/git-work/libsndfile/build/tests/header_test.c

CMakeFiles/header_test.dir/tests/header_test.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/header_test.dir/tests/header_test.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/rodriguez/git-work/libsndfile/build/tests/header_test.c > CMakeFiles/header_test.dir/tests/header_test.c.i

CMakeFiles/header_test.dir/tests/header_test.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/header_test.dir/tests/header_test.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/rodriguez/git-work/libsndfile/build/tests/header_test.c -o CMakeFiles/header_test.dir/tests/header_test.c.s

CMakeFiles/header_test.dir/tests/header_test.c.o.requires:

.PHONY : CMakeFiles/header_test.dir/tests/header_test.c.o.requires

CMakeFiles/header_test.dir/tests/header_test.c.o.provides: CMakeFiles/header_test.dir/tests/header_test.c.o.requires
	$(MAKE) -f CMakeFiles/header_test.dir/build.make CMakeFiles/header_test.dir/tests/header_test.c.o.provides.build
.PHONY : CMakeFiles/header_test.dir/tests/header_test.c.o.provides

CMakeFiles/header_test.dir/tests/header_test.c.o.provides.build: CMakeFiles/header_test.dir/tests/header_test.c.o


# Object files for target header_test
header_test_OBJECTS = \
"CMakeFiles/header_test.dir/tests/header_test.c.o"

# External object files for target header_test
header_test_EXTERNAL_OBJECTS =

header_test: CMakeFiles/header_test.dir/tests/header_test.c.o
header_test: CMakeFiles/header_test.dir/build.make
header_test: libsndfile.a
header_test: libtest_utils.a
header_test: /usr/lib/x86_64-linux-gnu/libm.so
header_test: libsndfile.a
header_test: /usr/lib/x86_64-linux-gnu/libm.so
header_test: /usr/lib/x86_64-linux-gnu/libogg.so
header_test: /usr/lib/x86_64-linux-gnu/libvorbisfile.so
header_test: /usr/lib/x86_64-linux-gnu/libvorbis.so
header_test: /usr/lib/x86_64-linux-gnu/libvorbisenc.so
header_test: /usr/lib/x86_64-linux-gnu/libFLAC.so
header_test: /usr/lib/x86_64-linux-gnu/libogg.so
header_test: /usr/lib/x86_64-linux-gnu/libvorbisfile.so
header_test: /usr/lib/x86_64-linux-gnu/libvorbis.so
header_test: /usr/lib/x86_64-linux-gnu/libvorbisenc.so
header_test: /usr/lib/x86_64-linux-gnu/libFLAC.so
header_test: CMakeFiles/header_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable header_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/header_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/header_test.dir/build: header_test

.PHONY : CMakeFiles/header_test.dir/build

CMakeFiles/header_test.dir/requires: CMakeFiles/header_test.dir/tests/header_test.c.o.requires

.PHONY : CMakeFiles/header_test.dir/requires

CMakeFiles/header_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/header_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/header_test.dir/clean

CMakeFiles/header_test.dir/depend: tests/header_test.c
	cd /home/rodriguez/git-work/libsndfile/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build/CMakeFiles/header_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/header_test.dir/depend

