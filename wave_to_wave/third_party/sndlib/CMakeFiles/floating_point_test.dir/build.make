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
include CMakeFiles/floating_point_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/floating_point_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/floating_point_test.dir/flags.make

tests/floating_point_test.c: ../cmake/CMakeAutoGenScript.cmake
tests/floating_point_test.c: ../tests/floating_point_test.def
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "CMakeAutoGen: generating floating_point_test.c"
	/home/rodriguez/cmake-3.10.0/bin/cmake -DDEFINITION=/home/rodriguez/git-work/libsndfile/tests/floating_point_test.def -DOUTPUTDIR=/home/rodriguez/git-work/libsndfile/build/tests -P /home/rodriguez/git-work/libsndfile/cmake//CMakeAutoGenScript.cmake

CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.o: CMakeFiles/floating_point_test.dir/flags.make
CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.o: ../tests/dft_cmp.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.o   -c /home/rodriguez/git-work/libsndfile/tests/dft_cmp.c

CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/rodriguez/git-work/libsndfile/tests/dft_cmp.c > CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.i

CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/rodriguez/git-work/libsndfile/tests/dft_cmp.c -o CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.s

CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.o.requires:

.PHONY : CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.o.requires

CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.o.provides: CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.o.requires
	$(MAKE) -f CMakeFiles/floating_point_test.dir/build.make CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.o.provides.build
.PHONY : CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.o.provides

CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.o.provides.build: CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.o


CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.o: CMakeFiles/floating_point_test.dir/flags.make
CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.o: tests/floating_point_test.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.o   -c /home/rodriguez/git-work/libsndfile/build/tests/floating_point_test.c

CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/rodriguez/git-work/libsndfile/build/tests/floating_point_test.c > CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.i

CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/rodriguez/git-work/libsndfile/build/tests/floating_point_test.c -o CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.s

CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.o.requires:

.PHONY : CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.o.requires

CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.o.provides: CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.o.requires
	$(MAKE) -f CMakeFiles/floating_point_test.dir/build.make CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.o.provides.build
.PHONY : CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.o.provides

CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.o.provides.build: CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.o


# Object files for target floating_point_test
floating_point_test_OBJECTS = \
"CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.o" \
"CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.o"

# External object files for target floating_point_test
floating_point_test_EXTERNAL_OBJECTS =

floating_point_test: CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.o
floating_point_test: CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.o
floating_point_test: CMakeFiles/floating_point_test.dir/build.make
floating_point_test: libsndfile.a
floating_point_test: libtest_utils.a
floating_point_test: /usr/lib/x86_64-linux-gnu/libm.so
floating_point_test: libsndfile.a
floating_point_test: /usr/lib/x86_64-linux-gnu/libm.so
floating_point_test: /usr/lib/x86_64-linux-gnu/libogg.so
floating_point_test: /usr/lib/x86_64-linux-gnu/libvorbisfile.so
floating_point_test: /usr/lib/x86_64-linux-gnu/libvorbis.so
floating_point_test: /usr/lib/x86_64-linux-gnu/libvorbisenc.so
floating_point_test: /usr/lib/x86_64-linux-gnu/libFLAC.so
floating_point_test: /usr/lib/x86_64-linux-gnu/libogg.so
floating_point_test: /usr/lib/x86_64-linux-gnu/libvorbisfile.so
floating_point_test: /usr/lib/x86_64-linux-gnu/libvorbis.so
floating_point_test: /usr/lib/x86_64-linux-gnu/libvorbisenc.so
floating_point_test: /usr/lib/x86_64-linux-gnu/libFLAC.so
floating_point_test: CMakeFiles/floating_point_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C executable floating_point_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/floating_point_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/floating_point_test.dir/build: floating_point_test

.PHONY : CMakeFiles/floating_point_test.dir/build

CMakeFiles/floating_point_test.dir/requires: CMakeFiles/floating_point_test.dir/tests/dft_cmp.c.o.requires
CMakeFiles/floating_point_test.dir/requires: CMakeFiles/floating_point_test.dir/tests/floating_point_test.c.o.requires

.PHONY : CMakeFiles/floating_point_test.dir/requires

CMakeFiles/floating_point_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/floating_point_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/floating_point_test.dir/clean

CMakeFiles/floating_point_test.dir/depend: tests/floating_point_test.c
	cd /home/rodriguez/git-work/libsndfile/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build/CMakeFiles/floating_point_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/floating_point_test.dir/depend

