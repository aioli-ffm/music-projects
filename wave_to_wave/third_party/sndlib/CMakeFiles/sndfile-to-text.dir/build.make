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
include CMakeFiles/sndfile-to-text.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sndfile-to-text.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sndfile-to-text.dir/flags.make

CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.o: CMakeFiles/sndfile-to-text.dir/flags.make
CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.o: ../examples/sndfile-to-text.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.o   -c /home/rodriguez/git-work/libsndfile/examples/sndfile-to-text.c

CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/rodriguez/git-work/libsndfile/examples/sndfile-to-text.c > CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.i

CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/rodriguez/git-work/libsndfile/examples/sndfile-to-text.c -o CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.s

CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.o.requires:

.PHONY : CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.o.requires

CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.o.provides: CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.o.requires
	$(MAKE) -f CMakeFiles/sndfile-to-text.dir/build.make CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.o.provides.build
.PHONY : CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.o.provides

CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.o.provides.build: CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.o


# Object files for target sndfile-to-text
sndfile__to__text_OBJECTS = \
"CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.o"

# External object files for target sndfile-to-text
sndfile__to__text_EXTERNAL_OBJECTS =

sndfile-to-text: CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.o
sndfile-to-text: CMakeFiles/sndfile-to-text.dir/build.make
sndfile-to-text: libsndfile.so.1.0.29
sndfile-to-text: CMakeFiles/sndfile-to-text.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable sndfile-to-text"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sndfile-to-text.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sndfile-to-text.dir/build: sndfile-to-text

.PHONY : CMakeFiles/sndfile-to-text.dir/build

CMakeFiles/sndfile-to-text.dir/requires: CMakeFiles/sndfile-to-text.dir/examples/sndfile-to-text.c.o.requires

.PHONY : CMakeFiles/sndfile-to-text.dir/requires

CMakeFiles/sndfile-to-text.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sndfile-to-text.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sndfile-to-text.dir/clean

CMakeFiles/sndfile-to-text.dir/depend:
	cd /home/rodriguez/git-work/libsndfile/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build/CMakeFiles/sndfile-to-text.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sndfile-to-text.dir/depend
