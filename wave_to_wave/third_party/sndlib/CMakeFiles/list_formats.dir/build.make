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
include CMakeFiles/list_formats.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/list_formats.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/list_formats.dir/flags.make

CMakeFiles/list_formats.dir/examples/list_formats.c.o: CMakeFiles/list_formats.dir/flags.make
CMakeFiles/list_formats.dir/examples/list_formats.c.o: ../examples/list_formats.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/list_formats.dir/examples/list_formats.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/list_formats.dir/examples/list_formats.c.o   -c /home/rodriguez/git-work/libsndfile/examples/list_formats.c

CMakeFiles/list_formats.dir/examples/list_formats.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/list_formats.dir/examples/list_formats.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/rodriguez/git-work/libsndfile/examples/list_formats.c > CMakeFiles/list_formats.dir/examples/list_formats.c.i

CMakeFiles/list_formats.dir/examples/list_formats.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/list_formats.dir/examples/list_formats.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/rodriguez/git-work/libsndfile/examples/list_formats.c -o CMakeFiles/list_formats.dir/examples/list_formats.c.s

CMakeFiles/list_formats.dir/examples/list_formats.c.o.requires:

.PHONY : CMakeFiles/list_formats.dir/examples/list_formats.c.o.requires

CMakeFiles/list_formats.dir/examples/list_formats.c.o.provides: CMakeFiles/list_formats.dir/examples/list_formats.c.o.requires
	$(MAKE) -f CMakeFiles/list_formats.dir/build.make CMakeFiles/list_formats.dir/examples/list_formats.c.o.provides.build
.PHONY : CMakeFiles/list_formats.dir/examples/list_formats.c.o.provides

CMakeFiles/list_formats.dir/examples/list_formats.c.o.provides.build: CMakeFiles/list_formats.dir/examples/list_formats.c.o


# Object files for target list_formats
list_formats_OBJECTS = \
"CMakeFiles/list_formats.dir/examples/list_formats.c.o"

# External object files for target list_formats
list_formats_EXTERNAL_OBJECTS =

list_formats: CMakeFiles/list_formats.dir/examples/list_formats.c.o
list_formats: CMakeFiles/list_formats.dir/build.make
list_formats: libsndfile.so.1.0.29
list_formats: CMakeFiles/list_formats.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable list_formats"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/list_formats.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/list_formats.dir/build: list_formats

.PHONY : CMakeFiles/list_formats.dir/build

CMakeFiles/list_formats.dir/requires: CMakeFiles/list_formats.dir/examples/list_formats.c.o.requires

.PHONY : CMakeFiles/list_formats.dir/requires

CMakeFiles/list_formats.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/list_formats.dir/cmake_clean.cmake
.PHONY : CMakeFiles/list_formats.dir/clean

CMakeFiles/list_formats.dir/depend:
	cd /home/rodriguez/git-work/libsndfile/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build/CMakeFiles/list_formats.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/list_formats.dir/depend
