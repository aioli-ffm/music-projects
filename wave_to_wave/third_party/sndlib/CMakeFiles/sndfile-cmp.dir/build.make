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
include CMakeFiles/sndfile-cmp.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sndfile-cmp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sndfile-cmp.dir/flags.make

CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.o: CMakeFiles/sndfile-cmp.dir/flags.make
CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.o: ../programs/sndfile-cmp.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.o   -c /home/rodriguez/git-work/libsndfile/programs/sndfile-cmp.c

CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/rodriguez/git-work/libsndfile/programs/sndfile-cmp.c > CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.i

CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/rodriguez/git-work/libsndfile/programs/sndfile-cmp.c -o CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.s

CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.o.requires:

.PHONY : CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.o.requires

CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.o.provides: CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.o.requires
	$(MAKE) -f CMakeFiles/sndfile-cmp.dir/build.make CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.o.provides.build
.PHONY : CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.o.provides

CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.o.provides.build: CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.o


CMakeFiles/sndfile-cmp.dir/programs/common.c.o: CMakeFiles/sndfile-cmp.dir/flags.make
CMakeFiles/sndfile-cmp.dir/programs/common.c.o: ../programs/common.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/sndfile-cmp.dir/programs/common.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sndfile-cmp.dir/programs/common.c.o   -c /home/rodriguez/git-work/libsndfile/programs/common.c

CMakeFiles/sndfile-cmp.dir/programs/common.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sndfile-cmp.dir/programs/common.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/rodriguez/git-work/libsndfile/programs/common.c > CMakeFiles/sndfile-cmp.dir/programs/common.c.i

CMakeFiles/sndfile-cmp.dir/programs/common.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sndfile-cmp.dir/programs/common.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/rodriguez/git-work/libsndfile/programs/common.c -o CMakeFiles/sndfile-cmp.dir/programs/common.c.s

CMakeFiles/sndfile-cmp.dir/programs/common.c.o.requires:

.PHONY : CMakeFiles/sndfile-cmp.dir/programs/common.c.o.requires

CMakeFiles/sndfile-cmp.dir/programs/common.c.o.provides: CMakeFiles/sndfile-cmp.dir/programs/common.c.o.requires
	$(MAKE) -f CMakeFiles/sndfile-cmp.dir/build.make CMakeFiles/sndfile-cmp.dir/programs/common.c.o.provides.build
.PHONY : CMakeFiles/sndfile-cmp.dir/programs/common.c.o.provides

CMakeFiles/sndfile-cmp.dir/programs/common.c.o.provides.build: CMakeFiles/sndfile-cmp.dir/programs/common.c.o


# Object files for target sndfile-cmp
sndfile__cmp_OBJECTS = \
"CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.o" \
"CMakeFiles/sndfile-cmp.dir/programs/common.c.o"

# External object files for target sndfile-cmp
sndfile__cmp_EXTERNAL_OBJECTS =

sndfile-cmp: CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.o
sndfile-cmp: CMakeFiles/sndfile-cmp.dir/programs/common.c.o
sndfile-cmp: CMakeFiles/sndfile-cmp.dir/build.make
sndfile-cmp: libsndfile.so.1.0.29
sndfile-cmp: CMakeFiles/sndfile-cmp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable sndfile-cmp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sndfile-cmp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sndfile-cmp.dir/build: sndfile-cmp

.PHONY : CMakeFiles/sndfile-cmp.dir/build

CMakeFiles/sndfile-cmp.dir/requires: CMakeFiles/sndfile-cmp.dir/programs/sndfile-cmp.c.o.requires
CMakeFiles/sndfile-cmp.dir/requires: CMakeFiles/sndfile-cmp.dir/programs/common.c.o.requires

.PHONY : CMakeFiles/sndfile-cmp.dir/requires

CMakeFiles/sndfile-cmp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sndfile-cmp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sndfile-cmp.dir/clean

CMakeFiles/sndfile-cmp.dir/depend:
	cd /home/rodriguez/git-work/libsndfile/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build/CMakeFiles/sndfile-cmp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sndfile-cmp.dir/depend

