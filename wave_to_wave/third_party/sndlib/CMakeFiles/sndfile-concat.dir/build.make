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
include CMakeFiles/sndfile-concat.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sndfile-concat.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sndfile-concat.dir/flags.make

CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.o: CMakeFiles/sndfile-concat.dir/flags.make
CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.o: ../programs/sndfile-concat.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.o   -c /home/rodriguez/git-work/libsndfile/programs/sndfile-concat.c

CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/rodriguez/git-work/libsndfile/programs/sndfile-concat.c > CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.i

CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/rodriguez/git-work/libsndfile/programs/sndfile-concat.c -o CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.s

CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.o.requires:

.PHONY : CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.o.requires

CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.o.provides: CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.o.requires
	$(MAKE) -f CMakeFiles/sndfile-concat.dir/build.make CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.o.provides.build
.PHONY : CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.o.provides

CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.o.provides.build: CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.o


CMakeFiles/sndfile-concat.dir/programs/common.c.o: CMakeFiles/sndfile-concat.dir/flags.make
CMakeFiles/sndfile-concat.dir/programs/common.c.o: ../programs/common.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/sndfile-concat.dir/programs/common.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sndfile-concat.dir/programs/common.c.o   -c /home/rodriguez/git-work/libsndfile/programs/common.c

CMakeFiles/sndfile-concat.dir/programs/common.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sndfile-concat.dir/programs/common.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/rodriguez/git-work/libsndfile/programs/common.c > CMakeFiles/sndfile-concat.dir/programs/common.c.i

CMakeFiles/sndfile-concat.dir/programs/common.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sndfile-concat.dir/programs/common.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/rodriguez/git-work/libsndfile/programs/common.c -o CMakeFiles/sndfile-concat.dir/programs/common.c.s

CMakeFiles/sndfile-concat.dir/programs/common.c.o.requires:

.PHONY : CMakeFiles/sndfile-concat.dir/programs/common.c.o.requires

CMakeFiles/sndfile-concat.dir/programs/common.c.o.provides: CMakeFiles/sndfile-concat.dir/programs/common.c.o.requires
	$(MAKE) -f CMakeFiles/sndfile-concat.dir/build.make CMakeFiles/sndfile-concat.dir/programs/common.c.o.provides.build
.PHONY : CMakeFiles/sndfile-concat.dir/programs/common.c.o.provides

CMakeFiles/sndfile-concat.dir/programs/common.c.o.provides.build: CMakeFiles/sndfile-concat.dir/programs/common.c.o


# Object files for target sndfile-concat
sndfile__concat_OBJECTS = \
"CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.o" \
"CMakeFiles/sndfile-concat.dir/programs/common.c.o"

# External object files for target sndfile-concat
sndfile__concat_EXTERNAL_OBJECTS =

sndfile-concat: CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.o
sndfile-concat: CMakeFiles/sndfile-concat.dir/programs/common.c.o
sndfile-concat: CMakeFiles/sndfile-concat.dir/build.make
sndfile-concat: libsndfile.so.1.0.29
sndfile-concat: CMakeFiles/sndfile-concat.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable sndfile-concat"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sndfile-concat.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sndfile-concat.dir/build: sndfile-concat

.PHONY : CMakeFiles/sndfile-concat.dir/build

CMakeFiles/sndfile-concat.dir/requires: CMakeFiles/sndfile-concat.dir/programs/sndfile-concat.c.o.requires
CMakeFiles/sndfile-concat.dir/requires: CMakeFiles/sndfile-concat.dir/programs/common.c.o.requires

.PHONY : CMakeFiles/sndfile-concat.dir/requires

CMakeFiles/sndfile-concat.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sndfile-concat.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sndfile-concat.dir/clean

CMakeFiles/sndfile-concat.dir/depend:
	cd /home/rodriguez/git-work/libsndfile/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build/CMakeFiles/sndfile-concat.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sndfile-concat.dir/depend

