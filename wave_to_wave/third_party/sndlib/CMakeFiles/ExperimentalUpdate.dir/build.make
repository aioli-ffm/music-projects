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

# Utility rule file for ExperimentalUpdate.

# Include the progress variables for this target.
include CMakeFiles/ExperimentalUpdate.dir/progress.make

CMakeFiles/ExperimentalUpdate:
	/home/rodriguez/cmake-3.10.0/bin/ctest -D ExperimentalUpdate

ExperimentalUpdate: CMakeFiles/ExperimentalUpdate
ExperimentalUpdate: CMakeFiles/ExperimentalUpdate.dir/build.make

.PHONY : ExperimentalUpdate

# Rule to build all files generated by this target.
CMakeFiles/ExperimentalUpdate.dir/build: ExperimentalUpdate

.PHONY : CMakeFiles/ExperimentalUpdate.dir/build

CMakeFiles/ExperimentalUpdate.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ExperimentalUpdate.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ExperimentalUpdate.dir/clean

CMakeFiles/ExperimentalUpdate.dir/depend:
	cd /home/rodriguez/git-work/libsndfile/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build/CMakeFiles/ExperimentalUpdate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ExperimentalUpdate.dir/depend

