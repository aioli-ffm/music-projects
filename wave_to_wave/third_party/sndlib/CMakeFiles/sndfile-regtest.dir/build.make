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
include CMakeFiles/sndfile-regtest.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sndfile-regtest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sndfile-regtest.dir/flags.make

CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.o: CMakeFiles/sndfile-regtest.dir/flags.make
CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.o: ../regtest/sndfile-regtest.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.o   -c /home/rodriguez/git-work/libsndfile/regtest/sndfile-regtest.c

CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/rodriguez/git-work/libsndfile/regtest/sndfile-regtest.c > CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.i

CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/rodriguez/git-work/libsndfile/regtest/sndfile-regtest.c -o CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.s

CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.o.requires:

.PHONY : CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.o.requires

CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.o.provides: CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.o.requires
	$(MAKE) -f CMakeFiles/sndfile-regtest.dir/build.make CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.o.provides.build
.PHONY : CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.o.provides

CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.o.provides.build: CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.o


CMakeFiles/sndfile-regtest.dir/regtest/database.c.o: CMakeFiles/sndfile-regtest.dir/flags.make
CMakeFiles/sndfile-regtest.dir/regtest/database.c.o: ../regtest/database.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/sndfile-regtest.dir/regtest/database.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sndfile-regtest.dir/regtest/database.c.o   -c /home/rodriguez/git-work/libsndfile/regtest/database.c

CMakeFiles/sndfile-regtest.dir/regtest/database.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sndfile-regtest.dir/regtest/database.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/rodriguez/git-work/libsndfile/regtest/database.c > CMakeFiles/sndfile-regtest.dir/regtest/database.c.i

CMakeFiles/sndfile-regtest.dir/regtest/database.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sndfile-regtest.dir/regtest/database.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/rodriguez/git-work/libsndfile/regtest/database.c -o CMakeFiles/sndfile-regtest.dir/regtest/database.c.s

CMakeFiles/sndfile-regtest.dir/regtest/database.c.o.requires:

.PHONY : CMakeFiles/sndfile-regtest.dir/regtest/database.c.o.requires

CMakeFiles/sndfile-regtest.dir/regtest/database.c.o.provides: CMakeFiles/sndfile-regtest.dir/regtest/database.c.o.requires
	$(MAKE) -f CMakeFiles/sndfile-regtest.dir/build.make CMakeFiles/sndfile-regtest.dir/regtest/database.c.o.provides.build
.PHONY : CMakeFiles/sndfile-regtest.dir/regtest/database.c.o.provides

CMakeFiles/sndfile-regtest.dir/regtest/database.c.o.provides.build: CMakeFiles/sndfile-regtest.dir/regtest/database.c.o


CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.o: CMakeFiles/sndfile-regtest.dir/flags.make
CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.o: ../regtest/checksum.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.o   -c /home/rodriguez/git-work/libsndfile/regtest/checksum.c

CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/rodriguez/git-work/libsndfile/regtest/checksum.c > CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.i

CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/rodriguez/git-work/libsndfile/regtest/checksum.c -o CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.s

CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.o.requires:

.PHONY : CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.o.requires

CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.o.provides: CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.o.requires
	$(MAKE) -f CMakeFiles/sndfile-regtest.dir/build.make CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.o.provides.build
.PHONY : CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.o.provides

CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.o.provides.build: CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.o


# Object files for target sndfile-regtest
sndfile__regtest_OBJECTS = \
"CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.o" \
"CMakeFiles/sndfile-regtest.dir/regtest/database.c.o" \
"CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.o"

# External object files for target sndfile-regtest
sndfile__regtest_EXTERNAL_OBJECTS =

sndfile-regtest: CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.o
sndfile-regtest: CMakeFiles/sndfile-regtest.dir/regtest/database.c.o
sndfile-regtest: CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.o
sndfile-regtest: CMakeFiles/sndfile-regtest.dir/build.make
sndfile-regtest: libsndfile.so.1.0.29
sndfile-regtest: /usr/lib/x86_64-linux-gnu/libsqlite3.so
sndfile-regtest: /usr/lib/x86_64-linux-gnu/libm.so
sndfile-regtest: CMakeFiles/sndfile-regtest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rodriguez/git-work/libsndfile/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C executable sndfile-regtest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sndfile-regtest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sndfile-regtest.dir/build: sndfile-regtest

.PHONY : CMakeFiles/sndfile-regtest.dir/build

CMakeFiles/sndfile-regtest.dir/requires: CMakeFiles/sndfile-regtest.dir/regtest/sndfile-regtest.c.o.requires
CMakeFiles/sndfile-regtest.dir/requires: CMakeFiles/sndfile-regtest.dir/regtest/database.c.o.requires
CMakeFiles/sndfile-regtest.dir/requires: CMakeFiles/sndfile-regtest.dir/regtest/checksum.c.o.requires

.PHONY : CMakeFiles/sndfile-regtest.dir/requires

CMakeFiles/sndfile-regtest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sndfile-regtest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sndfile-regtest.dir/clean

CMakeFiles/sndfile-regtest.dir/depend:
	cd /home/rodriguez/git-work/libsndfile/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build /home/rodriguez/git-work/libsndfile/build/CMakeFiles/sndfile-regtest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sndfile-regtest.dir/depend

