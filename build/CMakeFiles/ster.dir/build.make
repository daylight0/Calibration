# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wh/slam/Calibration/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wh/slam/Calibration/build

# Include any dependencies generated for this target.
include CMakeFiles/ster.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ster.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ster.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ster.dir/flags.make

CMakeFiles/ster.dir/stereo.cpp.o: CMakeFiles/ster.dir/flags.make
CMakeFiles/ster.dir/stereo.cpp.o: /home/wh/slam/Calibration/src/stereo.cpp
CMakeFiles/ster.dir/stereo.cpp.o: CMakeFiles/ster.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wh/slam/Calibration/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ster.dir/stereo.cpp.o"
	/usr/bin/clang++-11 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ster.dir/stereo.cpp.o -MF CMakeFiles/ster.dir/stereo.cpp.o.d -o CMakeFiles/ster.dir/stereo.cpp.o -c /home/wh/slam/Calibration/src/stereo.cpp

CMakeFiles/ster.dir/stereo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ster.dir/stereo.cpp.i"
	/usr/bin/clang++-11 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wh/slam/Calibration/src/stereo.cpp > CMakeFiles/ster.dir/stereo.cpp.i

CMakeFiles/ster.dir/stereo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ster.dir/stereo.cpp.s"
	/usr/bin/clang++-11 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wh/slam/Calibration/src/stereo.cpp -o CMakeFiles/ster.dir/stereo.cpp.s

# Object files for target ster
ster_OBJECTS = \
"CMakeFiles/ster.dir/stereo.cpp.o"

# External object files for target ster
ster_EXTERNAL_OBJECTS =

/home/wh/slam/Calibration/bin/ster: CMakeFiles/ster.dir/stereo.cpp.o
/home/wh/slam/Calibration/bin/ster: CMakeFiles/ster.dir/build.make
/home/wh/slam/Calibration/bin/ster: CMakeFiles/ster.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wh/slam/Calibration/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /home/wh/slam/Calibration/bin/ster"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ster.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ster.dir/build: /home/wh/slam/Calibration/bin/ster
.PHONY : CMakeFiles/ster.dir/build

CMakeFiles/ster.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ster.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ster.dir/clean

CMakeFiles/ster.dir/depend:
	cd /home/wh/slam/Calibration/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wh/slam/Calibration/src /home/wh/slam/Calibration/src /home/wh/slam/Calibration/build /home/wh/slam/Calibration/build /home/wh/slam/Calibration/build/CMakeFiles/ster.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ster.dir/depend
