# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_SOURCE_DIR = /home/babrakadabra/second_course/Parallelizm_NSU/task2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/babrakadabra/second_course/Parallelizm_NSU/task2/build

# Include any dependencies generated for this target.
include CMakeFiles/1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/1.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/1.dir/flags.make

CMakeFiles/1.dir/1.cpp.o: CMakeFiles/1.dir/flags.make
CMakeFiles/1.dir/1.cpp.o: /home/babrakadabra/second_course/Parallelizm_NSU/task2/1.cpp
CMakeFiles/1.dir/1.cpp.o: CMakeFiles/1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/babrakadabra/second_course/Parallelizm_NSU/task2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/1.dir/1.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/1.dir/1.cpp.o -MF CMakeFiles/1.dir/1.cpp.o.d -o CMakeFiles/1.dir/1.cpp.o -c /home/babrakadabra/second_course/Parallelizm_NSU/task2/1.cpp

CMakeFiles/1.dir/1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/1.dir/1.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/babrakadabra/second_course/Parallelizm_NSU/task2/1.cpp > CMakeFiles/1.dir/1.cpp.i

CMakeFiles/1.dir/1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/1.dir/1.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/babrakadabra/second_course/Parallelizm_NSU/task2/1.cpp -o CMakeFiles/1.dir/1.cpp.s

# Object files for target 1
1_OBJECTS = \
"CMakeFiles/1.dir/1.cpp.o"

# External object files for target 1
1_EXTERNAL_OBJECTS =

1 : CMakeFiles/1.dir/1.cpp.o
1 : CMakeFiles/1.dir/build.make
1 : CMakeFiles/1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/babrakadabra/second_course/Parallelizm_NSU/task2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable 1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/1.dir/build: 1
.PHONY : CMakeFiles/1.dir/build

CMakeFiles/1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/1.dir/clean

CMakeFiles/1.dir/depend:
	cd /home/babrakadabra/second_course/Parallelizm_NSU/task2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/babrakadabra/second_course/Parallelizm_NSU/task2 /home/babrakadabra/second_course/Parallelizm_NSU/task2 /home/babrakadabra/second_course/Parallelizm_NSU/task2/build /home/babrakadabra/second_course/Parallelizm_NSU/task2/build /home/babrakadabra/second_course/Parallelizm_NSU/task2/build/CMakeFiles/1.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/1.dir/depend

