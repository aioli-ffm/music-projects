cmake_minimum_required(VERSION 2.8)

project(tests)

include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(
    catch
    PREFIX ${CMAKE_BINARY_DIR}/catch
    GIT_REPOSITORY https://github.com/catchorg/Catch2
    TIMEOUT 10
    UPDATE_COMMAND ${GIT_EXECUTABLE} pull
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
   )

ExternalProject_Get_Property(catch source_dir)
set(CATCH_INCLUDE_DIR ${source_dir}/single_include CACHE INTERNAL "Path to include folder for Catch")
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

add_custom_target(test)
set_target_properties(test PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)

set(TEST_FILES utest_helpers.cpp utest_signal.cpp)

foreach(file ${TEST_FILES})

  get_filename_component(SOURCE_NAME ${file} NAME_WE)
  message(STATUS " Adding Test ${SOURCE_NAME} ")
  
  add_executable(${SOURCE_NAME} main_utest.cpp ${file} )
  add_dependencies(${SOURCE_NAME} catch)

  set_target_properties(${SOURCE_NAME}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
    FOLDER "${IDE_FOLDER}"
  )

  target_link_libraries(${SOURCE_NAME}
    PRIVATE
    ${DEFAULT_LINKER_OPTIONS}
  )

  target_link_libraries(${SOURCE_NAME}
    PRIVATE
    Catch
  )

  add_dependencies(test ${SOURCE_NAME})
  add_custom_command(TARGET test POST_BUILD 
      COMMAND $<TARGET_FILE:${SOURCE_NAME}>)

endforeach()


