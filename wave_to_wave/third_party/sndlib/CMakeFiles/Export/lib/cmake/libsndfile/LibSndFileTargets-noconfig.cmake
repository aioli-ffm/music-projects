#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "sndfile-static" for configuration ""
set_property(TARGET sndfile-static APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(sndfile-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsndfile.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS sndfile-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_sndfile-static "${_IMPORT_PREFIX}/lib/libsndfile.a" )

# Import target "sndfile-shared" for configuration ""
set_property(TARGET sndfile-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(sndfile-shared PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsndfile.so.1.0.29"
  IMPORTED_SONAME_NOCONFIG "libsndfile.so.1"
  )

list(APPEND _IMPORT_CHECK_TARGETS sndfile-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_sndfile-shared "${_IMPORT_PREFIX}/lib/libsndfile.so.1.0.29" )

# Import target "sndfile-info" for configuration ""
set_property(TARGET sndfile-info APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(sndfile-info PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/sndfile-info"
  )

list(APPEND _IMPORT_CHECK_TARGETS sndfile-info )
list(APPEND _IMPORT_CHECK_FILES_FOR_sndfile-info "${_IMPORT_PREFIX}/bin/sndfile-info" )

# Import target "sndfile-play" for configuration ""
set_property(TARGET sndfile-play APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(sndfile-play PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/sndfile-play"
  )

list(APPEND _IMPORT_CHECK_TARGETS sndfile-play )
list(APPEND _IMPORT_CHECK_FILES_FOR_sndfile-play "${_IMPORT_PREFIX}/bin/sndfile-play" )

# Import target "sndfile-convert" for configuration ""
set_property(TARGET sndfile-convert APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(sndfile-convert PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/sndfile-convert"
  )

list(APPEND _IMPORT_CHECK_TARGETS sndfile-convert )
list(APPEND _IMPORT_CHECK_FILES_FOR_sndfile-convert "${_IMPORT_PREFIX}/bin/sndfile-convert" )

# Import target "sndfile-cmp" for configuration ""
set_property(TARGET sndfile-cmp APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(sndfile-cmp PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/sndfile-cmp"
  )

list(APPEND _IMPORT_CHECK_TARGETS sndfile-cmp )
list(APPEND _IMPORT_CHECK_FILES_FOR_sndfile-cmp "${_IMPORT_PREFIX}/bin/sndfile-cmp" )

# Import target "sndfile-metadata-set" for configuration ""
set_property(TARGET sndfile-metadata-set APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(sndfile-metadata-set PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/sndfile-metadata-set"
  )

list(APPEND _IMPORT_CHECK_TARGETS sndfile-metadata-set )
list(APPEND _IMPORT_CHECK_FILES_FOR_sndfile-metadata-set "${_IMPORT_PREFIX}/bin/sndfile-metadata-set" )

# Import target "sndfile-metadata-get" for configuration ""
set_property(TARGET sndfile-metadata-get APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(sndfile-metadata-get PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/sndfile-metadata-get"
  )

list(APPEND _IMPORT_CHECK_TARGETS sndfile-metadata-get )
list(APPEND _IMPORT_CHECK_FILES_FOR_sndfile-metadata-get "${_IMPORT_PREFIX}/bin/sndfile-metadata-get" )

# Import target "sndfile-interleave" for configuration ""
set_property(TARGET sndfile-interleave APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(sndfile-interleave PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/sndfile-interleave"
  )

list(APPEND _IMPORT_CHECK_TARGETS sndfile-interleave )
list(APPEND _IMPORT_CHECK_FILES_FOR_sndfile-interleave "${_IMPORT_PREFIX}/bin/sndfile-interleave" )

# Import target "sndfile-deinterleave" for configuration ""
set_property(TARGET sndfile-deinterleave APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(sndfile-deinterleave PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/sndfile-deinterleave"
  )

list(APPEND _IMPORT_CHECK_TARGETS sndfile-deinterleave )
list(APPEND _IMPORT_CHECK_FILES_FOR_sndfile-deinterleave "${_IMPORT_PREFIX}/bin/sndfile-deinterleave" )

# Import target "sndfile-concat" for configuration ""
set_property(TARGET sndfile-concat APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(sndfile-concat PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/sndfile-concat"
  )

list(APPEND _IMPORT_CHECK_TARGETS sndfile-concat )
list(APPEND _IMPORT_CHECK_FILES_FOR_sndfile-concat "${_IMPORT_PREFIX}/bin/sndfile-concat" )

# Import target "sndfile-salvage" for configuration ""
set_property(TARGET sndfile-salvage APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(sndfile-salvage PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/sndfile-salvage"
  )

list(APPEND _IMPORT_CHECK_TARGETS sndfile-salvage )
list(APPEND _IMPORT_CHECK_FILES_FOR_sndfile-salvage "${_IMPORT_PREFIX}/bin/sndfile-salvage" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
