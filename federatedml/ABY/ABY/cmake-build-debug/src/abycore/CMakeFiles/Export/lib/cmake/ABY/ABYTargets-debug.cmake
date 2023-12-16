#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ABY::aby" for configuration "Debug"
set_property(TARGET ABY::aby APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(ABY::aby PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libaby.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS ABY::aby )
list(APPEND _IMPORT_CHECK_FILES_FOR_ABY::aby "${_IMPORT_PREFIX}/lib/libaby.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
