#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ENCRYPTO_utils::encrypto_utils" for configuration "Debug"
set_property(TARGET ENCRYPTO_utils::encrypto_utils APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(ENCRYPTO_utils::encrypto_utils PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libencrypto_utils.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS ENCRYPTO_utils::encrypto_utils )
list(APPEND _IMPORT_CHECK_FILES_FOR_ENCRYPTO_utils::encrypto_utils "${_IMPORT_PREFIX}/lib/libencrypto_utils.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
