# CMake generated Testfile for 
# Source directory: C:/Users/kaczok/Desktop/neural-network/project/test
# Build directory: C:/Users/kaczok/Desktop/neural-network/project/build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(NueralNetTest "C:/Users/kaczok/Desktop/neural-network/project/build/test/Debug/NueralNetTest.exe")
  set_tests_properties(NueralNetTest PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/kaczok/Desktop/neural-network/project/test/CMakeLists.txt;16;add_test;C:/Users/kaczok/Desktop/neural-network/project/test/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(NueralNetTest "C:/Users/kaczok/Desktop/neural-network/project/build/test/Release/NueralNetTest.exe")
  set_tests_properties(NueralNetTest PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/kaczok/Desktop/neural-network/project/test/CMakeLists.txt;16;add_test;C:/Users/kaczok/Desktop/neural-network/project/test/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(NueralNetTest "C:/Users/kaczok/Desktop/neural-network/project/build/test/MinSizeRel/NueralNetTest.exe")
  set_tests_properties(NueralNetTest PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/kaczok/Desktop/neural-network/project/test/CMakeLists.txt;16;add_test;C:/Users/kaczok/Desktop/neural-network/project/test/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(NueralNetTest "C:/Users/kaczok/Desktop/neural-network/project/build/test/RelWithDebInfo/NueralNetTest.exe")
  set_tests_properties(NueralNetTest PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/kaczok/Desktop/neural-network/project/test/CMakeLists.txt;16;add_test;C:/Users/kaczok/Desktop/neural-network/project/test/CMakeLists.txt;0;")
else()
  add_test(NueralNetTest NOT_AVAILABLE)
endif()
