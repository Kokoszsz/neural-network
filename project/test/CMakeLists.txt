cmake_minimum_required(VERSION 3.8)

set(This NueralNetTest)

set(Tests
    testtest.cpp
    test_cost_function.cpp
    test_connection.cpp
    test_layer.cpp
)

add_executable(${This} ${Tests})

target_link_libraries(${This} PUBLIC
    gtest_main
    neural_net
)

# Add the tests
foreach(test_src ${Tests})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_test(
        NAME ${test_name}
        COMMAND ${This} --gtest_filter=${test_name}:*
    )
endforeach()
