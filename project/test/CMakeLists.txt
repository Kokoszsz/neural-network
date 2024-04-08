cmake_minimum_required(VERSION 3.8)

set(This NueralNetTest)

set(Sources
    test_cost_function.cpp
)

add_executable(${This} ${Sources})

target_link_libraries(${This} PUBLIC
    gtest_main
    neural_net
)

add_test(
    NAME ${This}
    COMMAND ${This}
)

