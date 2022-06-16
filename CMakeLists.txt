cmake_minimum_required(VERSION 3.12)

project(llrt VERSION 1.0
    DESCRIPTION "Local Learning Rule Tensors Library"
    LANGUAGES CXX)

configure_file(Version.h.in Version.h)

# SANITIZETHREAD and SANITIZEADDRESS require DEBUG
if (SANITIZETHREAD AND NOT(SANITIZETHREAD_PREV) )
    set(SANITIZEADDRESS 0 CACHE STRING "" FORCE)
    set(DEBUG 1 CACHE STRING "" FORCE)
endif()

if (SANITIZEADDRESS AND NOT(SANITIZEADDRESS_PREV) )
    set(SANITIZETHREAD 0 CACHE STRING "" FORCE)
    set(DEBUG 1 CACHE STRING "" FORCE)
endif()

# If the user turned DEBUG off, turn off SANITIZETHREAD and SANITIZEADDRESS
if (NOT DEBUG)
    set(SANITIZETHREAD 0 CACHE STRING "" FORCE)
    set(SANITIZEADDRESS 0 CACHE STRING "" FORCE)
endif()

if (PROFILER)
    add_compile_definitions(PROFILER)
    message("Using PROFILER")
endif()

if (SANITIZETHREAD)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=thread")
    message("Thread sanitizer")
elseif (SANITIZEADDRESS)
    message("Address sanitizer")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
endif()

set (SANITIZETHREAD_PREV ${SANITIZETHREAD} CACHE STRING "" FORCE)
set (SANITIZEADDRESS_PREV ${SANITIZEADDRESS} CACHE STRING "" FORCE)
set (PROFILER_PREV ${PROFILER} CACHE STRING "" FORCE)

if (DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    message("Debug build")
else()
    set(CMAKE_BUILD_TYPE Release)
    message("Release build")
endif()


if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -Wfatal-errors")

# -fsanitize=address -fsanitize=pointer-compare -fsanitize=pointer-subtract -fsanitize=bounds-strict -fsanitize=alignment -fsanitize=object-size -Wfatal-errors")


set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wfatal-errors -DPERF_LOG_LEVEL=0")

add_library(Scheduler src/scheduler.cpp)

add_library(NetworkLib src/network.cpp)

add_library(NetworkPerfLogger src/network_perf_logger.cpp)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
# ${Python3_EXECUTABLE}
# ^^ required cmake 3.12

# find_package (PythonInterp 3 REQUIRED)


target_link_libraries(NetworkLib PRIVATE Scheduler NetworkPerfLogger)
target_link_libraries(Scheduler PRIVATE NetworkPerfLogger)

target_include_directories(Scheduler PRIVATE include)
target_include_directories(NetworkLib PRIVATE include)
target_include_directories(NetworkPerfLogger PRIVATE include)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(Scheduler PUBLIC Threads::Threads)

function(GenProcessLink ProcessLink SourcesList)
    add_custom_command(
        OUTPUT ${ProcessLink}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/genProcessLink.py -v -o ${ProcessLink} ${SourcesList}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/genProcessLink.py ${SourcesList}
    )
endfunction()

function(MakeLLRTObject TargetName SourcesList)
    set(TargetDir "${CMAKE_CURRENT_BINARY_DIR}/${TargetName}_gen")
    set(ProcessLink "${TargetDir}/process_link.hpp")
    set(ProcessLinkTarget "${TargetName}_gen")
    add_custom_target(${ProcessLinkTarget} DEPENDS ${ProcessLink})
    add_dependencies(${TargetName} ${ProcessLinkTarget})

    if (NOT EXISTS ${TargetDir})
        file(MAKE_DIRECTORY ${TargetDir})
    endif()
    target_include_directories(${TargetName} PRIVATE ${TargetDir})
    target_include_directories(${TargetName} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(${TargetName} PRIVATE NetworkLib)
    GenProcessLink(${ProcessLink} ${SourcesList})
endfunction()

function(MakeLLRTProgram TargetName SourcesList)
    add_executable(${TargetName} ${SourcesList})
    MakeLLRTObject(${TargetName} "${SourcesList}")
endfunction()

function(MakeLLRTLibrary TargetName SourcesList)
    add_library(${TargetName} ${SourcesList})
    MakeLLRTObject(${TargetName} "${SourcesList}")
endfunction()

MakeLLRTLibrary(SigmoidTest ${CMAKE_CURRENT_SOURCE_DIR}/tests/src/sigmoidtest.cpp)
target_include_directories(SigmoidTest PRIVATE tests/include)
MakeLLRTLibrary(AdjListTest ${CMAKE_CURRENT_SOURCE_DIR}/tests/src/adjlisttest.cpp)
target_include_directories(AdjListTest PRIVATE tests/include)
MakeLLRTLibrary(Local2DTest ${CMAKE_CURRENT_SOURCE_DIR}/tests/src/local2dtester.cpp)
target_include_directories(Local2DTest PRIVATE tests/include)
MakeLLRTProgram(Test ${CMAKE_CURRENT_SOURCE_DIR}/tests/src/test.cpp)
target_include_directories(Test PRIVATE tests/include)
target_link_libraries(Test PRIVATE SigmoidTest AdjListTest Local2DTest)

MakeLLRTProgram(ex1 "${CMAKE_CURRENT_SOURCE_DIR}/examples/ex1.cpp")
MakeLLRTProgram(ex2_linktypes "${CMAKE_CURRENT_SOURCE_DIR}/examples/ex2_linktypes.cpp")
MakeLLRTProgram(ex3_nonblocking "${CMAKE_CURRENT_SOURCE_DIR}/examples/ex3_nonblocking.cpp")
MakeLLRTProgram(ex4_combiners "${CMAKE_CURRENT_SOURCE_DIR}/examples/ex4_combiners.cpp")
MakeLLRTProgram(ex5_multineurontypes "${CMAKE_CURRENT_SOURCE_DIR}/examples/ex5_multineurontypes.cpp")

