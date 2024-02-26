add_subdirectory(pybind11)

set(DBOW_DIRS ../src/DBoW3/include)
set(DBOW_LIBS DBoW3)

set(DBOW2_DIRS ../src/DBoW2/include)
set(DBOW_LIBS2 DBoW2)

set(VLAD_DIRS ../src/VLAD/include)
set(VLAD_LIBS VLAD)

set(PYBIND_DIRS pybind11/include)

include_directories(${DBOW2_DIRS} ${DBOW_DIRS} ${VLAD_DIRS} ${PYBIND_DIRS})

pybind11_add_module(loopclosuretoolbox loopclosuretoolbox.cpp)
target_link_libraries(loopclosuretoolbox PRIVATE ${DBOW2_LIBS} ${DBOW_LIBS} ${VLAD_LIBS})