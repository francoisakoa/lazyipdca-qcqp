cmake_minimum_required(VERSION 3.18)
project(lazyipdca_qcqp VERSION 0.4.28 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)



# Exact sparse Phase-I Jv is trajectory-preserving only when the compiler
# retains strict reduction semantics.  Source-level macro detection remains
# authoritative; this warning exposes common inherited fast-FP flags early.
string(TOLOWER
  "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CMAKE_CXX_FLAGS_MINSIZEREL}"
  LAZYIPDCA_CONFIGURED_CXX_FLAGS_LOWER)
if(LAZYIPDCA_CONFIGURED_CXX_FLAGS_LOWER MATCHES
   "(^|[ ;])(-ffast-math|-ofast|-funsafe-math-optimizations|-fassociative-math|-fno-signed-zeros|-fno-trapping-math)([ ;]|$)")
  message(WARNING
    "Potentially reassociating floating-point flags detected: the exact "
    "runtime guard may latch Phase-I Jv onto the historical dense reduction.")
endif()

option(LAZYIPDCA_BUILD_TESTS "Build tests" ON)
if(LAZYIPDCA_BUILD_TESTS)
  enable_testing()
  add_executable(test_affine_face_rewrite tests/test_affine_face_rewrite.cpp)
  target_include_directories(test_affine_face_rewrite PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include ${LAZYIPDCA_BOOST_INCLUDE_DIR})
  add_test(NAME affine_face_rewrite COMMAND test_affine_face_rewrite)
  add_executable(test_adaptive_quadratic_probe tests/test_adaptive_quadratic_probe.cpp)
  target_include_directories(test_adaptive_quadratic_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME adaptive_quadratic_probe COMMAND test_adaptive_quadratic_probe)
  add_executable(test_historical_equality_watch tests/test_historical_equality_watch.cpp)
  target_include_directories(test_historical_equality_watch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME historical_equality_watch COMMAND test_historical_equality_watch)
endif()
option(LAZYIPDCA_WITH_HSL "Build the runtime MA27 adapter (external library required)" ON)
option(LAZYIPDCA_WITH_IPOPT "Build the optional Ipopt convex-subproblem and direct-QCQP interfaces" OFF)
option(LAZYIPDCA_WITH_LDL "Enable historical SuiteSparse LDL linear backend when found" ON)
option(LAZYIPDCA_WITH_QDLDL "Enable sparse QDLDL linear backend when found" ON)
option(LAZYIPDCA_REQUIRE_HISTORICAL_LDL "Fail configure unless SuiteSparse LDL+METIS is available" OFF)
option(LAZYIPDCA_WITH_MUMPS "Enable sparse MUMPS linear backend when found" ON)
option(LAZYIPDCA_WITH_METIS "Enable METIS fill-reducing ordering when found" ON)
option(LAZYIPDCA_REQUIRE_SPARSE_BACKENDS "Fail configure unless QDLDL+METIS and MUMPS+METIS are available" OFF)
option(LAZYIPDCA_NATIVE_ARCH "Use -march=native for local performance builds" OFF)
option(LAZYIPDCA_WITH_OPENMP "Enable optional equality-row OpenMP kernels" ON)
set(IPOPT_ROOT "" CACHE PATH "Optional Ipopt root (include/coin-or and lib/libipopt)")
set(LDL_ROOT "" CACHE PATH "Optional SuiteSparse LDL installation prefix")
set(QDLDL_ROOT "" CACHE PATH "Optional QDLDL installation prefix")
set(MUMPS_ROOT "" CACHE PATH "Optional MUMPS installation prefix")
set(METIS_ROOT "" CACHE PATH "Optional METIS installation prefix")

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_library(lazyipdca_qcqp
  src/condensed_retraction.cpp
  src/curvature.cpp
  src/direct_ipopt.cpp
  src/inner_ipm.cpp
  src/inner_lazyqp.cpp
  src/inner_socp.cpp
  src/inner_ipopt.cpp
  src/inner_sqp.cpp
  src/linear_solver.cpp
  src/lazy_newton.cpp
  src/atomic_newton.cpp
  src/io_compact.cpp
  src/inner_sparse_single_loop.cpp
  src/original_kkt_gmres.cpp
  src/io.cpp
  src/restoration.cpp
  src/rank_presolve.cpp
  src/free_epigraph_presolve.cpp
  src/solver.cpp
)
find_path(LAZYIPDCA_BOOST_INCLUDE_DIR boost/multiprecision/cpp_int.hpp
  HINTS ${BOOST_ROOT}/include $ENV{CONDA_PREFIX}/include)
if(NOT LAZYIPDCA_BOOST_INCLUDE_DIR)
  message(FATAL_ERROR "Candidate study requires Boost.Multiprecision headers; set LAZYIPDCA_BOOST_INCLUDE_DIR")
endif()
target_include_directories(lazyipdca_qcqp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(lazyipdca_qcqp PUBLIC ${LAZYIPDCA_BOOST_INCLUDE_DIR})
target_link_libraries(lazyipdca_qcqp PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
target_compile_options(lazyipdca_qcqp PRIVATE -Wall -Wextra -Wpedantic)
if(LAZYIPDCA_WITH_HSL)
  target_compile_definitions(lazyipdca_qcqp PUBLIC LAZYIPDCA_HAVE_HSL=1)
  target_link_libraries(lazyipdca_qcqp PUBLIC ${CMAKE_DL_LIBS})
  message(STATUS "MA27 runtime adapter enabled; external HSL library required")
endif()
if(LAZYIPDCA_WITH_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(lazyipdca_qcqp PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(lazyipdca_qcqp PUBLIC LAZYIPDCA_HAVE_OPENMP=1)
    message(STATUS "OpenMP equality-row kernels enabled")
  else()
    message(STATUS "OpenMP not found: equality-row kernels remain serial")
  endif()
endif()
if(LAZYIPDCA_NATIVE_ARCH AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(lazyipdca_qcqp PRIVATE -march=native)
  message(STATUS "Local native architecture optimizations enabled (-march=native)")
endif()

# Optional sparse linear algebra backends.  The native LAPACK backend is always
# available.  QDLDL is accepted only together with METIS because the public
# `ldl` backend contract includes METIS nested-dissection ordering.
if(LAZYIPDCA_WITH_METIS)
  find_path(METIS_INCLUDE_DIR NAMES metis.h
    HINTS ${METIS_ROOT} ${METIS_ROOT}/include $ENV{CONDA_PREFIX}/include
    PATH_SUFFIXES include metis coin-or/metis)
  find_library(METIS_LIBRARY NAMES metis coinmetis
    HINTS ${METIS_ROOT} ${METIS_ROOT}/lib ${METIS_ROOT}/lib64 $ENV{CONDA_PREFIX}/lib)
  if(METIS_INCLUDE_DIR AND METIS_LIBRARY)
    message(STATUS "METIS enabled: ${METIS_LIBRARY}")
    target_include_directories(lazyipdca_qcqp PRIVATE ${METIS_INCLUDE_DIR})
    target_link_libraries(lazyipdca_qcqp PUBLIC ${METIS_LIBRARY})
    target_compile_definitions(lazyipdca_qcqp PUBLIC LAZYIPDCA_HAVE_METIS=1)
  else()
    message(STATUS "METIS not found: sparse METIS ordering unavailable")
  endif()
endif()


if(LAZYIPDCA_WITH_LDL)
  find_path(LDL_INCLUDE_DIR NAMES ldl.h
    HINTS ${LDL_ROOT} ${LDL_ROOT}/include $ENV{CONDA_PREFIX}/include
    PATH_SUFFIXES suitesparse SuiteSparse)
  find_library(LDL_LIBRARY NAMES ldl
    HINTS ${LDL_ROOT} ${LDL_ROOT}/lib ${LDL_ROOT}/lib64 $ENV{CONDA_PREFIX}/lib)
  if(LDL_INCLUDE_DIR AND LDL_LIBRARY AND METIS_INCLUDE_DIR AND METIS_LIBRARY)
    message(STATUS "SuiteSparse LDL+METIS backend enabled: ${LDL_LIBRARY}")
    target_include_directories(lazyipdca_qcqp PRIVATE ${LDL_INCLUDE_DIR})
    target_link_libraries(lazyipdca_qcqp PUBLIC ${LDL_LIBRARY})
    target_compile_definitions(lazyipdca_qcqp PUBLIC LAZYIPDCA_HAVE_LDL=1)
  else()
    message(STATUS "SuiteSparse LDL backend not built (requires libldl headers/library + METIS)")
  endif()
endif()

# v0.4.28r3p28: optional SuiteSparse AMD ordering for the lazy-newton route
# (falls back to METIS at run time when absent).
option(LAZYIPDCA_WITH_AMD "Enable SuiteSparse AMD ordering for the lazy-newton route when found" ON)
if(LAZYIPDCA_WITH_AMD)
  find_path(AMD_INCLUDE_DIR NAMES amd.h
    HINTS ${LDL_ROOT} ${LDL_ROOT}/include $ENV{CONDA_PREFIX}/include
    PATH_SUFFIXES suitesparse SuiteSparse)
  find_library(AMD_LIBRARY NAMES amd
    HINTS ${LDL_ROOT} ${LDL_ROOT}/lib ${LDL_ROOT}/lib64 $ENV{CONDA_PREFIX}/lib)
  if(AMD_INCLUDE_DIR AND AMD_LIBRARY)
    message(STATUS "SuiteSparse AMD ordering enabled: ${AMD_LIBRARY}")
    target_include_directories(lazyipdca_qcqp PRIVATE ${AMD_INCLUDE_DIR})
    target_link_libraries(lazyipdca_qcqp PUBLIC ${AMD_LIBRARY})
    target_compile_definitions(lazyipdca_qcqp PUBLIC LAZYIPDCA_HAVE_AMD=1)
  else()
    message(STATUS "SuiteSparse AMD not found: lazy-newton falls back to METIS ordering")
  endif()
endif()

if(LAZYIPDCA_REQUIRE_HISTORICAL_LDL)
  if(NOT (LDL_INCLUDE_DIR AND LDL_LIBRARY AND METIS_INCLUDE_DIR AND METIS_LIBRARY))
    message(FATAL_ERROR "Required historical LDL backend missing: install SuiteSparse LDL development files and METIS")
  endif()
endif()

if(LAZYIPDCA_WITH_QDLDL)
  find_path(QDLDL_INCLUDE_DIR NAMES qdldl.h
    HINTS ${QDLDL_ROOT} ${QDLDL_ROOT}/include $ENV{CONDA_PREFIX}/include
    PATH_SUFFIXES include qdldl)
  find_library(QDLDL_LIBRARY NAMES qdldl
    HINTS ${QDLDL_ROOT} ${QDLDL_ROOT}/lib ${QDLDL_ROOT}/lib64 $ENV{CONDA_PREFIX}/lib)
  if(QDLDL_INCLUDE_DIR AND QDLDL_LIBRARY AND METIS_INCLUDE_DIR AND METIS_LIBRARY)
    message(STATUS "QDLDL+METIS backend enabled: ${QDLDL_LIBRARY}")
    target_include_directories(lazyipdca_qcqp PRIVATE ${QDLDL_INCLUDE_DIR})
    target_link_libraries(lazyipdca_qcqp PUBLIC ${QDLDL_LIBRARY})
    target_compile_definitions(lazyipdca_qcqp PUBLIC LAZYIPDCA_HAVE_QDLDL=1)
  else()
    message(STATUS "QDLDL backend not built (requires QDLDL + METIS)")
  endif()
endif()

if(LAZYIPDCA_WITH_MUMPS)
  find_path(MUMPS_INCLUDE_DIR NAMES dmumps_c.h
    HINTS ${MUMPS_ROOT} ${MUMPS_ROOT}/include $ENV{CONDA_PREFIX}/include
    PATH_SUFFIXES include mumps coin-or/mumps)
  find_library(MUMPS_DMUMPS_LIBRARY NAMES dmumps_seq dmumps coinmumps
    HINTS ${MUMPS_ROOT} ${MUMPS_ROOT}/lib ${MUMPS_ROOT}/lib64 $ENV{CONDA_PREFIX}/lib)
  find_library(MUMPS_COMMON_LIBRARY NAMES mumps_common_seq mumps_common
    HINTS ${MUMPS_ROOT} ${MUMPS_ROOT}/lib ${MUMPS_ROOT}/lib64 $ENV{CONDA_PREFIX}/lib)
  find_library(MUMPS_PORD_LIBRARY NAMES pord_seq pord
    HINTS ${MUMPS_ROOT} ${MUMPS_ROOT}/lib ${MUMPS_ROOT}/lib64 $ENV{CONDA_PREFIX}/lib)
  find_library(MUMPS_MPISEQ_LIBRARY NAMES mpiseq_seq mpiseq
    HINTS ${MUMPS_ROOT} ${MUMPS_ROOT}/lib ${MUMPS_ROOT}/lib64 $ENV{CONDA_PREFIX}/lib)

  if(MUMPS_INCLUDE_DIR AND MUMPS_DMUMPS_LIBRARY AND METIS_INCLUDE_DIR AND METIS_LIBRARY)
    message(STATUS "MUMPS+METIS backend enabled: ${MUMPS_DMUMPS_LIBRARY}")
    target_include_directories(lazyipdca_qcqp PRIVATE ${MUMPS_INCLUDE_DIR})
    target_link_libraries(lazyipdca_qcqp PUBLIC ${MUMPS_DMUMPS_LIBRARY})
    # Standard mumps-seq packages split these libraries; bundled coinmumps may
    # contain them already, so append only what was actually found.
    foreach(_lib MUMPS_COMMON_LIBRARY MUMPS_PORD_LIBRARY MUMPS_MPISEQ_LIBRARY)
      if(${_lib})
        target_link_libraries(lazyipdca_qcqp PUBLIC ${${_lib}})
      endif()
    endforeach()
    target_compile_definitions(lazyipdca_qcqp PUBLIC LAZYIPDCA_HAVE_MUMPS=1)
  else()
    message(STATUS "MUMPS backend not built (requires MUMPS + METIS)")
  endif()
endif()

if(LAZYIPDCA_REQUIRE_SPARSE_BACKENDS)
  if(NOT (QDLDL_INCLUDE_DIR AND QDLDL_LIBRARY AND METIS_INCLUDE_DIR AND METIS_LIBRARY))
    message(FATAL_ERROR "Required QDLDL backend missing: install QDLDL and METIS")
  endif()
  if(NOT (MUMPS_INCLUDE_DIR AND MUMPS_DMUMPS_LIBRARY AND METIS_INCLUDE_DIR AND METIS_LIBRARY))
    message(FATAL_ERROR "Required MUMPS backend missing: install mumps-seq/mumps-include and METIS")
  endif()
endif()

if(LAZYIPDCA_WITH_IPOPT)
  if(IPOPT_ROOT)
    find_path(IPOPT_INCLUDE_DIR coin-or/IpIpoptApplication.hpp HINTS ${IPOPT_ROOT}/include NO_DEFAULT_PATH)
    find_library(IPOPT_LIBRARY NAMES ipopt HINTS ${IPOPT_ROOT} ${IPOPT_ROOT}/lib ${IPOPT_ROOT}/lib64 NO_DEFAULT_PATH)
  else()
    find_path(IPOPT_INCLUDE_DIR coin-or/IpIpoptApplication.hpp
      HINTS $ENV{CONDA_PREFIX}/include)
    find_library(IPOPT_LIBRARY NAMES ipopt
      HINTS $ENV{CONDA_PREFIX}/lib $ENV{CONDA_PREFIX}/lib64)
  endif()
  if(IPOPT_INCLUDE_DIR AND IPOPT_LIBRARY)
    message(STATUS "Ipopt backend enabled: ${IPOPT_LIBRARY}")
    target_include_directories(lazyipdca_qcqp PRIVATE ${IPOPT_INCLUDE_DIR})
    target_link_libraries(lazyipdca_qcqp PUBLIC ${IPOPT_LIBRARY})
    target_compile_definitions(lazyipdca_qcqp PUBLIC LAZYIPDCA_HAVE_IPOPT=1)
  else()
    message(WARNING "Ipopt not found: native backend only")
  endif()
endif()


add_executable(lazyipdca_qcqp_cli src/main.cpp)
target_link_libraries(lazyipdca_qcqp_cli PRIVATE lazyipdca_qcqp)

# Explicit optional terminal protocol executable. Building this target does
# not enable finishing in the primary solver or change historical R3 modes.
add_executable(lazyipdca_feasible_kkt_cli tools/feasible_original_kkt_cli.cpp)
target_link_libraries(lazyipdca_feasible_kkt_cli PRIVATE lazyipdca_qcqp)
target_compile_options(lazyipdca_feasible_kkt_cli PRIVATE -Wall -Wextra -Wpedantic)

# Explicit optional terminal stage; never selected by the ordinary DCA CLI.
# Uses fresh signed ORIGINAL KKT matrices with MUMPS+METIS and native guards.
if(LAZYIPDCA_WITH_MUMPS AND MUMPS_INCLUDE_DIR AND MUMPS_DMUMPS_LIBRARY AND METIS_INCLUDE_DIR AND METIS_LIBRARY)
  add_executable(lazyipdca_sparse_equality_kkt_cli tools/sparse_equality_kkt_cli.cpp)
  target_include_directories(lazyipdca_sparse_equality_kkt_cli PRIVATE ${MUMPS_INCLUDE_DIR})
  target_link_libraries(lazyipdca_sparse_equality_kkt_cli PRIVATE lazyipdca_qcqp)
endif()


add_executable(ipopt_qcqp_cli src/ipopt_qcqp_main.cpp)
target_link_libraries(ipopt_qcqp_cli PRIVATE lazyipdca_qcqp)

if(LAZYIPDCA_WITH_HSL)
  add_executable(check_ma27 tools/check_ma27.cpp)
  target_link_libraries(check_ma27 PRIVATE lazyipdca_qcqp ${CMAKE_DL_LIBS})
  target_compile_options(check_ma27 PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  enable_testing()

  if(LAZYIPDCA_WITH_MUMPS AND MUMPS_INCLUDE_DIR AND MUMPS_DMUMPS_LIBRARY
      AND METIS_INCLUDE_DIR AND METIS_LIBRARY)
    add_test(NAME feasible_original_kkt
      COMMAND lazyipdca_feasible_kkt_cli --self-test)
    set_tests_properties(feasible_original_kkt PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=1;OMP_THREAD_LIMIT=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
  endif()
  if(TARGET lazyipdca_sparse_equality_kkt_cli)
    add_test(NAME sparse_original_kkt_guard COMMAND lazyipdca_sparse_equality_kkt_cli --self-test)
  endif()

  foreach(case_name IN ITEMS g10_memory_recovery g10_restoration_selection g10_progress_guard g10_terminal_policy g10_terminal_rescue g10_krylov_borders)
    add_executable(test_${case_name} tests/test_${case_name}.cpp)
    target_link_libraries(test_${case_name} PRIVATE lazyipdca_qcqp)
    target_compile_options(test_${case_name} PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME ${case_name} COMMAND test_${case_name})
  endforeach()

  if(LAZYIPDCA_WITH_IPOPT AND IPOPT_INCLUDE_DIR AND IPOPT_LIBRARY)
    add_test(NAME ipopt_qcq_v2_parse
      COMMAND ipopt_qcqp_cli --problem ${CMAKE_CURRENT_SOURCE_DIR}/tests/toy_v2.qcq
              --max-iter 100 --print-level 0 --quiet)
    set_tests_properties(ipopt_qcq_v2_parse PROPERTIES TIMEOUT 30)
    # An explicit hsllib path must be accepted by the direct-Ipopt CLI and
    # forwarded through Ipopt's registered option.  MUMPS deliberately keeps
    # this smoke test independent of the presence of a proprietary HSL binary.
    add_test(NAME ipopt_qcq_hsllib_option
      COMMAND ipopt_qcqp_cli --problem ${CMAKE_CURRENT_SOURCE_DIR}/tests/toy_v2.qcq
              --linear-solver mumps --hsl-library unused-in-mumps.so
              --max-iter 100 --print-level 0 --quiet)
    set_tests_properties(ipopt_qcq_hsllib_option PROPERTIES TIMEOUT 30)
  endif()

  add_executable(test_g10_filter_diagnostics tests/test_g10_filter_diagnostics.cpp)
  target_compile_options(test_g10_filter_diagnostics PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME g10_filter_diagnostics COMMAND test_g10_filter_diagnostics)
  if(LAZYIPDCA_WITH_HSL)
    add_library(mock_ma27_abi SHARED tests/mock_ma27_abi.cpp)
    target_link_libraries(mock_ma27_abi PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
    target_compile_options(mock_ma27_abi PRIVATE -Wall -Wextra -Wpedantic)
    add_executable(test_g10_ma27 tests/test_g10_ma27.cpp)
    target_link_libraries(test_g10_ma27 PRIVATE lazyipdca_qcqp ${CMAKE_DL_LIBS})
    target_compile_options(test_g10_ma27 PRIVATE -Wall -Wextra -Wpedantic)
    if(LAZYIPDCA_WITH_METIS AND METIS_INCLUDE_DIR AND METIS_LIBRARY)
      target_include_directories(test_g10_ma27 PRIVATE ${METIS_INCLUDE_DIR})
    endif()
    add_test(NAME g10_ma27_adapter COMMAND test_g10_ma27 $<TARGET_FILE:mock_ma27_abi>)
  endif()
  add_executable(test_direct_ipopt_structure
    tests/test_direct_ipopt_structure.cpp)
  target_link_libraries(test_direct_ipopt_structure PRIVATE lazyipdca_qcqp)
  add_executable(test_free_epigraph_presolve tests/test_free_epigraph_presolve.cpp)
  target_include_directories(test_free_epigraph_presolve PRIVATE ${LAZYIPDCA_BOOST_INCLUDE_DIR})
  target_link_libraries(test_free_epigraph_presolve PRIVATE lazyipdca_qcqp)
  add_test(NAME free_epigraph_presolve COMMAND test_free_epigraph_presolve)
  add_executable(test_direct_ipopt_value_accuracy
    tests/test_direct_ipopt_value_accuracy.cpp)
  target_include_directories(test_direct_ipopt_value_accuracy PRIVATE ${LAZYIPDCA_BOOST_INCLUDE_DIR})
  target_link_libraries(test_direct_ipopt_value_accuracy PRIVATE lazyipdca_qcqp)
  add_test(NAME direct_ipopt_value_accuracy COMMAND test_direct_ipopt_value_accuracy)
  add_test(NAME direct_ipopt_sparse_structure
    COMMAND test_direct_ipopt_structure)
  add_executable(test_direct_ipopt_acceptable
    tests/test_direct_ipopt_acceptable.cpp)
  target_link_libraries(test_direct_ipopt_acceptable PRIVATE lazyipdca_qcqp)
  add_test(NAME direct_ipopt_acceptable_status
    COMMAND test_direct_ipopt_acceptable)

  add_executable(test_toy tests/test_toy.cpp)
  target_link_libraries(test_toy PRIVATE lazyipdca_qcqp)
  add_test(NAME toy_qcqp COMMAND test_toy)

  add_executable(test_curvature_contract tests/test_curvature_contract.cpp)
  target_link_libraries(test_curvature_contract PRIVATE lazyipdca_qcqp)
  add_test(NAME curvature_contract COMMAND test_curvature_contract)

  add_executable(test_curvature_block_exact tests/test_curvature_block_exact.cpp)
  target_link_libraries(test_curvature_block_exact PRIVATE lazyipdca_qcqp)
  add_test(NAME curvature_block_exact COMMAND test_curvature_block_exact)

  add_executable(test_recovery_margin tests/test_recovery_margin.cpp)
  target_link_libraries(test_recovery_margin PRIVATE lazyipdca_qcqp)
  add_test(NAME recovery_objective_margin COMMAND test_recovery_margin)

  add_executable(test_guarded_inner_fallback tests/test_guarded_inner_fallback.cpp)
  target_link_libraries(test_guarded_inner_fallback PRIVATE lazyipdca_qcqp)
  add_test(NAME guarded_inner_fallback COMMAND test_guarded_inner_fallback)

  add_executable(test_cutest_regressions tests/test_cutest_regressions.cpp)
  target_link_libraries(test_cutest_regressions PRIVATE lazyipdca_qcqp)
  target_compile_definitions(test_cutest_regressions PRIVATE LAZYIPDCA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME cutest_qcqp_regressions COMMAND test_cutest_regressions)

  add_executable(test_abs_link_restoration tests/test_abs_link_restoration.cpp)
  target_link_libraries(test_abs_link_restoration PRIVATE lazyipdca_qcqp)
  add_test(NAME abs_link_restoration COMMAND test_abs_link_restoration)

  add_executable(test_corelike_abs_restoration tests/test_corelike_abs_restoration.cpp)
  target_link_libraries(test_corelike_abs_restoration PRIVATE lazyipdca_qcqp)
  add_test(NAME corelike_abs_reduced_restoration COMMAND test_corelike_abs_restoration)

  add_executable(test_aljazzaf_restoration tests/test_aljazzaf_restoration.cpp)
  target_link_libraries(test_aljazzaf_restoration PRIVATE lazyipdca_qcqp)
  add_test(NAME aljazzaf_exact_restoration COMMAND test_aljazzaf_restoration)

  add_executable(test_linear_backends tests/test_linear_backends.cpp)
  target_link_libraries(test_linear_backends PRIVATE lazyipdca_qcqp)
  target_compile_definitions(test_linear_backends PRIVATE LAZYIPDCA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME linear_backend_parity COMMAND test_linear_backends)

  add_executable(test_cholesk_backend tests/test_cholesk_backend.cpp)
  target_link_libraries(test_cholesk_backend PRIVATE lazyipdca_qcqp)
  target_compile_definitions(test_cholesk_backend PRIVATE LAZYIPDCA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME cholesk_backend COMMAND test_cholesk_backend)

  add_test(NAME cholesk_cli_surface COMMAND lazyipdca_qcqp_cli --linear-backends)
  set_tests_properties(cholesk_cli_surface PROPERTIES
    PASS_REGULAR_EXPRESSION "cholesk")

  add_executable(test_common_kkt_regularization tests/test_common_kkt_regularization.cpp)
  target_link_libraries(test_common_kkt_regularization PRIVATE lazyipdca_qcqp)
  target_compile_definitions(test_common_kkt_regularization PRIVATE LAZYIPDCA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME common_kkt_regularization COMMAND test_common_kkt_regularization)

  add_executable(test_minimax_restoration tests/test_minimax_restoration.cpp)
  target_link_libraries(test_minimax_restoration PRIVATE lazyipdca_qcqp)
  add_test(NAME minimax_restoration COMMAND test_minimax_restoration)

  add_executable(test_qp_dual_multiplier_path tests/test_qp_dual_multiplier_path.cpp)
  target_link_libraries(test_qp_dual_multiplier_path PRIVATE lazyipdca_qcqp)
  add_test(NAME qp_dual_multiplier_path COMMAND test_qp_dual_multiplier_path)

  add_executable(test_terminal_reduced_kkt tests/test_terminal_reduced_kkt.cpp)
  target_link_libraries(test_terminal_reduced_kkt PRIVATE lazyipdca_qcqp)
  add_test(NAME terminal_reduced_kkt COMMAND test_terminal_reduced_kkt)

  add_executable(test_working_set_drop tests/test_working_set_drop.cpp)
  target_link_libraries(test_working_set_drop PRIVATE lazyipdca_qcqp)
  add_test(NAME working_set_drop COMMAND test_working_set_drop)

  add_executable(test_terminal_guarded_no_pivot tests/test_terminal_guarded_no_pivot.cpp)
  target_link_libraries(test_terminal_guarded_no_pivot PRIVATE lazyipdca_qcqp)
  add_test(NAME terminal_guarded_no_pivot COMMAND test_terminal_guarded_no_pivot)

  add_executable(test_kkt_stop_certificate tests/test_kkt_stop_certificate.cpp)
  target_link_libraries(test_kkt_stop_certificate PRIVATE lazyipdca_qcqp)
  add_test(NAME kkt_stop_certificate COMMAND test_kkt_stop_certificate)

  add_executable(test_second_order_correction tests/test_second_order_correction.cpp)
  target_link_libraries(test_second_order_correction PRIVATE lazyipdca_qcqp)
  add_test(NAME second_order_correction COMMAND test_second_order_correction)

  add_executable(test_condensed_quadratic_retraction tests/test_condensed_quadratic_retraction.cpp)
  target_link_libraries(test_condensed_quadratic_retraction PRIVATE lazyipdca_qcqp)
  add_test(NAME condensed_quadratic_retraction COMMAND test_condensed_quadratic_retraction)

  add_executable(test_condensed_bound_lookahead tests/test_condensed_bound_lookahead.cpp)
  target_link_libraries(test_condensed_bound_lookahead PRIVATE lazyipdca_qcqp)
  add_test(NAME condensed_bound_lookahead COMMAND test_condensed_bound_lookahead)

  add_executable(test_condensed_multiplier_workingset_refit tests/test_condensed_multiplier_workingset_refit.cpp)
  target_link_libraries(test_condensed_multiplier_workingset_refit PRIVATE lazyipdca_qcqp)
  add_test(NAME condensed_multiplier_workingset_refit COMMAND test_condensed_multiplier_workingset_refit)

  add_executable(test_single_loop_transport_policy
    tests/test_single_loop_transport_policy.cpp)
  target_link_libraries(test_single_loop_transport_policy
    PRIVATE lazyipdca_qcqp)
  add_test(NAME single_loop_transport_policy
    COMMAND test_single_loop_transport_policy)

  add_executable(test_condensed_adaptive_normal_refinement
    tests/test_condensed_adaptive_normal_refinement.cpp)
  target_link_libraries(test_condensed_adaptive_normal_refinement
    PRIVATE lazyipdca_qcqp)
  add_test(NAME condensed_adaptive_normal_refinement
    COMMAND test_condensed_adaptive_normal_refinement)

  add_executable(test_condensed_alpha_predictor
    tests/test_condensed_alpha_predictor.cpp)
  target_link_libraries(test_condensed_alpha_predictor PRIVATE lazyipdca_qcqp)
  add_test(NAME condensed_alpha_predictor COMMAND test_condensed_alpha_predictor)

  add_executable(test_condensed_outer_extrapolation
    tests/test_condensed_outer_extrapolation.cpp)
  target_link_libraries(test_condensed_outer_extrapolation
    PRIVATE lazyipdca_qcqp)
  add_test(NAME condensed_outer_extrapolation
    COMMAND test_condensed_outer_extrapolation)

  add_executable(test_condensed_post_retraction_working_set
    tests/test_condensed_post_retraction_working_set.cpp)
  target_link_libraries(test_condensed_post_retraction_working_set
    PRIVATE lazyipdca_qcqp)
  add_test(NAME condensed_post_retraction_working_set
    COMMAND test_condensed_post_retraction_working_set)

  add_executable(test_condensed_modeled_side_filter
    tests/test_condensed_modeled_side_filter.cpp)
  target_link_libraries(test_condensed_modeled_side_filter
    PRIVATE lazyipdca_qcqp)
  add_test(NAME condensed_modeled_side_filter
    COMMAND test_condensed_modeled_side_filter)

  add_executable(test_condensed_flow_lift_schur
    tests/test_condensed_flow_lift_schur.cpp)
  target_link_libraries(test_condensed_flow_lift_schur PRIVATE lazyipdca_qcqp)
  add_test(NAME condensed_flow_lift_schur
    COMMAND test_condensed_flow_lift_schur)

  add_executable(test_projected_multi_ritz_inertia
    tests/test_projected_multi_ritz_inertia.cpp)
  target_link_libraries(test_projected_multi_ritz_inertia
    PRIVATE lazyipdca_qcqp)
  add_test(NAME projected_multi_ritz_inertia
    COMMAND test_projected_multi_ritz_inertia)

  add_executable(test_tangent_inertia_adversarial
    tests/test_tangent_inertia_adversarial.cpp)
  target_link_libraries(test_tangent_inertia_adversarial
    PRIVATE lazyipdca_qcqp)
  add_test(NAME tangent_inertia_adversarial
    COMMAND test_tangent_inertia_adversarial)

  add_executable(test_pglib_structural_hotpaths
    tests/test_pglib_structural_hotpaths.cpp)
  target_link_libraries(test_pglib_structural_hotpaths
    PRIVATE lazyipdca_qcqp)
  add_test(NAME pglib_structural_hotpaths
    COMMAND test_pglib_structural_hotpaths)

  # Diagnostic only: deliberately not registered with CTest because it needs
  # a caller-supplied, exported PGLib QCQP and reports timing measurements.
  add_executable(benchmark_flow_lift_schur
    tests/benchmark_flow_lift_schur.cpp)
  target_link_libraries(benchmark_flow_lift_schur PRIVATE lazyipdca_qcqp)

  add_test(NAME trust_region_cli
    COMMAND lazyipdca_qcqp_cli
      --problem ${CMAKE_CURRENT_SOURCE_DIR}/tests/toy.qcq
      --trust-region
      --tr-initial-radius 10
      --tr-max-radius 10
      --tr-min-radius 1e-10
      --tr-max-rejects 18
      --quiet)

  add_test(NAME no_tangential_cli
    COMMAND lazyipdca_qcqp_cli
      --problem ${CMAKE_CURRENT_SOURCE_DIR}/tests/toy.qcq
      --no-tangential
      --quiet)
endif()

install(TARGETS lazyipdca_qcqp lazyipdca_qcqp_cli ipopt_qcqp_cli)
install(DIRECTORY include/ DESTINATION include)

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_inner_warmstart_boundpush tests/test_inner_warmstart_boundpush.cpp)
  target_link_libraries(test_inner_warmstart_boundpush PRIVATE lazyipdca_qcqp)
  add_test(NAME inner_warmstart_boundpush COMMAND test_inner_warmstart_boundpush)

  add_executable(test_lazy_dca_majorants tests/test_lazy_dca_majorants.cpp)
  target_link_libraries(test_lazy_dca_majorants PRIVATE lazyipdca_qcqp)
  add_test(NAME lazy_dca_majorants COMMAND test_lazy_dca_majorants)

  add_executable(test_inner_convex_backends tests/test_inner_convex_backends.cpp)
  target_link_libraries(test_inner_convex_backends PRIVATE lazyipdca_qcqp)
  add_test(NAME inner_convex_backends COMMAND test_inner_convex_backends)

  add_executable(test_lazyqp_curvature_wall tests/test_lazyqp_curvature_wall.cpp)
  target_link_libraries(test_lazyqp_curvature_wall PRIVATE lazyipdca_qcqp)
  add_test(NAME lazyqp_curvature_wall COMMAND test_lazyqp_curvature_wall)

  add_executable(test_lazy_dca_affine_full_majorant tests/test_lazy_dca_affine_full_majorant.cpp)
  target_link_libraries(test_lazy_dca_affine_full_majorant PRIVATE lazyipdca_qcqp)
  add_test(NAME lazy_dca_affine_full_majorant COMMAND test_lazy_dca_affine_full_majorant)

  add_executable(test_lazy_dca_affine_full_shape tests/test_lazy_dca_affine_full_shape.cpp)
  target_link_libraries(test_lazy_dca_affine_full_shape PRIVATE lazyipdca_qcqp)
  add_test(NAME lazy_dca_affine_full_shape_130_1_384 COMMAND test_lazy_dca_affine_full_shape)

  add_executable(test_lazy_dca_affine_box_bounded_lazy tests/test_lazy_dca_affine_box_bounded_lazy.cpp)
  target_link_libraries(test_lazy_dca_affine_box_bounded_lazy PRIVATE lazyipdca_qcqp)
  add_test(NAME lazy_dca_affine_box_bounded_lazy COMMAND test_lazy_dca_affine_box_bounded_lazy)

  add_executable(test_lazyqp_dual_step_coupling tests/test_lazyqp_dual_step_coupling.cpp)
  target_link_libraries(test_lazyqp_dual_step_coupling PRIVATE lazyipdca_qcqp)
  # GCC 13 can emit a spurious -Wnonnull from its inlined zero-length
  # std::vector copy path in this optimized fixture.  Keep the diagnostic but
  # do not let that compiler-only false positive prevent the full test build.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
     CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    target_compile_options(test_lazyqp_dual_step_coupling PRIVATE
      -Wno-error=nonnull)
  endif()
  add_test(NAME lazyqp_dual_step_coupling COMMAND test_lazyqp_dual_step_coupling)

  add_executable(test_lazyqp_scaled_split_boundary tests/test_lazyqp_scaled_split_boundary.cpp)
  target_link_libraries(test_lazyqp_scaled_split_boundary PRIVATE lazyipdca_qcqp)
  add_test(NAME lazyqp_scaled_split_boundary COMMAND test_lazyqp_scaled_split_boundary)

  add_executable(test_lazyqp_fixed_native_shift tests/test_lazyqp_fixed_native_shift.cpp)
  target_link_libraries(test_lazyqp_fixed_native_shift PRIVATE lazyipdca_qcqp)
  add_test(NAME lazyqp_fixed_native_shift COMMAND test_lazyqp_fixed_native_shift)

  add_executable(test_lazyqp_lowrank_condensation tests/test_lazyqp_lowrank_condensation.cpp)
  target_link_libraries(test_lazyqp_lowrank_condensation PRIVATE lazyipdca_qcqp)
  add_test(NAME lazyqp_lowrank_condensation COMMAND test_lazyqp_lowrank_condensation)

  add_executable(test_lazyqp_reduced_kkt tests/test_lazyqp_reduced_kkt.cpp)
  target_link_libraries(test_lazyqp_reduced_kkt PRIVATE lazyipdca_qcqp)
  add_test(NAME lazyqp_reduced_kkt COMMAND test_lazyqp_reduced_kkt)

  add_executable(test_lazyqp_model_support_fastpath
    tests/test_lazyqp_model_support_fastpath.cpp)
  target_link_libraries(test_lazyqp_model_support_fastpath PRIVATE lazyipdca_qcqp)
  add_test(NAME lazyqp_model_support_fastpath
    COMMAND test_lazyqp_model_support_fastpath)

  add_executable(test_lazyqp_hybrid_kkt tests/test_lazyqp_hybrid_kkt.cpp)
  target_link_libraries(test_lazyqp_hybrid_kkt PRIVATE lazyipdca_qcqp)
  add_test(NAME lazyqp_hybrid_kkt COMMAND test_lazyqp_hybrid_kkt)

  add_executable(test_lazyqp_kkt_triplet_cache
    tests/test_lazyqp_kkt_triplet_cache.cpp)
  target_link_libraries(test_lazyqp_kkt_triplet_cache PRIVATE lazyipdca_qcqp)
  add_test(NAME lazyqp_kkt_triplet_cache
    COMMAND test_lazyqp_kkt_triplet_cache)

  add_executable(test_lazyqp_exact_affine_reduction
    tests/test_lazyqp_exact_affine_reduction.cpp)
  target_link_libraries(test_lazyqp_exact_affine_reduction PRIVATE lazyipdca_qcqp)
  add_test(NAME lazyqp_exact_affine_reduction
    COMMAND test_lazyqp_exact_affine_reduction)

  add_executable(test_lazyqp_reduced_fallback_scope tests/test_lazyqp_reduced_fallback_scope.cpp)
  target_link_libraries(test_lazyqp_reduced_fallback_scope PRIVATE lazyipdca_qcqp)
  add_test(NAME lazyqp_reduced_fallback_scope COMMAND test_lazyqp_reduced_fallback_scope)

  add_executable(test_lazyqp_guarded_continuation tests/test_lazyqp_guarded_continuation.cpp)
  target_link_libraries(test_lazyqp_guarded_continuation PRIVATE lazyipdca_qcqp)
  add_test(NAME lazyqp_guarded_continuation COMMAND test_lazyqp_guarded_continuation)

  add_executable(test_lazyqp_inexact_forcing tests/test_lazyqp_inexact_forcing.cpp)
  target_link_libraries(test_lazyqp_inexact_forcing PRIVATE lazyipdca_qcqp)
  add_test(NAME lazyqp_inexact_forcing COMMAND test_lazyqp_inexact_forcing)

  add_executable(test_dca_single_loop_primal_dual
    tests/test_dca_single_loop_primal_dual.cpp)
  target_link_libraries(test_dca_single_loop_primal_dual PRIVATE lazyipdca_qcqp)
  add_test(NAME dca_single_loop_primal_dual
    COMMAND test_dca_single_loop_primal_dual)

  add_executable(test_dca_single_loop_cold_dual_stall
    tests/test_dca_single_loop_cold_dual_stall.cpp)
  target_link_libraries(test_dca_single_loop_cold_dual_stall
    PRIVATE lazyipdca_qcqp)
  add_test(NAME dca_single_loop_cold_dual_stall
    COMMAND test_dca_single_loop_cold_dual_stall)

  add_executable(test_dca_single_loop_working_set_transport
    tests/test_dca_single_loop_working_set_transport.cpp)
  target_link_libraries(test_dca_single_loop_working_set_transport
    PRIVATE lazyipdca_qcqp)
  add_test(NAME dca_single_loop_working_set_transport
    COMMAND test_dca_single_loop_working_set_transport)

  add_executable(test_single_loop_slack_residual_transport
    tests/test_single_loop_slack_residual_transport.cpp)
  target_link_libraries(test_single_loop_slack_residual_transport
    PRIVATE lazyipdca_qcqp)
  add_test(NAME single_loop_slack_residual_transport
    COMMAND test_single_loop_slack_residual_transport)

  add_executable(test_dca_single_loop_direct_start
    tests/test_dca_single_loop_direct_start.cpp)
  target_link_libraries(test_dca_single_loop_direct_start
    PRIVATE lazyipdca_qcqp)
  add_test(NAME dca_single_loop_direct_start
    COMMAND test_dca_single_loop_direct_start)

  add_executable(test_dca_single_loop_fail_closed
    tests/test_dca_single_loop_fail_closed.cpp)
  target_link_libraries(test_dca_single_loop_fail_closed
    PRIVATE lazyipdca_qcqp)
  add_test(NAME dca_single_loop_fail_closed
    COMMAND test_dca_single_loop_fail_closed)

  add_test(NAME dca_single_loop_cli_reject_outer_alias
    COMMAND ${CMAKE_COMMAND}
      -DCLI=$<TARGET_FILE:lazyipdca_qcqp_cli>
      -DPROBLEM=${CMAKE_CURRENT_SOURCE_DIR}/tests/toy.qcq
      -DMODE=outer
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_single_loop_cli_rejection.cmake)

  add_test(NAME dca_single_loop_cli_reject_nonlazyqp
    COMMAND ${CMAKE_COMMAND}
      -DCLI=$<TARGET_FILE:lazyipdca_qcqp_cli>
      -DPROBLEM=${CMAKE_CURRENT_SOURCE_DIR}/tests/toy.qcq
      -DMODE=inner
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_single_loop_cli_rejection.cmake)

  add_test(NAME dca_single_loop_cli_reject_nonmehrotra
    COMMAND ${CMAKE_COMMAND}
      -DCLI=$<TARGET_FILE:lazyipdca_qcqp_cli>
      -DPROBLEM=${CMAKE_CURRENT_SOURCE_DIR}/tests/toy.qcq
      -DMODE=mehrotra
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_single_loop_cli_rejection.cmake)

  add_test(NAME dca_single_loop_cli_reject_retraction_fallback
    COMMAND ${CMAKE_COMMAND}
      -DCLI=$<TARGET_FILE:lazyipdca_qcqp_cli>
      -DPROBLEM=${CMAKE_CURRENT_SOURCE_DIR}/tests/toy.qcq
      -DMODE=retraction
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_single_loop_cli_rejection.cmake)

  add_test(NAME dca_single_loop_cli_reject_numeric_kkt_reuse
    COMMAND ${CMAKE_COMMAND}
      -DCLI=$<TARGET_FILE:lazyipdca_qcqp_cli>
      -DPROBLEM=${CMAKE_CURRENT_SOURCE_DIR}/tests/toy.qcq
      -DMODE=numeric-reuse
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_single_loop_cli_rejection.cmake)

  add_executable(test_lazyqp_persistent_symbolic_cache tests/test_lazyqp_persistent_symbolic_cache.cpp)
  target_link_libraries(test_lazyqp_persistent_symbolic_cache PRIVATE lazyipdca_qcqp)
  add_test(NAME lazyqp_persistent_symbolic_cache COMMAND test_lazyqp_persistent_symbolic_cache)

  add_executable(test_ldl_numeric_scatter_cache tests/test_ldl_numeric_scatter_cache.cpp)
  target_link_libraries(test_ldl_numeric_scatter_cache PRIVATE lazyipdca_qcqp)
  add_test(NAME ldl_numeric_scatter_cache COMMAND test_ldl_numeric_scatter_cache)

  add_executable(test_lazy_dca_persistent_symbolic_cache tests/test_lazy_dca_persistent_symbolic_cache.cpp)
  target_link_libraries(test_lazy_dca_persistent_symbolic_cache PRIVATE lazyipdca_qcqp)
  add_test(NAME lazy_dca_persistent_symbolic_cache COMMAND test_lazy_dca_persistent_symbolic_cache)

  add_executable(test_algorithm_single_route tests/test_algorithm_single_route.cpp)
  target_link_libraries(test_algorithm_single_route PRIVATE lazyipdca_qcqp)
  target_compile_definitions(test_algorithm_single_route PRIVATE LAZYIPDCA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME algorithm_single_route COMMAND test_algorithm_single_route)

  add_executable(test_lazy_dca_linearized_equalities tests/test_lazy_dca_linearized_equalities.cpp)
  target_link_libraries(test_lazy_dca_linearized_equalities PRIVATE lazyipdca_qcqp)
  add_test(NAME lazy_dca_linearized_equalities COMMAND test_lazy_dca_linearized_equalities)

  add_executable(test_lazy_dca_single_route_corrector tests/test_lazy_dca_single_route_corrector.cpp)
  target_link_libraries(test_lazy_dca_single_route_corrector PRIVATE lazyipdca_qcqp)
  target_compile_definitions(test_lazy_dca_single_route_corrector PRIVATE LAZYIPDCA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME lazy_dca_single_route_corrector COMMAND test_lazy_dca_single_route_corrector)

  add_executable(test_lazy_dca_iterated_soc tests/test_lazy_dca_iterated_soc.cpp)
  target_link_libraries(test_lazy_dca_iterated_soc PRIVATE lazyipdca_qcqp)
  add_test(NAME lazy_dca_iterated_soc COMMAND test_lazy_dca_iterated_soc)

  add_executable(test_dca_soc_defect_target tests/test_dca_soc_defect_target.cpp)
  target_link_libraries(test_dca_soc_defect_target PRIVATE lazyipdca_qcqp)
  add_test(NAME dca_soc_defect_target COMMAND test_dca_soc_defect_target)

  add_executable(test_dca_soc_boundary_continuity tests/test_dca_soc_boundary_continuity.cpp)
  target_link_libraries(test_dca_soc_boundary_continuity PRIVATE lazyipdca_qcqp)
  add_test(NAME dca_soc_boundary_continuity COMMAND test_dca_soc_boundary_continuity)

  add_executable(test_equality_qp_regularization tests/test_equality_qp_regularization.cpp)
  target_link_libraries(test_equality_qp_regularization PRIVATE lazyipdca_qcqp)
  add_test(NAME equality_qp_regularization COMMAND test_equality_qp_regularization)

  add_executable(test_equality_qp_roundoff_guard tests/test_equality_qp_roundoff_guard.cpp)
  target_link_libraries(test_equality_qp_roundoff_guard PRIVATE lazyipdca_qcqp)
  add_test(NAME equality_qp_roundoff_guard COMMAND test_equality_qp_roundoff_guard)

  add_executable(test_equality_qp_original_kkt_refinement tests/test_equality_qp_original_kkt_refinement.cpp)
  target_link_libraries(test_equality_qp_original_kkt_refinement PRIVATE lazyipdca_qcqp)
  add_test(NAME equality_qp_original_kkt_refinement COMMAND test_equality_qp_original_kkt_refinement)
  add_executable(test_lazy_dca_equality_retraction tests/test_lazy_dca_equality_retraction.cpp)
  target_link_libraries(test_lazy_dca_equality_retraction PRIVATE lazyipdca_qcqp)
  add_test(NAME lazy_dca_equality_retraction COMMAND test_lazy_dca_equality_retraction)

  add_executable(test_restoration_sparse_row_space tests/test_restoration_sparse_row_space.cpp)
  target_link_libraries(test_restoration_sparse_row_space PRIVATE lazyipdca_qcqp)
  add_test(NAME restoration_sparse_row_space COMMAND test_restoration_sparse_row_space)

  add_executable(test_near_feasible_sparse_inequality_restoration
    tests/test_near_feasible_sparse_inequality_restoration.cpp)
  target_link_libraries(test_near_feasible_sparse_inequality_restoration
    PRIVATE lazyipdca_qcqp)
  add_test(NAME near_feasible_sparse_inequality_restoration
    COMMAND test_near_feasible_sparse_inequality_restoration)

  add_executable(test_catenary_harmonic_restoration tests/test_catenary_harmonic_restoration.cpp)
  target_link_libraries(test_catenary_harmonic_restoration PRIVATE lazyipdca_qcqp)
  add_test(NAME catenary_harmonic_restoration COMMAND test_catenary_harmonic_restoration)

  add_executable(test_catenary_multiplier_transport tests/test_catenary_multiplier_transport.cpp)
  target_link_libraries(test_catenary_multiplier_transport PRIVATE lazyipdca_qcqp)
  add_test(NAME catenary_multiplier_transport COMMAND test_catenary_multiplier_transport)

  add_executable(test_lazy_dca_elastic_equalities tests/test_lazy_dca_elastic_equalities.cpp)
  target_link_libraries(test_lazy_dca_elastic_equalities PRIVATE lazyipdca_qcqp)
  add_test(NAME lazy_dca_elastic_equalities COMMAND test_lazy_dca_elastic_equalities)

  add_executable(test_catenary_tangent_activation tests/test_catenary_tangent_activation.cpp)
  target_link_libraries(test_catenary_tangent_activation PRIVATE lazyipdca_qcqp)
  add_test(NAME catenary_tangent_activation COMMAND test_catenary_tangent_activation)

  add_executable(test_lazy_dca_step_control tests/test_lazy_dca_step_control.cpp)
  target_link_libraries(test_lazy_dca_step_control PRIVATE lazyipdca_qcqp)
  add_test(NAME lazy_dca_step_control COMMAND test_lazy_dca_step_control)

  add_executable(test_unavailable_certificate tests/test_unavailable_certificate.cpp)
  target_link_libraries(test_unavailable_certificate PRIVATE lazyipdca_qcqp)
  add_test(NAME unavailable_certificate COMMAND test_unavailable_certificate)

  add_executable(test_terminal_multiplier_refinement tests/test_terminal_multiplier_refinement.cpp)
  target_link_libraries(test_terminal_multiplier_refinement PRIVATE lazyipdca_qcqp)
  add_test(NAME terminal_multiplier_refinement COMMAND test_terminal_multiplier_refinement)

  add_executable(test_hs108_lazyqp_regression tests/test_hs108_lazyqp_regression.cpp)
  target_link_libraries(test_hs108_lazyqp_regression PRIVATE lazyipdca_qcqp)
  add_test(NAME hs108_lazyqp_regression COMMAND test_hs108_lazyqp_regression)

  add_executable(test_aljazzaf_lazy_dca_regression tests/test_aljazzaf_lazy_dca_regression.cpp)
  target_link_libraries(test_aljazzaf_lazy_dca_regression PRIVATE lazyipdca_qcqp)
  add_test(NAME aljazzaf_lazy_dca_regression COMMAND test_aljazzaf_lazy_dca_regression)

  add_executable(test_aljazzaf_condensed_regression tests/test_aljazzaf_condensed_regression.cpp)
  target_link_libraries(test_aljazzaf_condensed_regression PRIVATE lazyipdca_qcqp)
  add_test(NAME aljazzaf_condensed_scale_guard COMMAND test_aljazzaf_condensed_regression)

  add_executable(test_live_iteration_display tests/test_live_iteration_display.cpp)
  target_link_libraries(test_live_iteration_display PRIVATE lazyipdca_qcqp)
  add_test(NAME live_iteration_display COMMAND test_live_iteration_display)

  add_executable(test_lazy_dca_working_set_retention
    tests/test_lazy_dca_working_set_retention.cpp)
  target_link_libraries(test_lazy_dca_working_set_retention PRIVATE lazyipdca_qcqp)
  add_test(NAME lazy_dca_working_set_retention
    COMMAND test_lazy_dca_working_set_retention)


  set_tests_properties(catenary_multiplier_transport lazy_dca_elastic_equalities PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;BLIS_NUM_THREADS=1")

  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME exporter_pycutest_contract
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_exporter_pycutest_contract.py)
  add_test(NAME exporter_quadratic_rejection
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_exporter_quadratic_rejection.py)
  add_test(NAME runner_export_rejection
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_runner_export_rejection.py)
  add_test(NAME core_xabs_lift_contract
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_core_xabs_lift_contract.py)
  add_test(NAME core_xabs_fixed_sign_contract
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_core_xabs_fixed_sign_contract.py)
  add_test(NAME exporter_pglib_contract_r3p16
           COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_exporter_pglib_contract_r3p16.py)
  set_tests_properties(exporter_pglib_contract_r3p16 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;BLIS_NUM_THREADS=1"
    TIMEOUT 120)
  add_test(NAME exporter_pglib_contract_r3p18
           COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_exporter_pglib_contract_r3p18.py)
  set_tests_properties(exporter_pglib_contract_r3p18 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;BLIS_NUM_THREADS=1"
    TIMEOUT 120)
  add_test(NAME pglib_voltage_start_continuation_r3p18
           COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pglib_voltage_start_continuation_r3p18.py)
  set_tests_properties(pglib_voltage_start_continuation_r3p18 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;BLIS_NUM_THREADS=1"
    TIMEOUT 120)
  add_test(NAME exporter_pglib_contract_r3p21
           COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_exporter_pglib_contract_r3p21.py)
  set_tests_properties(exporter_pglib_contract_r3p21 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;BLIS_NUM_THREADS=1"
    TIMEOUT 120)
  add_test(NAME pglib_voltage_start_continuation_r3p21
           COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pglib_voltage_start_continuation_r3p21.py)
  set_tests_properties(pglib_voltage_start_continuation_r3p21 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;BLIS_NUM_THREADS=1"
    TIMEOUT 120)
  add_test(NAME exporter_pglib_contract_r3p22
           COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_exporter_pglib_contract_r3p22.py
             --binary $<TARGET_FILE:lazyipdca_qcqp_cli>
             --linear-backend native)
  set_tests_properties(exporter_pglib_contract_r3p22 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;BLIS_NUM_THREADS=1"
    TIMEOUT 120)
  add_test(NAME pglib_voltage_start_continuation_r3p22
           COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pglib_voltage_start_continuation_r3p22.py)
  set_tests_properties(pglib_voltage_start_continuation_r3p22 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;BLIS_NUM_THREADS=1"
    TIMEOUT 120)
  add_test(NAME exporter_pglib_contract_r3p23
           COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_exporter_pglib_contract_r3p23.py)
  set_tests_properties(exporter_pglib_contract_r3p23 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;BLIS_NUM_THREADS=1"
    TIMEOUT 120)
  add_test(NAME qcq_model_r3p23
           COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_qcq_model_r3p23.py)
  set_tests_properties(qcq_model_r3p23 PROPERTIES TIMEOUT 60)
  add_test(NAME same_qcqp_protocol_r3p23
           COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_same_qcqp_protocol_r3p23.py)
  set_tests_properties(same_qcqp_protocol_r3p23 PROPERTIES TIMEOUT 60)
  add_test(NAME benchmark_same_qcqp_r3p24
           COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_benchmark_same_qcqp_r3p24.py)
  add_test(NAME verify_r3p26_candidate
           COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_verify_r3p26_candidate.py)
  set_tests_properties(verify_r3p26_candidate PROPERTIES TIMEOUT 60)
  add_test(NAME verify_r3p27_ab
           COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_verify_r3p27_ab.py)
  set_tests_properties(verify_r3p27_ab PROPERTIES TIMEOUT 60)
  add_test(NAME lazy_newton_cli_contract
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_lazy_newton_cli_contract.py
                   $<TARGET_FILE:lazyipdca_qcqp_cli>)
  add_test(NAME lazy_newton_regression
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_lazy_newton_regression.py
                   $<TARGET_FILE:lazyipdca_qcqp_cli>)
  add_test(NAME lazyqp_sparse_graft_regression
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_lazyqp_sparse_graft_regression.py
                   $<TARGET_FILE:lazyipdca_qcqp_cli>)
  set_tests_properties(
    lazy_newton_cli_contract
    lazy_newton_regression
    lazyqp_sparse_graft_regression
    PROPERTIES SKIP_RETURN_CODE 77)
  add_test(NAME r7d15_version_identity
           COMMAND lazyipdca_qcqp_cli --version)
  set_tests_properties(r7d15_version_identity PROPERTIES
    PASS_REGULAR_EXPRESSION
      "0.4.28r3p28g10-r7d15-majorants-r9-dual-tail.*API 4363.*r3p28g10-r7d15-majorants-r9-dual-tail-api-4363")
  set_tests_properties(benchmark_same_qcqp_r3p24 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;BLIS_NUM_THREADS=1"
    TIMEOUT 60)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_dca_soc_lift_projection_floor
    tests/test_dca_soc_lift_projection_floor.cpp)
  target_link_libraries(test_dca_soc_lift_projection_floor PRIVATE lazyipdca_qcqp)
  add_test(NAME dca_soc_lift_projection_floor
    COMMAND test_dca_soc_lift_projection_floor)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_phase1_persistent_openmp_parity
    tests/test_phase1_persistent_openmp_parity.cpp)
  target_link_libraries(test_phase1_persistent_openmp_parity PRIVATE lazyipdca_qcqp)
  add_test(NAME phase1_persistent_openmp_parity
    COMMAND test_phase1_persistent_openmp_parity)

  add_executable(test_equality_qp_original_kkt_gmres
    tests/test_equality_qp_original_kkt_gmres.cpp)
  target_link_libraries(test_equality_qp_original_kkt_gmres PRIVATE lazyipdca_qcqp)
  add_test(NAME equality_qp_original_kkt_gmres
    COMMAND test_equality_qp_original_kkt_gmres)

  add_executable(test_equality_qp_large_rho_original_kkt_gmres
    tests/test_equality_qp_large_rho_original_kkt_gmres.cpp)
  target_link_libraries(test_equality_qp_large_rho_original_kkt_gmres PRIVATE lazyipdca_qcqp)
  add_test(NAME equality_qp_large_rho_original_kkt_gmres
    COMMAND test_equality_qp_large_rho_original_kkt_gmres)

  add_test(NAME qp_prefixed_cli_surface
    COMMAND lazyipdca_qcqp_cli
      --problem ${CMAKE_CURRENT_SOURCE_DIR}/tests/toy.qcq
      --qp_option C
      --qp_amrs
      --qp_init 2
      --qp_backend native
      --qp_rho_trial_every 2
      --qp_max_atoms 500
      --qp_seed_coverage 0.25
      --qp_seed_stall_tol 0
      --qp_seed_patience 999
      --qp_atom_events
      --qp_atom_event_every 50
      --qp_bound_push 0
      --qp_bp_push_kappa 1e-4
      --qp_bp_margin_tol 1e-4
      --qp_escapes 3
      --qp_mehrotra
      --qp_sigma_max 1e14
      --qp_model_solved_tol 1e-12
      --qp_dca_stop
      --qp_dca_stall_tol_rel 1e-10
      --qp_dca_stall_iters 5
      --qp_dca_monotone_tol_abs 1e-6
      --qp_bp_adaptive
      --quiet)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_test(NAME qp_profile_c_amrs_defaults
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_qp_defaults.py
      $<TARGET_FILE:lazyipdca_qcqp_cli>)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_dca_soc_inequality_defect_continuity
    tests/test_dca_soc_inequality_defect_continuity.cpp)
  target_link_libraries(test_dca_soc_inequality_defect_continuity PRIVATE lazyipdca_qcqp)
  add_test(NAME dca_soc_inequality_defect_continuity
    COMMAND test_dca_soc_inequality_defect_continuity)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_test(NAME bt2_exact_lift
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_bt2_exact_lift.py)
  add_test(NAME eigen2_residual_lift
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_eigen2_residual_lift.py)
  add_test(NAME batch_runner_cli
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_batch_runner_cli.py)
  add_test(NAME cutest100_size_selector
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cutest100_selector.py)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_socp_formulation tests/test_socp_formulation.cpp)
  target_link_libraries(test_socp_formulation PRIVATE lazyipdca_qcqp)
  add_test(NAME socp_formulation COMMAND test_socp_formulation)

  add_executable(test_socp_equality_routes tests/test_socp_equality_routes.cpp)
  target_link_libraries(test_socp_equality_routes PRIVATE lazyipdca_qcqp)
  add_test(NAME socp_equality_routes COMMAND test_socp_equality_routes)

  add_executable(test_sparse_affine_tangent tests/test_sparse_affine_tangent.cpp)
  target_link_libraries(test_sparse_affine_tangent PRIVATE lazyipdca_qcqp)
  add_test(NAME sparse_affine_tangent COMMAND test_sparse_affine_tangent)
  set_tests_properties(sparse_affine_tangent PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME benchmark_case118_r3p27a1_certificate
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_benchmark_case118_r3p27a1.py)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_g1_newton_live_curvature tests/test_g1_newton_live_curvature.cpp)
  target_link_libraries(test_g1_newton_live_curvature PRIVATE lazyipdca_qcqp)
  add_test(NAME g1_newton_live_curvature COMMAND test_g1_newton_live_curvature)
  add_executable(test_g1_lazy_newton tests/test_g1_lazy_newton.cpp)
  target_link_libraries(test_g1_lazy_newton PRIVATE lazyipdca_qcqp)
  add_test(NAME g1_lazy_newton COMMAND test_g1_lazy_newton)
  add_executable(test_g1_original_certificate_sparse tests/test_g1_original_certificate_sparse.cpp)
  target_link_libraries(test_g1_original_certificate_sparse PRIVATE lazyipdca_qcqp)
  add_test(NAME g1_original_certificate_sparse COMMAND test_g1_original_certificate_sparse)
  add_executable(test_g1_curvature_probe tests/test_g1_curvature_probe.cpp)
  target_link_libraries(test_g1_curvature_probe PRIVATE lazyipdca_qcqp)
  add_test(NAME g1_curvature_probe COMMAND test_g1_curvature_probe)
  add_executable(test_g1_newton_atoms tests/test_g1_newton_atoms.cpp)
  target_link_libraries(test_g1_newton_atoms PRIVATE lazyipdca_qcqp)
  add_test(NAME g1_newton_atoms COMMAND test_g1_newton_atoms)
  add_executable(test_g1_newton_atom_projection tests/test_g1_newton_atom_projection.cpp)
  target_link_libraries(test_g1_newton_atom_projection PRIVATE lazyipdca_qcqp)
  add_test(NAME g1_newton_atom_projection COMMAND test_g1_newton_atom_projection)
  add_executable(test_g1_newton_display_levels tests/test_g1_newton_display_levels.cpp)
  target_link_libraries(test_g1_newton_display_levels PRIVATE lazyipdca_qcqp)
  add_test(NAME g1_newton_display_levels COMMAND test_g1_newton_display_levels)
  add_executable(test_g1_compact_qcq tests/test_g1_compact_qcq.cpp)
  target_link_libraries(test_g1_compact_qcq PRIVATE lazyipdca_qcqp)
  add_test(NAME g1_compact_qcq COMMAND test_g1_compact_qcq)
endif()
if(LAZYIPDCA_BUILD_TESTS AND Python3_Interpreter_FOUND)
  add_test(NAME g1_campaign_original_certificates
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_g1_campaign.py $<TARGET_FILE:lazyipdca_qcqp_cli>)
  set_tests_properties(g1_campaign_original_certificates PROPERTIES
    SKIP_RETURN_CODE 77)
endif()

if(LAZYIPDCA_BUILD_TESTS AND Python3_Interpreter_FOUND)
  add_test(NAME g1_compact_python_reader_parity
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_compact_python_reader.py)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_g3_filter_dca tests/test_g3_filter_dca.cpp)
  target_link_libraries(test_g3_filter_dca PRIVATE lazyipdca_qcqp)
  add_test(NAME g3_filter_dca_contract COMMAND test_g3_filter_dca)
endif()
if(LAZYIPDCA_BUILD_TESTS AND Python3_Interpreter_FOUND)
  add_test(NAME g3_campaign_contract
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_g3_campaign_contract.py $<TARGET_FILE:lazyipdca_qcqp_cli>)
  set_tests_properties(g3_campaign_contract PROPERTIES SKIP_RETURN_CODE 77)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_g4_persistent_dca tests/test_g4_persistent_dca.cpp)
  target_link_libraries(test_g4_persistent_dca PRIVATE lazyipdca_qcqp)
  add_test(NAME g4_collective_curvature_persistent_models COMMAND test_g4_persistent_dca)
  add_executable(test_g5_adaptive_dca tests/test_g5_adaptive_dca.cpp)
  target_link_libraries(test_g5_adaptive_dca PRIVATE lazyipdca_qcqp)
  add_test(NAME g5_adaptive_dca_original_certificates COMMAND test_g5_adaptive_dca)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  foreach(case_name IN ITEMS g6_block_curvature g6_anderson g6_equilibration g6_certificate_transport g6_tangent_curvature)
    add_executable(test_${case_name} tests/test_${case_name}.cpp)
    target_link_libraries(test_${case_name} PRIVATE lazyipdca_qcqp)
    add_test(NAME ${case_name} COMMAND test_${case_name})
    set_tests_properties(${case_name} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  endforeach()
endif()

if(LAZYIPDCA_BUILD_TESTS AND LAZYIPDCA_WITH_MUMPS AND LAZYIPDCA_WITH_METIS)
  add_executable(test_hybrid_mumps_border tests/test_hybrid_mumps_border.cpp)
  target_link_libraries(test_hybrid_mumps_border PRIVATE lazyipdca_qcqp)
  add_test(NAME hybrid_mumps_border_contract COMMAND test_hybrid_mumps_border)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  foreach(case_name IN ITEMS g7_local_active g7_pair_elimination g7_local_rollback g8_local_weak_affine g9_krylov g9_krylov_rollback g10_aggregate g10_aggregate_rollback)
    add_executable(test_${case_name} tests/test_${case_name}.cpp)
    target_link_libraries(test_${case_name} PRIVATE lazyipdca_qcqp)
    add_test(NAME ${case_name} COMMAND test_${case_name})
    set_tests_properties(${case_name} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  endforeach()
endif()

if(LAZYIPDCA_BUILD_TESTS AND Python3_Interpreter_FOUND)
  add_test(NAME g9_runner_contract COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_g9_runner_contract.py)
endif()

if(LAZYIPDCA_BUILD_TESTS AND Python3_Interpreter_FOUND)
  add_test(NAME g10_runner_contract COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_g10_runner_contract.py)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_g10_adaptive_dual tests/test_g10_adaptive_dual.cpp)
  target_link_libraries(test_g10_adaptive_dual PRIVATE lazyipdca_qcqp)
  add_test(NAME g10_adaptive_dual COMMAND test_g10_adaptive_dual)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_g10_acceptable tests/test_g10_acceptable.cpp)
  target_link_libraries(test_g10_acceptable PRIVATE lazyipdca_qcqp)
  add_test(NAME g10_acceptable COMMAND test_g10_acceptable)

  add_executable(test_g10_dual_step tests/test_g10_dual_step.cpp)
  target_link_libraries(test_g10_dual_step PRIVATE lazyipdca_qcqp)
  add_test(NAME g10_dual_step COMMAND test_g10_dual_step)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_ipopt_stop tests/test_ipopt_stop.cpp)
  target_link_libraries(test_ipopt_stop PRIVATE lazyipdca_qcqp)
  add_test(NAME ipopt_stop_contract COMMAND test_ipopt_stop)
endif()
if(LAZYIPDCA_BUILD_TESTS AND IPOPT_INCLUDE_DIR AND IPOPT_LIBRARY)
  add_executable(test_ipopt_stop_parity tests/test_ipopt_stop_parity.cpp)
  target_include_directories(test_ipopt_stop_parity PRIVATE ${IPOPT_INCLUDE_DIR} ${IPOPT_INCLUDE_DIR}/coin-or)
  target_link_libraries(test_ipopt_stop_parity PRIVATE lazyipdca_qcqp ${IPOPT_LIBRARY})
  add_test(NAME ipopt_stop_native_parity COMMAND test_ipopt_stop_parity)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(probe_endpoint tests/probe_endpoint.cpp)
  target_link_libraries(probe_endpoint PRIVATE lazyipdca_qcqp)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_g12_root_reuse tests/test_g12_root_reuse.cpp)
  target_link_libraries(test_g12_root_reuse PRIVATE lazyipdca_qcqp)
  add_test(NAME g12_root_reuse COMMAND test_g12_root_reuse)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  foreach(name IN ITEMS g14_aggregate_cache g14_strict_probe)
    add_executable(test_${name} tests/test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE lazyipdca_qcqp)
    target_compile_options(test_${name} PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME ${name} COMMAND test_${name})
  endforeach()
  if(Python3_Interpreter_FOUND)
    add_test(NAME g14_cache_cli COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_g14_cache_cli.py
      $<TARGET_FILE:lazyipdca_qcqp_cli> ${CMAKE_CURRENT_SOURCE_DIR}/tests/toy.qcq)
  endif()
endif()

if(LAZYIPDCA_BUILD_TESTS)
  foreach(name IN ITEMS g15_rho_grid g15_strict_finish)
    add_executable(test_${name} tests/test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE lazyipdca_qcqp)
    target_compile_options(test_${name} PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME ${name} COMMAND test_${name})
  endforeach()
  if(Python3_Interpreter_FOUND)
    add_test(NAME g15_cli COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_g15_cli.py
      $<TARGET_FILE:lazyipdca_qcqp_cli> ${CMAKE_CURRENT_SOURCE_DIR}/tests/toy.qcq)
  endif()
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_aggregate_only tests/test_aggregate_only.cpp)
  target_link_libraries(test_aggregate_only PRIVATE lazyipdca_qcqp)
  target_compile_options(test_aggregate_only PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME aggregate_only_contract COMMAND test_aggregate_only)
  if(Python3_Interpreter_FOUND)
    add_test(NAME aggregate_only_cli COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_aggregate_only_cli.py
      $<TARGET_FILE:lazyipdca_qcqp_cli> ${CMAKE_CURRENT_SOURCE_DIR}/tests/toy.qcq)
  endif()
endif()

add_executable(inspect_rank_presolve tools/inspect_rank_presolve.cpp)
target_link_libraries(inspect_rank_presolve PRIVATE lazyipdca_qcqp)
if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_rank_presolve tests/test_rank_presolve.cpp)
  target_link_libraries(test_rank_presolve PRIVATE lazyipdca_qcqp)
  add_test(NAME rank_presolve_coefficients COMMAND test_rank_presolve)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_test(NAME rank_presolve_cli COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rank_presolve_cli.py
    $<TARGET_FILE:lazyipdca_qcqp_cli> ${CMAKE_CURRENT_SOURCE_DIR}/tests/toy.qcq)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_custom_target(rank_release_targets DEPENDS
    inspect_rank_presolve
    test_rank_presolve
    test_aggregate_only
    lazyipdca_qcqp_cli
    test_g15_rho_grid
    test_g15_strict_finish
    test_g14_aggregate_cache
    test_g14_strict_probe
    test_ipopt_stop
    test_ipopt_stop_parity
    test_g12_root_reuse
    test_g10_terminal_policy
    test_g10_terminal_rescue
    test_g10_krylov_borders
    test_g9_krylov
    test_g9_krylov_rollback
    test_g10_acceptable
    test_g10_dual_step
    test_g10_adaptive_dual
    test_cholesk_backend
    test_linear_backends
    test_g10_memory_recovery
    test_g10_restoration_selection
    test_g10_progress_guard
    ipopt_qcqp_cli
    check_ma27
    mock_ma27_abi
    test_g10_ma27
    test_g10_filter_diagnostics
    test_g10_aggregate
    test_g10_aggregate_rollback
    test_g1_compact_qcq
    test_direct_ipopt_structure
    test_direct_ipopt_acceptable
    test_g3_filter_dca
  )
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_rank_audit tests/test_rank_audit.cpp)
  target_link_libraries(test_rank_audit PRIVATE lazyipdca_qcqp)
  add_test(NAME rank_audit_equivalence COMMAND test_rank_audit)
  add_dependencies(rank_release_targets test_rank_audit)
endif()

add_executable(benchmark_rank_audit tools/benchmark_rank_audit.cpp)
target_link_libraries(benchmark_rank_audit PRIVATE lazyipdca_qcqp)
if(LAZYIPDCA_BUILD_TESTS)
  add_dependencies(rank_release_targets benchmark_rank_audit)
  add_test(NAME rank_audit_cli COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rank_audit_cli.py
    $<TARGET_FILE:lazyipdca_qcqp_cli> ${CMAKE_CURRENT_SOURCE_DIR}/tests/toy.qcq)
endif()

# R3: orthogonal controller, structural equivalence, and exact fast paths.
if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_stable_border tests/test_stable_border.cpp)
  target_link_libraries(test_stable_border PRIVATE lazyipdca_qcqp)
  add_test(NAME r3_stable_border COMMAND test_stable_border)
  add_executable(test_g10_aggregate_controller tests/test_g10_aggregate_controller.cpp)
  target_link_libraries(test_g10_aggregate_controller PRIVATE lazyipdca_qcqp)
  add_test(NAME r3_aggregate_controller COMMAND test_g10_aggregate_controller)
  add_executable(test_g10_rejection_policy tests/test_g10_rejection_policy.cpp)
  target_link_libraries(test_g10_rejection_policy PRIVATE lazyipdca_qcqp)
  add_test(NAME r3_rejection_policy COMMAND test_g10_rejection_policy)
  add_executable(test_g16_aggregate_lean tests/test_g16_aggregate_lean.cpp)
  target_link_libraries(test_g16_aggregate_lean PRIVATE lazyipdca_qcqp)
  add_test(NAME r3_aggregate_lean COMMAND test_g16_aggregate_lean)
  add_executable(test_mumps_aggregate_assembly tests/test_mumps_aggregate_assembly.cpp)
  target_link_libraries(test_mumps_aggregate_assembly PRIVATE lazyipdca_qcqp)
  add_test(NAME mumps_aggregate_assembly COMMAND test_mumps_aggregate_assembly)
  add_dependencies(rank_release_targets test_stable_border test_g10_aggregate_controller test_g10_rejection_policy test_g16_aggregate_lean test_mumps_aggregate_assembly)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_r4_majorants tests/test_r4_majorants.cpp)
  target_link_libraries(test_r4_majorants PRIVATE lazyipdca_qcqp)
  add_test(NAME r4_majorants COMMAND test_r4_majorants)
  add_dependencies(rank_release_targets test_r4_majorants)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_r5_filter_path tests/test_r5_filter_path.cpp)
  target_link_libraries(test_r5_filter_path PRIVATE lazyipdca_qcqp)
  add_test(NAME r5_filter_path COMMAND test_r5_filter_path)
  add_dependencies(rank_release_targets test_r5_filter_path)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_r9_fused_direction tests/test_r9_fused_direction.cpp)
  target_link_libraries(test_r9_fused_direction PRIVATE lazyipdca_qcqp)
  add_test(NAME r9_fused_direction COMMAND test_r9_fused_direction)
  add_dependencies(rank_release_targets test_r9_fused_direction)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_original_filter_commit tests/test_original_filter_commit.cpp)
  target_link_libraries(test_original_filter_commit PRIVATE lazyipdca_qcqp)
  add_test(NAME original_filter_commit COMMAND test_original_filter_commit)
  add_dependencies(rank_release_targets test_original_filter_commit)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_kkt_precision_recovery tests/test_kkt_precision_recovery.cpp)
  target_link_libraries(test_kkt_precision_recovery PRIVATE lazyipdca_qcqp)
  add_test(NAME kkt_precision_recovery COMMAND test_kkt_precision_recovery)
  add_dependencies(rank_release_targets test_kkt_precision_recovery)
endif()

if(LAZYIPDCA_BUILD_TESTS)
 add_executable(test_spectral_majorant tests/test_spectral_majorant.cpp)
 target_link_libraries(test_spectral_majorant PRIVATE lazyipdca_qcqp)
 add_test(NAME spectral_majorant COMMAND test_spectral_majorant)
 add_dependencies(rank_release_targets test_spectral_majorant)
 add_executable(test_dual_tail tests/test_dual_tail.cpp)
 target_link_libraries(test_dual_tail PRIVATE lazyipdca_qcqp)
 add_test(NAME dual_tail COMMAND test_dual_tail)
 add_dependencies(rank_release_targets test_dual_tail)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_native_signed_square_reduction tests/test_native_signed_square_reduction.cpp)
  target_link_libraries(test_native_signed_square_reduction PRIVATE lazyipdca_qcqp)
  add_test(NAME native_signed_square_reduction COMMAND test_native_signed_square_reduction)
  add_executable(test_native_signed_square_path tests/test_native_signed_square_path.cpp)
  target_link_libraries(test_native_signed_square_path PRIVATE lazyipdca_qcqp)
  add_test(NAME native_signed_square_path COMMAND test_native_signed_square_path)
  add_dependencies(rank_release_targets test_native_signed_square_reduction test_native_signed_square_path)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_native_restoration tests/test_native_restoration.cpp)
  target_link_libraries(test_native_restoration PRIVATE lazyipdca_qcqp)
  add_test(NAME native_restoration COMMAND test_native_restoration)
  add_dependencies(rank_release_targets test_native_restoration)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_bounded_origin_certificate tests/test_bounded_origin_certificate.cpp)
  target_link_libraries(test_bounded_origin_certificate PRIVATE lazyipdca_qcqp)
  add_test(NAME bounded_origin_certificate COMMAND test_bounded_origin_certificate)
  add_dependencies(rank_release_targets test_bounded_origin_certificate)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_restoration_sparse_storage tests/test_restoration_sparse_storage.cpp)
  target_link_libraries(test_restoration_sparse_storage PRIVATE lazyipdca_qcqp)
  add_test(NAME restoration_sparse_storage COMMAND test_restoration_sparse_storage)
  add_executable(test_restoration_negative_curvature tests/test_restoration_negative_curvature.cpp)
  target_link_libraries(test_restoration_negative_curvature PRIVATE lazyipdca_qcqp)
  add_test(NAME restoration_negative_curvature COMMAND test_restoration_negative_curvature)
  add_dependencies(rank_release_targets test_restoration_sparse_storage test_restoration_negative_curvature)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_majorant_startup_policy tests/test_majorant_startup_policy.cpp)
  target_link_libraries(test_majorant_startup_policy PRIVATE lazyipdca_qcqp)
  add_test(NAME majorant_startup_policy COMMAND test_majorant_startup_policy)
  add_dependencies(rank_release_targets test_majorant_startup_policy)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_majorant_mixed_startup tests/test_majorant_mixed_startup.cpp)
  target_link_libraries(test_majorant_mixed_startup PRIVATE lazyipdca_qcqp)
  add_test(NAME majorant_mixed_startup COMMAND test_majorant_mixed_startup)
  add_dependencies(rank_release_targets test_majorant_mixed_startup)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_square_feasibility tests/test_square_feasibility.cpp)
  target_link_libraries(test_square_feasibility PRIVATE lazyipdca_qcqp)
  add_test(NAME square_feasibility COMMAND test_square_feasibility)
  add_dependencies(rank_release_targets test_square_feasibility)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_feasible_boundary_certificate tests/test_feasible_boundary_certificate.cpp)
  target_link_libraries(test_feasible_boundary_certificate PRIVATE lazyipdca_qcqp)
  target_compile_options(test_feasible_boundary_certificate PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME feasible_boundary_certificate COMMAND test_feasible_boundary_certificate)
  if(TARGET rank_release_targets)
    add_dependencies(rank_release_targets test_feasible_boundary_certificate)
  endif()
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_bounded_pivot_recovery tests/test_bounded_pivot_recovery.cpp)
  target_link_libraries(test_bounded_pivot_recovery PRIVATE lazyipdca_qcqp)
  add_test(NAME bounded_pivot_recovery COMMAND test_bounded_pivot_recovery)
  add_executable(test_failed_linearization_restoration tests/test_failed_linearization_restoration.cpp)
  target_link_libraries(test_failed_linearization_restoration PRIVATE lazyipdca_qcqp)
  add_test(NAME failed_linearization_restoration COMMAND test_failed_linearization_restoration)
  add_dependencies(rank_release_targets test_bounded_pivot_recovery test_failed_linearization_restoration)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_failure_recovery_policy tests/test_failure_recovery_policy.cpp)
  target_link_libraries(test_failure_recovery_policy PRIVATE lazyipdca_qcqp)
  add_test(NAME failure_recovery_policy COMMAND test_failure_recovery_policy)
  add_executable(test_overdetermined_equalities tests/test_overdetermined_equalities.cpp)
  target_link_libraries(test_overdetermined_equalities PRIVATE lazyipdca_qcqp)
  add_test(NAME overdetermined_equalities COMMAND test_overdetermined_equalities)
  add_executable(test_accurate_product_sum tests/test_accurate_product_sum.cpp)
  target_include_directories(test_accurate_product_sum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${LAZYIPDCA_BOOST_INCLUDE_DIR})
  add_test(NAME accurate_product_sum COMMAND test_accurate_product_sum)
  add_executable(test_affine_singleton_bounds tests/test_affine_singleton_bounds.cpp)
  target_link_libraries(test_affine_singleton_bounds PRIVATE lazyipdca_qcqp)
  add_test(NAME affine_singleton_bounds COMMAND test_affine_singleton_bounds)
  add_executable(test_rank_structural_retry tests/test_rank_structural_retry.cpp)
  target_link_libraries(test_rank_structural_retry PRIVATE lazyipdca_qcqp)
  add_test(NAME rank_structural_retry COMMAND test_rank_structural_retry)
  add_executable(test_rank_retry_statistics tests/test_rank_retry_statistics.cpp)
  target_link_libraries(test_rank_retry_statistics PRIVATE lazyipdca_qcqp)
  add_test(NAME rank_retry_statistics COMMAND test_rank_retry_statistics)
  add_executable(test_native_barrier_retry_statistics tests/test_native_barrier_retry_statistics.cpp)
  target_link_libraries(test_native_barrier_retry_statistics PRIVATE lazyipdca_qcqp)
  add_test(NAME native_barrier_retry_statistics COMMAND test_native_barrier_retry_statistics)
  add_executable(test_hybrid_slack_path tests/test_hybrid_slack_path.cpp)
  target_include_directories(test_hybrid_slack_path PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME hybrid_slack_path COMMAND test_hybrid_slack_path)
  add_executable(test_phase1_restart_policy tests/test_phase1_restart_policy.cpp)
  add_test(NAME phase1_restart_policy COMMAND test_phase1_restart_policy)
  add_executable(test_sparse_only_restoration tests/test_sparse_only_restoration.cpp)
  target_link_libraries(test_sparse_only_restoration PRIVATE lazyipdca_qcqp)
  add_test(NAME sparse_only_restoration COMMAND test_sparse_only_restoration)
  add_dependencies(rank_release_targets test_sparse_only_restoration)
  add_executable(test_restoration_shared_budget tests/test_restoration_shared_budget.cpp)
  target_link_libraries(test_restoration_shared_budget PRIVATE lazyipdca_qcqp)
  add_test(NAME restoration_shared_budget COMMAND test_restoration_shared_budget)
  add_dependencies(rank_release_targets test_restoration_shared_budget)
  add_executable(test_pair_inertia_lifting tests/test_pair_inertia_lifting.cpp)
  target_link_libraries(test_pair_inertia_lifting PRIVATE lazyipdca_qcqp)
  add_test(NAME pair_inertia_lifting COMMAND test_pair_inertia_lifting)
  add_dependencies(rank_release_targets test_failure_recovery_policy test_pair_inertia_lifting)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_column_lm tests/test_column_lm.cpp)
  target_link_libraries(test_column_lm PRIVATE lazyipdca_qcqp)
  add_test(NAME column_lm COMMAND test_column_lm)
  add_dependencies(rank_release_targets test_column_lm)
endif()

if(TARGET rank_release_targets AND TARGET test_g7_pair_elimination)
  add_dependencies(rank_release_targets test_g7_pair_elimination)
endif()

# CORE2 controller regressions: bounded retry lifecycle and exact frozen input.
if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_filter_retry_lifecycle tests/test_filter_retry_lifecycle.cpp)
  target_link_libraries(test_filter_retry_lifecycle PRIVATE lazyipdca_qcqp)
  add_dependencies(rank_release_targets test_filter_retry_lifecycle)
  add_test(NAME filter_retry_lifecycle_native COMMAND test_filter_retry_lifecycle native)
  foreach(_backend IN ITEMS mumps cholesk)
    if((_backend STREQUAL "mumps" AND LAZYIPDCA_WITH_MUMPS AND MUMPS_DMUMPS_LIBRARY AND METIS_LIBRARY)
       OR (_backend STREQUAL "cholesk" AND LAZYIPDCA_WITH_QDLDL AND QDLDL_LIBRARY AND METIS_LIBRARY))
      add_test(NAME filter_retry_lifecycle_${_backend} COMMAND test_filter_retry_lifecycle ${_backend})
      foreach(_variant IN ITEMS current rho tail)
        add_test(NAME core2_frozen_${_backend}_${_variant}
          COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_core2_frozen.py
            --binary $<TARGET_FILE:lazyipdca_qcqp_cli> --backend ${_backend} --variant ${_variant})
        set_tests_properties(core2_frozen_${_backend}_${_variant} PROPERTIES TIMEOUT 1810)
      endforeach()
    endif()
  endforeach()
endif()

# R9 targeted recovery and sparse-order safeguards.
if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_recovery_controller_isolation tests/test_recovery_controller_isolation.cpp)
  target_link_libraries(test_recovery_controller_isolation PRIVATE lazyipdca_qcqp)
  add_test(NAME recovery_controller_isolation COMMAND test_recovery_controller_isolation)
  if(LAZYIPDCA_WITH_QDLDL AND LAZYIPDCA_WITH_METIS)
    add_executable(test_cholesk_deferred_ordering tests/test_cholesk_deferred_ordering.cpp)
    target_link_libraries(test_cholesk_deferred_ordering PRIVATE lazyipdca_qcqp)
    add_test(NAME cholesk_deferred_ordering COMMAND test_cholesk_deferred_ordering)
    add_dependencies(rank_release_targets test_cholesk_deferred_ordering)
  endif()
  add_dependencies(rank_release_targets test_recovery_controller_isolation)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_mss_dual_guard tests/test_mss_dual_guard.cpp)
  target_link_libraries(test_mss_dual_guard PRIVATE lazyipdca_qcqp)
  # This test includes the private workspace solely for regression testing.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_mss_dual_guard PRIVATE -O0)
  endif()
  add_test(NAME mss_dual_candidate_guard COMMAND test_mss_dual_guard)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_pglib_filter_recenter tests/test_pglib_filter_recenter.cpp)
  target_include_directories(test_pglib_filter_recenter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME pglib_filter_recenter_policy COMMAND test_pglib_filter_recenter)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_auto_startup tests/test_auto_startup.cpp)
  target_link_libraries(test_auto_startup PRIVATE lazyipdca_qcqp)
  add_test(NAME auto_startup_policy COMMAND test_auto_startup)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_auto_startup_retry tests/test_auto_startup_retry.cpp)
  target_include_directories(test_auto_startup_retry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME auto_startup_retry_policy COMMAND test_auto_startup_retry)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_test(NAME auto_startup_retry_synthetic
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_auto_retry_integration.py
      --binary $<TARGET_FILE:lazyipdca_qcqp_cli>
      --input ${CMAKE_CURRENT_SOURCE_DIR}/tests/regression/auto_retry_stall_synthetic.qcq
      --command ${CMAKE_CURRENT_SOURCE_DIR}/tests/auto_retry_stall_command.json)
  set_tests_properties(auto_startup_retry_synthetic PROPERTIES TIMEOUT 130
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;BLIS_NUM_THREADS=1")
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_terminal_central_retry tests/test_terminal_central_retry.cpp)
  target_include_directories(test_terminal_central_retry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME terminal_central_retry_policy COMMAND test_terminal_central_retry)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_convexification_study tests/test_convexification_study.cpp)
  target_link_libraries(test_convexification_study PRIVATE lazyipdca_qcqp)
  add_test(NAME convexification_study_certificates COMMAND test_convexification_study)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  foreach(study_test hybrid_border hybrid_inexact hybrid_controller hybrid_selection hybrid_spectrum)
    add_executable(test_${study_test} tests/test_${study_test}.cpp)
    target_include_directories(test_${study_test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(study_test STREQUAL "hybrid_controller" OR study_test STREQUAL "hybrid_selection" OR study_test STREQUAL "hybrid_spectrum")
      target_link_libraries(test_${study_test} PRIVATE lazyipdca_qcqp)
    endif()
    add_test(NAME ${study_test}_contract COMMAND test_${study_test})
  endforeach()
  add_executable(test_hybrid_soc tests/test_hybrid_soc.cpp)
  target_link_libraries(test_hybrid_soc PRIVATE lazyipdca_qcqp)
  add_test(NAME hybrid_soc COMMAND test_hybrid_soc)
  add_executable(test_hybrid_original_bounds tests/test_hybrid_original_bounds.cpp)
  target_link_libraries(test_hybrid_original_bounds PRIVATE lazyipdca_qcqp)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_hybrid_original_bounds PRIVATE -O0)
  endif()
  add_test(NAME hybrid_original_bounds COMMAND test_hybrid_original_bounds)
  add_executable(test_h4_correction_guards tests/test_h4_correction_guards.cpp)
  target_link_libraries(test_h4_correction_guards PRIVATE lazyipdca_qcqp)
  add_test(NAME h4_correction_guards COMMAND test_h4_correction_guards)
  add_executable(test_h4_domain tests/test_h4_domain.cpp)
  target_include_directories(test_h4_domain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_h4_domain PRIVATE lazyipdca_qcqp)
  add_test(NAME h4_domain COMMAND test_h4_domain)
  add_executable(test_h4_domain_certificate tests/test_h4_domain_certificate.cpp)
  target_link_libraries(test_h4_domain_certificate PRIVATE lazyipdca_qcqp)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_h4_domain_certificate PRIVATE -O0)
  endif()
  add_test(NAME h4_domain_certificate COMMAND test_h4_domain_certificate)
  add_executable(test_objective_intrinsic_margin tests/test_objective_intrinsic_margin.cpp)
  target_link_libraries(test_objective_intrinsic_margin PRIVATE lazyipdca_qcqp)
  add_test(NAME objective_intrinsic_margin COMMAND test_objective_intrinsic_margin)
  add_executable(test_implicit_interval_psd tests/test_implicit_interval_psd.cpp)
  target_link_libraries(test_implicit_interval_psd PRIVATE lazyipdca_qcqp)
  add_test(NAME implicit_interval_psd COMMAND test_implicit_interval_psd)

  add_executable(test_hybrid_query_cache tests/test_hybrid_query_cache.cpp)
  target_link_libraries(test_hybrid_query_cache PRIVATE lazyipdca_qcqp)
  add_test(NAME hybrid_query_cache COMMAND test_hybrid_query_cache)

  add_executable(test_verified_sparse_psd tests/test_verified_sparse_psd.cpp)
  target_link_libraries(test_verified_sparse_psd PRIVATE lazyipdca_qcqp)
  add_test(NAME verified_sparse_psd COMMAND test_verified_sparse_psd)

  add_executable(test_hybrid_recovery tests/test_hybrid_recovery.cpp)
  target_link_libraries(test_hybrid_recovery PRIVATE lazyipdca_qcqp)
  add_test(NAME hybrid_recovery COMMAND test_hybrid_recovery)

  add_executable(test_h4_initial_metric tests/test_h4_initial_metric.cpp)
  target_link_libraries(test_h4_initial_metric PRIVATE lazyipdca_qcqp)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_h4_initial_metric PRIVATE -O0)
  endif()
  add_test(NAME h4_initial_metric COMMAND test_h4_initial_metric)
  add_executable(test_hybrid_terminal_certificate tests/test_hybrid_terminal_certificate.cpp)
  target_link_libraries(test_hybrid_terminal_certificate PRIVATE lazyipdca_qcqp)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_hybrid_terminal_certificate PRIVATE -O0)
  endif()
  add_test(NAME hybrid_terminal_certificate COMMAND test_hybrid_terminal_certificate)
endif()

if(LAZYIPDCA_BUILD_TESTS)
 add_executable(test_elastic_phase1 tests/test_elastic_phase1.cpp)
 target_link_libraries(test_elastic_phase1 PRIVATE lazyipdca_qcqp)
 add_test(NAME elastic_phase1 COMMAND test_elastic_phase1)
 add_dependencies(rank_release_targets test_elastic_phase1)
endif()

if(LAZYIPDCA_BUILD_TESTS)
 add_executable(test_convex_affine_pullback tests/test_convex_affine_pullback.cpp)
 target_link_libraries(test_convex_affine_pullback PRIVATE lazyipdca_qcqp)
 add_test(NAME convex_affine_pullback COMMAND test_convex_affine_pullback)
 add_dependencies(rank_release_targets test_convex_affine_pullback)
endif()

if(LAZYIPDCA_BUILD_TESTS)
 add_executable(test_native_inexact_contract tests/test_native_inexact_contract.cpp)
 target_link_libraries(test_native_inexact_contract PRIVATE lazyipdca_qcqp)
 add_test(NAME native_inexact_contract COMMAND test_native_inexact_contract)
 add_dependencies(rank_release_targets test_native_inexact_contract)
endif()

if(LAZYIPDCA_BUILD_TESTS)
 add_executable(test_hybrid_split_step tests/test_hybrid_split_step.cpp)
 target_link_libraries(test_hybrid_split_step PRIVATE lazyipdca_qcqp)
 add_test(NAME hybrid_split_step COMMAND test_hybrid_split_step)
 add_dependencies(rank_release_targets test_hybrid_split_step)
endif()

if(LAZYIPDCA_BUILD_TESTS)
 add_executable(test_hybrid_merit_recovery tests/test_hybrid_merit_recovery.cpp)
 target_link_libraries(test_hybrid_merit_recovery PRIVATE lazyipdca_qcqp)
 add_test(NAME hybrid_merit_recovery COMMAND test_hybrid_merit_recovery)
 add_dependencies(rank_release_targets test_hybrid_merit_recovery)
endif()

if(LAZYIPDCA_BUILD_TESTS)
 add_executable(test_hybrid_source_soc tests/test_hybrid_source_soc.cpp)
 target_link_libraries(test_hybrid_source_soc PRIVATE lazyipdca_qcqp)
 add_test(NAME hybrid_source_soc COMMAND test_hybrid_source_soc)
 add_dependencies(rank_release_targets test_hybrid_source_soc)
endif()

if(LAZYIPDCA_BUILD_TESTS)
 add_executable(test_terminal_multiplier_fit tests/test_terminal_multiplier_fit.cpp)
 target_link_libraries(test_terminal_multiplier_fit PRIVATE lazyipdca_qcqp)
 add_test(NAME terminal_multiplier_fit COMMAND test_terminal_multiplier_fit)
 add_executable(test_bounded_normal_restart tests/test_bounded_normal_restart.cpp)
 target_link_libraries(test_bounded_normal_restart PRIVATE lazyipdca_qcqp)
 add_test(NAME bounded_normal_restart COMMAND test_bounded_normal_restart)
 add_executable(test_bounded_normal_active_set tests/test_bounded_normal_active_set.cpp)
 target_link_libraries(test_bounded_normal_active_set PRIVATE lazyipdca_qcqp ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
 add_test(NAME bounded_normal_active_set COMMAND test_bounded_normal_active_set)
 add_dependencies(rank_release_targets test_bounded_normal_restart test_bounded_normal_active_set)
 add_dependencies(rank_release_targets test_terminal_multiplier_fit)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_majorant_barrier_retry tests/test_majorant_barrier_retry.cpp)
  target_include_directories(test_majorant_barrier_retry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME majorant_barrier_retry_policy COMMAND test_majorant_barrier_retry)
  add_dependencies(rank_release_targets test_majorant_barrier_retry)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_affine_reduction_budget tests/test_affine_reduction_budget.cpp)
  add_test(NAME affine_reduction_budget COMMAND test_affine_reduction_budget)
  add_dependencies(rank_release_targets test_affine_reduction_budget)
  add_executable(test_nonlinear_trial_soc_policy tests/test_nonlinear_trial_soc_policy.cpp)
  add_test(NAME nonlinear_trial_soc_policy COMMAND test_nonlinear_trial_soc_policy)
  add_dependencies(rank_release_targets test_nonlinear_trial_soc_policy)
  add_executable(test_small_step_soc_policy tests/test_small_step_soc_policy.cpp)
  add_test(NAME small_step_soc_policy COMMAND test_small_step_soc_policy)
  add_dependencies(rank_release_targets test_small_step_soc_policy)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_rho_decay_feasibility_guard tests/test_rho_decay_feasibility_guard.cpp)
  add_test(NAME rho_decay_feasibility_guard COMMAND test_rho_decay_feasibility_guard)
  add_dependencies(rank_release_targets test_rho_decay_feasibility_guard)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_trial_soc_full_primal tests/test_trial_soc_full_primal.cpp)
  target_link_libraries(test_trial_soc_full_primal PRIVATE lazyipdca_qcqp)
  target_compile_options(test_trial_soc_full_primal PRIVATE -O0)
  add_test(NAME trial_soc_full_primal COMMAND test_trial_soc_full_primal)
  add_executable(test_trial_soc_arc tests/test_trial_soc_arc.cpp)
  target_link_libraries(test_trial_soc_arc PRIVATE lazyipdca_qcqp)
  target_compile_options(test_trial_soc_arc PRIVATE -O0)
  add_test(NAME trial_soc_arc COMMAND test_trial_soc_arc)
  add_executable(test_soc_defect_control tests/test_soc_defect_control.cpp)
  target_link_libraries(test_soc_defect_control PRIVATE lazyipdca_qcqp)
  target_compile_options(test_soc_defect_control PRIVATE -O0)
  add_test(NAME soc_defect_control COMMAND test_soc_defect_control)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_cold_dual_balance tests/test_cold_dual_balance.cpp)
  target_include_directories(test_cold_dual_balance PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME cold_dual_balance COMMAND test_cold_dual_balance)
  add_executable(test_cold_dual_balance_workspace tests/test_cold_dual_balance_workspace.cpp)
  target_link_libraries(test_cold_dual_balance_workspace PRIVATE lazyipdca_qcqp)
  target_compile_options(test_cold_dual_balance_workspace PRIVATE -O0)
  add_test(NAME cold_dual_balance_workspace COMMAND test_cold_dual_balance_workspace)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_accepted_checkpoint_writer tests/test_accepted_checkpoint_writer.cpp)
  target_include_directories(test_accepted_checkpoint_writer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME accepted_checkpoint_writer COMMAND test_accepted_checkpoint_writer)
  add_executable(test_accepted_checkpoint_scope tests/test_accepted_checkpoint_scope.cpp)
  target_link_libraries(test_accepted_checkpoint_scope PRIVATE lazyipdca_qcqp)
  target_compile_definitions(test_accepted_checkpoint_scope PRIVATE CHECKPOINT_RANK_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/data/checkpoint_rank_leaf.qcq")
  add_test(NAME accepted_checkpoint_scope COMMAND test_accepted_checkpoint_scope)
  add_custom_target(cold_start_qualification DEPENDS lazyipdca_qcqp_cli
    test_nonlinear_trial_soc_policy test_hybrid_query_cache test_affine_reduction_budget
    test_small_step_soc_policy test_failure_recovery_policy test_hybrid_recovery
    test_implicit_interval_psd test_verified_sparse_psd test_h4_initial_metric
    test_hybrid_original_bounds test_rank_presolve test_overdetermined_equalities
    test_hybrid_soc test_original_filter_commit test_trial_soc_full_primal test_trial_soc_arc test_soc_defect_control
    test_rho_decay_feasibility_guard test_cold_dual_balance test_cold_dual_balance_workspace
    test_accepted_checkpoint_writer test_accepted_checkpoint_scope)
endif()

if(LAZYIPDCA_BUILD_TESTS AND LAZYIPDCA_WITH_MUMPS)
  add_executable(test_ss_composition tests/test_ss_composition.cpp)
  target_link_libraries(test_ss_composition PRIVATE lazyipdca_qcqp)
  add_test(NAME ss_composition COMMAND test_ss_composition)
  add_executable(test_frozen_start_json tests/test_frozen_start_json.cpp)
  target_link_libraries(test_frozen_start_json PRIVATE lazyipdca_qcqp)
  add_test(NAME frozen_start_json COMMAND test_frozen_start_json)
endif()

if(LAZYIPDCA_BUILD_TESTS AND LAZYIPDCA_WITH_MUMPS AND MUMPS_INCLUDE_DIR AND MUMPS_DMUMPS_LIBRARY AND METIS_INCLUDE_DIR AND METIS_LIBRARY)
  add_executable(test_native_augmented_repair tests/test_native_augmented_repair.cpp)
  target_link_libraries(test_native_augmented_repair PRIVATE lazyipdca_qcqp)
  target_compile_options(test_native_augmented_repair PRIVATE -UNDEBUG)
  add_test(NAME native_augmented_repair COMMAND test_native_augmented_repair)
  add_executable(test_native_bilinear_start tests/test_native_bilinear_start.cpp)
  target_link_libraries(test_native_bilinear_start PRIVATE lazyipdca_qcqp)
  target_compile_options(test_native_bilinear_start PRIVATE -UNDEBUG)
  add_test(NAME native_bilinear_start COMMAND test_native_bilinear_start)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_native_al_fallback_policy tests/test_native_al_fallback_policy.cpp)
  target_include_directories(test_native_al_fallback_policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME native_al_fallback_policy COMMAND test_native_al_fallback_policy)
  add_executable(test_signed_quadratic_equality_domain tests/test_signed_quadratic_equality_domain.cpp)
  target_include_directories(test_signed_quadratic_equality_domain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME signed_quadratic_equality_domain COMMAND test_signed_quadratic_equality_domain)
  add_executable(test_standard_exact_presolve tests/test_standard_exact_presolve.cpp)
  target_include_directories(test_standard_exact_presolve PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME standard_exact_presolve COMMAND test_standard_exact_presolve)
  add_executable(test_standard_auto_policy tests/test_standard_auto_policy.cpp)
  target_include_directories(test_standard_auto_policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME standard_auto_policy COMMAND test_standard_auto_policy)
  add_executable(test_gr_fixed_centering tests/test_gr_fixed_centering.cpp)
  target_include_directories(test_gr_fixed_centering PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME gr_fixed_centering COMMAND test_gr_fixed_centering)
  add_test(NAME r10_full_step_controller
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/r10_bound_controller/run_actual_r10_control_test.py
      --source ${CMAKE_CURRENT_SOURCE_DIR}
      --output ${CMAKE_CURRENT_BINARY_DIR}/r10_bound_controller
      --cxx ${CMAKE_CXX_COMPILER})
  set_tests_properties(r10_full_step_controller PROPERTIES TIMEOUT 60)
  add_executable(test_gr_proximal_policy tests/test_gr_proximal_policy.cpp)
  target_include_directories(test_gr_proximal_policy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME gr_proximal_policy COMMAND test_gr_proximal_policy)
  add_test(NAME gr_proximal_controller
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gr_proximal_controller.py
      --source ${CMAKE_CURRENT_SOURCE_DIR}
      --output ${CMAKE_CURRENT_BINARY_DIR}/gr_proximal_controller
      --cxx ${CMAKE_CXX_COMPILER})
  set_tests_properties(gr_proximal_controller PROPERTIES TIMEOUT 60)
  add_executable(test_coefficient_constraint_increment tests/test_coefficient_constraint_increment.cpp)
  target_include_directories(test_coefficient_constraint_increment PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME coefficient_constraint_increment COMMAND test_coefficient_constraint_increment)
  add_test(NAME gr_coefficient_actual_wiring
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gr_coefficient_actual_wiring.py
      --source ${CMAKE_CURRENT_SOURCE_DIR}
      --output ${CMAKE_CURRENT_BINARY_DIR}/gr_coefficient_actual_wiring
      --cxx ${CMAKE_CXX_COMPILER})
  set_tests_properties(gr_coefficient_actual_wiring PROPERTIES TIMEOUT 60)
endif()

# Domain and epoch contracts for equality-only stabilized systems.
if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_zero_barrier_merit tests/test_zero_barrier_merit.cpp)
  target_include_directories(test_zero_barrier_merit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME zero_barrier_merit COMMAND test_zero_barrier_merit)
  add_executable(test_zero_barrier_epoch tests/test_zero_barrier_epoch.cpp)
  target_include_directories(test_zero_barrier_epoch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME zero_barrier_epoch COMMAND test_zero_barrier_epoch)
endif()

# Fixed-primal original KKT completion with original bounds and signs.
if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_initial_original_fit_guard tests/test_initial_original_fit_guard.cpp)
  add_test(NAME homogeneous_spectral_envelope
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_homogeneous_spectral_envelope.py
      $<TARGET_FILE:lazyipdca_qcqp_cli>)
  target_link_libraries(test_initial_original_fit_guard PRIVATE lazyipdca_qcqp)
  add_test(NAME initial_original_fit_guard COMMAND test_initial_original_fit_guard)
  add_executable(test_initial_original_fit_certificate tests/test_initial_original_fit_certificate.cpp)
  target_link_libraries(test_initial_original_fit_certificate PRIVATE lazyipdca_qcqp)
  add_test(NAME initial_original_fit_certificate COMMAND test_initial_original_fit_certificate)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_initial_original_fit_guard PRIVATE -UNDEBUG)
    target_compile_options(test_initial_original_fit_certificate PRIVATE -UNDEBUG)
  endif()
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_test(NAME constant_objective_guard
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_constant_objective_guard.py
      --source ${CMAKE_CURRENT_SOURCE_DIR} --cxx ${CMAKE_CXX_COMPILER})
  add_test(NAME constant_feasibility_cli
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_constant_feasibility_cli.py
      $<TARGET_FILE:lazyipdca_qcqp_cli>)
  set_tests_properties(constant_objective_guard constant_feasibility_cli PROPERTIES TIMEOUT 90)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_test(NAME initial_original_fit_provenance
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_initial_original_fit_provenance.py
      $<TARGET_FILE:lazyipdca_qcqp_cli>)
  set_tests_properties(initial_original_fit_provenance PROPERTIES TIMEOUT 60)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_psd_equality_route tests/test_psd_equality_route.cpp)
  target_include_directories(test_psd_equality_route PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME diagonal_psd_equality_route COMMAND test_psd_equality_route)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_square_continuation tests/test_square_continuation.cpp)
  target_link_libraries(test_square_continuation PRIVATE lazyipdca_qcqp)
  add_test(NAME square_continuation COMMAND test_square_continuation)
  add_executable(test_zero_barrier_forcing tests/test_zero_barrier_forcing.cpp)
  target_link_libraries(test_zero_barrier_forcing PRIVATE lazyipdca_qcqp)
  add_test(NAME zero_barrier_forcing COMMAND test_zero_barrier_forcing)
  add_executable(test_constant_square_policy tests/test_constant_square_policy.cpp)
  target_link_libraries(test_constant_square_policy PRIVATE lazyipdca_qcqp)
  add_test(NAME constant_square_policy COMMAND test_constant_square_policy)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_positive_square_epigraph_restoration tests/test_positive_square_epigraph_restoration.cpp)
  target_include_directories(test_positive_square_epigraph_restoration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_positive_square_epigraph_restoration PRIVATE lazyipdca_qcqp)
  add_test(NAME positive_square_epigraph_restoration COMMAND test_positive_square_epigraph_restoration)
endif()

# Qualification binaries retain assertions even when the solver is Release.
if(LAZYIPDCA_BUILD_TESTS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  get_property(_qualification_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_target IN LISTS _qualification_targets)
    if(_target MATCHES "^test_")
      target_compile_options(${_target} PRIVATE -UNDEBUG)
    endif()
  endforeach()
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_coefficient_graph_certificate tests/test_coefficient_graph_certificate.cpp)
  target_include_directories(test_coefficient_graph_certificate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_coefficient_graph_certificate PRIVATE lazyipdca_qcqp)
  target_compile_options(test_coefficient_graph_certificate PRIVATE -UNDEBUG)
  add_test(NAME coefficient_graph_certificate COMMAND test_coefficient_graph_certificate)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_gr_bound_stagnation_watch tests/test_gr_bound_stagnation_watch.cpp)
  target_include_directories(test_gr_bound_stagnation_watch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_options(test_gr_bound_stagnation_watch PRIVATE -UNDEBUG)
  add_test(NAME gr_bound_stagnation_watch COMMAND test_gr_bound_stagnation_watch)

  add_executable(test_convex_aggregate_certificate tests/test_convex_aggregate_certificate.cpp)
  target_include_directories(test_convex_aggregate_certificate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_convex_aggregate_certificate PRIVATE lazyipdca_qcqp)
  target_compile_options(test_convex_aggregate_certificate PRIVATE -UNDEBUG)
  add_test(NAME convex_aggregate_certificate COMMAND test_convex_aggregate_certificate)

  add_executable(test_affine_cycle_certificate tests/test_affine_cycle_certificate.cpp)
  target_include_directories(test_affine_cycle_certificate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_affine_cycle_certificate PRIVATE lazyipdca_qcqp)
  target_compile_options(test_affine_cycle_certificate PRIVATE -UNDEBUG)
  add_test(NAME affine_cycle_certificate COMMAND test_affine_cycle_certificate)

  add_executable(test_common_factor_substitution tests/test_common_factor_substitution.cpp)
  target_include_directories(test_common_factor_substitution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_common_factor_substitution PRIVATE lazyipdca_qcqp)
  target_compile_options(test_common_factor_substitution PRIVATE -UNDEBUG)
  add_test(NAME common_factor_substitution COMMAND test_common_factor_substitution)

  add_executable(test_positive_square_epigraph_substitution tests/test_positive_square_epigraph_substitution.cpp)
  target_include_directories(test_positive_square_epigraph_substitution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_positive_square_epigraph_substitution PRIVATE lazyipdca_qcqp)
  target_compile_options(test_positive_square_epigraph_substitution PRIVATE -UNDEBUG)
  add_test(NAME positive_square_epigraph_substitution COMMAND test_positive_square_epigraph_substitution)
endif()

if(LAZYIPDCA_BUILD_TESTS)
  add_executable(test_native_closed_bound_restoration tests/test_native_closed_bound_restoration.cpp)
  target_link_libraries(test_native_closed_bound_restoration PRIVATE lazyipdca_qcqp)
  target_compile_options(test_native_closed_bound_restoration PRIVATE -UNDEBUG)
  add_test(NAME native_closed_bound_restoration COMMAND test_native_closed_bound_restoration)

  add_executable(test_shared_restoration_budget tests/test_shared_restoration_budget.cpp)
  target_include_directories(test_shared_restoration_budget PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_options(test_shared_restoration_budget PRIVATE -UNDEBUG)
  add_test(NAME shared_restoration_budget COMMAND test_shared_restoration_budget)

  add_executable(test_shared_restoration_route tests/test_shared_restoration_route.cpp)
  target_link_libraries(test_shared_restoration_route PRIVATE lazyipdca_qcqp)
  target_compile_options(test_shared_restoration_route PRIVATE -UNDEBUG)
  add_test(NAME shared_restoration_route COMMAND test_shared_restoration_route)

  add_executable(test_historical_stagnation_handoff tests/test_historical_stagnation_handoff.cpp)
  target_link_libraries(test_historical_stagnation_handoff PRIVATE lazyipdca_qcqp)
  target_compile_options(test_historical_stagnation_handoff PRIVATE -UNDEBUG)
  add_test(NAME historical_stagnation_handoff COMMAND test_historical_stagnation_handoff)

  add_executable(test_kkt_expanded_precision_recovery tests/test_kkt_expanded_precision_recovery.cpp)
  target_link_libraries(test_kkt_expanded_precision_recovery PRIVATE lazyipdca_qcqp)
  target_compile_options(test_kkt_expanded_precision_recovery PRIVATE -UNDEBUG)
  add_test(NAME kkt_expanded_precision_recovery COMMAND test_kkt_expanded_precision_recovery)

  add_executable(test_hybrid_affine_numeric_recovery tests/test_hybrid_affine_numeric_recovery.cpp)
  target_link_libraries(test_hybrid_affine_numeric_recovery PRIVATE lazyipdca_qcqp)
  target_compile_options(test_hybrid_affine_numeric_recovery PRIVATE -UNDEBUG)
  add_test(NAME hybrid_affine_numeric_recovery COMMAND test_hybrid_affine_numeric_recovery)

  add_executable(test_gr_kkt_soc tests/test_gr_kkt_soc.cpp)
  target_link_libraries(test_gr_kkt_soc PRIVATE lazyipdca_qcqp)
  target_compile_options(test_gr_kkt_soc PRIVATE -UNDEBUG)
  add_test(NAME gr_kkt_soc COMMAND test_gr_kkt_soc)

  if(Python3_Interpreter_FOUND)
    add_test(NAME shared_restoration_soc_cli
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_shared_restoration_soc_cli.py
              --binary $<TARGET_FILE:lazyipdca_qcqp_cli> --native)
  endif()
endif()

# One aggregate target permits a single bounded parallel build of the protected
# native test executables; registered CTest identities and options stay intact.
if(LAZYIPDCA_BUILD_TESTS)
  add_custom_target(r3_protected_native_tests DEPENDS
    test_affine_face_rewrite
    test_spectral_majorant
    test_pglib_filter_recenter
    test_rank_presolve
    test_rank_audit
    test_rank_structural_retry
    test_rank_retry_statistics
    test_restoration_shared_budget
    test_shared_restoration_budget
    test_shared_restoration_route
    test_historical_stagnation_handoff
    test_kkt_expanded_precision_recovery
    test_hybrid_soc
    test_hybrid_affine_numeric_recovery
    test_gr_kkt_soc
    test_standard_auto_policy
  )
endif()
