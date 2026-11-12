if(NOT DEFINED CLI OR NOT DEFINED PROBLEM OR NOT DEFINED MODE)
  message(FATAL_ERROR "CLI, PROBLEM, and MODE are required")
endif()

if(MODE STREQUAL "outer")
  set(extra --algorithm auto)
  set(expected "requires --algorithm lazy-dca")
elseif(MODE STREQUAL "inner")
  set(extra --inner-backend filter-sqp)
  set(expected "requires --inner-backend lazyqp")
elseif(MODE STREQUAL "mehrotra")
  set(extra --no-qp-mehrotra)
  set(expected "requires --qp-mehrotra")
elseif(MODE STREQUAL "retraction")
  set(extra --dca-single-loop-retraction-fallback-only)
  set(expected "forbids --dca-single-loop-retraction-fallback-only")
elseif(MODE STREQUAL "numeric-reuse")
  set(extra --lazyqp-numeric-kkt-reuse)
  set(expected "forbids --lazyqp-numeric-kkt-reuse")
else()
  message(FATAL_ERROR "unknown rejection mode: ${MODE}")
endif()

execute_process(
  COMMAND "${CLI}" --problem "${PROBLEM}"
          --dca-single-loop-primal-dual ${extra} --quiet
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)

if(rc EQUAL 0)
  message(FATAL_ERROR "invalid single-loop CLI configuration succeeded")
endif()
set(output "${stdout}${stderr}")
string(FIND "${output}" "${expected}" found)
if(found EQUAL -1)
  message(FATAL_ERROR
    "CLI failed for the wrong reason; expected '${expected}', got: ${output}")
endif()
