# Caller must admit idle hardware/resources and hold the owner launch lease before GPU CTests.
# A fresh path on every invocation preserves earlier passing AND failing inputs/outputs.
if(NOT DEFINED PROBE OR NOT DEFINED DEVICE OR NOT DEFINED EVIDENCE_ROOT)
  message(FATAL_ERROR "Explicit probe, requested card and evidence root required")
endif()
string(TIMESTAMP stamp "%Y%m%dT%H%M%S" UTC)
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef nonce)
set(path "${EVIDENCE_ROOT}/mmq-device${DEVICE}-${stamp}-${nonce}")
execute_process(COMMAND "${PROBE}" --device "${DEVICE}" --dump-dir "${path}"
                RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "gfx906 MMQ numerical gate failed (${result}); retained fixtures: ${path}")
endif()
