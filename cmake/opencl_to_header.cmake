# opencl_to_header.cmake - Script executed at build time
# Do not call by itself
file(READ "${INPUT_FILE}" FILE_CONTENTS)

set(RAW_DELIM "myDelimiter")

set(HEADER_TEXT
"#pragma once

namespace ${NAMESPACE} {

inline constexpr const char ${VAR_NAME}[] = R\"${RAW_DELIM}(
${FILE_CONTENTS}
)${RAW_DELIM}\";

} // namespace ${NAMESPACE}
")

file(WRITE "${OUTPUT_FILE}" "${HEADER_TEXT}")
