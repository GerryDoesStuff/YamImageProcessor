#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <locale> [<locale>...]" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$ROOT_DIR/yam_processor"
TS_DIR="$ROOT_DIR/translations"
I18N_DIR="$ROOT_DIR/translations"

mkdir -p "$TS_DIR" "$I18N_DIR"

readarray -t SOURCES < <(find "$SOURCE_DIR" -type f -name '*.py' ! -path '*/__pycache__/*' ! -name 'resources_rc.py' -print)

if [ "${#SOURCES[@]}" -eq 0 ]; then
    echo "No source files found for translation extraction." >&2
    exit 1
fi

PARSER_COMMAND=${PYLUPDATE5:-pylupdate5}
LRELEASE_COMMAND=${LRELEASE:-lrelease}

for locale in "$@"; do
    normalized_locale=${locale//-/_}
    ts_file="$TS_DIR/yam_processor_${normalized_locale}.ts"
    qm_file="$I18N_DIR/yam_processor_${normalized_locale}.qm"

    "$PARSER_COMMAND" "${SOURCES[@]}" -ts "$ts_file"
    "$LRELEASE_COMMAND" "$ts_file" -qm "$qm_file"

    printf 'Generated %s\n' "$qm_file"
    printf 'Updated source catalogue %s\n' "$ts_file"

done
