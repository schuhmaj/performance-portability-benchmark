#!/bin/bash
# =========================================================================
# intel-smi.sh — nvidia-smi-style status table for Intel GPUs
# =========================================================================
# Renders Intel GPU information (name, PCI address, utilization, temperature,
# power draw and memory usage) in a compact table reminiscent of `nvidia-smi`,
# using Intel's `xpu-smi` data-center management tool.
#
# Hardware this was written for: Intel Data Center GPU Max (PVC), e.g. the
# Sapphire Rapids + 4x PVC node ("sap"). Works for any device xpu-smi enumerates.
# =========================================================================
# REQUIREMENTS
# =========================================================================
#   - xpu-smi   (https://github.com/intel/xpumanager) — primary data source
#   - jq        — OPTIONAL. Used to parse xpu-smi's JSON when present; without
#                 it a portable awk/grep parser is used, so only xpu-smi and
#                 coreutils (awk, grep, tr) are strictly required.
# Fallbacks when those are unavailable:
#   - --raw mode passes xpu-smi output through verbatim (needs only xpu-smi)
#   - if xpu-smi itself is missing, the script falls back to `sycl-ls` /
#     `clinfo` for a bare device listing.
# =========================================================================
# USAGE
# =========================================================================
#   ./intel-smi.sh            # one-shot table
#   ./intel-smi.sh -l 2       # refresh every 2 seconds (like `nvidia-smi -l`)
#   ./intel-smi.sh -r         # raw xpu-smi discovery + stats passthrough
#   ./intel-smi.sh -h         # help
#
# NOTE ON UNITS: xpu-smi reports some values with version-dependent units. This
# script assumes `XPUM_STATS_MEMORY_USED` is in bytes and the discovery field
# `memory_physical_size_byte` is in MiB (the common case on recent xpu-smi). If
# your version differs, cross-check with `./intel-smi.sh -r` and adjust the
# to_mib_* helpers below.
# =========================================================================

set -uo pipefail

XPU_SMI="${XPU_SMI:-xpu-smi}"
LOOP_INTERVAL=0
RAW=0

print_help() {
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while getopts "l:rh" opt; do
  case "$opt" in
    l) LOOP_INTERVAL="$OPTARG" ;;
    r) RAW=1 ;;
    h) print_help ;;
    *) echo "Usage: $0 [-l <seconds>] [-r] [-h]" >&2; exit 1 ;;
  esac
done

have() { command -v "$1" >/dev/null 2>&1; }

# Repeat a character N times, e.g. `rep - 77`.
rep() { printf '%*s' "$2" '' | tr ' ' "$1"; }

# JSON backend: prefer jq when present, otherwise a portable awk/grep parser so
# the table still renders with just xpu-smi + coreutils (no jq required).
if have jq; then JSON_BACKEND=jq; else JSON_BACKEND=portable; fi

# Collapse JSON to a single line so the portable parser handles both compact and
# pretty-printed xpu-smi output.
json_norm() { tr -d '\n\r' | tr -s ' '; }

# ------------------------------------------------------------------------- #
# Fallback path: no xpu-smi available                                       #
# ------------------------------------------------------------------------- #
fallback_listing() {
  echo "xpu-smi not found in PATH." >&2
  if have sycl-ls; then
    echo "Falling back to 'sycl-ls':"
    sycl-ls
  elif have clinfo; then
    echo "Falling back to 'clinfo --list':"
    clinfo --list
  else
    echo "No Intel GPU tooling found (tried xpu-smi, sycl-ls, clinfo)." >&2
    return 1
  fi
}

# ------------------------------------------------------------------------- #
# Raw passthrough                                                           #
# ------------------------------------------------------------------------- #
raw_dump() {
  echo "===== xpu-smi discovery ====="
  "$XPU_SMI" discovery
  echo
  # Per-device stats; -1 means "all devices" on most xpu-smi versions.
  echo "===== xpu-smi stats (all devices) ====="
  "$XPU_SMI" stats -d -1 2>/dev/null || {
    # Older versions need an explicit device id; iterate over discovered ids.
    for id in $(device_ids); do
      echo "--- device ${id} ---"
      "$XPU_SMI" stats -d "$id"
    done
  }
}

# Device ids from `discovery -j`, one per line.
device_ids() {
  local js
  js="$("$XPU_SMI" discovery -j 2>/dev/null)"
  if [ "$JSON_BACKEND" = jq ]; then
    printf '%s' "$js" | jq -r '(.device_list // [])[] | .device_id' 2>/dev/null
  else
    printf '%s' "$js" | json_norm \
      | grep -oE '"device_id"[[:space:]]*:[[:space:]]*[0-9]+' \
      | grep -oE '[0-9]+$'
  fi
}

# json_scalar KEY  — extract a top-level scalar (string or number) from a flat
# JSON object on stdin; prints "N/A" if absent. Used for discovery fields.
json_scalar() {
  local key="$1"
  if [ "$JSON_BACKEND" = jq ]; then
    jq -r --arg k "$key" '.[$k] // "N/A"' 2>/dev/null
  else
    json_norm | awk -v k="$key" '{
      pat = "\"" k "\"[[:space:]]*:[[:space:]]*"
      if (match($0, pat "\"[^\"]*\"")) {           # quoted string value
        s = substr($0, RSTART, RLENGTH); sub(pat "\"", "", s); sub(/"$/, "", s); print s; exit
      }
      if (match($0, pat "-?[0-9.]+")) {            # numeric value
        s = substr($0, RSTART, RLENGTH); sub(pat, "", s); print s; exit
      }
      print "N/A"
    }'
  fi
}

# Convert a possibly-byte value to MiB (integer). Heuristic guard for versions
# that already report MiB: values below 1 MiB are assumed to be MiB already.
to_mib_bytes() {
  awk -v v="$1" 'BEGIN{
    if (v=="N/A" || v=="" ) { print "N/A"; exit }
    if (v+0 >= 1048576) printf "%d", v/1048576; else printf "%d", v
  }'
}
# Discovery memory field: assumed already in MiB.
to_mib_plain() {
  awk -v v="$1" 'BEGIN{ if (v=="N/A"||v==""){print "N/A";exit} printf "%d", v+0 }'
}

# Pull one stats metric (value/scale) from a stats JSON blob on stdin.
get_metric() {
  local type="$1"
  if [ "$JSON_BACKEND" = jq ]; then
    jq -r --arg t "$type" '
      [ (.device_level // [])[] | select(.metrics_type==$t) ] | .[0] |
      if . == null then "N/A"
      else ( (.value // 0) / ( (.scale // 1) | if . == 0 then 1 else . end) )
      end' 2>/dev/null || echo "N/A"
  else
    # Split the array into per-object chunks at "}", then read value/scale from
    # the chunk whose metrics_type matches exactly.
    json_norm | awk -v t="$type" '{
      n = split($0, parts, "}")
      for (i = 1; i <= n; i++) {
        if (parts[i] ~ ("\"metrics_type\"[[:space:]]*:[[:space:]]*\"" t "\"")) {
          val = "N/A"; scl = 1
          if (match(parts[i], /"value"[[:space:]]*:[[:space:]]*-?[0-9.]+/)) {
            s = substr(parts[i], RSTART, RLENGTH); sub(/.*:[[:space:]]*/, "", s); val = s
          }
          if (match(parts[i], /"scale"[[:space:]]*:[[:space:]]*-?[0-9.]+/)) {
            s = substr(parts[i], RSTART, RLENGTH); sub(/.*:[[:space:]]*/, "", s); scl = s
          }
          if (scl + 0 == 0) scl = 1
          # %.15g keeps full precision for large byte counts (a plain %g would
          # truncate to ~6 sig-figs and skew the MiB conversion).
          if (val == "N/A") print "N/A"; else printf "%.15g", val / scl
          found = 1; exit
        }
      }
      if (!found) print "N/A"
    }'
  fi
}

round() { awk -v v="$1" 'BEGIN{ if(v=="N/A"||v==""){print "N/A";exit} printf "%d", v+0 }'; }

# ------------------------------------------------------------------------- #
# Pretty table (uses jq when available, otherwise the portable parser)       #
# ------------------------------------------------------------------------- #
pretty_table() {
  local ts driver
  ts="$(date '+%a %b %e %H:%M:%S %Y')"

  # Driver version: taken from the first device's detailed discovery.
  local first_id
  first_id="$(device_ids | head -n1)"
  driver="N/A"
  if [ -n "${first_id:-}" ]; then
    driver="$("$XPU_SMI" discovery -d "$first_id" -j 2>/dev/null | json_scalar driver_version)"
  fi

  # Column widths (content incl. one leading pad space). Bars/pluses align
  # because 31 + 1 + 21 + 1 + 23 == 77 == inner width.
  local W1=31 W2=21 W3=23
  local d1 d2 d3
  d1="$(rep - $W1)"; d2="$(rep - $W2)"; d3="$(rep - $W3)"
  local e1 e2 e3
  e1="$(rep = $W1)"; e2="$(rep = $W2)"; e3="$(rep = $W3)"

  printf '+%s+\n'        "$(rep - 77)"
  printf '|%-77s|\n'     " $ts   Driver Version: $driver"
  printf '|%s+%s+%s|\n'  "$d1" "$d2" "$d3"
  printf '|%-*s|%-*s|%-*s|\n' "$W1" " GPU  Name" "$W2" " PCI Bus-Id" "$W3" " Temp  Pwr    Util"
  printf '|%-*s|%-*s|%-*s|\n' "$W1" " "          "$W2" " "           "$W3" " Memory-Usage (MiB)"
  printf '|%s+%s+%s|\n'  "$e1" "$e2" "$e3"

  local id
  for id in $(device_ids); do
    # Static info from discovery.
    local name pci memtotal disc
    disc="$("$XPU_SMI" discovery -d "$id" -j 2>/dev/null)"
    name="$(printf '%s' "$disc"     | json_scalar device_name)"
    pci="$(printf '%s' "$disc"      | json_scalar pci_bdf_address)"
    memtotal="$(printf '%s' "$disc" | json_scalar memory_physical_size_byte)"
    memtotal="$(to_mib_plain "$memtotal")"
    name="${name#Intel(R) }"; name="${name//Data Center GPU/DC GPU}"

    # Dynamic info from stats.
    local stats util power temp memused
    stats="$("$XPU_SMI" stats -d "$id" -j 2>/dev/null)"
    util="$(round  "$(printf '%s' "$stats" | get_metric XPUM_STATS_GPU_UTILIZATION)")"
    power="$(round "$(printf '%s' "$stats" | get_metric XPUM_STATS_POWER)")"
    temp="$(round  "$(printf '%s' "$stats" | get_metric XPUM_STATS_GPU_CORE_TEMPERATURE)")"
    memused="$(to_mib_bytes "$(printf '%s' "$stats" | get_metric XPUM_STATS_MEMORY_USED)")"

    printf '|%-*s|%-*s|%-*s|\n' \
      "$W1" "$(printf ' %-3s %-25.25s' "$id" "$name")" \
      "$W2" "$(printf ' %-19s' "$pci")" \
      "$W3" "$(printf ' %3sC %4sW %5s%%' "$temp" "$power" "$util")"
    printf '|%-*s|%-*s|%-*s|\n' \
      "$W1" " " "$W2" " " \
      "$W3" "$(printf ' %9s / %-8s' "$memused" "$memtotal")"
    printf '+%s+%s+%s+\n' "$d1" "$d2" "$d3"
  done
}

render() {
  if ! have "$XPU_SMI"; then
    fallback_listing
    return $?
  fi
  if [ "$RAW" -eq 1 ]; then
    raw_dump
    return $?
  fi
  pretty_table
}

# ------------------------------------------------------------------------- #
# Main                                                                      #
# ------------------------------------------------------------------------- #
if [ "$LOOP_INTERVAL" -gt 0 ] 2>/dev/null; then
  while true; do
    clear
    render
    sleep "$LOOP_INTERVAL"
  done
else
  render
fi
