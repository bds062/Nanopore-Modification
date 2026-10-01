# Sourced by every step: load the user's config and fail loudly on unset vars.
set -euo pipefail
: "${CFG:?CFG (path to the genome-map config) must be exported}"
# shellcheck disable=SC1090
source "$CFG"
PRE=$OUT/preprocess
sdir () { [[ "$1" == "+" ]] && echo plus || echo minus; }
comp () { case "$1" in A) echo T;; T) echo A;; C) echo G;; G) echo C;; esac; }
