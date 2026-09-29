#!/usr/bin/env bash
# Fetch the incident-neutron HDF5 files mcslab needs from an official OpenMC
# data library, without keeping the rest of the archive.
#
#   scripts/fetch_data.sh [library] [dest]
#       library : endfb-viii.0 (default) | fendl-3.2
#       dest    : parent directory (default ~/nuclear_data)
#
# The archives are single .tar.xz streams with no random access, so the whole
# archive crosses the network (ENDF/B-VIII.0: 3.38 GB), but only the listed
# nuclide files (plus cross_sections.xml) are written to disk. bsdtar's -q
# stops reading once every pattern has matched.
#
# Afterwards, point mcslab at the library root, e.g.
#   export MCSLAB_DATA=~/nuclear_data/endfb-viii.0-hdf5
#
# A sha256 of every extracted file is written to <root>/SHA256SUMS. If the repo
# holds checksums for this library (scripts/checksums/<library>.sha256), the
# extracted files are verified against them.
#
# Data are never committed to the repository (*.h5 is gitignored).
set -euo pipefail

LIBRARY="${1:-endfb-viii.0}"
DEST="${2:-$HOME/nuclear_data}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Download links from https://openmc.org/data (checked 2026-09-29).
case "$LIBRARY" in
  endfb-viii.0)
    # endfb80.tar.xz, 3,383,607,420 bytes; top-level dir endfb-viii.0-hdf5/
    URL="https://anl.box.com/shared/static/uhbxlrx7hvxqw27psymfbhi7bx7s6u6a.xz"
    ;;
  fendl-3.2)
    # FENDL-3.2 (OpenMC "other libraries", 293.6 K only). Not yet fetched or
    # tested with mcslab; kept here for the planned cross-library comparison.
    URL="https://anl.box.com/shared/static/3cb7jetw7tmxaw6nvn77x6c578jnm2ey.xz"
    ;;
  *)
    echo "unknown library '$LIBRARY' (expected endfb-viii.0 or fendl-3.2)" >&2
    exit 2
    ;;
esac

# Phase 2a nuclides: H1 is the test nuclide; Fe, W for structure; Li, Be, F
# for FLiBe. Natural W includes W180 (0.12 atom%).
NUCLIDES=(H1 Fe54 Fe56 Fe57 Fe58 W180 W182 W183 W184 W186 Li6 Li7 Be9 F19)

PATTERNS=("*/cross_sections.xml")
for n in "${NUCLIDES[@]}"; do
  PATTERNS+=("*/neutron/${n}.h5")
done

mkdir -p "$DEST"
echo "Streaming $LIBRARY from $URL"
echo "  extracting ${#NUCLIDES[@]} nuclides + cross_sections.xml into $DEST"

# With -q, tar exits as soon as all patterns have matched, and curl then
# fails to write ("curl: (23)" or "(56) Failure writing output"). That is
# expected, so curl's status is not checked; tar's status, the file check and
# the checksums below are what count.
echo "  (a curl 'Failure writing output' message at the end is expected)"
set +o pipefail
curl -fsSL "$URL" | tar -xJqf - -C "$DEST" "${PATTERNS[@]}"
TAR_STATUS=${PIPESTATUS[1]}
set -o pipefail
if [ "$TAR_STATUS" -ne 0 ]; then
  echo "tar failed with status $TAR_STATUS" >&2
  exit 1
fi

# Locate the library root (the directory holding neutron/).
ROOT="$(dirname "$(find "$DEST" -maxdepth 3 -path "*/neutron/${NUCLIDES[0]}.h5" | head -1)")"
ROOT="$(dirname "$ROOT")"
missing=0
for n in "${NUCLIDES[@]}"; do
  if [ ! -f "$ROOT/neutron/$n.h5" ]; then
    echo "MISSING: $ROOT/neutron/$n.h5" >&2
    missing=1
  fi
done
[ "$missing" -eq 0 ] || exit 1

(cd "$ROOT" && shasum -a 256 cross_sections.xml neutron/*.h5 > SHA256SUMS)
echo "wrote $ROOT/SHA256SUMS"

EXPECTED="$HERE/checksums/$LIBRARY.sha256"
if [ -f "$EXPECTED" ]; then
  (cd "$ROOT" && shasum -a 256 -c "$EXPECTED")
else
  echo "no committed checksums at $EXPECTED; copy SHA256SUMS there to pin them"
fi

echo
echo "Done. export MCSLAB_DATA=$ROOT"
