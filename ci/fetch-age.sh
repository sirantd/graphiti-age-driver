#!/usr/bin/env bash
# Lay out the AGE files of the pinned PGDG package as production mounts them:
#   <dest>/lib/age.so
#   <dest>/share/extension/age.control and age--*.sql
# Usage: ci/fetch-age.sh <dest>
# Environment: AGE_ARCH=arm64 takes the arm64 package, for a laptop only.
set -euo pipefail

dest=${1:?usage: fetch-age.sh <dest>}
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pins.env
source "$here/pins.env"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

deb_url=$AGE_DEB_URL
deb_sha256=$AGE_DEB_SHA256
if [ "${AGE_ARCH:-amd64}" = arm64 ]; then
  deb_url=$AGE_DEB_URL_ARM64
  deb_sha256=$AGE_DEB_SHA256_ARM64
fi
curl -fsSL -o "$work/age.deb" "$deb_url"
echo "$deb_sha256  $work/age.deb" | sha256sum -c -

# ar and tar, not dpkg-deb, so the script also runs on a Mac.
(cd "$work" && ar x age.deb && mkdir root && tar -xJf data.tar.xz -C root)

mkdir -p "$dest/lib" "$dest/share/extension"
cp "$work/root/usr/lib/postgresql/18/lib/age.so" "$dest/lib/"
cp "$work/root/usr/share/postgresql/18/extension/"age* "$dest/share/extension/"
grep -q "^default_version = '1.8.0'$" "$dest/share/extension/age.control"
# The server runs as uid 999 and must read every file.
chmod -R a+rX "$dest"
find "$dest" -type f | sort
