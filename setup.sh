#!/bin/bash
# Prepare this checkout for building the app with your own Apple ID.
#
#   ./setup.sh
#
# Checks Xcode and its Metal toolchain, finds your Apple team in Xcode's accounts, asks for one word to make a
# bundle id that is yours (com.<word>.phonestore), and writes ios/PhoneStoreApp/Local.xcconfig (git-ignored).
# Nothing is sent anywhere and no Apple account is changed. Run it again to change the values.
#
# Non-interactive: TEAM=ABCDE12345 BUNDLE_ID=com.yourname.phonestore ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"
OUT=ios/PhoneStoreApp/Local.xcconfig

say() { printf '%s\n' "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

# 1. Xcode
command -v xcodebuild >/dev/null 2>&1 || fail "Xcode is not installed. Install Xcode from the App Store, open it once, then run this again."
XV=$(xcodebuild -version 2>/dev/null | awk 'NR==1{print $2}') || true
[ -n "${XV:-}" ] || fail "xcodebuild does not run. Open Xcode once (it installs its components), or run: sudo xcode-select -s /Applications/Xcode.app"
XMAJOR=${XV%%.*}
if [ "$XMAJOR" -lt 26 ]; then fail "Xcode $XV found; this project needs Xcode 26 or later."; fi
say "Xcode $XV"

# 2. Metal toolchain (Xcode 26 downloads it separately; MLX's GPU kernels need it)
if xcrun -sdk iphoneos metal --version >/dev/null 2>&1; then
  say "Metal toolchain: installed"
else
  say "Metal toolchain: missing. It is needed to compile the model's GPU kernels (about 700 MB, once)."
  if [ -t 0 ]; then
    read -r -p "Download it now with 'xcodebuild -downloadComponent MetalToolchain'? [Y/n] " yn
    case "${yn:-y}" in
      [Nn]*) say "Skipped. Run it yourself before building: xcodebuild -downloadComponent MetalToolchain" ;;
      *) xcodebuild -downloadComponent MetalToolchain ;;
    esac
  else
    say "Run before building: xcodebuild -downloadComponent MetalToolchain"
  fi
fi

# 3. Team: from Xcode > Settings > Accounts
TEAM=${TEAM:-}
if [ -z "$TEAM" ]; then
  TEAMS=$(defaults export com.apple.dt.Xcode - 2>/dev/null | /usr/bin/python3 -c '
import plistlib, sys
try:
    d = plistlib.loads(sys.stdin.buffer.read())
except Exception:
    sys.exit(0)
seen = set()
for teams in (d.get("IDEProvisioningTeamByIdentifier") or {}).values():
    for t in teams:
        if t.get("teamID") and t["teamID"] not in seen:
            seen.add(t["teamID"])
            print(t["teamID"] + "\t" + t.get("teamName", "") + "\t" + t.get("teamType", ""))
' || true)
  N=$(printf '%s' "$TEAMS" | grep -c . || true)
  if [ "$N" = 1 ]; then
    TEAM=$(printf '%s' "$TEAMS" | cut -f1)
    say "Team: $(printf '%s' "$TEAMS" | cut -f2) ($TEAM)"
  elif [ "$N" -gt 1 ]; then
    say "Teams in Xcode:"
    printf '%s\n' "$TEAMS" | awk -F'\t' '{printf "  %d) %s  %s (%s)\n", NR, $1, $2, $3}'
    read -r -p "Which one? [1-$N] " k
    TEAM=$(printf '%s\n' "$TEAMS" | sed -n "${k}p" | cut -f1)
    [ -n "$TEAM" ] || fail "no team chosen"
  else
    say "Team: none found. Add your Apple ID in Xcode > Settings > Accounts and run this again,"
    say "      or pick the team later in Xcode (PhoneStoreApp target > Signing & Capabilities)."
  fi
fi

# 4. Bundle id: a word of yours, so the id is not already taken by someone else's team
BUNDLE_ID=${BUNDLE_ID:-}
if [ -z "$BUNDLE_ID" ]; then
  [ -t 0 ] || fail "set BUNDLE_ID=com.yourname.phonestore (no terminal to ask)"
  while :; do
    read -r -p "One word for your bundle id, e.g. your name (gives com.<word>.phonestore): " W
    W=$(printf '%s' "$W" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9-')
    [ -n "$W" ] && break
  done
  BUNDLE_ID="com.$W.phonestore"
fi
printf '%s' "$BUNDLE_ID" | grep -Eq '^[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$' || fail "not a valid bundle id: $BUNDLE_ID"

cat > "$OUT" <<EOF
// Written by setup.sh. Git-ignored: your team id stays on this Mac.
DEVELOPMENT_TEAM = $TEAM
BUNDLE_ID = $BUNDLE_ID
EOF
say "Wrote $OUT (bundle id $BUNDLE_ID)"
say ""
say "Next: open ios/PhoneStoreApp/PhoneStoreApp.xcodeproj, plug in your iPhone, choose it at the top of the window"
say "and press Run (the triangle). The first build takes several minutes."
