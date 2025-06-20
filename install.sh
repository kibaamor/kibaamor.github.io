#!/bin/sh

set -euf

TMPDIR_BACKUP="${TMPDIR:-}"

mkdir -p "$HOME"/.install_dotfiles
export TMPDIR="$HOME"/.install_dotfiles

cleanup() {
    [ -d "$TMPDIR" ] && rm -fr "$TMPDIR"

    export TMPDIR="$TMPDIR_BACKUP"
    [ -z "${TMPDIR}" ] && unset TMPDIR
}

trap cleanup EXIT INT

# shellcheck source=/dev/null
[ -s "$HOME"/.customrc.pre.sh ] && \. "$HOME"/.customrc.pre.sh

sh -c "$(curl -fsLS get.chezmoi.io)" -- -b "$HOME"/.local/bin init --apply kibaamor

trap '' EXIT INT
cleanup

exec /usr/bin/zsh -l
