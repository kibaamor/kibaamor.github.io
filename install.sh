#!/bin/sh

set -euf

mkdir -p "$HOME"/.install_dotfiles
TMPDIR="$HOME"/.install_dotfiles

cleanup() {
  [ -d "$TMPDIR" ] && rm -fr "$TMPDIR"
}

trap cleanup EXIT INT

# shellcheck source=/dev/null
[ -s "$HOME/.customrc.pre.sh" ] && \. "$HOME/.customrc.pre.sh"

sh -c "$(curl -fsLS get.chezmoi.io)" -- -b ~/.local/bin init --apply kibaamor && exec zsh -l
