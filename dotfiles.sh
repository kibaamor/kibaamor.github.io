#!/bin/sh
set -efu

TMPDIR_BACKUP="${TMPDIR:-}"
INSTALL_TMP="$HOME/.install_dotfiles"
CUSTOM_RC="$HOME/.customrc.pre.sh"

mkdir -p -- "$INSTALL_TMP"
export TMPDIR="$INSTALL_TMP"

cleanup() {
    if [ -n "${INSTALL_TMP:-}" ] && [ -d "$INSTALL_TMP" ] && [ "$INSTALL_TMP" != "/" ]; then
        rm -rf -- "$INSTALL_TMP"
    fi

    if [ -n "${TMPDIR_BACKUP:-}" ]; then
        export TMPDIR="$TMPDIR_BACKUP"
    else
        unset TMPDIR
    fi
}

trap cleanup EXIT INT TERM

if [ ! -s "$CUSTOM_RC" ]; then
    [ -n "${DOTFILES_INSTALL_EXTRA_BINS:-}" ] && echo "export DOTFILES_INSTALL_EXTRA_BINS='$DOTFILES_INSTALL_EXTRA_BINS'" >>"$CUSTOM_RC"
    [ -n "${DOTFILES_INSTALL_ARKADE_BINS:-}" ] && echo "export DOTFILES_INSTALL_ARKADE_BINS='$DOTFILES_INSTALL_ARKADE_BINS'" >>"$CUSTOM_RC"
    [ -n "${GIT_USERNAME:-}" ] && echo "export GIT_USERNAME='$GIT_USERNAME'" >>"$CUSTOM_RC"
    [ -n "${GIT_EMAIL:-}" ] && echo "export GIT_EMAIL='$GIT_EMAIL'" >>"$CUSTOM_RC"
    [ -n "${GITHUB_USERNAME:-}" ] && echo "export GITHUB_USERNAME='$GITHUB_USERNAME'" >>"$CUSTOM_RC"
    [ -n "${GITHUB_EMAIL:-}" ] && echo "export GITHUB_EMAIL='$GITHUB_EMAIL'" >>"$CUSTOM_RC"
    [ -n "${GITLAB_USERNAME:-}" ] && echo "export GITLAB_USERNAME='$GITLAB_USERNAME'" >>"$CUSTOM_RC"
    [ -n "${GITLAB_EMAIL:-}" ] && echo "export GITLAB_EMAIL='$GITLAB_EMAIL'" >>"$CUSTOM_RC"
    [ -n "${default_proxy:-}" ] && echo "export default_proxy='$default_proxy'" >>"$CUSTOM_RC"
    [ -n "${default_no_proxy:-}" ] && echo "export default_no_proxy='$default_no_proxy'" >>"$CUSTOM_RC"
fi

# shellcheck source=/dev/null
[ -s "$CUSTOM_RC" ] && \. "$CUSTOM_RC"

curl -fsLS get.chezmoi.io | sh -s -- -b "$HOME/.local/bin" init --apply kibaamor "$@"

trap '' EXIT INT TERM
cleanup

exec /usr/bin/zsh -l
