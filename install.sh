#!/bin/sh

set -ef

TMPDIR_BACKUP="${TMPDIR:-}"

mkdir -p "$HOME"/.install_dotfiles
export TMPDIR="$HOME"/.install_dotfiles

cleanup() {
    [ -d "$TMPDIR" ] && rm -fr "$TMPDIR"

    export TMPDIR="$TMPDIR_BACKUP"
    [ -z "${TMPDIR}" ] && unset TMPDIR
}

trap cleanup EXIT INT

if [ ! -s "$HOME"/.customrc.pre.sh ]; then
    if [ -n "$DOTFILES_INSTALL_EXTRA_BINS" ]; then
        echo "export DOTFILES_INSTALL_EXTRA_BINS='$DOTFILES_INSTALL_EXTRA_BINS'" >>"$HOME"/.customrc.pre.sh
    fi
    if [ -n "$DOTFILES_INSTALL_ARKADE_BINS" ]; then
        echo "export DOTFILES_INSTALL_ARKADE_BINS='$DOTFILES_INSTALL_ARKADE_BINS'" >>"$HOME"/.customrc.pre.sh
    fi

    if [ -n "$GIT_USERNAME" ]; then
        echo "export GIT_USERNAME='$GIT_USERNAME'" >>"$HOME"/.customrc.pre.sh
    fi
    if [ -n "$GIT_EMAIL" ]; then
        echo "export GIT_EMAIL='$GIT_EMAIL'" >>"$HOME"/.customrc.pre.sh
    fi
    if [ -n "$GITHUB_USERNAME" ]; then
        echo "export GITHUB_USERNAME='$GITHUB_USERNAME'" >>"$HOME"/.customrc.pre.sh
    fi
    if [ -n "$GITHUB_EMAIL" ]; then
        echo "export GITHUB_EMAIL='$GITHUB_EMAIL'" >>"$HOME"/.customrc.pre.sh
    fi
    if [ -n "$GITLAB_USERNAME" ]; then
        echo "export GITLAB_USERNAME='$GITLAB_USERNAME'" >>"$HOME"/.customrc.pre.sh
    fi
    if [ -n "$GITLAB_EMAIL" ]; then
        echo "export GITLAB_EMAIL='$GITLAB_EMAIL'" >>"$HOME"/.customrc.pre.sh
    fi

    if [ -n "$default_proxy" ]; then
        echo "export default_proxy='$default_proxy'" >>"$HOME"/.customrc.pre.sh
    fi
    if [ -n "$default_no_proxy" ]; then
        echo "export default_no_proxy='$default_no_proxy'" >>"$HOME"/.customrc.pre.sh
    fi
fi

# shellcheck source=/dev/null
[ -s "$HOME"/.customrc.pre.sh ] && \. "$HOME"/.customrc.pre.sh

sh -c "$(curl -fsLS get.chezmoi.io)" -- -b "$HOME"/.local/bin init --apply kibaamor "$@"

trap '' EXIT INT
cleanup

exec /usr/bin/zsh -l
