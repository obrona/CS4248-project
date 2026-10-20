#!/usr/bin/env bash

set -Eeuo pipefail

USERNAME="theodore"
JUMP_HOST="$USERNAME@stujump.comp.nus.edu.sg"
REMOTE_HOST="$USERNAME@xlogin.comp.nus.edu.sg"

# OpenSSH calls this script as its password-prompt helper. Other prompts (for
# example, first-time host-key confirmation or MFA) are relayed to the terminal.
if [[ ${TRANSFER_ASKPASS_MODE:-0} == 1 ]]; then
	prompt=${1:-"SSH prompt"}
	if [[ ${prompt,,} == *password* ]]; then
		printf '%s\n' "${TRANSFER_PASSWORD:?}"
	else
		printf '%s ' "$prompt" > /dev/tty
		IFS= read -r response < /dev/tty
		printf '%s\n' "$response"
	fi
	exit 0
fi

for command_name in ssh rsync; do
	if ! command -v "$command_name" >/dev/null 2>&1; then
		printf 'Error: required command "%s" is not installed.\n' "$command_name" >&2
		exit 1
	fi
done

printf 'Transfer direction:\n'
printf '  1) Push: local -> remote\n'
printf '  2) Pull: remote -> local\n'
read -r -p 'Choose 1 or 2: ' direction

case "$direction" in
	1 | push | PUSH)
		direction=push
		source_side=Local
		destination_side=Remote
		;;
	2 | pull | PULL)
		direction=pull
		source_side=Remote
		destination_side=Local
		;;
	*)
		printf 'Error: choose 1 (push) or 2 (pull).\n' >&2
		exit 1
		;;
esac

# Read one source path per line so paths containing spaces work. An empty
# line ends the list.
sources=()
printf '%s source files or folders (one per line, empty line to finish):\n' \
	"$source_side"
while IFS= read -r -p '  > ' source_path; do
	[[ -z $source_path ]] && break
	# A trailing slash means "copy the contents" to rsync. Remove it so this
	# script consistently copies the source folder itself. This has no effect
	# on ordinary file paths.
	if [[ $source_path != / ]]; then
		source_path=${source_path%/}
	fi
	sources+=("$source_path")
done
read -r -p "$destination_side destination folder: " destination_folder

if [[ ${#sources[@]} -eq 0 || -z $destination_folder ]]; then
	printf 'Error: source and destination paths cannot be empty.\n' >&2
	exit 1
fi

if [[ $direction == push ]]; then
	for source_path in "${sources[@]}"; do
		if [[ ! -f $source_path && ! -d $source_path ]]; then
			printf 'Error: local source is not a regular file or directory: %s\n' \
				"$source_path" >&2
			exit 1
		fi
	done
fi

if [[ $direction == pull ]]; then
	mkdir -p -- "$destination_folder"
fi

read -r -s -p "Password for $USERNAME: " TRANSFER_PASSWORD
printf '\n'
if [[ -z $TRANSFER_PASSWORD ]]; then
	printf 'Error: password cannot be empty.\n' >&2
	exit 1
fi

export TRANSFER_PASSWORD
export TRANSFER_ASKPASS_MODE=1
export SSH_ASKPASS_REQUIRE=force
export DISPLAY=${DISPLAY:-transfer-script}
SSH_ASKPASS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
export SSH_ASKPASS

cleanup() {
	unset TRANSFER_PASSWORD TRANSFER_ASKPASS_MODE SSH_ASKPASS_REQUIRE SSH_ASKPASS
}
trap cleanup EXIT

ssh_command="ssh -J $JUMP_HOST"

if [[ $direction == push ]]; then
	printf 'Pushing to %s:"%s":\n' "$REMOTE_HOST" "$destination_folder"
	printf '  "%s"\n' "${sources[@]}"
	rsync --archive --human-readable --info=progress2 --protect-args \
		--rsh="$ssh_command" -- \
		"${sources[@]}" "$REMOTE_HOST:$destination_folder/"
else
	printf 'Pulling from %s to "%s":\n' "$REMOTE_HOST" "$destination_folder"
	printf '  "%s"\n' "${sources[@]}"
	rsync --archive --human-readable --info=progress2 --protect-args \
		--rsh="$ssh_command" -- \
		"${sources[@]/#/$REMOTE_HOST:}" "$destination_folder/"
fi

printf 'Transfer complete.\n'