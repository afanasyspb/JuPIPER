#!/usr/bin/env bash
set -eu -o pipefail

usage() {
    cat <<'HELP'
Usage: fetch_structures.sh [--verify] [--all | --bound | ID ...]
       fetch_structures.sh --fetch-new ID ...

By default, fetch missing structures and verify their published SHA-256.
Without IDs, use the core example set. --bound selects bound complexes;
--all also includes their unbound partners. --verify checks local files only.
--fetch-new accepts unlisted PDB IDs and prints their unverified digests.
The published digest file is read-only in every mode.
HELP
}

ids_from_cases() {   # $1 = "all" | "bound"
    [ -f "$CASES" ] || { echo "missing $CASES -- generate it with scripts/fetch_benchmark4.jl" >&2; return 1; }
    awk -F'\t' -v mode="$1" '
        /^#/ || NF < 12 { next }
        $1 == "case"    { next }
        {
            print $2
            if (mode == "all") {
                if ($5 != "-" && $5 != "") print $5
                if ($7 != "-" && $7 != "") print $7
            }
        }' "$CASES" | sort -u
}

recorded_digest() {
    awk -v id="$1" '$2 == id { print $1; exit }' "$DIGESTS"
}

cd "$(dirname "${BASH_SOURCE[0]}")/.."
DEST=data/structures
DIGESTS=data/structures.sha256
CASES=data/benchmark51.tsv
mode=fetch
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    --verify) mode=verify; shift ;;
    --fetch-new) mode=new; shift ;;
esac
if [[ $mode == new && ( $# -eq 0 || $1 == -* ) ]]; then
    usage >&2; exit 2
fi
case "${1:-}" in
    --all|--bound)
        [[ $# == 1 ]] || { usage >&2; exit 2; }
        selected=$(ids_from_cases "${1#--}")
        ids=()
        while IFS= read -r id; do ids+=("$id"); done <<< "$selected" ;;
    "") ids=(1GGR 1F3G 1HDN 1AVX 1QQU 1BA7 4XAU) ;;
    *) ids=("$@") ;;
esac

[[ -r $DIGESTS ]] || { echo "Missing $DIGESTS" >&2; exit 1; }
[[ $mode == verify ]] || mkdir -p "$DEST"
tmp=''
cleanup() { if [[ -n $tmp ]]; then rm -f -- "$tmp"; fi; }
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

status=0
for input in "${ids[@]}"; do
    if [[ ! $input =~ ^[0-9][a-zA-Z0-9]{3}$ ]]; then
        echo "Invalid PDB ID: $input" >&2; status=1; continue
    fi
    id=$(printf '%s' "$input" | tr '[:lower:]' '[:upper:]')
    want=$(recorded_digest "$id")
    if [[ -z $want && $mode != new ]]; then
        echo "No published digest for $id. Use --fetch-new for an unverified structure." >&2
        status=1; continue
    fi
    out="$DEST/$id.pdb"
    source=$out
    if [[ ! -f $out ]]; then
        if [[ $mode == verify ]]; then
            echo "Missing $out" >&2; status=1; continue
        fi
        tmp=$(mktemp "$DEST/.$id.XXXXXX")
        if ! curl -fsSL -m 180 -o "$tmp" "https://files.rcsb.org/download/$id.pdb"; then
            echo "Download failed for $id" >&2
            cleanup; tmp=''; status=1; continue
        fi
        source=$tmp
    fi
    digest=$(sha256sum "$source")
    got=${digest%% *}
    if [[ -n $want && $got != "$want" ]]; then
        echo "SHA-256 mismatch for $id: expected $want, got $got" >&2
        cleanup; tmp=''; status=1; continue
    fi
    if [[ -n $tmp ]]; then mv -- "$tmp" "$out"; tmp=''; fi
    if [[ -n $want ]]; then
        echo "Verified $id  $got"
    else
        echo "Unverified $id  $got"
    fi
done
exit "$status"
