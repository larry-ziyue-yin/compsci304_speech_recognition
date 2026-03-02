#!/usr/bin/env python3
"""Prepare SG-whisper wTIMIT manifests for Project 7.

The split is by sentence ID (400/25/25) to avoid text leakage.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


UTT_SENT_RE = re.compile(r"u(\d{3})(?:[a-z])?$", re.IGNORECASE)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def sentence_id_from_utt_key(utt_key: str) -> Optional[int]:
    match = UTT_SENT_RE.search(utt_key.lower())
    if match:
        return int(match.group(1))
    return None


def sentence_id_from_text(text: str) -> Optional[int]:
    candidate = text.strip()
    if not candidate:
        return None

    sid = sentence_id_from_utt_key(Path(candidate).stem)
    if sid is not None:
        return sid

    match = re.search(r"u(\d{3})", candidate.lower())
    if match:
        return int(match.group(1))

    numbers = re.findall(r"\d+", candidate)
    if numbers:
        return int(numbers[-1])

    return None


def load_sentence_texts(sentences_path: Path) -> Dict[int, str]:
    if not sentences_path.is_file():
        raise FileNotFoundError(f"sentences file not found: {sentences_path}")

    sentence_texts: Dict[int, str] = {}
    running_sid = 0
    with sentences_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            running_sid += 1
            sid: Optional[int] = None
            text: Optional[str] = None

            if "\t" in line:
                cols = [c.strip() for c in line.split("\t") if c.strip()]
                if cols:
                    if cols[0].isdigit():
                        sid = int(cols[0])
                        text = " ".join(cols[1:]) if len(cols) > 1 else ""
                    else:
                        text = cols[-1]

            if sid is None:
                match = re.match(r"^(\d+)\s+(.+)$", line)
                if match:
                    sid = int(match.group(1))
                    text = match.group(2)

            if sid is None:
                sid = running_sid
                text = line

            text = normalize_text(text if text is not None else "")
            if text:
                sentence_texts[sid] = text

    return sentence_texts


def load_sentence_map(
    sentence_map_path: Path,
) -> Tuple[Dict[str, str], Dict[int, str]]:
    utt_to_text: Dict[str, str] = {}
    sid_to_text: Dict[int, str] = {}

    if not sentence_map_path.is_file():
        return utt_to_text, sid_to_text

    with sentence_map_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            cols = line.split("\t")
            if len(cols) < 2:
                continue

            key_field = cols[0].strip()
            text = normalize_text(cols[-1])
            if not text:
                continue

            utt_key = Path(key_field).stem.lower()
            if utt_key:
                utt_to_text[utt_key] = text

            sid = sentence_id_from_text(key_field)
            if sid is not None and sid not in sid_to_text:
                sid_to_text[sid] = text

    return utt_to_text, sid_to_text


def load_sent_id_file(path: Path) -> Set[int]:
    sent_ids: Set[int] = set()
    if not path.is_file():
        return sent_ids

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            sid = sentence_id_from_text(line)
            if sid is not None:
                sent_ids.add(sid)
    return sent_ids


def build_audio_lookup(wtimit_root: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for path in wtimit_root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".wav":
            rel = path.relative_to(wtimit_root).as_posix().lower()
            lookup[rel] = path
    return lookup


def resolve_audio_path(
    wtimit_root: Path,
    rel_or_abs_path: str,
    lookup: Optional[Dict[str, Path]],
) -> Optional[Path]:
    path = Path(rel_or_abs_path)
    if not path.is_absolute():
        path = wtimit_root / path

    if path.is_file():
        return path

    if path.suffix:
        swapped = path.with_suffix(".wav" if path.suffix != ".wav" else ".WAV")
        if swapped.is_file():
            return swapped

    if lookup is None:
        return None

    try:
        rel = path.relative_to(wtimit_root).as_posix().lower()
    except ValueError:
        rel = path.as_posix().lower()
    return lookup.get(rel)


def collect_wavs_from_list(wtimit_root: Path, list_path: Path) -> List[Path]:
    if not list_path.is_file():
        return []

    wav_paths: Set[Path] = set()
    lookup: Optional[Dict[str, Path]] = None

    with list_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            rel_path = raw_line.strip().replace("\\", "/")
            if not rel_path:
                continue
            rel_path_l = rel_path.lower()
            if not rel_path_l.endswith(".wav"):
                continue
            if "/whisper/" not in rel_path_l or "/sg/" not in rel_path_l:
                continue

            resolved = resolve_audio_path(wtimit_root, rel_path, lookup)
            if resolved is None:
                if lookup is None:
                    lookup = build_audio_lookup(wtimit_root)
                resolved = resolve_audio_path(wtimit_root, rel_path, lookup)

            if resolved is not None:
                wav_paths.add(resolved)

    return sorted(wav_paths)


def collect_wavs_by_scan(wtimit_root: Path) -> List[Path]:
    nist_root = wtimit_root / "nist"
    if not nist_root.is_dir():
        return []

    wav_paths: List[Path] = []
    for path in nist_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        rel_l = path.relative_to(wtimit_root).as_posix().lower()
        if "/whisper/" in rel_l and "/sg/" in rel_l:
            wav_paths.append(path)
    return sorted(wav_paths)


def choose_sentence_splits(
    all_sentence_ids: Sequence[int],
    train_sent_ids_path: Path,
    dev_sent_ids_path: Path,
    test_sent_ids_path: Path,
) -> Tuple[Set[int], Set[int], Set[int]]:
    train_ids = load_sent_id_file(train_sent_ids_path)
    dev_ids = load_sent_id_file(dev_sent_ids_path)
    test_ids = load_sent_id_file(test_sent_ids_path)

    if train_ids and dev_ids and test_ids:
        overlap = (train_ids & dev_ids) | (train_ids & test_ids) | (dev_ids & test_ids)
        if overlap:
            raise ValueError(
                "split sentence id files overlap, cannot continue safely"
            )
        return train_ids, dev_ids, test_ids

    ordered = sorted(set(all_sentence_ids))
    if len(ordered) < 450:
        raise ValueError(
            f"not enough sentence ids ({len(ordered)}) for a 400/25/25 split"
        )
    ordered = ordered[:450]
    return set(ordered[:400]), set(ordered[400:425]), set(ordered[425:450])


def deduplicate_entries(entries: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    deduped: List[Tuple[str, str]] = []
    seen_paths: Set[str] = set()
    for audio_path, text in sorted(entries, key=lambda x: x[0]):
        if audio_path in seen_paths:
            continue
        seen_paths.add(audio_path)
        deduped.append((audio_path, text))
    return deduped


def write_manifest(path: Path, entries: Sequence[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for audio_path, text in entries:
            f.write(f"{audio_path}\t{text}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wtimit-root",
        type=Path,
        default=Path("/dkucc/group/courses/compsci304-2526-s3/cl688/wTIMIT"),
        help="Root path of wTIMIT dataset",
    )
    parser.add_argument(
        "--train-manifest", type=Path, default=Path("data/train"), help="Output train manifest"
    )
    parser.add_argument(
        "--dev-manifest", type=Path, default=Path("data/dev"), help="Output dev manifest"
    )
    parser.add_argument(
        "--test-manifest", type=Path, default=Path("data/test"), help="Output test manifest"
    )
    args = parser.parse_args()

    wtimit_root = args.wtimit_root
    if not wtimit_root.is_dir():
        print(f"[ERROR] wTIMIT root not found: {wtimit_root}", file=sys.stderr)
        return 1

    sentences_path = wtimit_root / "doc" / "sentences.txt"
    sentence_map_path = wtimit_root / "lists" / "sentences_map.tsv"
    sg_whisper_list_path = wtimit_root / "lists" / "wtimit_sg_w.lst"
    train_sent_ids_path = wtimit_root / "lists" / "train_sent_ids.txt"
    dev_sent_ids_path = wtimit_root / "lists" / "dev_sent_ids.txt"
    test_sent_ids_path = wtimit_root / "lists" / "test_sent_ids.txt"

    sentence_texts = load_sentence_texts(sentences_path)
    utt_to_text, sid_to_text = load_sentence_map(sentence_map_path)
    sentence_texts.update(sid_to_text)

    wav_paths = collect_wavs_from_list(wtimit_root, sg_whisper_list_path)
    if not wav_paths:
        wav_paths = collect_wavs_by_scan(wtimit_root)
    if not wav_paths:
        print("[ERROR] no SG-whisper wav files found", file=sys.stderr)
        return 1

    all_sentence_ids = set(sentence_texts.keys())
    if not all_sentence_ids:
        for wav_path in wav_paths:
            sid = sentence_id_from_utt_key(wav_path.stem)
            if sid is not None:
                all_sentence_ids.add(sid)

    train_ids, dev_ids, test_ids = choose_sentence_splits(
        sorted(all_sentence_ids),
        train_sent_ids_path,
        dev_sent_ids_path,
        test_sent_ids_path,
    )
    split_by_sid = {sid: "train" for sid in train_ids}
    split_by_sid.update({sid: "dev" for sid in dev_ids})
    split_by_sid.update({sid: "test" for sid in test_ids})

    split_entries: Dict[str, List[Tuple[str, str]]] = {"train": [], "dev": [], "test": []}
    split_used_ids: Dict[str, Set[int]] = {"train": set(), "dev": set(), "test": set()}
    missing_text = 0

    for wav_path in wav_paths:
        utt_key = wav_path.stem.lower()
        sid = sentence_id_from_utt_key(utt_key)
        if sid is None:
            continue

        split = split_by_sid.get(sid)
        if split is None:
            continue

        text = utt_to_text.get(utt_key) or sentence_texts.get(sid)
        if not text:
            missing_text += 1
            continue

        split_entries[split].append((wav_path.resolve().as_posix(), text))
        split_used_ids[split].add(sid)

    train_entries = deduplicate_entries(split_entries["train"])
    dev_entries = deduplicate_entries(split_entries["dev"])
    test_entries = deduplicate_entries(split_entries["test"])

    write_manifest(args.train_manifest, train_entries)
    write_manifest(args.dev_manifest, dev_entries)
    write_manifest(args.test_manifest, test_entries)

    print(
        "[INFO] sentence split sizes (train/dev/test): "
        f"{len(train_ids)}/{len(dev_ids)}/{len(test_ids)}",
        file=sys.stderr,
    )
    print(
        "[INFO] manifest utterances (train/dev/test): "
        f"{len(train_entries)}/{len(dev_entries)}/{len(test_entries)}",
        file=sys.stderr,
    )
    print(
        "[INFO] covered sentence ids (train/dev/test): "
        f"{len(split_used_ids['train'])}/{len(split_used_ids['dev'])}/{len(split_used_ids['test'])}",
        file=sys.stderr,
    )
    if missing_text:
        print(f"[WARN] skipped {missing_text} wavs without transcript", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
