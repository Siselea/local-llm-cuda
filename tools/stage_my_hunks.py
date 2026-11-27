#!/usr/bin/env python3
"""Stage only *my* hunks out of a working tree shared with other agents.

Usage: python tools/stage_my_hunks.py <file> <predicate> [<file> <predicate> ...]
A hunk is selected when any added/removed line contains one of the '|'-separated
substrings of its predicate.  The patch is rebuilt with the original file header and
only the selected hunks, then applied to the index with `git apply --cached`.
"""
import subprocess
import sys


def diff_for(path):
    out = subprocess.run(['git', 'diff', '--no-color', '-U3', '--', path],
                         capture_output=True, text=True, errors='replace')
    if out.returncode != 0:
        raise SystemExit(f'git diff failed for {path}: {out.stderr[:400]}')
    return out.stdout


def split_patch(text):
    lines = text.splitlines(keepends=True)
    header, hunks, cur = [], [], None
    for ln in lines:
        if ln.startswith('@@'):
            if cur is not None:
                hunks.append(cur)
            cur = [ln]
        elif cur is None:
            header.append(ln)
        else:
            cur.append(ln)
    if cur is not None:
        hunks.append(cur)
    return header, hunks


def main():
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2:
        raise SystemExit(__doc__)
    pair = list(zip(args[0::2], args[1::2]))
    total_sel = 0
    for path, pred in pair:
        text = diff_for(path)
        if not text.strip():
            print(f'{path}: no diff')
            continue
        header, hunks = split_patch(text)
        preds = [p for p in pred.split('|') if p]
        selected = []
        for i, h in enumerate(hunks):
            body = ''.join(h)
            content = '\n'.join(l for l in h[1:] if l[:1] in '+-')
            if any(p in content for p in preds):
                selected.append(h)
            else:
                print(f'  {path}: skip hunk {i} ({h[0].strip()[:70]})')
        if not selected:
            print(f'{path}: nothing selected')
            continue
        patch = ''.join(header) + ''.join(''.join(h) for h in selected)
        import os
        os.makedirs('build/stage', exist_ok=True)
        pfile = f'build/stage/stage_{abs(hash(path)) % 10**8}.patch'
        with open(pfile, 'w', encoding='utf-8', newline='') as f:
            f.write(patch)
        chk = subprocess.run(['git', 'apply', '--cached', '--check', pfile],
                             capture_output=True, text=True)
        if chk.returncode != 0:
            raise SystemExit(f'git apply --check failed for {path}: {chk.stderr[:800]}')
        subprocess.run(['git', 'apply', '--cached', pfile], check=True)
        print(f'{path}: staged {len(selected)}/{len(hunks)} hunks')
        total_sel += len(selected)
    print(f'total staged hunks: {total_sel}')


if __name__ == '__main__':
    main()
