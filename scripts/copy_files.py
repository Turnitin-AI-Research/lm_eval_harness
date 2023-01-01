#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import time
import fire


def copy_files(src_dir, dest_dir, *, dry_run=True):
    files3 = {f for f in os.listdir(src_dir) if f.endswith('.json')}
    files2 = {f for f in os.listdir(dest_dir) if f.endswith('.json')}
    PROMPT = "dry run: " if dry_run else ""

    for f in files3:
        src_path = f'{src_dir}/{f}'
        src_stat = os.stat(src_path)
        if f not in files2:
            print(f'{PROMPT}copying file {f}: modified on {time.ctime(src_stat.st_mtime)}')
            if not dry_run:
                shutil.copy2(src_path, dest_dir)
        else:
            dst_path = f'{dest_dir}/{f}'
            dst_stat = os.stat(dst_path)
            if src_stat.st_mtime > dst_stat.st_mtime:
                print(f'{PROMPT}overwriting file {f}: modified on {time.ctime(src_stat.st_mtime)}')
                if not dry_run:
                    shutil.copy2(src_path, dest_dir)
            else:
                print(f'{PROMPT}skipping older file {f}: modified on {time.ctime(src_stat.st_mtime)}')


if __name__ == '__main__':
    fire.Fire(copy_files)
