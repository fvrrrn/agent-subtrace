"""Checksum-verified relocation of this user's experiment directories; no result deletion."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/var/local/danefedov/experiments')
PROJECT=ROOT/'agent-safety'


def remove_verified_copy(path):
    # Triton cache directories can be read-only. Only unlock our own directories.
    for folder,dirs,files in os.walk(path,followlinks=False):
        p=Path(folder)
        assert p.stat().st_uid==os.getuid()
        p.chmod(p.stat().st_mode|0o700)
    shutil.rmtree(path)


def main():
    assert Path.home()==Path('/home/danefedov')
    for name in ('archive/legacy-home','archive/legacy-shm','datasets','runs','models','envs','vendors','audits'):
        (PROJECT/name).mkdir(parents=True,exist_ok=True)
    entry=Path.home()/'experiments'
    if not entry.exists():entry.symlink_to(ROOT,target_is_directory=True)
    assert entry.resolve()==ROOT
    ledger=PROJECT/'audits/storage_migration.jsonl'
    def log(record):
        with ledger.open('a') as f:f.write(json.dumps(dict(time=time.time(),**record))+'\n')
    previous=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    for record in previous:
        if record['phase']!='checksum_verified':continue
        src=Path(record['source']);q=src.with_name('.'+src.name+'.verified-relocation')
        if q.exists():
            assert Path(record['destination']).is_dir()
            try:
                remove_verified_copy(q)
                log({'phase':'verified_copy_cleanup_resumed','source':str(src),'destination':record['destination']})
            except (AssertionError,OSError) as error:
                log({'phase':'protected_leftover','path':str(q),'reason':repr(error)})
    sources=[p for p in Path.home().iterdir() if p.is_dir() and not p.is_symlink() and p.name.startswith(('kev','laya-rjudge-'))]
    homework=Path.home()/'homework-2-resnet-tuning-20261006'
    if homework.is_dir() and not homework.is_symlink():sources.append(homework)
    sources+=sorted(p for p in Path('/dev/shm').glob('danefedov-*') if p.is_dir() and not p.is_symlink())
    for src in sources:
        assert src.stat().st_uid==os.getuid()
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                if proc.stat().st_uid!=os.getuid() or int(proc.name)==os.getpid():continue
                cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                cwd=(proc/'cwd').resolve()
                if str(src) in cmd or cwd==src or src in cwd.parents:
                    raise RuntimeError(f'Active process {proc.name} references {src}')
            except (FileNotFoundError,PermissionError,ProcessLookupError):pass
        if src==homework:
            dst=ROOT/'other-projects'/src.name
        else:dst=PROJECT/'archive'/('legacy-home' if src.parent==Path.home() else 'legacy-shm')/src.name
        dst.mkdir(parents=True,exist_ok=False)
        size=subprocess.check_output(['du','-sb',str(src)],text=True).split()[0]
        log({'phase':'copy_started','source':str(src),'destination':str(dst),'bytes':int(size)})
        print('COPY',src,'->',dst,flush=True)
        subprocess.run(['rsync','-aH',str(src)+'/',str(dst)+'/'],check=True)
        verify=subprocess.check_output(['rsync','-aHnc','--delete','--itemize-changes',str(src)+'/',str(dst)+'/'],text=True)
        if verify.strip():raise RuntimeError(f'Checksum verification failed: {src}: {verify}')
        log({'phase':'checksum_verified','source':str(src),'destination':str(dst)})
        # Keep compatibility paths for interpreter shebangs, HF caches and historical scripts.
        compatibility=src.parent==Path('/dev/shm') or src.name=='kev-rjudge-baseline-20261013'
        quarantine=src.with_name('.'+src.name+'.verified-relocation')
        assert not quarantine.exists()
        src.rename(quarantine)
        if compatibility:src.symlink_to(dst,target_is_directory=True)
        try:remove_verified_copy(quarantine)
        except (AssertionError,OSError) as error:
            log({'phase':'protected_leftover','path':str(quarantine),'reason':repr(error)})
        log({'phase':'relocated','source':str(src),'destination':str(dst),'compatibility_symlink':compatibility,'removed_only_verified_source_copy':True})
    links={'envs/kev':PROJECT/'archive/legacy-shm/danefedov-kev-rjudge-20261013/venv',
        'models/huggingface':PROJECT/'archive/legacy-shm/danefedov-kev-rjudge-20261013/huggingface',
        'vendors/kev':PROJECT/'archive/legacy-home/kev-rjudge-baseline-20261013/source'}
    for name,target in links.items():
        link=PROJECT/name
        if not link.exists():link.symlink_to(target,target_is_directory=True)
    log({'phase':'complete'})
    print('MIGRATION COMPLETE',flush=True)


if __name__=='__main__':main()
