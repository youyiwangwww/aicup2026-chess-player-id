"""Create an isolated CUDA-capable Python location, leaving the CPU environment intact."""
import shutil
from pathlib import Path


def main():
    """Copy Python and identical non-Torch dependencies before a CUDA wheel install."""
    source = Path(r'C:\Users\abbywang\.codex\tmp\aicup-python')
    target = Path(r'C:\Users\abbywang\.codex\tmp\aicup-python-cuda')
    project = Path(__file__).resolve().parents[1]
    if target.exists():
        raise ValueError('CUDA environment already exists; refusing to overwrite it')
    target.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and (path.suffix in ['.exe', '.dll', '.pyd'] or path.name == 'python312.zip'):
            shutil.copy2(path, target / path.name)
    packages = target / 'Lib/site-packages'
    packages.mkdir(parents=True)
    for path in (source / 'Lib/site-packages').iterdir():
        if path.name.lower().startswith(('torch', 'functorch')):
            continue
        destination = packages / path.name
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)
    (target / 'python312._pth').write_text(
        'python312.zip\n.\nLib/site-packages\n' + str(project) + '\nimport site\n', encoding='utf-8')
    print(f'Created isolated interpreter: {target / "python.exe"}')


if __name__ == '__main__':
    main()
