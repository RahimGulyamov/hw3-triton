#!/usr/bin/env python3
"""Create a source-only archive for uploading to a new Git repository."""
from pathlib import Path
import zipfile


def main():
    project = Path(__file__).resolve().parent.parent
    items = [
        '.github', '.dockerignore', '.gitignore', 'Dockerfile', 'compose.yaml',
        'requirements.txt', 'requirements-analyzer.txt', 'README.md',
        'model_repository', 'clients', 'scripts', 'profiling', 'docs',
        'examples/.gitkeep', 'results/.gitkeep', 'tests',
    ]
    output = project / 'results/source.zip'
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for item in items:
            path = project / item
            files = sorted(path.rglob('*')) if path.is_dir() else [path]
            for file in files:
                if not file.is_file() or file.is_symlink():
                    continue
                if '__pycache__' in file.parts or file.suffix in ('.pyc', '.pt', '.pdf'):
                    continue
                archive.write(file, file.relative_to(project))
    print(f'Source archive: {output} ({output.stat().st_size:,} bytes)')
    print('Contains code and workflow; measurements must be added after profiling.')


if __name__ == '__main__':
    main()
