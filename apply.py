#!/usr/bin/env python3
"""One-shot, reversible consolidation for model-multiplicity-explanations.

Applies active-data cleanup, four loaders, margin-quadrant analysis, full SHAP
ranks, relative top-k, selective plotting with BA rankings, and optional
SHAP / IG / LIME methods (without changing existing training/checkpoints).

Designed for the Nov 27, 2026 main revision inspected via the GitHub connector.
It stages and syntax-checks changes before writing and retains backups.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC / 'internal'))
from internal.datasets_patch import (  # noqa: E402
    patch_data, patch_preflight, patch_tests, patch_pyproject, replace_once,
)
from internal.plot_patch_v2 import apply as apply_plot_v2  # noqa: E402


FILES = [
    'mmcyber/data.py', 'mmcyber/preflight.py', 'mmcyber/cli.py',
    'mmcyber/explanation_disagreement.py', 'mmcyber/shap_worker.py',
    'mmcyber/plotting.py', 'tests/test_reliability.py',
    'pyproject.toml', 'requirements.txt', 'README.md',
    'configs/default.yaml', 'configs/smoke.yaml',
    'scripts/run_all_full_datasets.sh',
]

ADDITIONS = [
    'mmcyber/extra_datasets.py',
    'mmcyber/method_comparison.py',
    'analysis/prediction_explanation_quadrants.py',
    'analysis/compare_methods.py',
    'analysis/rebuild_shap_summary.py',
    'tests/test_extra_datasets.py',
    'tests/test_quadrants.py',
    'tests/test_method_comparison.py',
    'configs/compas_full.yaml',
    'configs/german_credit_full.yaml',
    'configs/taiwan_credit_full.yaml',
    'configs/breast_cancer_full.yaml',
    'scripts/run_all_full_datasets.sh',
    'requirements-explain.txt',
    'README.md',
    'GENERAL_ML_MIGRATION.md',
]

CLI_PARSER_ANCHOR = '''    subparsers.add_parser(
        "preflight",
'''
CLI_DISPATCH_ANCHOR = '''    elif args.command == "preflight":
        preflight_full_configs()'''


def patch_unique_ranking_files(code: str) -> str:
    """Avoid overwriting Adult <=50K vs >50K rank plots after slug sanitization."""
    start = code.index('def plot_ba_feature_rankings_from_summary(')
    end = code.index('def plot_paper_summary(', start)
    section = code[start:end]
    before = '    for class_name, class_frame in summary.groupby("class_name"):\n'
    after = ('    grouped_classes = list(summary.groupby("class_name"))\n'
             '    sanitized = [_safe_name(label) for label, _ in grouped_classes]\n'
             '    for class_index, (class_name, class_frame) in enumerate(grouped_classes):\n'
             '        slug = _safe_name(class_name)\n'
             '        if sanitized.count(slug) > 1:\n'
             '            slug = f"class{class_index}_{slug}"\n')
    if section.count(before) != 1:
        raise ValueError('Unexpected ranking plot structure')
    section = section.replace(before, after, 1)
    section = section.replace('_safe_name(class_name)}.png', 'slug}.png')
    return code[:start] + section + code[end:]


def patch_cli(code: str) -> str:
    code = replace_once(code,
        'from mmcyber.shap_analysis import compute_shap\n',
        'from mmcyber.shap_analysis import compute_shap\n'
        'from mmcyber.method_comparison import compute_method_comparison\n', 'CLI method import')
    code = replace_once(code, CLI_PARSER_ANCHOR, '''    multi_parser = subparsers.add_parser(
        "multi-explain", help="Compare SHAP margin, Integrated Gradients and LIME"
    )
    multi_parser.add_argument("--run-dir", required=True)
    multi_parser.add_argument("--methods", nargs="+", choices=["shap", "ig", "lime"],
                              default=["shap", "ig", "lime"])
    multi_parser.add_argument("--max-models", type=int)
    multi_parser.add_argument("--max-samples", type=int, default=64)
    multi_parser.add_argument("--device", default="cpu")
    multi_parser.add_argument("--ig-steps", type=int, default=64)
    multi_parser.add_argument("--ig-batch", type=int, default=16)
    multi_parser.add_argument("--lime-samples", type=int, default=1000)
    multi_parser.add_argument("--lime-reference-size", type=int, default=2000)
    multi_parser.add_argument("--random-state", type=int, default=42)
    multi_parser.add_argument("--overwrite", action="store_true")

''' + CLI_PARSER_ANCHOR, 'CLI method parser')
    code = replace_once(code, CLI_DISPATCH_ANCHOR, '''    elif args.command == "multi-explain":
        output = compute_method_comparison(
            args.run_dir, methods=tuple(args.methods), max_models=args.max_models,
            max_samples=args.max_samples, device=args.device, ig_steps=args.ig_steps,
            ig_batch=args.ig_batch, lime_samples=args.lime_samples,
            lime_reference_size=args.lime_reference_size,
            random_state=args.random_state, overwrite=args.overwrite,
        )
        print(f"Method attributions written to {output}")
''' + CLI_DISPATCH_ANCHOR, 'CLI method dispatch')
    return code


def patch_pyproject_extras(code: str) -> str:
    code = replace_once(code, '  "folktables>=0.0.12",',
                        '  "folktables>=0.0.12",\n  "xlrd>=2.0",', 'xlrd base dependency')
    code = replace_once(code, '[project.scripts]',
        '''[project.optional-dependencies]
explain = ["captum>=0.7", "lime>=0.2.0.1"]

[project.scripts]''', 'explanation extras')
    return code


def patch_requirements(code: str) -> str:
    result = code.rstrip() + '\n'
    if 'xlrd>=' not in result:
        result += 'xlrd>=2.0\n'
    return result


def check_clean_checkout(root: Path) -> None:
    if not all((root / path).is_file() for path in FILES):
        missing = [path for path in FILES if not (root / path).is_file()]
        raise FileNotFoundError(f'Not a compatible model-multiplicity-explanations checkout: {missing}')
    # Prior one-off patches may have overlapping anchors. This installer
    # deliberately refuses them; apply on a fresh branch to avoid mixed state.
    current = (root / 'mmcyber/cli.py').read_text(encoding='utf-8')
    if '--plot-profile' in current or 'multi-explain' in current:
        raise RuntimeError('This checkout already has prior plotting/method patches. '
                           'Use a clean checkout of main for the unified installer; '
                           'your existing changes and runs will remain untouched.')
    data = (root / 'mmcyber/data.py').read_text(encoding='utf-8')
    if 'from mmcyber.extra_datasets import PREPARERS' in data:
        raise RuntimeError('General-dataset patch already installed. '
                           'Use a clean main checkout for the unified installer.')


def install(root: Path) -> None:
    check_clean_checkout(root)
    for path in ADDITIONS:
        if path not in ('README.md', 'scripts/run_all_full_datasets.sh') and not (SRC / path).is_file():
            raise FileNotFoundError(f'Bundle missing {path}')

    with tempfile.TemporaryDirectory(prefix='mm-combined-') as tmp:
        stage = Path(tmp)
        for name in FILES:
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, destination)
        # Apply plotting first. It also fixes the old top-50 SHAP truncation and
        # replaces fixed top-k Jaccard with 10%+minimum-three feature selection.
        apply_plot_v2(stage)
        fig_file = stage / 'mmcyber/plotting.py'
        fig_file.write_text(patch_unique_ranking_files(fig_file.read_text(encoding='utf-8')), encoding='utf-8')
        updates = {
            'mmcyber/data.py': patch_data,
            'mmcyber/preflight.py': patch_preflight,
            'tests/test_reliability.py': patch_tests,
            'pyproject.toml': lambda c: patch_pyproject_extras(patch_pyproject(c)),
            'requirements.txt': patch_requirements,
            'mmcyber/cli.py': patch_cli,
        }
        for name, transform in updates.items():
            path = stage / name
            path.write_text(transform(path.read_text(encoding='utf-8')), encoding='utf-8')
        shutil.copy2(root / 'configs/folktables_income_full.yaml', stage / 'configs/default.yaml')
        shutil.copy2(root / 'configs/folktables_income_smoke.yaml', stage / 'configs/smoke.yaml')
        for name in ADDITIONS:
            if name in ('README.md', 'scripts/run_all_full_datasets.sh'):
                continue
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SRC / name, dest)
        shutil.copy2(SRC / 'README_GENERAL.md', stage / 'README.md')
        shutil.copy2(SRC / 'scripts/run_all_full_datasets.sh', stage / 'scripts/run_all_full_datasets.sh')

        # Preflight syntax; all original files stay unchanged if any anchor fails.
        for python_path in stage.rglob('*.py'):
            compile(python_path.read_text(encoding='utf-8'), str(python_path), 'exec')
        updates_and_adds = list(dict.fromkeys([
            'mmcyber/data.py', 'mmcyber/preflight.py', 'tests/test_reliability.py',
            'pyproject.toml', 'requirements.txt', 'mmcyber/cli.py',
            'mmcyber/plotting.py', 'mmcyber/shap_worker.py',
            'mmcyber/explanation_disagreement.py',
            'configs/default.yaml', 'configs/smoke.yaml',
            *ADDITIONS,
        ]))
        backups = root / 'docs' / 'migration_backups' / 'complete_2026-11-27'
        if backups.exists():
            raise FileExistsError(f'Backup folder already exists: {backups}')
        backups.mkdir(parents=True)
        for name in updates_and_adds:
            orig = root / name
            if orig.exists():
                backup = backups / name
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(orig, backup)
        for name in updates_and_adds:
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stage / name, dest)
        (backups / 'manifest.json').write_text(json.dumps({
            'installer': 'unified 2026-11-27', 'modified': updates_and_adds,
            'original_backup': 'same relative paths in this folder',
        }, indent=2) + '\n')
    print('Applied unified general-ML dataset, plot and method-comparison fixes.')
    print('Old NSL-KDD checkpoints / papers were NOT deleted.')
    print("Next: pip install -e '.[explain]' && python -m unittest discover -s tests -v")
    print('Then: python -m mmcyber.cli preflight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path, help='Clean model-multiplicity-explanations clone')
    args = parser.parse_args()
    install(args.repository.resolve())
