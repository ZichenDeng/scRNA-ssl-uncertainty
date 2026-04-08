from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

ROOT = Path('/home/zichende/scRNA-ssl-uncertainty')
DELIV = ROOT / 'deliverables' / 'ms2_ab'
FIG = DELIV / 'figures'
DATA_RAW = ROOT / 'data' / 'raw' / 'GSE96583'
DATA_PROC = ROOT / 'data' / 'processed'

B1 = DATA_PROC / 'GSE96583_batch1_qc_shared_annotated_singlets.h5ad'
B2 = DATA_PROC / 'GSE96583_batch2_qc_shared_annotated_singlets.h5ad'

SLIDE_TITLE = 'MS2 Sections 1-2: Dataset Motivation and Data Wrangling'
PROJECT_NUMBER = 'Canvas Project #: TODO'
GROUP_MEMBERS = 'Group Members: TODO'


def ensure_dirs() -> None:
    DELIV.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def load_data():
    adata1 = ad.read_h5ad(B1)
    adata2 = ad.read_h5ad(B2)
    return adata1, adata2


def raw_file_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(DATA_RAW.glob('*')):
        if path.is_file():
            rows.append(
                {
                    'file': path.name,
                    'size_mb': round(path.stat().st_size / (1024 * 1024), 2),
                    'kind': infer_kind(path.name),
                }
            )
    return pd.DataFrame(rows)


def infer_kind(name: str) -> str:
    if name.endswith('.mtx.gz') or name.endswith('.mat.gz'):
        return 'count matrix'
    if 'genes' in name:
        return 'gene metadata'
    if 'tsne' in name:
        return 'cell metadata'
    if name.endswith('.txt.gz'):
        return 'gene list'
    return 'other'


def get_label_col(adata) -> str:
    for candidate in ['cell.type', 'cell']:
        if candidate in adata.obs.columns:
            return candidate
    raise KeyError(f'No label column found in {list(adata.obs.columns)}')


def sample_counts(adata1, adata2) -> pd.DataFrame:
    rows = []
    for batch_name, adata in [('batch1', adata1), ('batch2', adata2)]:
        counts = adata.obs['sample'].value_counts().sort_index()
        for sample, count in counts.items():
            rows.append({'batch': batch_name, 'sample': sample, 'cells': int(count)})
    return pd.DataFrame(rows)


def label_counts(adata1, adata2) -> pd.DataFrame:
    label_col1 = get_label_col(adata1)
    label_col2 = get_label_col(adata2)
    labels = sorted(set(adata1.obs[label_col1]).union(set(adata2.obs[label_col2])))
    rows = []
    for label in labels:
        rows.append(
            {
                'cell_type': label,
                'batch1': int((adata1.obs[label_col1] == label).sum()),
                'batch2': int((adata2.obs[label_col2] == label).sum()),
            }
        )
    return pd.DataFrame(rows)


def save_figures(files_df: pd.DataFrame, samples_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax = plt.subplots(figsize=(8, 4.8))
    files_sorted = files_df.sort_values('size_mb', ascending=False)
    ax.bar(files_sorted['file'], files_sorted['size_mb'], color='#4C78A8')
    ax.set_title('GSE96583 Raw Files by Size')
    ax.set_ylabel('Size (MB)')
    ax.set_xlabel('Raw file')
    ax.tick_params(axis='x', rotation=75, labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'raw_file_sizes.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    pivot = samples_df.pivot(index='sample', columns='batch', values='cells').fillna(0)
    pivot.plot(kind='bar', ax=ax, color=['#72B7B2', '#F58518'])
    ax.set_title('Singlet Cell Counts by Sample and Batch')
    ax.set_ylabel('Cells')
    ax.set_xlabel('Sample accession')
    ax.legend(title='Batch')
    fig.tight_layout()
    fig.savefig(FIG / 'sample_counts.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    labels_plot = labels_df.set_index('cell_type')[['batch1', 'batch2']]
    labels_plot.plot(kind='barh', ax=ax, color=['#54A24B', '#E45756'])
    ax.set_title('Cell-Type Distribution After QC and Singlet Filtering')
    ax.set_xlabel('Cells')
    ax.set_ylabel('Cell type')
    fig.tight_layout()
    fig.savefig(FIG / 'label_counts.png', dpi=200)
    plt.close(fig)


def md_cell(text: str) -> dict:
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.strip().splitlines()],
    }


def code_cell(code: str) -> dict:
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in code.strip().splitlines()],
    }


def build_notebook(files_df: pd.DataFrame, samples_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
    notebook = {
        'cells': [
            md_cell(f'''\\
# {SLIDE_TITLE}
{PROJECT_NUMBER}

{GROUP_MEMBERS}

## Scope of This Notebook
This notebook supports presentation sections 1 and 2 only:
1. Introduction, motivation, and why `GSE96583` is a defensible dataset choice.
2. Data access, raw structure, wrangling, and preprocessing decisions.
            '''),
            md_cell('''\\
## Data Description
`GSE96583` is a PBMC single-cell RNA-seq dataset from GEO with directly downloadable raw count matrices and cell-level metadata. It is a strong MS2 dataset because it already contains usable cell-type annotations, clear batch structure (`batch1` vs `batch2`), and a condition shift inside `batch2` (`ctrl` vs `stim`).

This makes it suitable for studying distribution shift without introducing a second dataset whose labels or accessibility are uncertain.
            '''),
            code_cell('''\\
from pathlib import Path
import pandas as pd

raw_dir = Path('/home/zichende/scRNA-ssl-uncertainty/data/raw/GSE96583')
files = []
for path in sorted(raw_dir.glob('*')):
    if path.is_file():
        files.append({'file': path.name, 'size_mb': round(path.stat().st_size / (1024 * 1024), 2)})
pd.DataFrame(files)
            '''),
            md_cell('''\\
## Understanding the Raw Data Structure
The raw data are not a single clean table. Instead, the dataset is split across multiple compressed files:
- count matrices for different biological or technical subsets
- gene metadata files
- cell-level metadata files with t-SNE coordinates and annotations

That raw structure is one reason wrangling matters here. Before we can model anything, we have to unify matrices, reconcile gene spaces, attach metadata, and decide what cells to keep.
            '''),
            code_cell('''\\
import anndata as ad

adata_b1 = ad.read_h5ad('/home/zichende/scRNA-ssl-uncertainty/data/processed/GSE96583_batch1_qc_shared_annotated_singlets.h5ad')
adata_b2 = ad.read_h5ad('/home/zichende/scRNA-ssl-uncertainty/data/processed/GSE96583_batch2_qc_shared_annotated_singlets.h5ad')

print('batch1 shape:', adata_b1.shape)
print('batch2 shape:', adata_b2.shape)
print('batch1 samples:', adata_b1.obs['sample'].value_counts().to_dict())
print('batch2 samples:', adata_b2.obs['sample'].value_counts().to_dict())
            '''),
            md_cell('''\\
## Wrangling and Preprocessing Decisions
The current processed benchmark reflects these choices:
- load and standardize the raw matrices from GEO
- attach available metadata to each cell
- restrict analyses to shared genes where cross-batch comparisons are valid
- apply QC-oriented filtering already captured in the processed `.h5ad` files
- keep only singlets to avoid doublets contaminating downstream cell-type classification

These are not cosmetic steps. They directly determine whether cross-batch evaluation is meaningful.
            '''),
            code_cell('''\\
label_col_b1 = 'cell.type' if 'cell.type' in adata_b1.obs.columns else 'cell'
label_col_b2 = 'cell.type' if 'cell.type' in adata_b2.obs.columns else 'cell'

label_counts = pd.DataFrame({
    'batch1': adata_b1.obs[label_col_b1].value_counts(),
    'batch2': adata_b2.obs[label_col_b2].value_counts(),
}).fillna(0).astype(int)
label_counts
            '''),
            md_cell('''\\
## Meaningful Insights from Sections 1 and 2
- `GSE96583` is accessible, labeled, and already contains both batch shift and condition shift, so it is enough for a clean milestone dataset.
- The dataset is class-imbalanced: broad immune populations dominate while dendritic cells and megakaryocytes are rare.
- The raw file layout is fragmented, so wrangling is necessary before any valid analysis.
- Singlet filtering is important because downstream classification assumes one biological cell state per observation.
            '''),
            md_cell('''\\
## Revised Research Question
Given a well-wrangled and annotated `GSE96583` PBMC benchmark, can self-supervised representations improve cell-type prediction robustness under batch and condition shift, and can uncertainty estimates help identify unreliable predictions?
            '''),
        ],
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {
                'name': 'python',
                'version': '3.11',
            },
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }

    with (DELIV / 'MS2_AB_Data_Wrangling.ipynb').open('w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)

    files_df.to_csv(DELIV / 'raw_file_inventory.csv', index=False)
    samples_df.to_csv(DELIV / 'sample_counts.csv', index=False)
    labels_df.to_csv(DELIV / 'cell_type_counts.csv', index=False)


def wrap_lines(text: str, width: int = 80) -> str:
    words = text.split()
    lines = []
    current = []
    count = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if count + extra > width:
            lines.append(' '.join(current))
            current = [word]
            count = len(word)
        else:
            current.append(word)
            count += extra
    if current:
        lines.append(' '.join(current))
    return '\n'.join(lines)


def slide(ax, title: str, bullets: list[str], footer: str | None = None) -> None:
    ax.axis('off')
    ax.text(0.03, 0.93, title, fontsize=24, fontweight='bold', va='top')
    y = 0.82
    for bullet in bullets:
        wrapped = wrap_lines(bullet, 76)
        ax.text(0.05, y, f'• {wrapped}', fontsize=15, va='top')
        y -= 0.12 + 0.02 * wrapped.count('\n')
    if footer:
        ax.text(0.03, 0.05, footer, fontsize=10, color='dimgray')


def build_slides(files_df: pd.DataFrame, samples_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
    pdf_path = DELIV / 'MS2_AB_Sections_1_2_Slides.pdf'
    total_raw_mb = round(files_df['size_mb'].sum(), 2)
    batch1_cells = int(samples_df.loc[samples_df['batch'] == 'batch1', 'cells'].sum())
    batch2_cells = int(samples_df.loc[samples_df['batch'] == 'batch2', 'cells'].sum())
    rare = labels_df.assign(total=labels_df['batch1'] + labels_df['batch2']).sort_values('total').iloc[0]['cell_type']

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(13.33, 7.5))
        slide(ax, '1. Why GSE96583?', [
            'We rescoped the project to one benchmark dataset so that the milestone is about defensible data wrangling, not about uncertain cross-dataset labels.',
            'GSE96583 is public on GEO, directly downloadable, and already contains usable PBMC cell-type metadata.',
            'It also contains meaningful shift structure: batch1 versus batch2, plus control versus stimulation inside batch2.',
            f'This gives us a realistic benchmark without adding label-accessibility risk from a second dataset. Total raw download size is about {total_raw_mb} MB.'
        ], footer='Section 1: introduction, motivation, and dataset choice')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(13.33, 7.5))
        slide(ax, '2. Raw Data Structure', [
            'The GEO download is not a single tidy table. It is fragmented across compressed count matrices, gene files, and metadata tables.',
            'Batch1 is assembled from three source matrices: GSM2560245_A, GSM2560246_B, and GSM2560247_C.',
            'Batch2 is assembled from GSM2560248_2.1 and GSM2560249_2.2, which also support control-versus-stimulation comparisons.',
            'The raw structure itself motivates wrangling: before modeling, we need consistent genes, aligned metadata, and one clean cell-by-gene object per benchmark split.'
        ], footer='Section 2: data access and raw dataset structure')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(13.33, 7.5))
        slide(ax, '3. Wrangling Pipeline', [
            'Step 1: download GEO count matrices and metadata and inspect file layout.',
            'Step 2: standardize matrices and attach cell-level annotations.',
            'Step 3: restrict analyses to shared genes so that cross-batch comparisons are valid.',
            'Step 4: keep QC-passed singlets only, which removes likely doublets that would distort cell-type classification.',
            'Step 5: save processed AnnData objects that can be reused by EDA, baselines, and later SSL models.'
        ], footer='Section 2: preprocessing decisions and why they matter')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(13.33, 7.5))
        slide(ax, '4. Wrangling Outcomes', [
            f'After QC and singlet filtering, batch1 contains {batch1_cells} cells and batch2 contains {batch2_cells} cells.',
            'The processed data preserve major immune cell types needed for downstream classification.',
            f'The class distribution is imbalanced, with broad T-cell and monocyte populations dominating while {rare} and other rare types are scarce.',
            'This matters for later modeling because macro-F1 and per-class behavior will be more informative than accuracy alone.'
        ], footer='Bridge to later sections: EDA and baseline evaluation')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def build_notes() -> None:
    notes = '''# MS2 Sections 1-2 Speaker Notes

## Slide 1: Why GSE96583?
- Open with the rescoping decision.
- Emphasize that MS2 rewards a defensible dataset and a clear wrangling story.
- Say explicitly that GSE96583 already gives us both labels and shift.

## Slide 2: Raw Data Structure
- Explain that the GEO download is fragmented rather than analysis-ready.
- Name the batch1 and batch2 source files once so the TF sees that you understand the actual data layout.
- Transition into why preprocessing is necessary.

## Slide 3: Wrangling Pipeline
- Keep this procedural and concrete.
- The important point is that each preprocessing step protects validity of cross-batch comparisons.

## Slide 4: Wrangling Outcomes
- End with concrete cell counts and class imbalance.
- Set up the next presenter by saying these processed objects are now ready for EDA and baseline modeling.
'''
    (DELIV / 'MS2_AB_Speaker_Notes.md').write_text(notes, encoding='utf-8')


def main() -> None:
    ensure_dirs()
    adata1, adata2 = load_data()
    files_df = raw_file_inventory()
    samples_df = sample_counts(adata1, adata2)
    labels_df = label_counts(adata1, adata2)
    save_figures(files_df, samples_df, labels_df)
    build_notebook(files_df, samples_df, labels_df)
    build_slides(files_df, samples_df, labels_df)
    build_notes()
    print('Built MS2 section 1-2 assets in', DELIV)


if __name__ == '__main__':
    main()
