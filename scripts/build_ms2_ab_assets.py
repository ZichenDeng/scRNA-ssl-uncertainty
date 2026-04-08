from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
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

BG = '#F6F4EF'
NAVY = '#16324F'
TEAL = '#2A7F8E'
GOLD = '#C58B2B'
INK = '#1F2933'
MUTED = '#5B6770'


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
            md_cell(f'''\
# {SLIDE_TITLE}
{PROJECT_NUMBER}

{GROUP_MEMBERS}

## Scope of This Notebook
This notebook supports presentation sections 1 and 2 only:
1. Introduction, motivation, and why `GSE96583` is a defensible dataset choice.
2. Data access, raw structure, wrangling, and preprocessing decisions.
            '''),
            md_cell('''\
## Data Description
`GSE96583` is a PBMC single-cell RNA-seq dataset from GEO with directly downloadable raw count matrices and cell-level metadata. It is a strong MS2 dataset because it already contains usable cell-type annotations, clear batch structure (`batch1` vs `batch2`), and a condition shift inside `batch2` (`ctrl` vs `stim`).

This makes it suitable for studying distribution shift without introducing a second dataset whose labels or accessibility are uncertain.
            '''),
            code_cell('''\
from pathlib import Path
import pandas as pd

raw_dir = Path('/home/zichende/scRNA-ssl-uncertainty/data/raw/GSE96583')
files = []
for path in sorted(raw_dir.glob('*')):
    if path.is_file():
        files.append({'file': path.name, 'size_mb': round(path.stat().st_size / (1024 * 1024), 2)})
pd.DataFrame(files)
            '''),
            md_cell('''\
## Understanding the Raw Data Structure
The raw data are not a single clean table. Instead, the dataset is split across multiple compressed files:
- count matrices for different biological or technical subsets
- gene metadata files
- cell-level metadata files with t-SNE coordinates and annotations

That raw structure is one reason wrangling matters here. Before we can model anything, we have to unify matrices, reconcile gene spaces, attach metadata, and decide what cells to keep.
            '''),
            code_cell('''\
import anndata as ad

adata_b1 = ad.read_h5ad('/home/zichende/scRNA-ssl-uncertainty/data/processed/GSE96583_batch1_qc_shared_annotated_singlets.h5ad')
adata_b2 = ad.read_h5ad('/home/zichende/scRNA-ssl-uncertainty/data/processed/GSE96583_batch2_qc_shared_annotated_singlets.h5ad')

print('batch1 shape:', adata_b1.shape)
print('batch2 shape:', adata_b2.shape)
print('batch1 samples:', adata_b1.obs['sample'].value_counts().to_dict())
print('batch2 samples:', adata_b2.obs['sample'].value_counts().to_dict())
            '''),
            md_cell('''\
## Wrangling and Preprocessing Decisions
The current processed benchmark reflects these choices:
- load and standardize the raw matrices from GEO
- attach available metadata to each cell
- restrict analyses to shared genes where cross-batch comparisons are valid
- apply QC-oriented filtering already captured in the processed `.h5ad` files
- keep only singlets to avoid doublets contaminating downstream cell-type classification

These are not cosmetic steps. They directly determine whether cross-batch evaluation is meaningful.
            '''),
            code_cell('''\
label_col_b1 = 'cell.type' if 'cell.type' in adata_b1.obs.columns else 'cell'
label_col_b2 = 'cell.type' if 'cell.type' in adata_b2.obs.columns else 'cell'

label_counts = pd.DataFrame({
    'batch1': adata_b1.obs[label_col_b1].value_counts(),
    'batch2': adata_b2.obs[label_col_b2].value_counts(),
}).fillna(0).astype(int)
label_counts
            '''),
            md_cell('''\
## Meaningful Insights from Sections 1 and 2
- `GSE96583` is accessible, labeled, and already contains both batch shift and condition shift, so it is enough for a clean milestone dataset.
- The dataset is class-imbalanced: broad immune populations dominate while dendritic cells and megakaryocytes are rare.
- The raw file layout is fragmented, so wrangling is necessary before any valid analysis.
- Singlet filtering is important because downstream classification assumes one biological cell state per observation.
            '''),
            md_cell('''\
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


def new_slide():
    fig = plt.figure(figsize=(13.33, 7.5), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.035, 0.87), 0.16, 0.05, boxstyle='round,pad=0.01,rounding_size=0.01',
                                facecolor=NAVY, edgecolor='none'))
    return fig, ax


def add_header(ax, section: str, title: str, subtitle: str | None = None) -> None:
    ax.text(0.05, 0.895, section.upper(), color='white', fontsize=11, fontweight='bold', va='center')
    ax.text(0.05, 0.82, title, color=NAVY, fontsize=26, fontweight='bold', va='top')
    if subtitle:
        ax.text(0.05, 0.775, subtitle, color=MUTED, fontsize=12.5, va='top')


def add_bullets(ax, bullets: list[str], x: float, y: float, width: int = 48, line_gap: float = 0.088) -> None:
    cursor = y
    for bullet in bullets:
        wrapped = wrap_lines(bullet, width)
        ax.text(x, cursor, f'• {wrapped}', fontsize=15, color=INK, va='top')
        cursor -= line_gap + 0.02 * wrapped.count('\n')


def add_card(ax, xy: tuple[float, float], wh: tuple[float, float], title: str, body: str, color: str) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02',
                                facecolor='white', edgecolor='none'))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.02), w, 0.02, boxstyle='round,pad=0.012,rounding_size=0.02',
                                facecolor=color, edgecolor='none'))
    ax.text(x + 0.02, y + h - 0.05, title, fontsize=14.5, fontweight='bold', color=NAVY, va='top')
    ax.text(x + 0.02, y + h - 0.10, wrap_lines(body, 28), fontsize=12.2, color=INK, va='top')


def add_footer(ax, text: str, page: int) -> None:
    ax.text(0.05, 0.035, text, fontsize=10.5, color=MUTED, va='center')
    ax.text(0.95, 0.035, str(page), fontsize=10.5, color=MUTED, va='center', ha='right')


def build_slides(files_df: pd.DataFrame, samples_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
    pdf_path = DELIV / 'MS2_AB_Sections_1_2_Slides.pdf'
    total_raw_mb = round(files_df['size_mb'].sum(), 2)
    batch1_cells = int(samples_df.loc[samples_df['batch'] == 'batch1', 'cells'].sum())
    batch2_cells = int(samples_df.loc[samples_df['batch'] == 'batch2', 'cells'].sum())
    rare = labels_df.assign(total=labels_df['batch1'] + labels_df['batch2']).sort_values('total').iloc[0]['cell_type']

    with PdfPages(pdf_path) as pdf:
        fig, ax = new_slide()
        add_header(
            ax,
            'Section 1',
            'Why We Chose GSE96583',
            'A single well-labeled PBMC benchmark is stronger for MS2 than a broader but poorly grounded multi-dataset scope.',
        )
        add_bullets(ax, [
            'We rescoped to one dataset so the milestone centers on defensible wrangling rather than uncertain external labels.',
            'GSE96583 is fully public on GEO, script-downloadable, and already includes usable cell-level PBMC annotations.',
            'It contains both batch shift (batch1 vs batch2) and condition shift (ctrl vs stim inside batch2).',
        ], x=0.06, y=0.68, width=52)
        add_card(ax, (0.64, 0.56), (0.27, 0.16), 'Dataset facts',
                 f'Raw download size: {total_raw_mb} MB\nProcessed singlets: {batch1_cells + batch2_cells:,} cells', TEAL)
        add_card(ax, (0.64, 0.34), (0.27, 0.16), 'Why it fits MS2',
                 'Direct access, real batch structure, and immediate labels make it suitable for EDA and baseline modeling.', GOLD)
        add_footer(ax, 'Introduction, motivation, and dataset choice', 1)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = new_slide()
        add_header(
            ax,
            'Section 2',
            'The Raw GEO Release Is Fragmented',
            'The source data arrive as separate matrices, gene files, and metadata tables rather than one clean analysis table.',
        )
        add_bullets(ax, [
            'Batch1 is assembled from GSM2560245_A, GSM2560246_B, and GSM2560247_C.',
            'Batch2 is assembled from GSM2560248_2.1 and GSM2560249_2.2, which also enable ctrl-versus-stim comparisons.',
            'This raw layout forces explicit reconciliation of genes, samples, and annotations before modeling.',
        ], x=0.06, y=0.67, width=45)
        img = plt.imread(FIG / 'raw_file_sizes.png')
        ax_img = fig.add_axes([0.56, 0.16, 0.37, 0.58])
        ax_img.imshow(img)
        ax_img.axis('off')
        ax.text(0.56, 0.13, 'Raw file inventory and size profile from the GEO download.', fontsize=10.5, color=MUTED)
        add_footer(ax, 'Raw file structure and access pattern', 2)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = new_slide()
        add_header(
            ax,
            'Section 2',
            'Wrangling Pipeline',
            'Each preprocessing step protects validity of later cross-batch comparisons.',
        )
        steps = [
            ('1. Download + inspect', 'Pull GEO matrices and metadata, then verify the file layout before loading.'),
            ('2. Standardize + annotate', 'Load matrices into AnnData and attach per-cell metadata.'),
            ('3. Align gene space', 'Restrict to shared genes so batch-to-batch comparisons use the same feature set.'),
            ('4. Keep singlets only', 'Remove likely doublets to avoid mixed-cell labels in classification.'),
        ]
        x_positions = [0.06, 0.29, 0.52, 0.75]
        for (title, body), x in zip(steps, x_positions):
            add_card(ax, (x, 0.36), (0.17, 0.22), title, body, [NAVY, TEAL, GOLD, '#A23E48'][x_positions.index(x)])
        img = plt.imread(FIG / 'sample_counts.png')
        ax_img = fig.add_axes([0.08, 0.10, 0.36, 0.20])
        ax_img.imshow(img)
        ax_img.axis('off')
        ax.text(0.48, 0.18, wrap_lines(
            'After filtering and harmonization, batch1 contributes 11,432 singlets and batch2 contributes 24,250 singlets.',
            42
        ), fontsize=13, color=INK, va='center')
        add_footer(ax, 'Preprocessing steps and immediate outputs', 3)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = new_slide()
        add_header(
            ax,
            'Section 2',
            'What Wrangling Revealed',
            'The processed benchmark is usable, but it is clearly imbalanced across cell types.',
        )
        add_bullets(ax, [
            f'Batch1 ends with {batch1_cells:,} singlets and batch2 ends with {batch2_cells:,} singlets.',
            'Major immune populations are preserved and ready for downstream classification.',
            f'Class imbalance is substantial: abundant T-cell and monocyte groups dominate, while {rare} and other rare classes remain small.',
            'This is why later sections should emphasize macro-F1 and per-class behavior rather than accuracy alone.',
        ], x=0.06, y=0.66, width=49)
        img = plt.imread(FIG / 'label_counts.png')
        ax_img = fig.add_axes([0.57, 0.16, 0.34, 0.58])
        ax_img.imshow(img)
        ax_img.axis('off')
        add_footer(ax, 'Bridge to EDA and baseline evaluation', 4)
        pdf.savefig(fig)
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
