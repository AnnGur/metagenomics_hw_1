# Metagenomics Analysis Pipeline

A comprehensive bioinformatics pipeline for analyzing metagenomic data, including taxonomic profiling, abundance analysis, and sample metadata comparison.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Output Files](#output-files)
- [Analysis Scripts](#analysis-scripts)

## Overview

This pipeline performs multiple analyses on metagenomic data:

- **Taxonomic Analysis**: Generates visualizations and reports on taxonomic composition
- **Abundance Data Analysis**: Analyzes microbial abundance patterns across samples
- **Sample Metadata Analysis**: Correlates sample metadata with taxonomic and abundance data
- **Disease Comparison Analysis**: Compares microbiome composition between disease states

## Requirements

- **micromamba** or **conda** (for environment management)
- Python 3.9+

## Installation

### 1. Install micromamba (if not already installed)

```bash
# On macOS
brew install micromamba

# On Linux
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ~/
```

### 2. Set up the environment

```bash
source scripts/setup.sh
```

This script will:
- Create a new conda environment called `microbiome_analysis`
- Install all required dependencies (pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, etc.)
- Activate the environment

## Usage

### Quick Start

Run all analyses with a single command:

```bash
# Activate the environment
micromamba activate microbiome_analysis

# Run all analysis scripts
python abundance_data_analysis.py
python sample_metadata_analysis.py
python taxonomic_analysis.py
python desease_comparison_analysis.py
```

### Individual Analyses

#### Abundance Data Analysis
```bash
micromamba activate microbiome_analysis
python abundance_data_analysis.py
```
Analyzes microbial abundance patterns across samples.

#### Sample Metadata Analysis
```bash
micromamba activate microbiome_analysis
python sample_metadata_analysis.py
```
Examines relationships between sample metadata and microbiome composition.

#### Taxonomic Analysis
```bash
micromamba activate microbiome_analysis
python taxonomic_analysis.py
```
Generates taxonomic profiles and interactive visualizations.

#### Disease Comparison Analysis
```bash
micromamba activate microbiome_analysis
python desease_comparison_analysis.py
```
Compares microbiome composition between disease states.

## Project Structure

```
metagenomics_hw_1/
├── README.md                              # This file
├── environment.yml                        # Conda environment specification
├── scripts/
│   └── setup.sh                          # Environment setup script
├── input/
│   ├── abundance_table.csv               # Microbial abundance data
│   └── metadata_table.csv                # Sample metadata
├── output/
│   ├── abundance_data_analysis/          # Abundance analysis results
│   ├── sample_metadata_analysis/         # Metadata correlation results
│   ├── taxonomic_analysis/               # Taxonomic profiles and visualizations
│   └── desease_comparison_analysis/      # Disease comparison results
├── abundance_data_analysis.py            # Abundance analysis script
├── sample_metadata_analysis.py           # Metadata analysis script
├── taxonomic_analysis.py                 # Taxonomic analysis script
├── desease_comparison_analysis.py        # Disease comparison script
└── create_sunburst_diagram.py            # Utility for creating sunburst diagrams
```

## Output Files

### Abundance Data Analysis
- `output/abundance_data_analysis/analysis_results.txt` - Summary statistics and findings

### Sample Metadata Analysis
- `output/sample_metadata_analysis/metadata_analysis_summary.txt` - Metadata correlation results

### Taxonomic Analysis
- `output/taxonomic_analysis/taxonomic_analysis_report.txt` - Detailed taxonomic report
- `output/taxonomic_analysis/sunburst_simple.html` - Interactive sunburst visualization
- `output/taxonomic_analysis/sankey_taxonomy.html` - Interactive Sankey diagram

### Disease Comparison Analysis
- `output/desease_comparison_analysis/` - Comparative analysis results

## Analysis Scripts

### Dependencies

All scripts require the packages specified in `environment.yml`:

- **pandas**: Data manipulation and CSV handling
- **numpy**: Numerical computations
- **matplotlib & seaborn**: Static visualizations
- **plotly**: Interactive visualizations (HTML output)
- **scikit-learn**: Statistical analyses and machine learning
- **openpyxl & adjusttext**: Data formatting and text adjustment

### Input Data Format

- **abundance_table.csv**: Rows = taxa, Columns = samples, Values = abundance counts
- **metadata_table.csv**: Rows = samples, Columns = metadata attributes

## Notes

- All outputs are saved to the `output/` directory organized by analysis type
- Interactive HTML visualizations can be opened in any web browser
- Analysis reports are saved as plain text files for easy review