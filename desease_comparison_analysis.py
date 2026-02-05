import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

INPUT_DIR = 'input'
OUTPUT_DIR = 'output/desease_comparison_analysis.py'

def load_data():
    """Load all necessary data files."""
    print("Loading data files...")
    
    # Load metadata
    metadata_df = pd.read_csv(f'{INPUT_DIR}/metadata_table.csv')
    
    # Load abundance table
    abundance_df = pd.read_csv(f'{INPUT_DIR}/abundance_table.csv', index_col=0)
    
    # Load taxonomy
    taxonomy_df = pd.read_excel(f'{INPUT_DIR}/mag_data_taxa.xlsx')
    
    # Convert DiseaseStatus to categorical
    metadata_df['DiseaseStatus'] = metadata_df['DiseaseStatus'].map({0: 'Healthy', 1: 'Diseased'})
    metadata_df['Gender'] = metadata_df['Gender'].map({0: 'Male', 1: 'Female'})
    
    # Handle sample ID matching (MAG names vs metadata)
    print("\nChecking sample ID alignment...")
    metadata_samples = set(metadata_df['SampleID'])
    abundance_samples = set(abundance_df.index)
    exact_matches = metadata_samples.intersection(abundance_samples)
    
    if len(exact_matches) == 0:
        print("  - Extracting base sample IDs from MAG names...")
        abundance_df['BaseSampleID'] = abundance_df.index.str.split('_metabat').str[0]
        abundance_df['BaseSampleID'] = abundance_df['BaseSampleID'].str.split('_').str[0]
        abundance_df = abundance_df.set_index('BaseSampleID')
        print(f"  ✓ Using base sample ID matching")
    
    print(f"Loaded: {len(metadata_df)} metadata samples, {len(abundance_df)} abundance samples")
    
    return metadata_df, abundance_df, taxonomy_df

def aggregate_by_taxonomy(abundance_df, taxonomy_df, level='phylum'):
    """Aggregate species abundances at a specific taxonomic level."""
    species_to_tax = dict(zip(taxonomy_df['species'], taxonomy_df[level]))
    
    abundance_renamed = abundance_df.copy()
    abundance_renamed.columns = [species_to_tax.get(col, 'Unknown') for col in abundance_renamed.columns]
    
    aggregated = abundance_renamed.groupby(abundance_renamed.columns, axis=1).sum()
    
    return aggregated

def merge_with_metadata(abundance_df, metadata_df):
    """Merge abundance data with metadata, handling sample ID matching."""
    
    if 'SampleID' in metadata_df.columns:
        metadata_indexed = metadata_df.set_index('SampleID')
    else:
        metadata_indexed = metadata_df
    
    merged = abundance_df.merge(
        metadata_indexed[['DiseaseStatus', 'Gender', 'Age', 'BMI']], 
        left_index=True, 
        right_index=True,
        how='inner'
    )
    
    print(f"  - Merged samples: {len(merged)}")
    print(f"    • Healthy: {(merged['DiseaseStatus'] == 'Healthy').sum()}")
    print(f"    • Diseased: {(merged['DiseaseStatus'] == 'Diseased').sum()}")
    
    return merged

def compare_alpha_diversity(merged_data, output_dir):
    """Compare alpha diversity between disease states."""
    
    print("\nCalculating alpha diversity metrics...")
    
    # Get only abundance columns (exclude metadata)
    metadata_cols = ['DiseaseStatus', 'Gender', 'Age', 'BMI']
    abundance_cols = [col for col in merged_data.columns if col not in metadata_cols]
    abundance_only = merged_data[abundance_cols]
    
    # Calculate diversity metrics
    diversity_metrics = pd.DataFrame(index=merged_data.index)
    
    # Species richness (number of species present)
    diversity_metrics['Richness'] = (abundance_only > 0).sum(axis=1)
    
    # Shannon diversity
    def calculate_shannon(row):
        row = row[row > 0]
        if len(row) == 0:
            return 0
        p = row / row.sum()
        return -np.sum(p * np.log(p))
    
    diversity_metrics['Shannon'] = abundance_only.apply(calculate_shannon, axis=1)
    
    # Simpson diversity (1 - sum(p^2))
    def calculate_simpson(row):
        row = row[row > 0]
        if len(row) == 0:
            return 0
        p = row / row.sum()
        return 1 - np.sum(p ** 2)
    
    diversity_metrics['Simpson'] = abundance_only.apply(calculate_simpson, axis=1)
    
    # Add disease status
    diversity_metrics['DiseaseStatus'] = merged_data['DiseaseStatus']
    
    # Statistical tests
    print("\nAlpha diversity comparisons:")
    stats_results = {}
    
    for metric in ['Richness', 'Shannon', 'Simpson']:
        healthy = diversity_metrics[diversity_metrics['DiseaseStatus'] == 'Healthy'][metric]
        diseased = diversity_metrics[diversity_metrics['DiseaseStatus'] == 'Diseased'][metric]
        
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = mannwhitneyu(healthy, diseased, alternative='two-sided')
        
        stats_results[metric] = {
            'healthy_mean': healthy.mean(),
            'healthy_std': healthy.std(),
            'diseased_mean': diseased.mean(),
            'diseased_std': diseased.std(),
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        print(f"\n{metric}:")
        print(f"  Healthy: {healthy.mean():.3f} ± {healthy.std():.3f}")
        print(f"  Diseased: {diseased.mean():.3f} ± {diseased.std():.3f}")
        print(f"  P-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(['Richness', 'Shannon', 'Simpson']):
        ax = axes[idx]
        
        # Boxplot
        sns.boxplot(data=diversity_metrics, x='DiseaseStatus', y=metric, ax=ax, palette='Set2')
        sns.swarmplot(data=diversity_metrics, x='DiseaseStatus', y=metric, ax=ax, 
                     color='black', alpha=0.5, size=3)
        
        # Add significance
        p_val = stats_results[metric]['p_value']
        if p_val < 0.05:
            y_max = diversity_metrics[metric].max()
            y_pos = y_max * 1.1
            ax.plot([0, 1], [y_pos, y_pos], 'k-', linewidth=1)
            sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
            ax.text(0.5, y_pos * 1.02, sig_text, ha='center', fontsize=14)
        
        ax.set_title(f'{metric}\n(p = {p_val:.4f})')
        ax.set_ylabel(metric)
        ax.set_xlabel('')
    
    plt.suptitle('Alpha Diversity Comparison: Healthy vs. Diseased', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/alpha_diversity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return diversity_metrics, stats_results

def compare_taxonomic_composition(merged_data, taxonomy_df, output_dir):
    """Compare taxonomic composition between disease states at multiple levels."""
    
    print("\n" + "="*80)
    print("TAXONOMIC COMPOSITION COMPARISON")
    print("="*80)
    
    metadata_cols = ['DiseaseStatus', 'Gender', 'Age', 'BMI']
    abundance_cols = [col for col in merged_data.columns if col not in metadata_cols]
    
    results = {}
    
    for level in ['phylum', 'class', 'order', 'family', 'genus']:
        print(f"\nAnalyzing {level} level...")
        
        # Aggregate to taxonomic level
        abundance_only = merged_data[abundance_cols]
        agg_abundance = aggregate_by_taxonomy(abundance_only, taxonomy_df, level)
        
        # Add disease status back
        agg_with_metadata = agg_abundance.copy()
        agg_with_metadata['DiseaseStatus'] = merged_data['DiseaseStatus'].values
        
        # Calculate mean abundance by disease state
        healthy_mean = agg_with_metadata[agg_with_metadata['DiseaseStatus'] == 'Healthy'].drop('DiseaseStatus', axis=1).mean()
        diseased_mean = agg_with_metadata[agg_with_metadata['DiseaseStatus'] == 'Diseased'].drop('DiseaseStatus', axis=1).mean()
        
        # Statistical testing for each taxon
        taxa_stats = []
        
        for taxon in agg_abundance.columns:
            healthy_values = agg_with_metadata[agg_with_metadata['DiseaseStatus'] == 'Healthy'][taxon]
            diseased_values = agg_with_metadata[agg_with_metadata['DiseaseStatus'] == 'Diseased'][taxon]
            
            # Mann-Whitney U test
            if len(healthy_values) > 0 and len(diseased_values) > 0:
                statistic, p_value = mannwhitneyu(healthy_values, diseased_values, alternative='two-sided')
                
                # Calculate fold change (log2)
                h_mean = healthy_values.mean() + 1e-10
                d_mean = diseased_values.mean() + 1e-10
                log2_fc = np.log2(d_mean / h_mean)
                
                taxa_stats.append({
                    'taxon': taxon,
                    'healthy_mean': healthy_mean[taxon],
                    'diseased_mean': diseased_mean[taxon],
                    'log2_fold_change': log2_fc,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
        
        taxa_df = pd.DataFrame(taxa_stats)
        taxa_df = taxa_df.sort_values('p_value')
        
        results[level] = {
            'stats': taxa_df,
            'healthy_mean': healthy_mean,
            'diseased_mean': diseased_mean
        }
        
        # Print top significant taxa
        sig_taxa = taxa_df[taxa_df['significant']]
        print(f"  - Significant taxa: {len(sig_taxa)} / {len(taxa_df)}")
        
        if len(sig_taxa) > 0:
            print(f"  - Top 5 differentially abundant:")
            for _, row in sig_taxa.head(5).iterrows():
                direction = "↑" if row['log2_fold_change'] > 0 else "↓"
                print(f"    {direction} {row['taxon']}: log2FC = {row['log2_fold_change']:.2f}, p = {row['p_value']:.4f}")
    
    return results

def create_comparison_barplots(comparison_results, output_dir):
    """Create bar plots comparing taxonomic composition between disease states."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    levels = ['phylum', 'class', 'order', 'family']
    
    for idx, level in enumerate(levels):
        ax = axes[idx]
        
        healthy_mean = comparison_results[level]['healthy_mean'].sort_values(ascending=False).head(10)
        diseased_mean = comparison_results[level]['diseased_mean'][healthy_mean.index]
        
        x = np.arange(len(healthy_mean))
        width = 0.35
        
        ax.bar(x - width/2, healthy_mean * 100, width, label='Healthy', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, diseased_mean * 100, width, label='Diseased', color='coral', alpha=0.8)
        
        ax.set_ylabel('Mean Relative Abundance (%)')
        ax.set_title(f'Top 10 {level.capitalize()}s by Disease State')
        ax.set_xticks(x)
        ax.set_xticklabels(healthy_mean.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/taxonomic_comparison_barplots.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_volcano_plot(comparison_results, level='phylum', output_dir='output'):
    """Create volcano plot showing differentially abundant taxa."""
    
    stats_df = comparison_results[level]['stats'].copy()
    
    # Add -log10(p-value)
    stats_df['-log10_p'] = -np.log10(stats_df['p_value'] + 1e-300)
    
    # Define significance thresholds
    p_threshold = 0.05
    fc_threshold = 1.0  # log2 fold change threshold
    
    # Categorize points
    stats_df['category'] = 'Not significant'
    stats_df.loc[(stats_df['p_value'] < p_threshold) & (stats_df['log2_fold_change'] > fc_threshold), 'category'] = 'Enriched in Diseased'
    stats_df.loc[(stats_df['p_value'] < p_threshold) & (stats_df['log2_fold_change'] < -fc_threshold), 'category'] = 'Enriched in Healthy'
    stats_df.loc[(stats_df['p_value'] < p_threshold) & (abs(stats_df['log2_fold_change']) <= fc_threshold), 'category'] = 'Significant but small FC'
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        'Not significant': 'lightgray',
        'Significant but small FC': 'yellow',
        'Enriched in Healthy': 'steelblue',
        'Enriched in Diseased': 'coral'
    }
    
    for category in colors.keys():
        data = stats_df[stats_df['category'] == category]
        ax.scatter(data['log2_fold_change'], data['-log10_p'], 
                  c=colors[category], label=category, alpha=0.6, s=50)
    
    # Add threshold lines
    ax.axhline(y=-np.log10(p_threshold), color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=fc_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=-fc_threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Label significant points
    sig_data = stats_df[(stats_df['p_value'] < p_threshold) & (abs(stats_df['log2_fold_change']) > fc_threshold)]
    for _, row in sig_data.head(10).iterrows():
        ax.annotate(row['taxon'], 
                   xy=(row['log2_fold_change'], row['-log10_p']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Log2 Fold Change (Diseased / Healthy)')
    ax.set_ylabel('-Log10 P-value')
    ax.set_title(f'Volcano Plot: Differential Abundance at {level.capitalize()} Level')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/volcano_plot_{level}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats_df

def perform_pca(merged_data, taxonomy_df, output_dir):
    """Perform PCA and visualize sample clustering by disease state."""
    
    print("\nPerforming PCA analysis...")
    
    metadata_cols = ['DiseaseStatus', 'Gender', 'Age', 'BMI']
    abundance_cols = [col for col in merged_data.columns if col not in metadata_cols]
    
    # Get abundance data
    abundance_data = merged_data[abundance_cols]
    
    # Filter low abundance species (present in < 10% of samples)
    prevalence = (abundance_data > 0).sum() / len(abundance_data)
    abundant_species = prevalence[prevalence > 0.1].index
    abundance_filtered = abundance_data[abundant_species]
    
    print(f"  - Using {len(abundant_species)} abundant species (> 10% prevalence)")
    
    # Standardize
    scaler = StandardScaler()
    abundance_scaled = scaler.fit_transform(abundance_filtered)
    
    # PCA
    pca = PCA(n_components=min(10, abundance_scaled.shape[1]))
    pca_coords = pca.fit_transform(abundance_scaled)
    
    # Create DataFrame
    pca_df = pd.DataFrame(
        pca_coords[:, :5],
        columns=[f'PC{i+1}' for i in range(5)],
        index=merged_data.index
    )
    pca_df['DiseaseStatus'] = merged_data['DiseaseStatus'].values
    pca_df['Gender'] = merged_data['Gender'].values
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PC1 vs PC2
    for status in ['Healthy', 'Diseased']:
        data = pca_df[pca_df['DiseaseStatus'] == status]
        axes[0].scatter(data['PC1'], data['PC2'], 
                       label=status, alpha=0.6, s=80)
    
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].set_title('PCA: Disease State')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Scree plot
    axes[1].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
               pca.explained_variance_ratio_ * 100)
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Variance Explained (%)')
    axes[1].set_title('Scree Plot')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  - PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
    print(f"  - PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")
    
    return pca_df, pca

def create_heatmap(merged_data, taxonomy_df, level='phylum', output_dir='output'):
    """Create heatmap showing taxonomic profiles across samples grouped by disease state."""
    
    print(f"\nCreating heatmap at {level} level...")
    
    metadata_cols = ['DiseaseStatus', 'Gender', 'Age', 'BMI']
    abundance_cols = [col for col in merged_data.columns if col not in metadata_cols]
    
    # Aggregate to taxonomic level
    abundance_only = merged_data[abundance_cols]
    agg_abundance = aggregate_by_taxonomy(abundance_only, taxonomy_df, level)
    
    # Keep top 20 most abundant taxa
    mean_abundance = agg_abundance.mean()
    top_taxa = mean_abundance.sort_values(ascending=False).head(20).index
    agg_top = agg_abundance[top_taxa]
    
    # Sort samples by disease status
    sample_order = merged_data.sort_values('DiseaseStatus').index
    agg_sorted = agg_top.loc[sample_order]
    
    # Create color bar for disease status
    disease_colors = merged_data.loc[sample_order, 'DiseaseStatus'].map({
        'Healthy': 'steelblue',
        'Diseased': 'coral'
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(agg_sorted.T * 100, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    ax.set_yticks(range(len(top_taxa)))
    ax.set_yticklabels(top_taxa)
    ax.set_xlabel('Samples (sorted by Disease Status)')
    ax.set_ylabel(f'{level.capitalize()}')
    ax.set_title(f'Taxonomic Heatmap: Top 20 {level.capitalize()}s')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Abundance (%)', rotation=270, labelpad=20)
    
    # Add disease status bar at top
    for i, (idx, color) in enumerate(zip(sample_order, disease_colors)):
        ax.plot([i-0.5, i+0.5], [-1, -1], color=color, linewidth=3)
    
    # Add legend for disease status
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Healthy'),
        Patch(facecolor='coral', label='Diseased')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_{level}.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(alpha_results, comparison_results, output_dir):
    """Generate comprehensive report of disease state comparisons."""
    
    report_path = f'{output_dir}/disease_comparison_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DISEASE STATE COMPARISON ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # 1. Alpha Diversity Results
        f.write("1. ALPHA DIVERSITY COMPARISON\n")
        f.write("-"*80 + "\n\n")
        
        diversity_metrics, stats = alpha_results
        
        for metric in ['Richness', 'Shannon', 'Simpson']:
            f.write(f"{metric}:\n")
            f.write(f"  Healthy: {stats[metric]['healthy_mean']:.3f} ± {stats[metric]['healthy_std']:.3f}\n")
            f.write(f"  Diseased: {stats[metric]['diseased_mean']:.3f} ± {stats[metric]['diseased_std']:.3f}\n")
            f.write(f"  P-value: {stats[metric]['p_value']:.4f}")
            
            if stats[metric]['significant']:
                f.write(" *SIGNIFICANT*")
                if stats[metric]['diseased_mean'] > stats[metric]['healthy_mean']:
                    f.write(" (Higher in diseased)")
                else:
                    f.write(" (Lower in diseased)")
            f.write("\n\n")
        
        # 2. Taxonomic Composition Results
        f.write("\n2. DIFFERENTIAL ABUNDANCE ANALYSIS\n")
        f.write("-"*80 + "\n\n")
        
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            stats_df = comparison_results[level]['stats']
            sig_taxa = stats_df[stats_df['significant']]
            
            f.write(f"{level.upper()}:\n")
            f.write(f"  Total taxa tested: {len(stats_df)}\n")
            f.write(f"  Significantly different: {len(sig_taxa)} ({len(sig_taxa)/len(stats_df)*100:.1f}%)\n")
            
            if len(sig_taxa) > 0:
                enriched_diseased = sig_taxa[sig_taxa['log2_fold_change'] > 0]
                enriched_healthy = sig_taxa[sig_taxa['log2_fold_change'] < 0]
                
                f.write(f"    - Enriched in diseased: {len(enriched_diseased)}\n")
                f.write(f"    - Enriched in healthy: {len(enriched_healthy)}\n")
                
                f.write(f"\n  Top 5 most differentially abundant:\n")
                for _, row in sig_taxa.head(5).iterrows():
                    direction = "Diseased" if row['log2_fold_change'] > 0 else "Healthy"
                    f.write(f"    • {row['taxon']}: log2FC = {row['log2_fold_change']:.2f}, ")
                    f.write(f"p = {row['p_value']:.4f} ({direction})\n")
            
            f.write("\n")
        
        # 3. Key Findings
        f.write("\n3. KEY FINDINGS AND BIOLOGICAL INTERPRETATION\n")
        f.write("-"*80 + "\n\n")
        
        # Diversity patterns
        f.write("a) Alpha Diversity Patterns:\n")
        shannon_diff = stats['Shannon']['diseased_mean'] - stats['Shannon']['healthy_mean']
        if stats['Shannon']['significant']:
            if shannon_diff > 0:
                f.write("   The diseased group shows SIGNIFICANTLY HIGHER Shannon diversity, ")
                f.write("suggesting increased microbial diversity. This could indicate:\n")
                f.write("   - Loss of dominant beneficial species\n")
                f.write("   - Overgrowth of opportunistic taxa\n")
                f.write("   - Disrupted microbial community structure\n")
            else:
                f.write("   The diseased group shows SIGNIFICANTLY LOWER Shannon diversity, ")
                f.write("suggesting decreased microbial diversity. This could indicate:\n")
                f.write("   - Dominance by few pathogenic species\n")
                f.write("   - Loss of beneficial diversity\n")
                f.write("   - Simplified microbial ecosystem\n")
        else:
            f.write("   No significant difference in Shannon diversity between groups.\n")
            f.write("   Disease may affect specific taxa rather than overall diversity.\n")
        
        # Taxonomic patterns
        f.write("\nb) Taxonomic Composition Changes:\n")
        
        phylum_stats = comparison_results['phylum']['stats']
        sig_phyla = phylum_stats[phylum_stats['significant']]
        
        if len(sig_phyla) > 0:
            f.write(f"   {len(sig_phyla)} phyla show significant differences between disease states.\n\n")
            
            for _, row in sig_phyla.head(3).iterrows():
                if row['log2_fold_change'] > 0:
                    f.write(f"   - {row['taxon']} is ENRICHED in diseased samples ")
                    f.write(f"({row['diseased_mean']*100:.1f}% vs {row['healthy_mean']*100:.1f}%)\n")
                else:
                    f.write(f"   - {row['taxon']} is DEPLETED in diseased samples ")
                    f.write(f"({row['diseased_mean']*100:.1f}% vs {row['healthy_mean']*100:.1f}%)\n")
        else:
            f.write("   No significant phylum-level differences detected.\n")
            f.write("   Disease-associated changes may occur at lower taxonomic levels.\n")
        
        # 4. Clinical Implications
        f.write("\n4. CLINICAL AND BIOLOGICAL IMPLICATIONS\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Based on the observed taxonomic shifts:\n\n")
        
        # Check for common dysbiosis markers
        phylum_stats_dict = phylum_stats.set_index('taxon').to_dict('index')
        
        proteobacteria_enriched = False
        if 'Proteobacteria' in phylum_stats_dict:
            if phylum_stats_dict['Proteobacteria']['log2_fold_change'] > 0:
                proteobacteria_enriched = True
        
        if proteobacteria_enriched:
            f.write("• PROTEOBACTERIA ENRICHMENT DETECTED:\n")
            f.write("  This is a common marker of gut dysbiosis and inflammation.\n")
            f.write("  Clinical relevance: Associated with IBD, metabolic disorders, and systemic inflammation.\n\n")
        
        f.write("• MICROBIOME-DISEASE RELATIONSHIP:\n")
        f.write("  The detected differences suggest that disease status is associated with\n")
        f.write("  distinct microbial signatures. These changes could be:\n")
        f.write("  - Causative: Microbiome changes drive disease\n")
        f.write("  - Consequential: Disease alters microbiome\n")
        f.write("  - Confounded: Shared factors affect both\n\n")
        
        # 5. Recommendations
        f.write("\n5. RECOMMENDATIONS FOR FURTHER INVESTIGATION\n")
        f.write("-"*80 + "\n\n")
        f.write("• Validate findings with larger independent cohort\n")
        f.write("• Control for confounding factors (diet, medications, age, BMI)\n")
        f.write("• Perform functional analysis (metabolic pathways, KEGG)\n")
        f.write("• Investigate strain-level differences within significant species\n")
        f.write("• Conduct longitudinal studies to assess causality\n")
        f.write("• Consider multi-omics integration (metatranscriptomics, metabolomics)\n")
        
    print(f"\nComprehensive report saved to {report_path}")

def main():
    """Main analysis pipeline."""
    
    print("\n" + "="*80)
    print("DISEASE STATE COMPARISON ANALYSIS")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    metadata_df, abundance_df, taxonomy_df = load_data()
    
    # Merge abundance with metadata
    print("\nMerging abundance data with metadata...")
    merged_data = merge_with_metadata(abundance_df, metadata_df)
    
    if len(merged_data) == 0:
        print("ERROR: No samples matched between metadata and abundance!")
        print("Please check sample ID alignment.")
        return
    
    # 1. Alpha diversity comparison
    print("\n" + "="*80)
    print("ALPHA DIVERSITY ANALYSIS")
    print("="*80)
    alpha_results = compare_alpha_diversity(merged_data, OUTPUT_DIR)
    
    # 2. Taxonomic composition comparison
    comparison_results = compare_taxonomic_composition(merged_data, taxonomy_df, OUTPUT_DIR)
    
    # 3. Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    print("\n1. Comparison bar plots...")
    create_comparison_barplots(comparison_results, OUTPUT_DIR)
    
    print("\n2. Volcano plots...")
    for level in ['phylum', 'genus']:
        create_volcano_plot(comparison_results, level, OUTPUT_DIR)
    
    print("\n3. PCA analysis...")
    perform_pca(merged_data, taxonomy_df, OUTPUT_DIR)
    
    print("\n4. Heatmaps...")
    for level in ['phylum', 'genus']:
        create_heatmap(merged_data, taxonomy_df, level, OUTPUT_DIR)
    
    # 4. Generate report
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)
    generate_report(alpha_results, comparison_results, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - alpha_diversity_comparison.png")
    print("  - taxonomic_comparison_barplots.png")
    print("  - volcano_plot_phylum.png")
    print("  - volcano_plot_genus.png")
    print("  - pca_analysis.png")
    print("  - heatmap_phylum.png")
    print("  - heatmap_genus.png")
    print("  - disease_comparison_report.txt")

if __name__ == "__main__":
    main()