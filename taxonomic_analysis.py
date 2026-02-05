import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import colorsys
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

INPUT_DIR = 'input'
OUTPUT_DIR = 'output/taxonomic_analysis/'

def load_data():
    """Load abundance and taxonomic data."""
    print("Loading data files...")
    
    # Load abundance table
    abundance_df = pd.read_csv(f'{INPUT_DIR}/abundance_table.csv', index_col=0)
    
    # Load taxonomic table
    taxonomy_df = pd.read_excel(f'{INPUT_DIR}/mag_data_taxa.xlsx')
    
    print(f"Loaded {len(abundance_df)} samples with {len(abundance_df.columns)} species")
    print(f"Loaded taxonomic information for {len(taxonomy_df)} species")
    
    return abundance_df, taxonomy_df

def aggregate_by_taxonomy(abundance_df, taxonomy_df, level='phylum'):
    """
    Aggregate species abundances at a specific taxonomic level.
    
    Parameters:
    -----------
    abundance_df : DataFrame
        Species abundance table (samples x species)
    taxonomy_df : DataFrame
        Taxonomic annotations (species x taxonomic levels)
    level : str
        Taxonomic level to aggregate at (phylum, class, order, family, genus)
    
    Returns:
    --------
    DataFrame : Aggregated abundances at specified taxonomic level
    """
    # Create mapping from species to taxonomic level
    species_to_tax = dict(zip(taxonomy_df['species'], taxonomy_df[level]))
    
    # Rename columns in abundance table to taxonomic level
    abundance_renamed = abundance_df.copy()
    abundance_renamed.columns = [species_to_tax.get(col, 'Unknown') for col in abundance_renamed.columns]
    
    # Sum abundances for each taxonomic group
    aggregated = abundance_renamed.groupby(abundance_renamed.columns, axis=1).sum()
    
    return aggregated

def calculate_taxonomic_statistics(abundance_df, taxonomy_df):
    """Calculate summary statistics at each taxonomic level."""
    stats = {}
    
    taxonomic_levels = ['phylum', 'class', 'order', 'family', 'genus']
    
    for level in taxonomic_levels:
        print(f"\nAnalyzing {level} level...")
        
        # Aggregate abundances
        agg_df = aggregate_by_taxonomy(abundance_df, taxonomy_df, level)
        
        # Calculate statistics
        mean_abundance = agg_df.mean(axis=0).sort_values(ascending=False)
        prevalence = (agg_df > 0).sum(axis=0) / len(agg_df) * 100
        
        stats[level] = {
            'count': len(agg_df.columns),
            'mean_abundance': mean_abundance,
            'prevalence': prevalence,
            'top_10': mean_abundance.head(10),
            'aggregated_data': agg_df
        }
        
        print(f"  - Number of {level}s: {len(agg_df.columns)}")
        print(f"  - Top {level}: {mean_abundance.index[0]} ({mean_abundance.iloc[0]:.2%})")
    
    return stats

def plot_phylum_dominance(stats, output_dir):
    """Create visualizations showing phylum-level dominance patterns."""
    
    phylum_stats = stats['phylum']
    mean_abundance = phylum_stats['mean_abundance']
    
    # 1. Bar plot of top phyla
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top 10 phyla by mean abundance
    top_phyla = mean_abundance.head(10)
    axes[0].bar(range(len(top_phyla)), top_phyla.values * 100, color='steelblue')
    axes[0].set_xticks(range(len(top_phyla)))
    axes[0].set_xticklabels(top_phyla.index, rotation=45, ha='right')
    axes[0].set_ylabel('Mean Relative Abundance (%)')
    axes[0].set_title('Top 10 Most Abundant Phyla')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Cumulative abundance
    cumsum = mean_abundance.sort_values(ascending=False).cumsum() * 100
    axes[1].plot(range(1, len(cumsum) + 1), cumsum.values, marker='o', linewidth=2)
    axes[1].axhline(y=80, color='r', linestyle='--', label='80% threshold')
    axes[1].axhline(y=95, color='orange', linestyle='--', label='95% threshold')
    axes[1].set_xlabel('Number of Phyla (ranked by abundance)')
    axes[1].set_ylabel('Cumulative Relative Abundance (%)')
    axes[1].set_title('Cumulative Abundance Curve - Phylum Level')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/phylum_dominance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate dominance metrics
    top_3_abundance = mean_abundance.head(3).sum()
    top_5_abundance = mean_abundance.head(5).sum()
    n_phyla_80pct = (cumsum <= 80).sum() + 1
    n_phyla_95pct = (cumsum <= 95).sum() + 1
    
    return {
        'top_3_abundance': top_3_abundance,
        'top_5_abundance': top_5_abundance,
        'n_phyla_80pct': n_phyla_80pct,
        'n_phyla_95pct': n_phyla_95pct
    }

def create_stacked_barplot(stats, output_dir):
    """Create stacked bar plots at different taxonomic ranks."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    levels = ['phylum', 'class', 'order', 'family']
    
    for idx, level in enumerate(levels):
        agg_data = stats[level]['aggregated_data']
        mean_abundance = stats[level]['mean_abundance']
        
        # Get top taxa (keep top 10, group rest as "Other")
        top_taxa = mean_abundance.head(10).index.tolist()
        
        plot_data = agg_data.copy()
        other_cols = [col for col in plot_data.columns if col not in top_taxa]
        if other_cols:
            plot_data['Other'] = plot_data[other_cols].sum(axis=1)
            plot_data = plot_data.drop(columns=other_cols)
        
        # Calculate mean for plotting
        plot_means = plot_data.mean(axis=0).sort_values(ascending=False)
        
        # Create stacked bar (single bar showing composition)
        bottom = 0
        colors = plt.cm.tab20(np.linspace(0, 1, len(plot_means)))
        
        for i, (taxon, value) in enumerate(plot_means.items()):
            axes[idx].bar(0, value * 100, bottom=bottom * 100, 
                         color=colors[i], label=taxon, width=0.6)
            bottom += value
        
        axes[idx].set_ylabel('Relative Abundance (%)')
        axes[idx].set_title(f'Taxonomic Composition - {level.capitalize()} Level')
        axes[idx].set_xlim(-0.5, 0.5)
        axes[idx].set_xticks([])
        axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stacked_barplots_taxonomy.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_sunburst_diagram(abundance_df, taxonomy_df, output_dir):
    """Create sunburst plot with simplified logic."""
    
    print("Loading data...")
    
    # Load files
    abundance_df = pd.read_csv(f'{INPUT_DIR}/abundance_table.csv', index_col=0)
    taxonomy_df = pd.read_excel(f'{INPUT_DIR}/mag_data_taxa.xlsx')
    
    # Calculate mean abundance
    species_abundance = abundance_df.mean(axis=0)
    
    print(f"Total species: {len(species_abundance)}")
    
    # Create taxonomy table with abundance
    tax_data = taxonomy_df.copy()
    tax_data['abundance'] = tax_data['species'].map(species_abundance)
    
    # Remove missing
    tax_data = tax_data.dropna(subset=['abundance'])
    
    # Fill NaN taxonomy
    for col in ['phylum', 'class', 'order', 'family', 'genus']:
        if col in tax_data.columns:
            tax_data[col] = tax_data[col].fillna(f'Unknown')
    
    # Keep top 50 species
    top_species = species_abundance.nlargest(50).index
    tax_data = tax_data[tax_data['species'].isin(top_species)]
    
    print(f"Using top {len(tax_data)} species")
    
    # Prepare data in format Plotly expects
    # Create full path strings
    tax_data['path'] = (
        tax_data['phylum'].astype(str) + '/' +
        tax_data['class'].astype(str) + '/' +
        tax_data['order'].astype(str) + '/' +
        tax_data['family'].astype(str) + '/' +
        tax_data['genus'].astype(str) + '/' +
        tax_data['species'].astype(str)
    )
    
    # Create figure using px.sunburst (simpler!)
    fig = px.sunburst(
        tax_data,
        path=['phylum', 'class', 'order', 'family', 'genus', 'species'],
        values='abundance',
        title='Taxonomic Hierarchy - Sunburst Plot (Top 50 Species)',
        color='phylum',
        hover_data=['abundance'],
        height=1000,
        width=1000
    )
    
    fig.update_traces(
        textfont_size=10,
        marker=dict(line=dict(color='white', width=1))
    )
    
    fig.update_layout(
        margin=dict(t=50, l=0, r=0, b=0)
    )
    
    # Save HTML
    print(f"\nSaving outputs to {OUTPUT_DIR}/")
    fig.write_html(f'{OUTPUT_DIR}/sunburst_simple.html')
    print("✓ Saved: sunburst_simple.html")
    
    # Try to save PNG
    try:
        fig.write_image(f'{OUTPUT_DIR}/sunburst_simple.png', width=1200, height=1200, scale=2)
        print("✓ Saved: sunburst_simple.png")
    except:
        print("⚠ Could not save PNG (install: pip install -U kaleido)")
        create_matplotlib_plot(tax_data)

def create_matplotlib_plot(tax_data):
    """Create matplotlib-based hierarchical visualization."""
    
    print("\nCreating matplotlib alternative...")
    
    # Nested pie charts for different levels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Phylum level
    phylum_abund = tax_data.groupby('phylum')['abundance'].sum().sort_values(ascending=False)
    if len(phylum_abund) > 10:
        other = phylum_abund[10:].sum()
        phylum_abund = pd.concat([phylum_abund[:10], pd.Series({'Other': other})])
    
    axes[0].pie(phylum_abund.values, labels=phylum_abund.index, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Phylum Level', fontweight='bold')
    
    # Class level (top phylum only)
    top_phylum = phylum_abund.index[0]
    class_data = tax_data[tax_data['phylum'] == top_phylum]
    class_abund = class_data.groupby('class')['abundance'].sum().sort_values(ascending=False).head(10)
    
    axes[1].pie(class_abund.values, labels=class_abund.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title(f'Class Level\n(within {top_phylum})', fontweight='bold')
    
    # Genus level (top class)
    if len(class_abund) > 0:
        top_class = class_abund.index[0]
        genus_data = tax_data[(tax_data['phylum'] == top_phylum) & (tax_data['class'] == top_class)]
        genus_abund = genus_data.groupby('genus')['abundance'].sum().sort_values(ascending=False).head(10)
        
        axes[2].pie(genus_abund.values, labels=genus_abund.index, autopct='%1.1f%%', startangle=90)
        axes[2].set_title(f'Genus Level\n(within {top_class})', fontweight='bold')
    
    plt.suptitle('Taxonomic Composition - Hierarchical View', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sunburst_matplotlib.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved: sunburst_matplotlib.png")

def create_sankey_diagram_enhanced(abundance_df, taxonomy_df, output_dir, top_n_species=5):
    """
    Create enhanced colorful Sankey diagram showing all taxonomic levels.
    Follows the path of the top N most abundant species through the full hierarchy.
    
    Parameters:
    -----------
    abundance_df : DataFrame
        Species abundance table
    taxonomy_df : DataFrame
        Taxonomic annotations
    output_dir : str
        Output directory path
    top_n_species : int
        Number of top species to visualize (default: 5)
    """
    
    print(f"\nCreating enhanced Sankey diagram for top {top_n_species} species...")
    
    # Calculate mean abundance for each species
    species_abundance = abundance_df.mean(axis=0)
    
    # Get top N species
    top_species_list = species_abundance.nlargest(top_n_species).index.tolist()
    print(f"  - Top {top_n_species} species selected:")
    for i, sp in enumerate(top_species_list, 1):
        print(f"    {i}. {sp} ({species_abundance[sp]:.4f})")
    
    # Merge with taxonomy - keep only top species
    taxonomy_with_abundance = taxonomy_df.copy()
    taxonomy_with_abundance['abundance'] = taxonomy_with_abundance['species'].map(species_abundance)
    taxonomy_with_abundance = taxonomy_with_abundance.dropna(subset=['abundance'])
    
    # Filter to top species only
    top_species_data = taxonomy_with_abundance[taxonomy_with_abundance['species'].isin(top_species_list)]
    
    # Fill missing taxonomy
    hierarchy_levels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    for level in hierarchy_levels:
        if level in top_species_data.columns:
            top_species_data[level] = top_species_data[level].fillna(f'Unknown_{level}')
            top_species_data[level] = top_species_data[level].astype(str)
    
    print(f"  - Building hierarchical paths...")
    
    # Build comprehensive label and index mappings
    labels = []
    label_to_idx = {}
    node_levels = []  # Track which level each node belongs to
    node_colors = []
    
    def generate_vibrant_colors(n, hue_start=0):
        """Generate vibrant, distinct colors using HSV."""
        colors = []
        for i in range(n):
            hue = (hue_start + i / n) % 1.0
            saturation = 0.85 + (i % 3) * 0.05  # Vary saturation
            value = 0.95 - (i % 2) * 0.1  # Vary brightness
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            rgba = f'rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.9)'
            colors.append(rgba)
        return colors
    
    # Color schemes for each level with different hue ranges
    level_color_schemes = {
        'phylum': generate_vibrant_colors(20, hue_start=0.0),    # Red-orange
        'class': generate_vibrant_colors(20, hue_start=0.17),    # Yellow-green
        'order': generate_vibrant_colors(20, hue_start=0.35),    # Green-cyan
        'family': generate_vibrant_colors(20, hue_start=0.55),   # Cyan-blue
        'genus': generate_vibrant_colors(20, hue_start=0.7),     # Blue-purple
        'species': generate_vibrant_colors(top_n_species, hue_start=0.85)  # Purple-magenta
    }
    
    # Collect all unique taxa at each level from top species
    level_taxa = {}
    for level in hierarchy_levels:
        unique_taxa = top_species_data[level].unique()
        level_taxa[level] = sorted(unique_taxa)
    
    # Create nodes for each taxon at each level
    for level in hierarchy_levels:
        color_scheme = level_color_schemes[level]
        for idx, taxon in enumerate(level_taxa[level]):
            # Create display label
            if level == 'species':
                # Truncate long species names
                display_label = taxon if len(taxon) <= 30 else taxon[:27] + "..."
            else:
                display_label = taxon
            
            labels.append(display_label)
            label_to_idx[f"{level}::{taxon}"] = len(labels) - 1
            node_levels.append(level)
            node_colors.append(color_scheme[idx % len(color_scheme)])
    
    print(f"  - Created {len(labels)} nodes across {len(hierarchy_levels)} levels")
    
    # Build links between consecutive levels
    source = []
    target = []
    value = []
    link_colors = []
    link_labels = []
    
    for level_idx in range(len(hierarchy_levels) - 1):
        current_level = hierarchy_levels[level_idx]
        next_level = hierarchy_levels[level_idx + 1]
        
        # Group by both levels and sum abundance
        grouped = top_species_data.groupby([current_level, next_level])['abundance'].sum()
        
        for (current_taxon, next_taxon), abund in grouped.items():
            current_key = f"{current_level}::{current_taxon}"
            next_key = f"{next_level}::{next_taxon}"
            
            if current_key in label_to_idx and next_key in label_to_idx:
                source_idx = label_to_idx[current_key]
                target_idx = label_to_idx[next_key]
                
                source.append(source_idx)
                target.append(target_idx)
                value.append(abund)
                
                # Create gradient effect: use source color but more transparent
                source_color = node_colors[source_idx]
                link_color = source_color.replace('0.9)', '0.4)')
                link_colors.append(link_color)
                link_labels.append(f"{current_taxon} → {next_taxon}")
    
    print(f"  - Created {len(source)} links")
    
    # Calculate node positions for better layout
    node_x = []
    node_y = []
    
    level_x_positions = np.linspace(0.05, 0.95, len(hierarchy_levels))
    
    for node_idx, level in enumerate(node_levels):
        level_idx = hierarchy_levels.index(level)
        x_pos = level_x_positions[level_idx]
        
        # Count nodes at this level
        nodes_at_level = node_levels.count(level)
        node_position_in_level = [i for i, l in enumerate(node_levels) if l == level].index(node_idx)
        
        # Spread nodes vertically
        if nodes_at_level > 1:
            y_pos = node_position_in_level / (nodes_at_level - 1)
        else:
            y_pos = 0.5
        
        node_x.append(x_pos)
        node_y.append(y_pos)
    
    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=25,
            thickness=30,
            line=dict(color="white", width=3),
            label=labels,
            color=node_colors,
            x=node_x,
            y=node_y,
            customdata=[[label, level] for label, level in zip(labels, node_levels)],
            hovertemplate='<b>%{customdata[0]}</b><br>Level: %{customdata[1]}<br>Total: %{value:.4f}<extra></extra>'
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
            customdata=link_labels,
            hovertemplate='%{customdata}<br>Flow: %{value:.4f}<extra></extra>'
        )
    )])
    
    # Enhanced layout with level labels
    annotations = []
    for idx, level in enumerate(hierarchy_levels):
        annotations.append(
            dict(
                x=level_x_positions[idx],
                y=1.05,
                text=f"<b>{level.upper()}</b>",
                showarrow=False,
                font=dict(size=13, color='#2c3e50', family='Arial Black'),
                xanchor='center'
            )
        )
    
    fig.update_layout(
        title={
            'text': f"<b>Complete Taxonomic Hierarchy</b><br>" + 
                   f"<sub>Tracing Top {top_n_species} Most Abundant Species</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=18, color='#2c3e50', family='Arial')
        },
        font=dict(size=11, family='Arial', color='#34495e'),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#f8f9fa',
        height=1000,
        width=1600,
        margin=dict(t=120, l=50, r=50, b=50),
        annotations=annotations
    )
    
    # Save as HTML
    html_path = f'{output_dir}/sankey_enhanced.html'
    fig.write_html(html_path)
    print(f"  ✓ Enhanced Sankey HTML saved to {html_path}")
    
    # Save as PNG
    try:
        png_path = f'{output_dir}/sankey_enhanced.png'
        fig.write_image(png_path, width=1600, height=1000, scale=2)
        print(f"  ✓ Enhanced Sankey PNG saved to {png_path}")
    except Exception as e:
        print(f"  ⚠ Could not save PNG: {e}")
        print(f"    Install kaleido: pip install -U kaleido")

def plot_diversity_across_levels(stats, output_dir):
    """Plot taxonomic diversity metrics across different levels."""
    
    levels = ['phylum', 'class', 'order', 'family', 'genus']
    n_taxa = [stats[level]['count'] for level in levels]
    
    # Calculate Shannon diversity at each level
    shannon_div = []
    for level in levels:
        mean_abundance = stats[level]['mean_abundance']
        # Shannon diversity: -sum(p * log(p))
        p = mean_abundance / mean_abundance.sum()
        shannon = -np.sum(p * np.log(p + 1e-10))
        shannon_div.append(shannon)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Number of taxa
    axes[0].plot(levels, n_taxa, marker='o', linewidth=2, markersize=10)
    axes[0].set_ylabel('Number of Taxa')
    axes[0].set_xlabel('Taxonomic Level')
    axes[0].set_title('Taxonomic Richness Across Levels')
    axes[0].grid(alpha=0.3)
    
    # Shannon diversity
    axes[1].plot(levels, shannon_div, marker='s', linewidth=2, markersize=10, color='coral')
    axes[1].set_ylabel('Shannon Diversity Index')
    axes[1].set_xlabel('Taxonomic Level')
    axes[1].set_title('Taxonomic Evenness Across Levels')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/diversity_across_levels.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return shannon_div

def generate_report(stats, dominance_metrics, shannon_div, output_dir):
    """Generate comprehensive text report of findings."""
    
    report_path = f'{output_dir}/taxonomic_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TAXONOMIC ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # 1. Summary Statistics
        f.write("1. SUMMARY STATISTICS BY TAXONOMIC LEVEL\n")
        f.write("-"*80 + "\n\n")
        
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            f.write(f"{level.upper()}:\n")
            f.write(f"  Total number: {stats[level]['count']}\n")
            f.write(f"  Top 5 most abundant:\n")
            for taxon, abundance in stats[level]['top_10'].head(5).items():
                prev = stats[level]['prevalence'][taxon]
                f.write(f"    - {taxon}: {abundance*100:.2f}% (present in {prev:.1f}% of samples)\n")
            f.write("\n")
        
        # 2. Phylum-Level Dominance Analysis
        f.write("\n2. PHYLUM-LEVEL DOMINANCE ANALYSIS\n")
        f.write("-"*80 + "\n\n")
        
        phylum_stats = stats['phylum']['mean_abundance']
        
        f.write(f"Total number of phyla detected: {len(phylum_stats)}\n\n")
        
        f.write(f"Dominance Metrics:\n")
        f.write(f"  - Top 3 phyla account for: {dominance_metrics['top_3_abundance']*100:.2f}% of total abundance\n")
        f.write(f"  - Top 5 phyla account for: {dominance_metrics['top_5_abundance']*100:.2f}% of total abundance\n")
        f.write(f"  - Number of phyla needed for 80% of abundance: {dominance_metrics['n_phyla_80pct']}\n")
        f.write(f"  - Number of phyla needed for 95% of abundance: {dominance_metrics['n_phyla_95pct']}\n\n")
        
        # 3. Diversity Analysis
        f.write("\n3. DIVERSITY ANALYSIS\n")
        f.write("-"*80 + "\n\n")
        
        levels = ['phylum', 'class', 'order', 'family', 'genus']
        f.write("Shannon Diversity Index by taxonomic level:\n")
        for level, shannon in zip(levels, shannon_div):
            f.write(f"  - {level.capitalize()}: {shannon:.3f}\n")
        
    print(f"\nComprehensive report saved to {report_path}")

def main():
    """Main analysis pipeline."""
    
    print("\n" + "="*80)
    print("TAXONOMIC ANALYSIS PIPELINE")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    abundance_df, taxonomy_df = load_data()
    
    # Calculate summary statistics
    print("\n" + "="*80)
    print("CALCULATING TAXONOMIC STATISTICS")
    print("="*80)
    stats = calculate_taxonomic_statistics(abundance_df, taxonomy_df)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    print("\n1. Phylum dominance analysis...")
    dominance_metrics = plot_phylum_dominance(stats, OUTPUT_DIR)
    
    print("\n2. Stacked bar plots...")
    create_stacked_barplot(stats, OUTPUT_DIR)
    
    print("\n3. Diversity across taxonomic levels...")
    shannon_div = plot_diversity_across_levels(stats, OUTPUT_DIR)
    
    print("\n4. Sunburst plot (interactive)...")
    create_sunburst_diagram(abundance_df, taxonomy_df, OUTPUT_DIR)
    
    print("\n5. Sankey diagram (interactive)...")
    create_sankey_diagram_enhanced(abundance_df, taxonomy_df, OUTPUT_DIR)
    
    # Generate comprehensive report
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)
    generate_report(stats, dominance_metrics, shannon_div, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - phylum_dominance.png")
    print("  - stacked_barplots_taxonomy.png")
    print("  - diversity_across_levels.png")
    print("  - sunburst_taxonomy.html (interactive)")
    print("  - sankey_taxonomy.html (interactive)")
    print("  - taxonomic_analysis_report.txt")

if __name__ == "__main__":
    main()