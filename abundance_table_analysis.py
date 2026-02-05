import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

INPUT_DIR = 'input'
OUTPUT_DIR = 'output/abundance_data_analysis/'

def load_data():
    """Load and return the data files."""
    try:
        abundance_df = pd.read_csv(f'{INPUT_DIR}/abundance_table.csv', index_col=0)
        taxa_df = pd.read_excel(f'{INPUT_DIR}/mag_data_taxa.xlsx')
        metadata_df = pd.read_csv(f'{INPUT_DIR}/metadata_table.csv')
        return abundance_df, taxa_df, metadata_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def calculate_statistics(abundance_df):
    """Calculate basic statistics from the abundance table."""
    n_samples = abundance_df.shape[0]  # Number of rows (samples)
    n_species = abundance_df.shape[1]  # Number of columns (species)
    
    # Calculate species prevalence (proportion of samples where species is present)
    presence_absence = (abundance_df > 0).astype(int)
    species_prevalence = presence_absence.sum() / n_samples
    
    # Calculate mean relative abundance
    mean_rel_abundance = abundance_df.mean()
    
    return n_samples, n_species, species_prevalence, mean_rel_abundance

def create_species_stats(abundance_df, species_prevalence, mean_rel_abundance):
    """Create a DataFrame with species statistics."""
    return pd.DataFrame({
        'Species': abundance_df.columns,
        'Prevalence': species_prevalence,
        'Mean_Relative_Abundance': mean_rel_abundance
    })

def plot_top_prevalent_species(top_prevalent, output_dir):
    """Create bar plot of top prevalent species."""
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    ax = sns.barplot(data=top_prevalent, x='Species', y='Prevalence')
    
    # Customize the plot
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 5 Most Prevalent Species', pad=20)
    plt.xlabel('Species', labelpad=10)
    plt.ylabel('Prevalence', labelpad=10)
    
    # Add value labels on top of bars
    for i, v in enumerate(top_prevalent['Prevalence']):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_prevalent_species.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_prevalence_vs_abundance(species_stats, output_dir):
    """Create scatter plot of prevalence vs mean relative abundance with non-overlapping labels."""
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(species_stats['Prevalence'], 
               species_stats['Mean_Relative_Abundance'],
               alpha=0.6,
               c='blue',
               edgecolors='white')
    
    # Prepare annotations for top 5 species by abundance
    top_5_abundant = species_stats.nlargest(5, 'Mean_Relative_Abundance')
    texts = []
    
    for _, species in top_5_abundant.iterrows():
        texts.append(plt.text(species['Prevalence'], 
                            species['Mean_Relative_Abundance'],
                            species['Species'],
                            fontsize=8,
                            bbox=dict(facecolor='white', 
                                    edgecolor='none', 
                                    alpha=0.7,
                                    pad=2)))
    
    # Adjust text positions to avoid overlap
    adjust_text(texts,
               arrowprops=dict(arrowstyle='->', color='red', lw=0.5),
               expand_points=(1.5, 1.5),
               force_points=(0.1, 0.1))
    
    # Customize the plot
    plt.xlabel('Species Prevalence', labelpad=10)
    plt.ylabel('Mean Relative Abundance', labelpad=10)
    plt.title('Species Prevalence vs Mean Relative Abundance', pad=20)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prevalence_vs_abundance.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    plt.close()

def plot_results(species_stats, top_prevalent, output_dir):
    """Create and save all visualization plots."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create individual plots
    plot_top_prevalent_species(top_prevalent, output_dir)
    plot_prevalence_vs_abundance(species_stats, output_dir)


def main():
    # Load data
    abundance_df, taxa_df, metadata_df = load_data()
    if abundance_df is None:
        return
    
    # Calculate statistics
    n_samples, n_species, species_prevalence, mean_rel_abundance = calculate_statistics(abundance_df)
    
    # Create species statistics DataFrame
    species_stats = create_species_stats(abundance_df, species_prevalence, mean_rel_abundance)
    
    # Get top 5 species by prevalence and abundance
    top_prevalent = species_stats.nlargest(5, 'Prevalence')
    top_abundant = species_stats.nlargest(5, 'Mean_Relative_Abundance')
    
    # Create visualizations
    plot_results(species_stats, top_prevalent, OUTPUT_DIR)
    
    # Print results
    print("\nDataset Summary:")
    print(f"Number of samples: {n_samples}")
    print(f"Number of species: {n_species}")
    
    print("\nTop 5 most prevalent species:")
    print(top_prevalent.to_string())
    
    print("\nTop 5 species by mean relative abundance:")
    print(top_abundant.to_string())
    
    # Save results to file
    with open(f'{OUTPUT_DIR}/analysis_results.txt', 'w') as f:
        f.write("Dataset Summary:\n")
        f.write(f"Number of samples: {n_samples}\n")
        f.write(f"Number of species: {n_species}\n\n")
        f.write("Top 5 most prevalent species:\n")
        f.write(top_prevalent.to_string())
        f.write("\n\nTop 5 species by mean relative abundance:\n")
        f.write(top_abundant.to_string())

if __name__ == "__main__":
    main()