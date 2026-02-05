import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

INPUT_DIR = 'input'
OUTPUT_DIR = 'output/sample_metadata_analysis/'

def load_and_prepare_data():
    """Load and prepare all necessary data."""
    # Load data
    metadata_df = pd.read_csv(f'{INPUT_DIR}/metadata_table.csv')
    abundance_df = pd.read_csv(f'{INPUT_DIR}/abundance_table.csv', index_col=0)
    
    print(f"Loaded metadata: {len(metadata_df)} samples")
    print(f"Loaded abundance: {len(abundance_df)} samples")
    
    # Check sample ID matching
    print("\nChecking sample ID alignment...")
    metadata_samples = set(metadata_df['SampleID'])
    abundance_samples = set(abundance_df.index)
    exact_matches = metadata_samples.intersection(abundance_samples)
    
    print(f"  - Exact matches: {len(exact_matches)}")
    
    # If no exact matches, try extracting base sample ID from abundance table
    if len(exact_matches) == 0:
        print("  - No exact matches found. Attempting base ID extraction...")
        
        # Extract base ID (everything before '_metabat' or first underscore)
        abundance_df['BaseSampleID'] = abundance_df.index.str.split('_metabat').str[0]
        abundance_df['BaseSampleID'] = abundance_df['BaseSampleID'].str.split('_').str[0]
        
        # Check if base IDs match metadata
        base_matches = set(abundance_df['BaseSampleID']).intersection(metadata_samples)
        print(f"  - Base ID matches: {len(base_matches)}")
        
        if len(base_matches) > 0:
            print("  ✓ Using base sample ID matching")
            # Set BaseSampleID as index for merging
            abundance_df = abundance_df.set_index('BaseSampleID')
        else:
            print("  ✗ WARNING: No matches found even with base ID extraction!")
    
    # Convert Gender to categorical (0=Male, 1=Female)
    metadata_df['Gender'] = metadata_df['Gender'].map({0: 'Male', 1: 'Female'})
    
    # Convert DiseaseStatus to categorical
    metadata_df['DiseaseStatus'] = metadata_df['DiseaseStatus'].map({0: 'Healthy', 1: 'Diseased'})
    
    return metadata_df, abundance_df

def analyze_sex_distribution(metadata_df):
    """Analyze and visualize sex distribution."""
    # Count sex distribution
    sex_counts = metadata_df['Gender'].value_counts()
    
    # Create pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%')
    plt.title('Sex Distribution in Cohort')
    plt.savefig(f'{OUTPUT_DIR}/sex_distribution.png')
    plt.close()
    
    return sex_counts

def analyze_age_distribution(metadata_df):
    """Analyze and visualize age distribution."""
    # Overall age distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=metadata_df, x='Age', bins=20)
    plt.title('Age Distribution in Cohort')
    plt.savefig(f'{OUTPUT_DIR}/age_distribution.png')
    plt.close()
    
    # Age distribution by sex
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=metadata_df, x='Gender', y='Age')
    plt.title('Age Distribution by Sex')
    plt.savefig(f'{OUTPUT_DIR}/age_by_sex_boxplot.png')
    plt.close()
    
    # Calculate statistics
    age_stats = metadata_df['Age'].describe()
    age_by_sex = metadata_df.groupby('Gender')['Age'].describe()
    
    # Statistical test for age difference between sexes
    male_ages = metadata_df[metadata_df['Gender'] == 'Male']['Age']
    female_ages = metadata_df[metadata_df['Gender'] == 'Female']['Age']
    
    # Use t-test if both groups have data
    if len(male_ages) > 0 and len(female_ages) > 0:
        t_stat, p_value = stats.ttest_ind(male_ages, female_ages)
        test_name = "Independent t-test"
    else:
        p_value = None
        test_name = "Not applicable (insufficient data)"
    
    # Create population pyramid
    create_population_pyramid(metadata_df, OUTPUT_DIR)
    
    return {
        'age_stats': age_stats,
        'age_by_sex': age_by_sex,
        'statistical_test': {
            'test_name': test_name,
            'p_value': p_value
        }
    }

def create_population_pyramid(metadata_df, output_dir):
    """Create population pyramid showing age distribution by gender."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bins for age groups
    bins = np.arange(0, metadata_df['Age'].max() + 10, 10)
    
    # Separate data by gender
    male_data = metadata_df[metadata_df['Gender'] == 'Male']['Age']
    female_data = metadata_df[metadata_df['Gender'] == 'Female']['Age']
    
    # Create histograms
    male_hist = np.histogram(male_data, bins=bins)[0]
    female_hist = np.histogram(female_data, bins=bins)[0]
    
    # Plot pyramid (males on left as negative, females on right as positive)
    ax.barh(bins[:-1], -male_hist, height=9, align='edge', color='lightblue', label='Male')
    ax.barh(bins[:-1], female_hist, height=9, align='edge', color='pink', label='Female')
    
    # Set labels
    ax.set_xlabel('Count')
    ax.set_ylabel('Age')
    ax.set_title('Population Pyramid')
    ax.legend()
    
    # Add center line
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Fix x-axis to show absolute values
    max_count = max(male_hist.max(), female_hist.max())
    ax.set_xlim(-max_count - 5, max_count + 5)
    
    # Create custom x-axis labels with absolute values
    ticks = ax.get_xticks()
    ax.set_xticklabels([abs(int(x)) for x in ticks])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/population_pyramid.png', bbox_inches='tight')
    plt.close()

def analyze_species_by_sex(metadata_df, abundance_df):
    """Analyze most prevalent and abundant species by sex."""
    
    print("\n  Merging metadata with abundance data...")
    
    # Set SampleID as index in metadata if not already
    if 'SampleID' in metadata_df.columns:
        metadata_indexed = metadata_df.set_index('SampleID')
    else:
        metadata_indexed = metadata_df
    
    # Merge metadata with abundance data
    # Use outer join first to see what's happening
    merged_data = abundance_df.merge(metadata_indexed[['Gender']], 
                                   left_index=True, 
                                   right_index=True,
                                   how='inner')
    
    print(f"    - Samples after merge: {len(merged_data)}")
    
    if len(merged_data) == 0:
        print("    ✗ WARNING: No samples matched between metadata and abundance!")
        print("    → Returning empty results")
        return {
            'Male': {'top_prevalent': pd.Series(dtype=float), 'top_abundant': pd.Series(dtype=float)},
            'Female': {'top_prevalent': pd.Series(dtype=float), 'top_abundant': pd.Series(dtype=float)}
        }
    
    # Check what we merged
    male_count = (merged_data['Gender'] == 'Male').sum()
    female_count = (merged_data['Gender'] == 'Female').sum()
    print(f"    - Male samples: {male_count}")
    print(f"    - Female samples: {female_count}")
    
    # Function to get top species for each metric
    def get_top_species(data, metric='prevalence'):
        if metric == 'prevalence':
            return (data > 0).mean()
        else:  # abundance
            return data.mean()
    
    # Get top species by sex
    results = {}
    for sex in ['Male', 'Female']:
        sex_data = merged_data[merged_data['Gender'] == sex].drop('Gender', axis=1)
        
        if len(sex_data) > 0:  # Only calculate if there are samples
            results[sex] = {
                'top_prevalent': get_top_species(sex_data, 'prevalence').nlargest(5),
                'top_abundant': get_top_species(sex_data, 'abundance').nlargest(5)
            }
        else:
            results[sex] = {
                'top_prevalent': pd.Series(dtype=float),
                'top_abundant': pd.Series(dtype=float)
            }
    
    return results

def calculate_alpha_diversity(abundance_df, metadata_df):
    """
    Calculate alpha diversity metrics for each sample.
    
    Alpha diversity measures the diversity within individual samples:
    - Richness: Simply counts how many different species are present (presence/absence)
    - Shannon Index: Accounts for both richness and evenness - how evenly distributed species are
    - Simpson Index: Probability that two randomly selected individuals belong to different species
    
    Higher values = more diverse communities
    """
    
    print("\n  Calculating alpha diversity metrics...")
    
    # Align samples between abundance and metadata
    if 'SampleID' in metadata_df.columns:
        metadata_indexed = metadata_df.set_index('SampleID')
    else:
        metadata_indexed = metadata_df
    
    # Get common samples
    common_samples = abundance_df.index.intersection(metadata_indexed.index)
    
    if len(common_samples) == 0:
        print("    ✗ WARNING: No matching samples for diversity calculation!")
        return None
    
    abundance_aligned = abundance_df.loc[common_samples]
    metadata_aligned = metadata_indexed.loc[common_samples]
    
    print(f"    - Analyzing {len(common_samples)} samples")
    
    # Initialize diversity dataframe
    diversity_df = pd.DataFrame(index=common_samples)
    
    # 1. RICHNESS (Species count)
    # Simply counts how many species are present (abundance > 0)
    diversity_df['Richness'] = (abundance_aligned > 0).sum(axis=1)
    
    # 2. SHANNON DIVERSITY INDEX
    # H = -Σ(p_i * ln(p_i))
    # where p_i is the proportion of species i
    # Ranges from 0 (no diversity) to ln(S) where S is total species
    # Interpretation: Higher = more diverse and even community
    def shannon_diversity(row):
        """
        Shannon Index measures both richness and evenness.
        - Accounts for how many species AND how evenly distributed they are
        - More sensitive to rare species than Simpson
        - Typical range in microbiomes: 1.5-4.0
        """
        row = row[row > 0]  # Remove zeros
        if len(row) == 0:
            return 0
        proportions = row / row.sum()
        return -np.sum(proportions * np.log(proportions))
    
    diversity_df['Shannon'] = abundance_aligned.apply(shannon_diversity, axis=1)
    
    # 3. SIMPSON DIVERSITY INDEX
    # D = 1 - Σ(p_i^2)
    # where p_i is the proportion of species i
    # Ranges from 0 to 1
    # Interpretation: Higher = more diverse, represents probability that two 
    # randomly selected individuals are from different species
    def simpson_diversity(row):
        """
        Simpson Index emphasizes dominant species.
        - Less sensitive to rare species than Shannon
        - Easier to interpret: probability two random picks are different species
        - Typical range in microbiomes: 0.7-0.95
        """
        row = row[row > 0]  # Remove zeros
        if len(row) == 0:
            return 0
        proportions = row / row.sum()
        return 1 - np.sum(proportions ** 2)
    
    diversity_df['Simpson'] = abundance_aligned.apply(simpson_diversity, axis=1)
    
    # Add metadata
    diversity_df['Gender'] = metadata_aligned['Gender'].values
    diversity_df['DiseaseStatus'] = metadata_aligned.get('DiseaseStatus', 'Unknown')
    diversity_df['Age'] = metadata_aligned['Age'].values
    diversity_df['BMI'] = metadata_aligned['BMI'].values
    
    print(f"    - Mean Richness: {diversity_df['Richness'].mean():.1f} species")
    print(f"    - Mean Shannon: {diversity_df['Shannon'].mean():.3f}")
    print(f"    - Mean Simpson: {diversity_df['Simpson'].mean():.3f}")
    
    return diversity_df

def visualize_diversity_metrics(diversity_df, output_dir):
    """Create visualizations for diversity metrics."""
    
    print("\n  Creating diversity visualizations...")
    
    if diversity_df is None or len(diversity_df) == 0:
        print("    ✗ No diversity data to visualize")
        return
    
    # Create comprehensive diversity figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    metrics = ['Richness', 'Shannon', 'Simpson']
    
    # Row 1: Distribution histograms
    for idx, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[0, idx])
        sns.histplot(data=diversity_df, x=metric, bins=20, kde=True, ax=ax, color='steelblue')
        ax.set_title(f'{metric} Distribution', fontweight='bold')
        ax.set_xlabel(metric)
        ax.set_ylabel('Count')
        ax.grid(alpha=0.3)
    
    # Row 2: By Gender
    for idx, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[1, idx])
        
        if 'Gender' in diversity_df.columns and diversity_df['Gender'].notna().any():
            sns.boxplot(data=diversity_df, x='Gender', y=metric, ax=ax, palette='Set2')
            sns.swarmplot(data=diversity_df, x='Gender', y=metric, ax=ax, 
                         color='black', alpha=0.3, size=3)
            
            # Statistical test
            male = diversity_df[diversity_df['Gender'] == 'Male'][metric].dropna()
            female = diversity_df[diversity_df['Gender'] == 'Female'][metric].dropna()
            
            if len(male) > 0 and len(female) > 0:
                _, p_val = stats.mannwhitneyu(male, female, alternative='two-sided')
                sig_text = f'p = {p_val:.3f}'
                if p_val < 0.05:
                    sig_text += ' *'
                ax.text(0.5, 0.95, sig_text, transform=ax.transAxes, 
                       ha='center', va='top', fontsize=9, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f'{metric} by Gender', fontweight='bold')
        ax.set_xlabel('')
        ax.grid(alpha=0.3)
    
    # Row 3: Correlations with Age and BMI
    ax1 = fig.add_subplot(gs[2, 0])
    for metric in metrics:
        if 'Age' in diversity_df.columns:
            ax1.scatter(diversity_df['Age'], diversity_df[metric], 
                       alpha=0.5, s=50, label=metric)
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Diversity Value')
    ax1.set_title('Diversity vs Age', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[2, 1])
    for metric in metrics:
        if 'BMI' in diversity_df.columns:
            ax2.scatter(diversity_df['BMI'], diversity_df[metric], 
                       alpha=0.5, s=50, label=metric)
    ax2.set_xlabel('BMI')
    ax2.set_ylabel('Diversity Value')
    ax2.set_title('Diversity vs BMI', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Correlation heatmap
    ax3 = fig.add_subplot(gs[2, 2])
    corr_vars = ['Richness', 'Shannon', 'Simpson', 'Age', 'BMI']
    available_vars = [v for v in corr_vars if v in diversity_df.columns]
    
    if len(available_vars) > 2:
        corr_matrix = diversity_df[available_vars].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax3, square=True, cbar_kws={'shrink': 0.8})
        ax3.set_title('Correlation Matrix', fontweight='bold')
    
    plt.suptitle('Alpha Diversity Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(f'{output_dir}/alpha_diversity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved to {output_dir}/alpha_diversity_analysis.png")

def analyze_diversity_by_groups(diversity_df, output_dir):
    """Perform statistical analysis of diversity across groups."""
    
    print("\n  Analyzing diversity differences between groups...")
    
    if diversity_df is None or len(diversity_df) == 0:
        return {}
    
    results = {}
    
    # Gender comparison
    if 'Gender' in diversity_df.columns:
        print("\n  By Gender:")
        for metric in ['Richness', 'Shannon', 'Simpson']:
            male = diversity_df[diversity_df['Gender'] == 'Male'][metric].dropna()
            female = diversity_df[diversity_df['Gender'] == 'Female'][metric].dropna()
            
            if len(male) > 0 and len(female) > 0:
                stat, p_val = stats.mannwhitneyu(male, female, alternative='two-sided')
                
                print(f"    {metric}:")
                print(f"      Male: {male.mean():.3f} ± {male.std():.3f}")
                print(f"      Female: {female.mean():.3f} ± {female.std():.3f}")
                print(f"      P-value: {p_val:.4f} {'*' if p_val < 0.05 else 'ns'}")
                
                results[f'gender_{metric}'] = {
                    'male_mean': male.mean(),
                    'female_mean': female.mean(),
                    'p_value': p_val
                }
    
    # Disease status comparison
    if 'DiseaseStatus' in diversity_df.columns and diversity_df['DiseaseStatus'].notna().any():
        print("\n  By Disease Status:")
        for metric in ['Richness', 'Shannon', 'Simpson']:
            healthy = diversity_df[diversity_df['DiseaseStatus'] == 'Healthy'][metric].dropna()
            diseased = diversity_df[diversity_df['DiseaseStatus'] == 'Diseased'][metric].dropna()
            
            if len(healthy) > 0 and len(diseased) > 0:
                stat, p_val = stats.mannwhitneyu(healthy, diseased, alternative='two-sided')
                
                print(f"    {metric}:")
                print(f"      Healthy: {healthy.mean():.3f} ± {healthy.std():.3f}")
                print(f"      Diseased: {diseased.mean():.3f} ± {diseased.std():.3f}")
                print(f"      P-value: {p_val:.4f} {'*' if p_val < 0.05 else 'ns'}")
                
                results[f'disease_{metric}'] = {
                    'healthy_mean': healthy.mean(),
                    'diseased_mean': diseased.mean(),
                    'p_value': p_val
                }
    
    return results

def main():
    """Main analysis function."""
    print("Starting metadata analysis...\n")
    
    # Load data
    metadata_df, abundance_df = load_and_prepare_data()
    
    # Create output directory for plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Sex Distribution Analysis
    print("\n=== Sex Distribution Analysis ===")
    sex_counts = analyze_sex_distribution(metadata_df)
    print("Number of participants by sex:")
    for sex, count in sex_counts.items():
        print(f"{sex}: {count} ({(count/len(metadata_df)*100):.1f}%)")
    
    # 2. Age Distribution Analysis
    print("\n=== Age Distribution Analysis ===")
    age_results = analyze_age_distribution(metadata_df)
    
    print("\nOverall age statistics:")
    print(age_results['age_stats'])
    
    print("\nAge statistics by sex:")
    print(age_results['age_by_sex'])
    
    if age_results['statistical_test']['p_value'] is not None:
        print(f"\nStatistical test for age difference between sexes:")
        print(f"Test used: {age_results['statistical_test']['test_name']}")
        print(f"P-value: {age_results['statistical_test']['p_value']:.4f}")
    
    # 3. Species Analysis by Sex
    print("\n=== Species Analysis by Sex ===")
    species_results = analyze_species_by_sex(metadata_df, abundance_df)
    
    for sex, results in species_results.items():
        print(f"\n{sex} participants:")
        
        if len(results['top_prevalent']) > 0:
            print("\nTop 5 most prevalent species:")
            for species, prev in results['top_prevalent'].items():
                print(f"{species}: {prev:.3f}")
            
            print("\nTop 5 most abundant species:")
            for species, abund in results['top_abundant'].items():
                print(f"{species}: {abund:.3f}")
        else:
            print("  No data available for this group")
    
    # 4. Alpha Diversity Analysis
    print("\n=== Alpha Diversity Analysis ===")
    diversity_df = calculate_alpha_diversity(abundance_df, metadata_df)
    
    if diversity_df is not None:
        visualize_diversity_metrics(diversity_df, OUTPUT_DIR)
        diversity_stats = analyze_diversity_by_groups(diversity_df, OUTPUT_DIR)
        
        # Save diversity data
        diversity_df.to_csv(f'{OUTPUT_DIR}/alpha_diversity_data.csv')
        print(f"\n  ✓ Diversity data saved to {OUTPUT_DIR}/alpha_diversity_data.csv")
    
    # 5. Summary and Limitations
    print("\n=== Study Limitations and Representativeness ===")
    
    # Calculate basic demographic metrics
    age_range = metadata_df['Age'].max() - metadata_df['Age'].min()
    median_age = metadata_df['Age'].median()
    sex_ratio = sex_counts['Male'] / sex_counts['Female'] if 'Female' in sex_counts else float('inf')
    
    print("\nKey demographic indicators:")
    print(f"Age range: {age_range:.1f} years")
    print(f"Median age: {median_age:.1f} years")
    print(f"Sex ratio (M/F): {sex_ratio:.2f}")
    
    # Save all results to a text file
    with open(f'{OUTPUT_DIR}/metadata_analysis_summary.txt', 'w') as f:
        f.write("=== Metadata Analysis Summary ===\n\n")
        
        f.write("1. Sample Size and Sex Distribution\n")
        f.write(f"Total samples: {len(metadata_df)}\n")
        for sex, count in sex_counts.items():
            f.write(f"{sex}: {count} ({(count/len(metadata_df)*100):.1f}%)\n")
        
        f.write("\n2. Age Distribution\n")
        f.write("Overall age statistics:\n")
        f.write(str(age_results['age_stats']))
        f.write("\n\nAge statistics by sex:\n")
        f.write(str(age_results['age_by_sex']))
        
        f.write("\n\n3. Statistical Testing\n")
        if age_results['statistical_test']['p_value'] is not None:
            f.write(f"Test for age difference between sexes:\n")
            f.write(f"Test used: {age_results['statistical_test']['test_name']}\n")
            f.write(f"P-value: {age_results['statistical_test']['p_value']:.4f}\n")
        
        # Add diversity results if available
        if diversity_df is not None:
            f.write("\n\n4. Alpha Diversity Metrics\n")
            
            f.write("RICHNESS (Species Count):\n")
            f.write(f"  - Mean: {diversity_df['Richness'].mean():.1f} ± {diversity_df['Richness'].std():.1f} species\n")
            f.write(f"  - Range: {diversity_df['Richness'].min():.0f} - {diversity_df['Richness'].max():.0f}\n\n")
            
            f.write("SHANNON INDEX:\n")
            f.write(f"  - Mean: {diversity_df['Shannon'].mean():.3f} ± {diversity_df['Shannon'].std():.3f}\n")
            f.write(f"  - Range: {diversity_df['Shannon'].min():.2f} - {diversity_df['Shannon'].max():.2f}\n\n")
            
            f.write("SIMPSON INDEX:\n")
            f.write(f"  - Mean: {diversity_df['Simpson'].mean():.3f} ± {diversity_df['Simpson'].std():.3f}\n")
            f.write(f"  - Range: {diversity_df['Simpson'].min():.2f} - {diversity_df['Simpson'].max():.2f}\n\n")
            
            # Add statistical comparisons
            if 'diversity_stats' in locals() and diversity_stats:
                f.write("Diversity Differences by Groups:\n")
                
                if any('gender' in key for key in diversity_stats.keys()):
                    f.write("\n  By Gender:\n")
                    for metric in ['Richness', 'Shannon', 'Simpson']:
                        key = f'gender_{metric}'
                        if key in diversity_stats:
                            stats = diversity_stats[key]
                            f.write(f"    {metric}:\n")
                            f.write(f"      Male: {stats['male_mean']:.3f}\n")
                            f.write(f"      Female: {stats['female_mean']:.3f}\n")
                            f.write(f"      P-value: {stats['p_value']:.4f}")
                            if stats['p_value'] < 0.05:
                                f.write(" *SIGNIFICANT*")
                            f.write("\n")
                
                if any('disease' in key for key in diversity_stats.keys()):
                    f.write("\n  By Disease Status:\n")
                    for metric in ['Richness', 'Shannon', 'Simpson']:
                        key = f'disease_{metric}'
                        if key in diversity_stats:
                            stats = diversity_stats[key]
                            f.write(f"    {metric}:\n")
                            f.write(f"      Healthy: {stats['healthy_mean']:.3f}\n")
                            f.write(f"      Diseased: {stats['diseased_mean']:.3f}\n")
                            f.write(f"      P-value: {stats['p_value']:.4f}")
                            if stats['p_value'] < 0.05:
                                f.write(" *SIGNIFICANT*")
                            f.write("\n")
    
    print(f"\nAnalysis complete. Results saved in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()