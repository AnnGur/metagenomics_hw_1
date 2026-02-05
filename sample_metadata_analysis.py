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
    # Set SampleID as index in metadata if not already
    if 'SampleID' in metadata_df.columns:
        metadata_df = metadata_df.set_index('SampleID')
    
    # Merge metadata with abundance data
    merged_data = abundance_df.merge(metadata_df[['Gender']], 
                                   left_index=True, 
                                   right_index=True,
                                   how='inner')
    
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

def main():
    """Main analysis function."""
    print("Starting metadata analysis...\n")
    
    # Load data
    metadata_df, abundance_df = load_and_prepare_data()
    
    # Create output directory for plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Sex Distribution Analysis
    print("=== Sex Distribution Analysis ===")
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
        print("\nTop 5 most prevalent species:")
        for species, prev in results['top_prevalent'].items():
            print(f"{species}: {prev:.3f}")
        
        print("\nTop 5 most abundant species:")
        for species, abund in results['top_abundant'].items():
            print(f"{species}: {abund:.3f}")
    
    # 4. Summary and Limitations
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
        
        f.write("\n4. Study Limitations\n")
        f.write("Potential limitations in representativeness:\n")
        f.write("- Age range and distribution\n")
        f.write("- Sex ratio\n")
        f.write("- Missing metadata variables\n")
        f.write("- Potential sampling bias\n")
    
    print(f"\nAnalysis complete. Results saved in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()