#!/bin/bash

micromamba env create -f environment.yml
micromamba activate microbiome_analysis

echo "Environment 'microbiome_analysis' has been created and activated."
echo "You can now run the analysis script."