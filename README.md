# Cobweb: Enhanced Generation Coverage for Black-box Fairness Testing

## Abstract
Black-box fairness testing reveals the potential discriminatory behavior of released artificial intelligence software by searching for individual discriminatory instances, which is used to ensure credibility in critical areas such as smart education and talent recruitment. However, existing approaches usually suffer from limited instance space coverage due to their reliance on local neighborhood searches around predefined seed instances. To address this issue, we propose Cobweb, a novel hard-label black-box fairness testing framework that synergizes genetic algorithms with spatial diffusion mechanisms. Our framework employs two key innovations: (1) spatial uniform initialization to construct an initial population that maximizes instance diversity through pairwise distance optimization, and (2) a dual-fitness selection strategy that balances exploitation (selecting instances demonstrating discrimination) and exploration (prioritizing under-sampled regions of the instance space). Extensive experiments on four industry-standard tabular datasets (Bank, Credit, Census, and Meps), showed that Cobweb and its variant Cobweb-GAN outperform existing methods in terms of effectiveness (+106\%), efficiency (+86\%), and instance space coverage (+39\%). Furthermore, models retrained using Cobweb-generated discriminatory instances exhibit a 40\% reduction in individual fairness violation rates compared to baselines.

## Requirements
- Python 3.8+

Install the required packages using conda:
- conda env create -f cobweb.yml

## Usage
### Model Testing
> python experimentGAN.py
### Experimental Results
Results such as training logs, model weights, and visualizations are saved in the results/ directory. More detailed experimental analysis can be found in the evaulate.ipynb.

