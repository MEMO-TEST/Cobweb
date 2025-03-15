# Cobweb: Enhanced Generation Coverage for Black-box Fairness Testing

## Requirements
- Python 3.9
- Ubuntu OS

Install the required packages using conda:
```bash
conda env create -f cobweb.yml
```
## Usage
### Model Testing
```bash
python experiment.py
python experimentGAN.py
```
### Measure Individual Fairness
```bash
python MeasureFairness.py
```
### Experimental Results
Results such as training logs, model weights, and visualizations are saved in the results/ directory. More detailed experimental analysis can be found in the evaulate.ipynb.

## Directory Structure
```
- baseline/ # baselines
- Cobweb/ # source code folder
- data_preprocess/ # data preprocess
- datasets/ # raw data
- limictgan/ # GAN model for LIMI
- model/ # model structure
- model_info/ #model weights
- result/ # experiment results
- utils/ # universal functions
- experiment.py # Cobweb
- experiment.py # CobwebGAN
- evaluate.ipynb # Coverage and Validity Test
- GAN.py # our GAN method
- MeasureFairness.py # measure individual fairness rate 
```