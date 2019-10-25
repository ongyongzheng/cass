## CASS: CROSS ADVERSARIAL SOURCE SEPARATION VIA AUTOENCODER

This folder contains the codes used to generate results used in the paper. Provided sample codes are for the MUSDB18 and FECGSYNDB dataset.

CASS: CROSS ADVERSARIAL SOURCE SEPARATION VIA AUTOENCODER

## Usage

To generate datasets, download the dataset, then use correspondingly labelled ipynb files in /samples directory by updating relevant configurations for save directory of datasets.

Next, update the main_xxx.py file in root directory with the samples directory, as well as the save directory. For ease of use, the results folder can be used to save the models and results.

run the python code using:

python filename.py |& tee filename_results.txt

## Preparing the files for Bach10

A sample main file, main_cass_ws_bach10.py is included for reference to preparing the models for 4 component dataset of bach10. Cross reference with the other main_xxx files can be used to prepare the relevant files for Bach10 dataset.

## Note:

Please cite the below paper when using this repository.

CASS: CROSS ADVERSARIAL SOURCE SEPARATION VIA AUTOENCODER

## Dataset references

Bach10

Zhiyao Duan and Bryan A Pardo, “Soundprism: An online system for score-informed source separation of music audio,” IEEE Journal on Selected Topics in Signal Processing, vol. 5, no. 6, pp. 1205–1215, 10 2011.

MUSDB18

Zafar Rafii, Antoine Liutkus, Fabian-Robert St¨oter, Stylianos Ioannis Mimilakis, and Rachel Bittner, “The MUSDB18 corpus for music separation,” Dec. 2017.

FECGSYNDB

Joachim Behar, Fernando Andreotti, Sebastian Zaunseder, Qiao Li, Julien Oster, and Gari D Clifford, “An ECG simulator for generating maternal-foetal activity mixtures on abdominal ECG recordings,” Physiological Measurement, vol. 35, no. 8, pp. 1537–1550, jul 2014.

## Model references

SVSGAN

Zhe-Cheng Fan, Yen-Lin Lai, and Jyh-Shing Roger Jang, “SVSGAN: singing voice separation via generative adversarial network,” CoRR, 2017.

SCSS

Laxmi Pandey, Anurendra Kumar, and Vinay Namboodiri, “Monoaural audio source separation using variational autoencoders,” 09 2018, pp. 3489–3493.

U-Net

Nicola Montecchio Rachel Bittner Aparna Kumar Tillman Weyde1 Andreas Jansson, Eric Humphrey, “Singing voice separation with deep u-net convolutional networks,” in Proceedings of the 18th ISMIR Conference, Suzhou, China, 2017.