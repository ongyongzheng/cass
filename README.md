## CASS: CROSS ADVERSARIAL SOURCE SEPARATION VIA AUTOENCODER

This repository contains the codes used to generate results used in the above paper. Above are sample codes used for producing results used in the MUSDB18 [1] and FECGSYNDB [2] dataset. A sample code, named main_cass_ws_bach10.py provides a sample for reference to preparing the code to run the model for Bach10 [3] dataset.

CASS: CROSS ADVERSARIAL SOURCE SEPARATION VIA AUTOENCODER (Insert link for CASS paper)

## Model Structure

![alt text](https://github.com/ongyongzheng/cass/blob/master/yongz1.png "Framework of a single CASS component")

![alt text](https://github.com/ongyongzheng/cass/blob/master/yongz2.png "Framework of 2 CASS components")

## Usage Instructions

### Preparation of Datasets

Download links for datasets can be found in [1], [2], [3]. Extract the downloaded datasets and update relevant configuration files for the dataset directory and save locations, using the corresponding Jupyter notebook in /samples directory.

#### Preparing the files for Bach10

A sample main file, main_cass_ws_bach10.py is included for reference to preparing the models for 4 component dataset of bach10. Cross reference with the other main_xxx files can be used to prepare the relevant files for Bach10 dataset.

### Preparing configurations

The below table lists the models and their corresponding main files. Here, M1, M2 and M3 are as explained in the paper.

| Model Name    | Main File          |
| ------------- | ------------------ |
| M1            | main_cass_ws.py    |
| M2            | main_cass.py       |
| M3            | main_cass_unet2.py |
| SCSS [4]      | main_vae.py        |
| U-Net [5]     | main_unet.py       |
| SVSGAN [6]    | main_svsgan.py     |

Update the corresponding main_xxx.py file in root directory with the saved processed samples directory, as well as the save directory. For ease of use, the /results folder can be used to save the models and results.

Run the python code using:

python filename.py |& tee filename_results.txt

## Note:

Please cite the below paper when using this repository.

CASS: CROSS ADVERSARIAL SOURCE SEPARATION VIA AUTOENCODER (Insert link for CASS paper)

## References

[1] Zafar Rafii, Antoine Liutkus, Fabian-Robert St¨oter, Stylianos Ioannis Mimilakis, and Rachel Bittner, “[The MUSDB18 corpus for music separation](https://sigsep.github.io/datasets/musdb.html),” Dec. 2017.

[2] Joachim Behar, Fernando Andreotti, Sebastian Zaunseder, Qiao Li, Julien Oster, and Gari D Clifford, “[An ECG simulator for generating maternal-foetal activity mixtures on abdominal ECG recordings](https://github.com/fernandoandreotti/fecgsyn),” Physiological Measurement, vol. 35, no. 8, pp. 1537–1550, jul 2014.

[3] Zhiyao Duan and Bryan A Pardo, “[Soundprism: An online system for score-informed source separation of music audio](http://music.cs.northwestern.edu/data/Bach10_Dataset_Description.pdf),” IEEE Journal on Selected Topics in Signal Processing, vol. 5, no. 6, pp. 1205–1215, 10 2011.

[4] Laxmi Pandey, Anurendra Kumar, and Vinay Namboodiri, “Monoaural audio source separation using variational autoencoders,” 09 2018, pp. 3489–3493.

[5] Nicola Montecchio Rachel Bittner Aparna Kumar Tillman Weyde1 Andreas Jansson, Eric Humphrey, “Singing voice separation with deep u-net convolutional networks,” in Proceedings of the 18th ISMIR Conference, Suzhou, China, 2017.

[6] Laxmi Pandey, Anurendra Kumar, and Vinay Namboodiri, “Monoaural audio source separation using variational autoencoders,” 09 2018, pp. 3489–3493.