Audio Denoising Using Deep CNN and U-Net Architecture
This repository contains an implementation of an audio denoising system using a deep Convolutional Neural Network (CNN) and U-Net architecture. The system is designed to clean noisy audio signals by extracting the clean speech from noisy environments. The model is trained on a dataset of clean audio and noisy audio augmented with environmental sounds, such as wind, sirens, dog barks, and more. The denoising process is accomplished by using a deep autoencoder built on a U-Net architecture.

**Overview**
The goal of this project is to suppress background noise in audio signals while preserving the clean speech. The proposed approach leverages deep CNNs for their ability to capture complex features in noisy signals. The U-Net architecture is utilized, which is well-known for its success in image segmentation tasks and is adapted here for audio denoising.

Final Dataset Specifications
2000 noisy and clean audio WAV files
22,179 samples of noisy and clean spectrograms, each of size 128x128
Total size: 2.71 GB
U-Net Autoencoder Architecture
The U-Net architecture consists of two main parts: the Encoder and the Decoder. The Encoder compresses the input data, while the Decoder reconstructs it. The architecture consists of a series of convolutional layers, activation functions, pooling operations, and up-sampling operations to achieve the desired result.

Decoder Network Architecture
Input: 128x128x1
The decoder consists of two 3x3 Conv2D layers (16 filters each), followed by 2x2 max pooling layers. The activation function used after each convolutional layer is LeakyReLU. After each max pooling layer, the size of the first and second dimensions is reduced by half, and the number of filters is doubled.
This pattern is repeated across four sets of layers in the decoder. In the final set, a dropout layer is applied before the max pooling operation.
After these sets, two 3x3 Conv2D layers with 256 filters are applied, followed by a second dropout layer.
The output of the decoder is 8x8x256.
Encoder Network Architecture
Input: 8x8x256 (output of the decoder's dropout layer)
The encoder consists of an up-sampling layer, followed by a 2x2 Conv2D layer with 128 filters. This is followed by a concatenation layer to merge the weights of the preceding Conv2D layer with the dropout output from the decoder.
After concatenation, two 3x3 Conv2D layers with 128 filters are applied. These five layers form one set.
The activation function used after each convolutional layer is LeakyReLU.
After each up-sampling operation, the first and second dimensions are doubled, and the number of filters is halved. This pattern is repeated across four sets in the encoder.
The final layer consists of a 3x3 Conv2D with 2 filters, followed by a 1x1 Conv2D with 1 filter, resulting in the final output of 128x128x1.
Dataset
The following publicly available datasets were used for this project:

Mozilla Common Voice Dataset: A large, free, and publicly available dataset of human speech that was crowdsourced. This dataset is used for the clean audio.
VoxCeleb Dataset: VoxCeleb1 contains over 100,000 speech samples from 1,251 celebrities. It was combined with the Mozilla dataset to provide diverse speech samples.
UrbanSound8K Dataset: This dataset contains 8,732 labeled sound excerpts of urban noises (e.g., traffic, sirens, dog barks), which were used as background noise for the mixing process.
Data Preprocessing
Loading the Data: Both the clean speech and noise samples were loaded using the Librosa audio library at a sampling rate of 22050 samples/sec. Each audio file was imported as a numpy array.
Cleaning the Data: Silent frames were removed from both the clean speech and noise samples to reduce training time. Clean audio samples longer than 10 seconds were discarded due to poor quality.
Mixing the Data: The noise samples were added to the clean speech samples to create noisy audio. The length of the noise sample was adjusted to match the length of the clean sample.
Feature Extraction: The audio data was transformed into spectrograms using Short-Time Fourier Transform (STFT) to convert the one-dimensional audio data into a two-dimensional format suitable for CNNs.
Training on Google Colaboratory
Google Colaboratory (Colab) is a cloud-based Jupyter notebook environment that provides access to free resources such as CPU, GPU, and TPU. It allows for faster model training by utilizing GPU acceleration. Colab also supports integration with Google Drive, making it easier to load datasets and save trained model weights.

The dataset was uploaded to Google Drive for easy access from Colab. The model took approximately 3 hours to train for 60 epochs using 21,000 training samples and 1,179 validation samples on the GPU.
![WhatsApp Image 2024-11-10 at 22 44 14_b40f2edf](https://github.com/user-attachments/assets/68fe866f-9db1-4016-8d72-0c7381c6cb8d)


Training Results
The U-Net model took about 3 hours to train for 60 epochs on a dataset of 21,000 training samples and 1,179 validation samples using the GPU.

Usage
Once trained, the model can be used to denoise audio on both the client and server sides, depending on the specific application. You can find the pre-trained model and code in this repository to apply the denoising process to other audio files.

Installation
1) Clone this repository:

git clone https://github.com/yourusername/audiodenoising.git

2) Install required dependencies:

pip install -r requirements.txt

Upload your dataset to Google Drive and link it to Colab for easy access.
Run the training script in Google Colaboratory to start training the model.
