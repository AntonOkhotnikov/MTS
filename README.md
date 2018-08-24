## MTS
Number of speakers detection

### How it works
The baseline is next:
* Voice Activity Detection (VAD) first to identify the speech regions
* Mixing the voices and extracting the MFCC features (13-dimensional) per speech frame
* Classification of sequence of N frames using LSTM and CNN classifiers

### Files description:
* **code/main.py** - main code to detect the number of speakers in an audio file
* **code/prepare_training_data.ipynb** - vizualisations and data preprocessing, dataset preparation
* **code/train_model_LSTM.ipynb** - trains LSTM classifier
* **code/train_model_CNN.ipynb** - trains CNN classifier
* **code/vad.py** and **code/utils/estnoise_ms.py** - are borrowed from source [1] - Voice Activity Detector


### To install the environment
* `conda config --add channels anaconda`
* `conda config --append channels conda-forge`
* `conda create -n "name_of_new_environment" --file package-list.txt`
* acivate the environment by `source activate "name_of_new_environment"`
* run `pip install python-speech-features`


### Download and extract dataset
* To download the dataset run in terminal
`wget --mirror --no-parent http://www.repository.voxforge1.org/downloads/Russian/Trunk/Audio/Main/16kHz_16bit/`

* Navigate to the `16kHz_16bit/` directory and run in terminal `for i in *.tgz; do echo working on $i; tar xvzf $i ; done` to unzip all the archives

* Move folder `16kHz_16bit/` with all the nested folders to the folder `data/`


### To run the code
* Navigate to the folder `code/`
* Run in terminal `python main.py`


### References
1. https://github.com/eesungkim/Voice_Activity_Detector
2. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298570
3. https://mycourses.aalto.fi/pluginfile.php/146209/mod_resource/content/1/slides_07_vad.pdf


### Python version
Python 3.6.3

