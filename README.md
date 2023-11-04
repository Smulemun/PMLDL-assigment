## Made by
Artyom Makarov DS21-02 
a.makarov@innopolis.university

# How to run
First of all run `pip install -r requirements.txt`
## Transform Data
1. Download [this dataset](https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip) into `data/raw`
2. Run `python src/data/make_dataset.py` to preprocess and split the data (or alternatively run notebook `notebooks/1.1_initial_data_exploration`)
## Train Models
To train models run respective Jyputer notebook (for example to train t5-small model run `4_style_transfer.ipynb`)
However training takes a long time, instead download [pretrained model](https://github.com/Smulemun/PMLDL-assigment/releases/tag/model)
## Make Predictions
To detoxify given text run `python src/models/detoxify.py --text="your toxic text here"`
