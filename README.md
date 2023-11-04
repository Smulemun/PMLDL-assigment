## Made by
Artyom Makarov DS21-02 
a.makarov@innopolis.university

# How to run
First of all run `pip install -r requirements.txt`
## Transform Data
Run `python src/data/make_dataset.py` to download and preprocess
## Train Models
To train models run respective Jyputer notebook (for example to train t5-small model run `4_style_transfer.ipynb`)
However training takes a long time, instead download [pretrained model](https://github.com/Smulemun/PMLDL-assigment/releases/tag/model)
## Make Predictions
To detoxify given text run `python src/models/detoxify.py --text="your toxic text here"`
