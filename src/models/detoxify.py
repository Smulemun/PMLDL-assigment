from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse


from predict import detoxificate_style_transfer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detoxify a given text')
    parser.add_argument('--text', type=str, help='Text to detoxify')
    args = parser.parse_args()
    model_name = 'models/detoxificator'
    best_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = args.text
    detoxified_text = detoxificate_style_transfer([text], best_model, tokenizer)[0]
    print(detoxified_text)
