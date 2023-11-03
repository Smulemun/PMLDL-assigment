def postprocess(text):
    # deleting unnecessery white spaces 
    text = [x.strip() for x in text]
    return text