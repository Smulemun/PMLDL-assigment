def postprocess(text):
    '''Postprocess text to remove unnecessary white spaces.'''
    # deleting unnecessery white spaces 
    text = [x.strip() for x in text]
    return text