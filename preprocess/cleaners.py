import re
import spacy

nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
nlp.max_length = 4000000

def cleaner(text, spacy=True, qqq=True, lower=True) :
    text = re.sub(r'\s+', ' ', text.strip())
    if spacy :
        text = [t.text for t in nlp(text)]
    else :
        text = text.split()

    if lower :
        text = [t.lower() for t in text]

    if qqq :
        text = ['qqq' if any(char.isdigit() for char in word) else word for word in text]

    return " ".join(text)

def cleaner_mimic(text, spacy=True) :
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\[\s*\*\s*\*(.*?)\*\s*\*\s*\]', ' DEIDENT ', text)
    text = [t.text.lower() for t in nlp(text)]
    text = " ".join(text)
    text = re.sub(r'([^a-zA-Z0-9])(\s*\1\s*)+', r'\1 ', text)
    text = re.sub(r'([^a-zA-Z0-9])', ' \1 ', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = ['qqq' if any(char.isdigit() for char in word) else word for word in text.split(' ')]
    return " ".join(text)

regex_punctuation  = re.compile('([\',\-/\n])')
regex_alphanum     = re.compile('([^a-zA-Z0-9_ \.])')
regex_num          = re.compile('\d[\d ]+')
regex_sectionbreak = re.compile('____+')

def cleaner_whatinnote(text):
    text = text.strip()

    # remove phi tags
    tags = re.findall('\[\*\*.*?\*\*\]', text)
    for tag in set(tags):
        text = text.replace(tag, ' ')

    # collapse phrases (including diagnoses) into single tokens
    if text != text.upper():
        caps_matches = set(re.findall('([A-Z][A-Z_ ]+[A-Z])', text))
        for caps_match in caps_matches:
            caps_match = re.sub(' +', ' ', caps_match)
            if len(caps_match) < 35:
                replacement = caps_match.replace(' ','_')
                text = text.replace(caps_match,replacement)

    text = re.sub('_+', '_', text)

    text = [t.text.lower() for t in nlp(text)]
    text = " ".join(text)
    
#     text = ['qqq' if any(char.isdigit() for char in word) else word for word in text.split(' ')]
    text = re.sub(r'(\d+(\.\d+)?)', r' \1 ', text)
    text = re.sub(regex_alphanum , r' \1 ', text)
    text = re.sub(r'\s+', ' ', text.strip())
    
    text = text.split(' ')
    return " ".join(text).strip()