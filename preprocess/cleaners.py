import re
import spacy

# nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
# nlp.max_length = 4000000
# nlp.add_pipe('sentencizer')

nlp = spacy.load('en')
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
    tag_pattern = re.compile(r'\[\*\*([^*]*)\*\*\]')
    text = tag_pattern.sub(lambda m : ('deid_' + m.groups()[0] + '_deid').replace(' ', '_'), text)

    # collapse phrases (including diagnoses) into single tokens
    if text != text.upper():
        caps_matches = set(re.findall('([A-Z][A-Z_ ]+[A-Z])', text))
        for caps_match in caps_matches:
            caps_match = re.sub(' +', ' ', caps_match)
            if len(caps_match) < 35:
                replacement = caps_match.replace(' ','_')
                text = text.replace(caps_match,replacement)

    text = re.sub('_+', '_', text)

    sentences = []
    for s in nlp(text).sents :
        s = " ".join([t.text.lower() for t in s])
        s = re.sub(r'(\d+(\.\d+)?)', r' \1 ', s)
        s = re.sub(regex_alphanum , r' \1 ', s)
        s = re.sub(r'\s+', ' ', s.strip())
        s = re.sub(r'deid_(.*?)_deid', lambda s : s.group(0).replace(' ', '').replace('deid_', '[**').replace('_deid', '**]'), s)
        if re.match(r'\d+\s*\.', s) :
            continue
        sentences.append(s)
    
    text = " <ExpSBD> ".join(sentences)
    
#     text = ['qqq' if any(char.isdigit() for char in word) else word for word in text.split(' ')]
    
    
    text = text.split(' ')
    return " ".join(text).strip()