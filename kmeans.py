import nltk
import re 
from spacy.lang.bn import Bengali
import numpy as np
import string
import sklearn.cluster

def coef_sim(word1, word2):

    # makes a list of characters from the words
    ch_fst = [ch for ch in word1]
    ch_sec = [ch for ch in word2]

    # crates biagram (2-gram) list out of those list of characters
    ngram_fst = list(nltk.ngrams(ch_fst, 2))
    ngram_sec = list(nltk.ngrams(ch_sec, 2))
    #length of bigram word 
    len1 = len(ngram_fst)
    len2 = len(ngram_sec)
    #unique bigram
    ngram_fst = list(set(ngram_fst))
    ngram_sec = list(set(ngram_sec))
    #unique bigram length
    x=len(ngram_fst)
    y=len(ngram_sec)
    
    
    # count similar digramss
    count = 0
    for i in range(x):
        for j in range(y):
            if ngram_fst[i] == ngram_sec[j]:
                count += 1

    # coefficient similarity measure
    if count!=0:
        sim = float(2*count)/float(len1 + len2)
    else:
        sim= float(count)

    return sim

#corpus file load 1
def read_files(file_path):

    try:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            corpus = file.readlines()   # copy the content of the file in a list
    except:
        print("Usage: $ python BN_Tokenizer.py <corpus_file>")

    return corpus


#punctuation remove 2 
def clean_text(corpus, pattern):

    # clean_word = [s.translate(string.punctuation+string.ascii_letters+string.digits+pattern) for s in corpus]
    clean_word = []

    for w in corpus:
        text = re.sub(r"[A-Za-z!-()\"#/@;_:<>’’{}+-=~-।|.?,*–—…  →√॥\d+^\\`\[\]‘৷]+", " ", w)
        text = re.sub(r"["+string.punctuation+string.ascii_letters+string.digits+pattern+"]+", " ", text)
        clean_word.append(text)

    return clean_word

    
#tokenization 3
def tokenize(corpus):

    nlp = Bengali() 
    tokenizer = nlp.tokenizer

    word_tokens = []

    for line in corpus:
        tokens = tokenizer(line.strip())    
        for token in tokens:
            word_tokens.append(str(token))  

    return word_tokens

#corpus file write 4
def write_file(word_tokens, file_path):

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(word_tokens))

    file.close()


def main():

    # corpus part
    corpus = read_files('sample_input.txt')

    # Reading the punctuation symbols (as list) and making them string 'pattern'
    corpus_punch = read_files('punctuations.txt')
    pattern = corpus_punch.pop()
    corpus_clean = clean_text(corpus, pattern)

    # Tokenize the corpus
    word_tokens = tokenize(corpus_clean)

    # Taking only the unique tokens and removing whitespaces
    text = list(set(word_tokens))
    #text = [w.strip() for w in text]
    #text.remove('')
    text.sort()

    print("Tokens = {}".format(len(text)))

    print('Please wait for the program to finish...')
    #dic = create_dictionary(text, threshold=0.5)
    
    token = np.asarray(text) #So that indexing with a list will work
    lev_similarity =np.array([[coef_sim(w1,w2) for w1 in token] for w2 in token])

    affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
    affprop.fit(lev_similarity)
    for cluster_id in np.unique(affprop.labels_):
        exemplar = token[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(token[np.nonzero(affprop.labels_==cluster_id)])
        cluster_str = ", ".join(cluster)
        print("%s ::: %s" % (exemplar, cluster_str))

    #TODO: Need to write the file in a different way... I'm too lazy to do it ;)
    #write_file(text, 'final_tokens.txt')
    
        
#punctuation part
if __name__ == '__main__':
    main()


