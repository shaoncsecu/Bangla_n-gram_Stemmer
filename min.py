import re 
from spacy.lang.bn import Bengali
import numpy as np
import string
import sklearn.cluster


def read_files(file_path):

    corpus = []
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
        text = re.sub(r"[A-Za-z!-()\"#/@;_:<>’’{}+-=~-।|.?,*–—…  →√॥\d+^\\`\[\]‘৷🎊😜]+", " ", w)
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


def min_edit_distance(s1, s2):
    if len(s1) < len(s2):
        return min_edit_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1

            if c1==c2:
                substitutions = previous_row[j]
            else:
                substitutions = previous_row[j] + 2     # Levenshtein Substitution cost for

            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    # print("{0} - {1} = Distance {2}".format(s1, s2, previous_row[-1]))

    # If the minimum edit distance is greater than the minimum length of either of the strings
    # return a big value (say 200) so that they could be categorized as a different cluster
    dist = previous_row[-1]

    if dist >= min(len(s1), len(s2)):
        dist = 200

    return dist


def initial_cluster(token, out_filename):

    # sorting is not necessary - just can make the output pretty
    # token.sort()
    # lev_similarity = -1*np.array([[min_edit_distance(w1, w2) for w1 in token] for w2 in token])

    len_tok = len(token)
    lev_similarity = [[0] * len_tok for i in range(len_tok)]

    for i in range(len_tok):
        for j in range(i, len_tok):
            # This will reduce the number of iterations since the matrix is a mirror
            lev_similarity[i][j] = lev_similarity[j][i] = min_edit_distance(token[i], token[j])

    # Comment out this print - just to show you the matrix
    # print(lev_similarity[:2])

    affprop = sklearn.cluster.AffinityPropagation(affinity="euclidean", damping=0.5)
    affprop.fit(lev_similarity)

    with open(out_filename, 'w', encoding='utf-8') as file:

        # Calculating the biggest cluster so that we can use it letter to again cluster our data
        biggest_exemplar = token[affprop.cluster_centers_indices_[np.unique(affprop.labels_)[0]]]
        biggest_cluster = np.unique(token[np.nonzero(affprop.labels_ == np.unique(affprop.labels_)[0])])
        max_cluster_len = len(biggest_cluster)

        for cluster_id in np.unique(affprop.labels_):
            exemplar = token[affprop.cluster_centers_indices_[cluster_id]]
            cluster = np.unique(token[np.nonzero(affprop.labels_ == cluster_id)])

            cluster_str = ''
            len_cluster = len(cluster)
            if max_cluster_len < len_cluster:
                cluster_str = ", ".join(biggest_cluster)
                cluster_str = biggest_exemplar+" ::: "+cluster_str
                file.write("{0}\n".format(cluster_str))

                biggest_cluster = cluster
                biggest_exemplar = exemplar
                max_cluster_len = len_cluster

            elif max_cluster_len > len_cluster:
                cluster_str = ", ".join(cluster)
                cluster_str = exemplar+" ::: "+cluster_str
                file.write("{0}\n".format(cluster_str))

            print("{0}".format(cluster_str))

    file.close()

    return biggest_cluster


def final_cluster(token, out_filename):

    # sorting is not necessary - just can make the output pretty
    # token.sort()
    # lev_similarity = -1*np.array([[min_edit_distance(w1, w2) for w1 in token] for w2 in token])

    len_tok = len(token)
    lev_similarity = [[0] * len_tok for i in range(len_tok)]

    for i in range(len_tok):
        for j in range(i, len_tok):
            # This will reduce the number of iterations since the matrix is a mirror
            lev_similarity[i][j] = lev_similarity[j][i] = min_edit_distance(token[i], token[j])

    # Comment out this print - just to show you the matrix
    # print(lev_similarity[:2])

    affprop = sklearn.cluster.AffinityPropagation(affinity="euclidean", damping=0.5)
    affprop.fit(lev_similarity)

    with open(out_filename, 'w', encoding='utf-8') as file:
        for cluster_id in np.unique(affprop.labels_):
            exemplar = token[affprop.cluster_centers_indices_[cluster_id]]
            cluster = np.unique(token[np.nonzero(affprop.labels_ == cluster_id)])

            cluster_str = ", ".join(cluster)
            cluster_str = exemplar+" ::: "+cluster_str

            file.write("{0}\n".format(cluster_str))
            print("{0}".format(cluster_str))

    file.close()


def main():
    corpus = read_files('sample_input.txt')

    # Reading the punctuation symbols (as list) and making them string 'pattern'
    corpus_punch = read_files('punctuations.txt')
    pattern = corpus_punch.pop()
    corpus_clean = clean_text(corpus, pattern)

    # Tokenize the corpus
    word_tokens = tokenize(corpus_clean)

    # Taking only the unique tokens and removing whitespaces
    # So that indexing with a list will work
    token = np.asarray(list(set(word_tokens)))

    tok_len = len(token)
    print("Tokens = {}".format(tok_len))

    print('Please wait for the program to finish...')

    # clustering the tokens and writing them into specified filename
    # returns a biggest cluster of unwanted items (which we can re-use)
    biggest_cluster = initial_cluster(token, out_filename='sample_out_first.txt')

    # using the biggest cluster calling a simpler version of clustering
    final_cluster(biggest_cluster, out_filename='sample_out_second.txt')


if __name__ == '__main__':
    main()


