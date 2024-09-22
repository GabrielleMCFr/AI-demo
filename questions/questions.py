import nltk
import sys
import string
import math
import os

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    if len(sys.argv) != 2:
        sys.exit("Wrong arguments count")

    # calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # prompt user for query
    query = set(tokenize(input("Query: ")))

    # get top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # get top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    takes a directory name as input and returns a dictionary. In the dictionary, each key is the name of a .txt file 
    found in the directory, and the corresponding value is the content of that file as a string.
    """
    filesDict = dict()
    for dirpath, dirs, files in os.walk(directory): 
        for filename in files:
            if '.txt' in filename:
                filepath = os.path.join(dirpath,filename)

                with open(filepath, encoding="utf8") as f:
                    file_content = f.read()
                    filesDict[filename] = file_content

    return filesDict

def tokenize(document):
    """
    Takes a document (string) as input and returns a list of words in the order they appear. 
    It converts all words to lowercase, removes punctuation, and filters out English stopwords.
    """
    tokens =  nltk.tokenize.word_tokenize(document.lower())
    wordslist = list()
    for token in tokens :
        #check if tokens arent only punctuation or part of useless words. if not append to our list
        if token not in nltk.corpus.stopwords.words("english") and token not in string.punctuation:   
            wordslist.append(token)
        
    return wordslist


def compute_idfs(documents):
    """
    Takes a dictionary of documents, where each document name maps to a list of words. 
    Returns a dictionary that maps each word to its IDF value, including any word that appears in at least one document.
    """

    idfs = dict()
    # keeps track of the number of documents containing a same word
    wordscount = dict()
    
    # iterate throught all documents
    for document in documents:
        # keep only 1 instance of a word
        words = set(documents[document])
        # iterate through all the words it contains, add the count to the wordscount dict
        for word in words:
            if word in wordscount:
                wordscount[word] += 1
            else:
                wordscount[word] = 1
    
    # iterate through all the different words in all docs, and calculate their idfs
    for word in wordscount:
        idfs[word] = math.log(len(documents) / wordscount[word])

    return idfs


def top_files(query, files, idfs, n):
    """   
    This function takes a query (a set of words), files (a dictionary mapping filenames to lists of words), 
    and idfs (a dictionary of word-to-IDF values). It returns a list of the top n filenames that best match the query,
    ranked by their TF-IDF scores.
    """
    tfidfs = dict()
    # iterate through files, calculate the tfidf of the word
    for f in files:
        #initial value of 0
        tfidfs[f] = 0
        for word in query:
            tf = files[f].count(word)
            # ignore tfidf if the word doesnt appear, else add it to our dict
            if tf > 0:
                tfidfs[f] += tf*idfs[word]
    
    # return a list of the files names sorted by desc value of their tfidf score
    files_sorted_by_tfidf = sorted([f for f in files], key=lambda x: tfidfs[x], reverse=True)
        
    return files_sorted_by_tfidf[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Takes a query (a set of words), sentences (a dictionary mapping sentences to lists of words), 
    and idfs (a dictionary mapping words to their IDF values). It returns a list of the top n sentences that best match 
    the query, ranked by IDF. In case of ties, sentences with higher query term density are prioritized.
    """

    idf_sentences = dict()
    for sentence in sentences:
        # dict organized that way: [idf of sentence, query term density of the sentence]
        idf_sentences[sentence] = {'idf': 0, 'density': 0}
        count = 0
        # update the idf score of the sentence for each query words, and keep track of the count of the query words in the sentence to get the query term density
        for word in query:
            if word in sentences[sentence]:
                idf_sentences[sentence]['idf'] += idfs[word]
                count += 1

        idf_sentences[sentence]['density'] = count / len(sentences[sentence])

    # sort sentences first by their idf, then if two idfs are equal, by their query term density
    sorted_sentences = sorted([s for s in sentences], key= lambda x: (idf_sentences[x]['idf'], idf_sentences[x]['density']), reverse=True )    
    
    return sorted_sentences[:n]

if __name__ == "__main__":
    main()
