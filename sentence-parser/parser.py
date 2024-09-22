import nltk
#nltk.download('punkt')
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | VP | S Conj S | PP S | S PP
PP -> P NP
VP -> V | V NP | V NP PP | V PP | VP PP | V NP Adv | V Adv | Adv V
AP -> Adj | Adj AP
NP -> N | Det NP | Det AP N | AP NP | N PP | C NP | Det N | Adv NP | N Adv
"""

# AVP ->  V Adv NP | Adv V NP | V Adv NP PP | Adv V NP PP

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()
    else:
        s = input("Sentence: ")

    # convert input into list of words
    s = preprocess(s)

    # parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence")
        return

    # print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    words = nltk.word_tokenize(sentence)
    outputWords = list()
    for word in words:
        count = 0
        for w in word:
            if w.isalpha():
                count += 1
        if count > 0:
            outputWords.append(word.lower())
            
    return outputWords


def np_chunk(tree):
    NPList = list()

    # find out lasts NP subtrees from tree
    # get label : tree.label()
    #check if there is no np subtree.

    ## check all subtrees with a label of NP
    for subtree in tree.subtrees(lambda t: t.label() == 'NP'):
        for child in subtree:
            if child.label() == 'N':
                NPList.append(subtree)
            if child.label() == 'PP':
                if subtree in NPList:
                    NPList.remove(subtree)

    return NPList


if __name__ == "__main__":
    main()
