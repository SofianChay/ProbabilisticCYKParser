# (MVA Speech & NLP assignment) Probabilistic Parser For French 

## Introduction
In this work, we develop a basic French
probabilistic parser that is based on the **Coke-Youger-
Kasami** (CYK) algorithm and a probabilistic context free
grammar learned with the **SEQUOIA treebank v6.0**. In
the following parts, I will present the different compo-
nents of my parser, i.e : the PCFG extractor, the OOV
module that handles out of vocabulary tokens and the
CYK algorithm. Eventually, I analyse the performances
of my system.

## ENGINEERING OF THE PARSER
### Preprocessing of Data 

A parsed sentence from the SEQUOIA dataset is in
the following form :
```
( (SENT (PP-MOD (P En) (NP (NC 1996))) (PONCT ,) (NP-SUJ (DET la) (NC municipalité)) .... 
```
As a preprocessing step, I remove the functional labels
so that PP-MOD becomes PP. In addition the first and
the last parentheses are removed so that the parsed
sentence that is processed looks like :
```
(SENT (PP (P En) (NP (NC 1996))) (PONCT ,) (NP (DET la) (NC municipalité)) .... 
```

### PCFG extraction

To extract the PCFG, I have chosen not to use the
Tree module from nltk because I found it interesting to
extract the rules from the trees and to transform into
Chomsky Normal Form myself. In addition, it allows
me to use my own formalism. Thus, I have build a
class named pcfg that contains useful objects such as
the start symbols, the terminals and non-terminals, the
rules ...

#### Terminal symbols and lexicon extraction

The first steps of the PCFG learning were to detect terminal
symbols and the lexicons. I have represented the lexi-
con as a dictionary : a key is a tuple (**part of speech
tag**, **token**) and the value is the probability of a token
to be a given part of speech tag.

#### Rules Extraction

The rules extraction was the
hard part of the PCFG building step. This process
follows this path : first the extraction of raw rules,
then the removal of unit productions and eventually the
binarization of the rules. The final rules are represented
as a dictionary : a key is a tuple (**left side non terminal
symbol**, (**first right side symbol**, **second right side
symbol**)) and the value is the probability of left symbol
to produce the right part of the rule.
My algorithm to learn the rules is based on the sequen-
tial reading of the sentences in the dataset and on the
counting of opening and closing parentheses.

#### Chomsky Normal Form of the rules

In order to parse the sentence thanks to the **CYK algorithm**,
transforming the raw rules into Chomsky Normal Form
is necessary. To do so, I have removed unit productions
thanks to an algorithm based on a pile that is filled
or emptied as unit rules are removed and replaced by
others. Then the rules are binarized : to do so the
algorithm creates new artificial rules and non terminal
symbols. For instance the rule : **SENT → NP VP
PONCT** will lead to **SENT → NP+VP PONCT** and
**NP+VP → NP VP**.

### OOV module

In this section, I present the choices I have made
to handle the problem of out of vocabulary tokens.
Concretely, my oov module is a class that takes the
**pcfg** and the the **Polyglot embedding lexicon for
French** as inputs. To handle a sequence of tokens, the
main idea is that I provide a substitute sequence of
tokens to my parser. To choose the substitute tokens,
the originals are read, for a given token, if it is in the
lexicon I keep it. If it is not in the lexicon but it is
in the French vocabulary, I compute the **k nearest
neighbors** (in my experiments I choose k = 4) in terms
of embeddings thanks to **Scikit-Learn** implementation,
then to choose the best candidate there are two cases
: if the token is in first position I choose **the most
probable unigram**, else I choose **the most probable
bigram** (with its predecessor) thanks to language model based on the dataset. Eventually, if it is neither
in the lexicon nor in the French vocabulary I compute
the words in the French vocabulary that are within a
minimum distance in terms of **Levenstein distance** and
I repeat the previous process for each of these words.
Consequently, the part of speech tag of each token is
the one of its substitute. 

As a toy example, I tried to substitute : *J’aime les
oranges*. and it gives *Il aimait les Sangsues*. 

### Parser based on the CKY algorithm

#### CYK algorithm 
The class is composed of the
**CYK algorithm** that takes as input the substitute sen-
tence and outputs the table containing the probabilities
of each sub-problems as well as the table containing
the back pointers which allows to build the parsing.

#### Parsing a sentence from the output backpointers of CYK

Once the **CYK algorithm** has run, the parsing
is retrieved thanks to the output **backpointers**. The
algorithm works recursively, we begin from the top
right element of the table and with the start symbol
and at each step the left and right node. The stopping
condition is when a node has no right and left element.

#### Remove artificial symbols created because of the Chomsky Normal Form

Eventually, the sentence is
parsed with the grammar in Chomsky Normal Form,
to go back to the previous form, we need to remove the
artificial symbols and the corresponding parentheses. It
is done thanks to a counter of opening and closing
parentheses. The problem that still need to be handled
is how the unit productions can be retrieved.

#### Whole system

Eventually, the final parser takes
the input **tokenized sentence** so it is substituted thanks
to the oov module, it is then passed to **CYK algorithm**
and eventually the parsing is transformed into the
original form.

## ERROR ANALYSIS AND IMPROVEMENTS 

### Training/test sets and evaluation metric 

### First results 

### Proper nouns handling and new results 

## CONCLUSION

