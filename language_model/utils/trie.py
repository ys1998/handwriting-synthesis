"""Trie implementation in Python3 for storing backoff ngram model"""

def log_to_ln(x):
    return x * 2.30258509299

class Trie(object):
    def __init__(self):
        """
        Initialize trie
        """
        self.backoff = 0
        self.log_prob = 0.0
        self.children = {}
        self.character = None

    def ptr(self):
        return self

    def load_arpa(self, filename, vocab):
        """
        filename  : Path to ngram file in ARPA format
        vocab     : Dictionary to store word-index mapping
        """
        with open(filename, 'r') as f:
            ngram_sizes = [0]
            stage = 0
            gram = 0
            for line in iter(f.readline, ''):
                words = line.strip('\n\r').split()
                if not words:
                    stage += 1
                else:
                    if stage == 1:
                        if words[0] == "\\data\\":
                            print("Reading number of ngrams ...")
                        elif words[0] == "ngram":
                            ngram_sizes.append(int(words[1].split('=')[1]))
                    else:
                        if words[0] == "\\end\\":
                            break
                        elif words[0][0] == "\\":
                            gram = int(words[0][1:].split('-')[0])
                            print("Loading %d %d-grams ..."%(ngram_sizes[gram], gram))
                        else:
                            log_prob = log_to_ln(float(words[0]))                            
                            try:
                                backoff = log_to_ln(float(words[-1]))
                            except:
                                backoff = 0.0
                            
                            ptr = self
                            created = False
                            for i in range(gram):
                                vocab_id = vocab[words[i+1]]
                                if vocab_id not in ptr.children:
                                    ptr.children[vocab_id] = Trie()
                                    created = True
                                ptr = ptr.children[vocab_id]
                            
                            if created == False:
                                print("Error")
                                exit(1)
                            
                            ptr.log_prob = log_prob
                            ptr.character = words[gram]
                            ptr.backoff = backoff

    def get_distro(self, context, distro):
        """
        context : Indices of words in context window for a given word
        distro  : Array to be filled with ngram probability distribution
        """
        backoff = 0.0
        context_size = len(context)
        for gram in range(context_size):
            i = gram
            current = self
            while i is not context_size:
                print("In while loop")
                if context[i] not in current.children:
                    break
                else:
                    current = current.children[context[i]]
                    if i == context_size-1:
                        for k,v in current.children.items():
                            if distro[k] == 0.0:
                                distro[k] = v.log_prob + backoff
                        
                        backoff += current.backoff
                i += 1
        
        current = self
        for i in range(len(distro)):
            if distro[i] == 0.0:
                distro[i] = current.children[i].log_prob + backoff      
