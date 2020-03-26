import re

# list of the stop words to be eliminated from the issue dataset
# Note that all the words have to be lower-cased in the dataset before filtering these words
STOP_WORDS = ['ve', 'ile', 'ki', 'ama', 
             # .....
             'en', 'bir', 'bizden', 'benim']

# Turkish upper-case characters are lower-cased seperately so as to be sure of them
lower_map_turkish = {
    ord(u'I'): u'ı',
    ord(u'İ'): u'i',
    ord(u'Ç'): u'ç',
    ord(u'Ş'): u'ş',
    ord(u'Ö'): u'ö',
    ord(u'Ü'): u'ü',
    ord(u'Ğ'): u'ğ'
    }

class TextPreProcessor(object):
    def filterNoiseAndStemWords(self, text):
        """
        converts words to lowercase, eliminates non-alphanumeric characters, eliminates stop-words

        """
        # Remove all non-alphanumeric characters from the text via the regex[\W]+,
        # Convert the text into lowercase characters
        text_tr = text.translate(lower_map_turkish)
        lowerText = re.sub('[\W]+', ' ', text_tr.lower())

        #remove stopwords
        noStopWordsText = [word for word in lowerText.split() if word not in STOP_WORDS]
        return ' '.join(noStopWordsText)
