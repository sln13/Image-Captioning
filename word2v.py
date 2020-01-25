from numpy import asarray
from pickle import dump
from gensim.models import Word2Vec

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# split a dataset into train/test elements
def train_test_split(dataset):
	# order keys so the split is consistent
	ordered = sorted(dataset)
	# return split dataset as two new sets
	return set(ordered[:200]), set(ordered[200:300])

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# store
			descriptions[image_id] = 'startseq ' + ' '.join(image_desc) + ' endseq'
	return descriptions

# load dev set
filename = '/home/lakshminarasimhan/Projectimages/Flickr_8k.devImages.txt'
dataset = load_set(filename)
print('Dataset: %d' % len(dataset))
# train-test split
train, test = train_test_split(dataset)
print('Train=%d, Test=%d' % (len(train), len(test)))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

# train word2vec model
lines = [s.split() for s in train_descriptions.values()]
model = Word2Vec(lines, size=100, window=5, workers=8, min_count=1)
# summarize vocabulary size in model
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))

# save model in ASCII (word2vec) format
filename = 'custom_embedding.txt'
model.wv.save_word2vec_format(filename, binary=False)

# load the whole embedding into memory
embedding = dict()
file = open('custom_embedding.txt')
for line in file:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embedding[word] = coefs
file.close()
print('Embedding Size: %d' % len(embedding))

# summarize vocabulary
all_tokens = ' '.join(train_descriptions.values()).split()
vocabulary = set(all_tokens)
print('Vocabulary Size: %d' % len(vocabulary))

# get the vectors for words in our vocab
cust_embedding = dict()
for word in vocabulary:
	# check if word in embedding
	if word not in embedding:
		continue
	cust_embedding[word] = embedding[word]
print('Custom Embedding %d' % len(cust_embedding))

# save
dump(cust_embedding, open('word2vec_embedding.pkl', 'wb'))
print('Saved Embedding')
