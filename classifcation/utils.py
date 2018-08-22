import codecs
import operator
from sklearn.model_selection import train_test_split

IO_DIR = 'data_dir'


def vocab_creation(domain_name, max_len=0, vocab_size=0):
    source = None
    try:
        source = '%s/%s/train.csv' % (IO_DIR, domain_name)
    except:
        print("Domain %s doesn't exist" % (domain_name))
    print('Vocabulary initialization...')
    total, unique = 0, 0
    word_freqs = {}
    top = 0
    text = codecs.open(source, 'r', 'utf-8')
    for line in text:
        words = line.split()
        if max_len > 0 and len(words) > max_len:
            continue

        for word in words:
            try:
                word_freqs[word] += 1
            except KeyError:
                unique += 1
                word_freqs[word] = 1
            total += 1
    print('Total amount of words %i with %i unique ones' % (total, unique))
    sorted_freq = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
    # TODO: simplify this part
    vocab = {'<pad>': 0, '<unk>': 1}
    index = len(vocab)
    for word, _ in sorted_freq:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print('Vocabulary size is %i' % vocab_size)

    ofile = codecs.open('%s/%s/vocab' % (IO_DIR, domain_name), mode='w', encoding='utf-8')
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    for word, index in sorted_vocab:
        # TODO: remove hardcore
        if index < 2:
            ofile.write(word + '\t' + str(0) + '\n')
            continue
        ofile.write(word + '\t' + str(word_freqs[word]) + '\n')
    ofile.close()
    print('Vocabulary is successfully created')

    return vocab


# TODO: implement me
def read_vocabulary(domain_name):
    ifile = codecs.open('%s/%s/vocab' % (IO_DIR, domain_name), mode='r', encoding='utf-8')
    ifile.read()


def read_set(domain_name, set_name, vocab, max_len):
    assert set_name in {'train', 'test'}
    source = '%s/%s/%s.csv' % (IO_DIR, domain_name, set_name)
    # TODO: refactor this
    unk, total = 0., 0.
    max_x = 0
    data_x = []
    text = codecs.open(source, 'r', 'utf-8')
    for line in text:
        # TODO: here was strip() but the purpose was vague
        words = line.split()
        if max_len > 0 and len(words) > max_len:
            continue
        indices = []
        for word in words:
            if word in vocab:
                indices.append(vocab[word])
            else:
                indices.append(vocab['<unk>'])
                unk += 1
            total += 1
        data_x.append(indices)
        if max_x < len(indices):
            max_x = len(indices)
    print('%s is processed' % domain_name)
    return data_x, max_x


def read_data(domain_name, vocab_size=0, max_len=0):
    vocab = vocab_creation(domain_name, max_len, vocab_size)
    print('Reading train set...')
    train, train_max = read_set(domain_name=domain_name, set_name='train', vocab=vocab, max_len=0)
    print('Success')
    print('Reading test set...')
    test, test_max = read_set(domain_name=domain_name, set_name='test', vocab=vocab, max_len=max_len)
    print('Success')
    max_len = max(train_max, test_max)
    return vocab, train, test, max_len


def train_test_split(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        labels, test_size=0.2)
    return
