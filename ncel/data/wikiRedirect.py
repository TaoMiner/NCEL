import re

def loadIdDic(wiki_id_file):
    label2id = dict()
    with open(wiki_id_file, 'r', encoding='UTF-8') as fin:
        for line in fin:
            items = re.split(r':', line.strip())
            if len(items) < 3: continue
            label2id[items[2]] = items[1]
    return label2id

def loadEntityDic(vocab_entity_file):
    label2id = dict()
    with open(vocab_entity_file, 'r', encoding='UTF-8') as fin:
        for line in fin:
            items = re.split(r'\t', line.strip())
            if len(items) < 2: continue
            label2id[items[1]] = items[0]
    return label2id

def buildRedirectId(vocab_redirect_file, label2id, entity_id_vocab):
    redirect_id_dict = dict()
    redirect_candidate = dict()
    with open(vocab_redirect_file, 'r', encoding='UTF-8') as fin:
        for line in fin:
            items = re.split(r'\t', line.strip())
            if len(items) < 2 or items[1] not in entity_id_vocab: continue
            ent_id = entity_id_vocab[items[1]]
            if items[0] in label2id:
                redirect_id_dict[label2id[items[0]]] = ent_id
            redirect_candidate[items[0]] = ent_id
    return redirect_id_dict, redirect_candidate

wiki_id_file = '/home/caoyx/data/dump20170401/enwiki/enwiki-20170401-pages-articles-multistream-index.txt'
vocab_redirect_file = '/home/caoyx/data/dump20170401/enwiki_cl/vocab_redirects.dat'
vocab_entity_file = '/home/caoyx/data/dump20170401/enwiki_cl/vocab_entity.dat'

redirect_id_file = '/home/caoyx/data/dump20170401/enwiki_cl/redirect_id_vocab'
redirect_candidate_file = '/home/caoyx/data/dump20170401/enwiki_cl/redirect_candidate'

label2id = loadIdDic(wiki_id_file)
entity_id_vocab = loadEntityDic(vocab_entity_file)
redirect_id_dict, redirect_candidate = buildRedirectId(vocab_redirect_file, label2id, entity_id_vocab)

with open(redirect_id_file, 'w', encoding='UTF-8') as fout:
    for rid in redirect_id_dict:
        fout.write("{}\t{}\n".format(rid,redirect_id_dict[rid]))

with open(redirect_candidate_file, 'w', encoding='UTF-8') as fout:
    for r_str in redirect_candidate:
        fout.write("{}\t{}\n".format(r_str,redirect_candidate[r_str]))