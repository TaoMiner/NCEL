import re

# enti_id \t gobal_prior \t cand_ment::=count \t ...
def buildCandidatesFromPrior(prior_file, wiki_candidate_file):
    mention2wiki_dict = dict()
    with open(prior_file, 'r', encoding='UTF-8') as fin:
        for line in fin:
            items = re.split(r'\t', line.strip())
            if len(items) < 3 : continue
            for mention in items[2:]:
                tmp_items = re.split(r'::=', mention)
                if len(tmp_items) != 2: continue
                tmp_set = mention2wiki_dict.get(tmp_items[0], set())
                tmp_set.add(items[0])
                mention2wiki_dict[tmp_items[0]] = tmp_set
    num_mentions = len(mention2wiki_dict)
    num_candidates = sum([len(mention2wiki_dict[m]) for m in mention2wiki_dict])
    print("num mentions:{}, num candidates:{}.".format(num_mentions, num_candidates))
    with open(wiki_candidate_file, 'w', encoding='UTF-8') as fout:
        for m in mention2wiki_dict:
            fout.write("{}\t{}\n".format(m,'\t'.join(mention2wiki_dict[m])))

wiki_candidate_file = '/data/caoyx/el_datasets/wiki_candidate'
prior_file = '/data/caoyx/el_datasets/entity_prior'
buildCandidatesFromPrior(prior_file, wiki_candidate_file)
