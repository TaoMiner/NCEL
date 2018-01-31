from ncel.data import load_conll_data
from ncel.data import load_xlwiki_data
from ncel.data import load_kbp_data
from ncel.data import load_wned_data
from ncel.utils.misc import inspectDoc
from ncel.data import load_ncel_data

if __name__ == "__main__":
    # conll demo:
    # docs = load_conll_data.load_data(mention_file='/data/caoyx/el_datasets/AIDA-YAGO2-dataset.tsv',
    #                                 genre=2, include_unresolved=False, lowercase=True)
    # wned demo:
    # docs = load_wned_data.load_data(text_path='/home/caoyx/data/WNED/wned-datasets/ace2004/RawText/',
    #                 mention_file='/home/caoyx/data/WNED/wned-datasets/ace2004/ace2004.xml')
    # wned demo:
    # docs = load_xlwiki_data.load_data(text_path='/Users/ethan/Downloads/xlwikifier-wikidata/data/it/train/')
    # kbp demo:
    # docs = load_kbp_data.load_data(text_path='/home/caoyx/data/kbp/kbp_cl/kbp16/eng/eval/nw',
    #                 mention_file='/home/caoyx/data/kbp/kbp_cl/kbp16/eng/eval/kbp16_nw_gold.xml',
    #                 kbp_id2wikiid_file='/home/caoyx/data/kbp/kbp_cl/id.key2015')
    # ncel data
    docs = load_ncel_data.load_data(text_path='/data/caoyx/el_datasets/ncel-wikidata/small_100000/abstracts_100000/',
                     mention_file='/data/caoyx/el_datasets/ncel-wikidata/small_100000/wikipedia_100000.xml')


    for doc in docs:
        inspectDoc(doc)
        input("press any key to continue ...")