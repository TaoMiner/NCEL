# Neural Collective Entity Linking

This is the code of the *Neural Collective Entity Linking* in COLING'18, which proposed a model that performs global Entity Linking (EL) combining deep neural networks with Graph Convolutional Network (GCN).

## Environment

python 3.6

Pytorch 0.3.x

numpy

gflags

### logging

We choose [Google protobuf](https://developers.google.com/protocol-buffers/docs/pythontutorial) for logging, you may need to generate your own logging_pb2.py for your computer by compiling *logging.proto*. There are two steps: 

1. install *protoc* compiler from [here](https://developers.google.com/protocol-buffers/docs/downloads);

2. use the following command to compile:

`protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/addressbook.proto`

More details can be found in [here](https://developers.google.com/protocol-buffers/docs/pythontutorial).

## Run our codes

We use the following command to run our model NCEL:

`python run_entity_linking.py --training_data DATASET:SUPPLEMENT:TEXTPATH:MENTIONPATH --eval_data DATASET:SUPPLEMENT:TEXTPATH:MENTIONPATH --candidates_file ncel:CANDIDATE_FILE --wiki_entity_vocab ENTITY_VOCAB --word_embedding_file WORD_VECTOR --entity_embedding_file ENTITY_VECTOR --log_path PATH_TO_LOG`

The required inputs consist of three parts: **datasets**, **candidates dictionary** and **embedding files**. We now describe each. 

We are sorry that the program is not easy to follow because an EL system always refers to multiple elements: words, entities, and an prelemenary dictionary indicating their mappings. Also, we try to include as many datasets (most of them have different format) as possible, but we cannot release them along with our codes due to the copyright, which makes the situation more severe. Therefore, you may want to use the model file only (*./models/subncel.py*), which we shall describe the inputs, outputs and loss later, or we may provide an simplication version in months.

## Datasets (--training_data and --eval_data)
We specify the datasets by *--training_data* and *--eval_data*. Paticurlarly, you can evaluate multiple datasets simultaneously, by setting --eval_data separated by ','.

We have test our codes on the following datasets: Conll (DATASET="conll"), Kbp (DATASET in ["kbp10", "kbp15", "kbp16"]), Xlwiki ( DATASET="xlwiki", Cross-lingual Wikification Using Multilingual Embeddings), WNED (DATASET in ["msnbc", "aquaint", "ace04", "wiki13", "clueweb12"]) in the paper *Robust Named Entity Disambiguation with RandomWalks*, and our data derived from wikipedia (DATASET="ncelwiki").

The five sources of data have different data format, and only two of them are public available: WNED and "ncelwiki". They have the same data format: a folder including separated files of plain text, and a xml file indicating their annotations. You may download our ncelwiki data from here, and see the details. Of course, you can also construct your own data with our data format and load it use ncelwiki as DATASET in the above command. Thus, you can ignore **SUPPLEMENT**, denote **TEXTPATH** with the folder, and **MENTIONPATH** with the xml annotation file.

We provide our ncelwiki data: [MENTIONPATH](https://drive.google.com/file/d/1mWPFOPgU1H-9FAtDveNf841MCFjOi2Oe/view?usp=sharing) and [TEXTPATH](https://drive.google.com/file/d/1E4AE8UpmS2LKtHFRocNuI9q4eNfvxZ0Q/view?usp=sharing).

## candidates dictionary

We specify the candidate dictionary by *--candidates_file ncel:CANDIDATE_FILE*, where *ncel* is the source, and CANDIDATE_FILE has the following format:

```
<mention><tab><probability of entity given mention><tab><entity>
```

The candidate dictionary provides an mapping from mention strings (several words) to entities. Note that mention can be repeated. That is, one mention can refer to multiple entities, and different mentions can refer to the same entity.

We provide our candidate file (top30) for ncelwiki data [here](https://drive.google.com/file/d/1z5BMncxRCD9phyIzniyEW6HCJrO5El9I/view?usp=sharing).

## Embedding Files

We take pre-trained word and entity embeddings in a unified vector space as features to measure their similarity. Particularly, we specify the entity vocab with **wiki_entity_vocab**, each line is:
```
wiki_id \t wiki_label
```
For the embedding file, the format is the same as the binary output of [word2vec](https://github.com/tmikolov/word2vec):  the first line is the vocab size and embedding size separated by tab or white space, each line of the rest starts with the word or wiki_id (entity) and its vector, separated by tab or white space.

Thus, the easist way to obtain the embedding file is to download wikipedia articles with anchors, replace all anchors with corresponding wiki_id (entity), and then run word2vec on it for jointly embedding words and entities.

In our settings, we use the embeddings of the paper *Bridging Text and Knowledge by Learning Multi-Prototype Entity Mention Embedding* as inputs (codes is [here](https://github.com/TaoMiner/bridgeGap) ), or the cross-lingual embeddings of the paper *joint representation learning of cross-lingual words and entities via attentive distant supervision* with the [codes](https://github.com/TaoMiner/MultiLingualEmbedding). Note that they shall output not only word and entity embedding files, but also an auxillary sense embedding file where each entity refers to a sense embedding and a mu embedding (trained based on all of the entity textual contexts targeting the global features compared to each local occurrence).

### General Flags

You can also specify the other training parameters, such as training_steps, learning_rate, l2_lambda. More details can be found in *./model/base.py*

## Reference
If you use our code, please cite our paper:
```
@inproceedings{cao2018neural,
  title={Neural Collective Entity Linking},
  author={Cao, Yixin and Hou, Lei and Li, Juanzi and Liu, Zhiyuan},
  booktitle={Proceedings of the 27th International Conference on Computational Linguistics},
  pages={675--686},
  year={2018}
}
```