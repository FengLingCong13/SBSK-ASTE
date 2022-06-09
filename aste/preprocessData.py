# author：FLC
# time:2022/3/24
import os
from data_utils import (
    LabelEnum,
    SplitEnum,
    Sentence,
    SentimentTriple,
    Data,
    ResultAnalyzer,
)
from pydantic import BaseModel
from pathlib import Path
from stanfordcorenlp import StanfordCoreNLP
stanfordcorenlp_dir = './stanford-corenlp-full-2018-10-05'
from typing import List, Tuple, Optional
import json
def getDepTree1(tokens, edges_list):
    nodes = [[] for token in tokens]
    nodes_dict = {"nodes":nodes}

    for edge in edges_list:
        if edge['dep'] == 'ROOT':
            continue
        if edge['dependent']-1 not in nodes[edge['governor']-1]:
            nodes[edge['governor']-1].append(edge['dependent']-1)
        if edge['governor']-1 not in nodes[edge['dependent']-1]:
            nodes[edge['dependent']-1].append(edge['governor']-1)

    return nodes_dict

class SpanModelDocument(BaseModel):
    sentences: List[List[str]]
    ner: List[List[Tuple[int, int, str]]]
    relations: List[List[Tuple[int, int, int, int, str]]]
    doc_key: str
    dep:List[List[dict]]

    @property
    def is_valid(self) -> bool:
        return len(set(map(len, [self.sentences, self.ner, self.relations]))) == 1

    @classmethod
    def from_sentence(cls, x: Sentence, nlp):
        ner: List[Tuple[int, int, str]] = []
        for t in x.triples:
            ner.append((t.o_start, t.o_end, LabelEnum.opinion))
            ner.append((t.t_start, t.t_end, LabelEnum.target))
        ner = sorted(set(ner), key=lambda n: n[0])
        relations = [
            (t.o_start, t.o_end, t.t_start, t.t_end, t.label) for t in x.triples
        ]
        dep = []
        # nlp_res_raw = nlp([x.tokens,])
        nlp_res_raw = nlp.annotate(' '.join(x.tokens), properties={'annotators': 'tokenize,ssplit,pos,parse'})
        nlp_res = json.loads(nlp_res_raw)
        # nlp_res = nlp_res_raw
        # dep_nodes = getDepTree1(nlp_res.sentences[0].tokens,
        #                         nlp_res.sentences[0].dependencies)
        dep_nodes = getDepTree1(nlp_res["sentences"][0]['tokens'], nlp_res["sentences"][0]['enhancedPlusPlusDependencies'])
        dep.append(dep_nodes)
        return cls(
            sentences=[x.tokens],
            ner=[ner],
            relations=[relations],
            doc_key=str(x.id),
            dep=[dep],
        )

class SpanModelData(BaseModel):
    root: Path
    data_split: SplitEnum
    documents: Optional[List[SpanModelDocument]]

    @classmethod
    def read(cls, path: Path) -> List[SpanModelDocument]:
        docs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                raw: dict = json.loads(line)
                docs.append(SpanModelDocument(**raw))
        return docs

    def load(self):
        if self.documents is None:
            path = self.root / f"{self.data_split}.json"
            self.documents = self.read(path)

    def dump(self, path: Path, sep="\n"):
        for d in self.documents:
            assert d.is_valid
        with open(path, "w") as f:
            f.write(sep.join([d.json() for d in self.documents]))
        assert all(
            [a.dict() == b.dict() for a, b in zip(self.documents, self.read(path))]
        )

    @classmethod
    def from_data(cls, x: Data):
        data = cls(root=x.root, data_split=x.data_split)
        with StanfordCoreNLP(stanfordcorenlp_dir) as nlp:
            data.documents = [SpanModelDocument.from_sentence(s, nlp) for s in x.sentences]
        return data

ospath = os.getcwd()
if "/" in ospath:
    root = ospath + "/aste/data/triplet_data/"
else:
    root = ospath + "\\aste\\data\\triplet_data\\"
print(root)
dataset_name = ['14lap','14res','15res','16res']
seeds = [0,1,2,3,4]
import os
for name in dataset_name:
    input_name = root+name
    for seed in seeds:
        if "/" in ospath:
            output_dir = "data_inputs/" + "_".join([name, str(seed)])
        else:
            cc = "_".join([name, str(seed)])
            output_dir = "model_outputs\\" + "_".join([name, str(seed)])
        for data_split in ["train", "dev", "test"]:
            data = Data(root=input_name, data_split=data_split)
            data.load()
            new = SpanModelData.from_data(data)

            if "/" in output_dir:
                final_path = output_dir+"/"+data_split+".json"
            else:
                final_path = output_dir+"\\"+data_split+".json"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            new.dump(Path(final_path))
