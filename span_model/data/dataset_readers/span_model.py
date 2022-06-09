import logging
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set, Union
import json
import pickle as pkl
import warnings

import numpy as np
import torch
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (ListField, TextField, SpanField, MetadataField,
                                  SequenceLabelField, AdjacencyField, LabelField)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans

from span_model.data.dataset_readers.document import Document, Sentence
import os
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# New
import sys
sys.path.append("aste")
from parsing import DependencyParser
from pydantic import BaseModel
from data_utils import BioesTagMaker
from span_model.data.dataset_readers.dep_parser import DepInstanceParser


class MissingDict(dict):

    def __init__(self, missing_val, generator=None) -> None:
        if generator:
            super().__init__(generator)
        else:
            super().__init__()
        self._missing_val = missing_val

    def __missing__(self, key):
        return self._missing_val


def format_label_fields(
                        dep_tree: Dict[str, Any],) :

    if 'nodes' in dep_tree:
        dep_children_dict = MissingDict("",(((node_idx, adj_node_idx), "1")
                                        for node_idx, adj_node_idxes in enumerate(dep_tree['nodes']) for adj_node_idx in adj_node_idxes))
    else:
        dep_children_dict = MissingDict("")

    return dep_children_dict


class Stats(BaseModel):
    entity_total:int = 0
    entity_drop:int = 0
    relation_total:int = 0
    relation_drop:int = 0
    graph_total:int=0
    graph_edges:int=0
    grid_total: int = 0
    grid_paired: int = 0


class SpanModelDataException(Exception):
    pass


@DatasetReader.register("span_model")
class SpanModelReader(DatasetReader):
    """
    Reads a single JSON-formatted file. This is the same file format as used in the
    scierc, but is preprocessed
    """
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        # New
        self.stats = Stats()
        self.is_train = False
        self.dep_parser = DependencyParser()
        self.tag_maker = BioesTagMaker()

        print("#" * 80)

        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def get_dep_labels(self, data_dir,direct=False):
        dep_labels = ["self_loop"]
        dep_type_path = os.path.join(data_dir, "dep_type.json")
        with open(dep_type_path, 'r', encoding='utf-8') as f:
            dep_types = json.load(f)
            for label in dep_types:
                if direct:
                    dep_labels.append("{}_in".format(label))
                    dep_labels.append("{}_out".format(label))
                else:
                    dep_labels.append(label)
        return dep_labels

    def get_tree_labels(self, data_dir):
        tree_labels = []
        tree_type_path = os.path.join(data_dir, "tree_type.json")
        with open(tree_type_path, 'r', encoding='utf-8') as f:
            tree_types = json.load(f)
            for label in tree_types:
                tree_labels.append(label)
        return tree_labels

    def prepare_type_dict(self, data_dir):
        dep_type_list = self.get_dep_labels(data_dir)
        types_dict = {"none": 0}
        for dep_type in dep_type_list:
            types_dict[dep_type] = len(types_dict)
        return types_dict

    def prepare_tree_dict(self,data_dir):
        tree_type_list = self.get_tree_labels(data_dir)
        types_dict = {}
        for tree_type in tree_type_list:
            types_dict[tree_type] = len(types_dict)
        return types_dict

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        # file_path = cached_path(file_path)
        path_list = file_path.split("model_outputs")
        input_file_path = path_list[0]+"data_inputs"+path_list[1]
        task_name = path_list[1].split(".json")
        if "\\" in file_path:
            data_name = task_name[0].split("\\")[1].split("_")[0]
            status = task_name[0].split("\\")[2]
            input_file_path = path_list[0]+"data_inputs\\"+data_name+"\\"+path_list[1].split("\\")[2]
            dep_file_path = path_list[0] + "data_inputs\\" + data_name+"\\"+status + ".txt.dep"
            tree_file_path = path_list[0] + "data_inputs\\" + data_name + "\\" + status + ".txt.tree"
        else:
            data_name = task_name[0].split("/")[1].split("_")[0]
            status = task_name[0].split("/")[2]
            input_file_path = path_list[0] + "data_inputs/" + data_name + "/" + path_list[1].split("/")[2]
            dep_file_path = path_list[0] + "data_inputs/" + data_name + "/" + status + ".txt.dep"
            tree_file_path = path_list[0] + "data_inputs/" + data_name + "/" + status + ".txt.tree"
        with open(input_file_path, "r") as f:
            lines = f.readlines()

        all_dep_info = self.load_depfile(dep_file_path)
        all_tree_info = self.load_treefile(tree_file_path)
        self.is_train = "train" in file_path  # New
        types_dict = self.prepare_type_dict(path_list[0]+"data_inputs")
        tree_dict = self.prepare_tree_dict(path_list[0] + "data_inputs")
        for i in range(len(lines)):
        # for line in lines:
            # Loop over the documents.
            doc_text = json.loads(lines[i])
            dep_info = all_dep_info[i]
            tree_info = all_tree_info[i]
            dep_children_dict = format_label_fields(doc_text["dep"][0][0])
            instance = self.text_to_instance(doc_text,dep_children_dict,dep_info,types_dict,tree_info,tree_dict)
            yield instance

        # New
        print(dict(file_path=input_file_path, stats=self.stats))
        self.stats = Stats()

    def load_depfile(self, filename):
        data = []
        with open(filename, 'r') as f:
            dep_info = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    items = line.split("\t")
                    dep_info.append({
                        "governor": int(items[0]),
                        "dependent": int(items[1]),
                        "dep": items[2],
                    })
                else:
                    if len(dep_info) > 0:
                        data.append(dep_info)
                        dep_info = []
            if len(dep_info) > 0:
                data.append(dep_info)
                dep_info = []
        return data


    def load_treefile(self, filename):
        data = []
        with open(filename, 'r') as f:
            tree_info = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    items = line.split("\t")
                    tree_info.append({
                        "word": items[0],
                        "tree_list": items[1:],
                    })
                else:
                    if len(tree_info) > 0:
                        data.append(tree_info)
                        tree_info = []
            if len(tree_info) > 0:
                data.append(tree_info)
                tree_info = []
        return data

    def _too_long(self, span):
        return span[1] - span[0] + 1 > self._max_span_width

    def _process_ner(self, span_tuples, sent):
        ner_labels = [""] * len(span_tuples)

        for span, label in sent.ner_dict.items():
            if self._too_long(span):
                continue
            # New
            self.stats.entity_total += 1
            if span not in span_tuples:
                self.stats.entity_drop += 1
                continue
            ix = span_tuples.index(span)
            ner_labels[ix] = label

        return ner_labels

    def _process_tags(self, sent) -> List[str]:
        if not sent.ner_dict:
            return []
        spans, labels = zip(*sent.ner_dict.items())
        return self.tag_maker.run(spans, labels, num_tokens=len(sent.text))

    def _process_relations(self, span_tuples, sent):
        relations = []
        relation_indices = []

        # Loop over the gold spans. Look up their indices in the list of span tuples and store
        # values.
        for (span1, span2), label in sent.relation_dict.items():
            # If either span is beyond the max span width, skip it.
            if self._too_long(span1) or self._too_long(span2):
                continue
            # New
            self.stats.relation_total += 1
            if (span1 not in span_tuples) or (span2 not in span_tuples):
                self.stats.relation_drop += 1
                continue
            ix1 = span_tuples.index(span1)
            ix2 = span_tuples.index(span2)
            relation_indices.append((ix1, ix2))
            relations.append(label)

        return relations, relation_indices

    def _process_grid(self, sent):
        indices = []
        for ((a_start, a_end), (b_start, b_end)), label in sent.relation_dict.items():
            for i in [a_start, a_end]:
                for j in [b_start, b_end]:
                    indices.append((i, j))
        indices = sorted(set(indices))
        assert indices
        self.stats.grid_paired += len(indices)
        self.stats.grid_total += len(sent.text) ** 2
        return indices

    def get_adj_with_value_matrix(self,max_words_num,dep_adj_matrix, dep_type_matrix,types_dict):
        final_dep_adj_matrix = np.zeros((max_words_num, max_words_num), dtype=np.int)
        final_dep_type_matrix = np.zeros((max_words_num, max_words_num), dtype=np.int)
        for pi in range(max_words_num):
            for pj in range(max_words_num):
                if dep_adj_matrix[pi][pj] == 0:
                    continue
                if pi >= max_words_num or pj >= max_words_num:
                    continue
                final_dep_adj_matrix[pi][pj] = dep_adj_matrix[pi][pj]
                final_dep_type_matrix[pi][pj] = types_dict[dep_type_matrix[pi][pj]]
        return final_dep_adj_matrix, final_dep_type_matrix

    def _process_sentence(self, sent: Sentence, dataset: str,dep_children_dict: Dict[Tuple[int, int],List[Tuple[int, int]]],dep_info,types_dict,tree_info,tree_dict):
        # Get the sentence text and define the `text_field`.
        sentence_text = [self._normalize_word(word) for word in sent.text]
        text_field = TextField([Token(word) for word in sentence_text], self._token_indexers)

        dep_instance_parser = DepInstanceParser(basicDependencies=dep_info, tokens=sent.text)
        dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_first_order(direct=False)
        dep_adj_matrix, dep_type_matrix = self.get_adj_with_value_matrix(len(sent.text),dep_adj_matrix,dep_type_matrix,types_dict)
        spans = []
        for start, end in enumerate_spans(sentence_text, max_span_width=self._max_span_width):
            spans.append(SpanField(start, end, text_field))


        n_tokens = len(sentence_text)
        candidate_indices = [(i, j) for i in range(n_tokens) for j in range(n_tokens)]
        dep_adjs = []
        dep_adjs_indices = []
        for token_pair in candidate_indices:
            dep_adj_label = dep_children_dict[token_pair]
            if dep_adj_label:
                dep_adjs_indices.append(token_pair)
                dep_adjs.append(dep_adj_label)

        # New
        # spans = spans[:len(spans)//2]  # bug: deliberately truncate
        # labeled:Set[Tuple[int, int]] = set([span for span,label in sent.ner_dict.items()])
        # for span_pair, label in sent.relation_dict.items():
        #     labeled.update(span_pair)
        # existing:Set[Tuple[int, int]] = set([(s.span_start, s.span_end) for s in spans])
        # for start, end in labeled:
        #     if (start, end) not in existing:
        #         spans.append(SpanField(start, end, text_field))

        span_field = ListField(spans)
        span_tuples = [(span.span_start, span.span_end) for span in spans]

        # Convert data to fields.
        # NOTE: The `ner_labels` and `coref_labels` would ideally have type
        # `ListField[SequenceLabelField]`, where the sequence labels are over the `SpanField` of
        # `spans`. But calling `as_tensor_dict()` fails on this specific data type. Matt G
        # recognized that this is an AllenNLP API issue and suggested that represent these as
        # `ListField[ListField[LabelField]]` instead.
        fields = {}
        fields["text"] = text_field
        fields["spans"] = span_field
        # fields["spacy_process_matrix"] = spacy_process_matrix
        # New
        graph = self.dep_parser.run([sentence_text])[0]
        self.stats.graph_total += graph.matrix.numel()
        self.stats.graph_edges += graph.matrix.sum()
        fields["dep_graph_labels"] = AdjacencyField(
            indices=graph.indices,
            sequence_field=text_field,
            labels=None,  # Pure adjacency matrix without dep labels for now
            label_namespace=f"{dataset}__dep_graph_labels",
        )

        if sent.ner is not None:
            ner_labels = self._process_ner(span_tuples, sent)
            fields["ner_labels"] = ListField(
                [LabelField(entry, label_namespace=f"{dataset}__ner_labels")
                 for entry in ner_labels])
            fields["tag_labels"] = SequenceLabelField(
                self._process_tags(sent), text_field, label_namespace=f"{dataset}__tag_labels"
            )
        if sent.relations is not None:
            relation_labels, relation_indices = self._process_relations(span_tuples, sent)
            fields["relation_labels"] = AdjacencyField(
                indices=relation_indices, sequence_field=span_field, labels=relation_labels,
                label_namespace=f"{dataset}__relation_labels")
            fields["grid_labels"] = AdjacencyField(
                indices=self._process_grid(sent), sequence_field=text_field, labels=None,
                label_namespace=f"{dataset}__grid_labels"
            )

        # Syntax
        dep_span_children_field = AdjacencyField(
            indices=dep_adjs_indices, sequence_field=text_field, labels=dep_adjs,
            label_namespace="dep_adj_labels")

        fields["dep_span_children"] = dep_span_children_field
        dep_type_matrix_tensor = torch.LongTensor(dep_type_matrix.reshape((1,dep_type_matrix.shape[0],dep_type_matrix.shape[1])))
        fields["dep_type_matrix"] = MetadataField(dep_type_matrix_tensor)
        fields["tree_info"] = MetadataField(tree_info)
        fields["tree_dict"] = MetadataField(tree_dict)
        fields["max_span_width"] = MetadataField(self._max_span_width)

        return fields

    def _process_sentence_fields(self, doc: Document, dep_children_dict,dep_info,types_dict,tree_info,tree_dict):
        # Process each sentence.


        sentence_fields = [self._process_sentence(sent, doc.dataset, dep_children_dict,dep_info,types_dict,tree_info,tree_dict) for sent in doc.sentences]
        # Make sure that all sentences have the same set of keys.
        first_keys = set(sentence_fields[0].keys())
        for entry in sentence_fields:
            if set(entry.keys()) != first_keys:
                raise SpanModelDataException(
                    f"Keys do not match across sentences for document {doc.doc_key}.")

        # For each field, store the data from all sentences together in a ListField.
        fields = {}
        keys = sentence_fields[0].keys()
        for key in keys:
            this_field = ListField([sent[key] for sent in sentence_fields])
            fields[key] = this_field

        return fields


    @overrides
    def text_to_instance(self, doc_text: Dict[str, Any],dep_children_dict,dep_info,types_dict,tree_info,tree_dict):
        """
        Convert a Document object into an instance.
        """
        doc = Document.from_json(doc_text)

        # Make sure there are no single-token sentences; these break things.
        sent_lengths = [len(x) for x in doc.sentences]
        if min(sent_lengths) < 2:
            msg = (f"Document {doc.doc_key} has a sentence with a single token or no tokens. "
                   "This may break the modeling code.")
            warnings.warn(msg)

        fields = self._process_sentence_fields(doc,dep_children_dict,dep_info,types_dict,tree_info,tree_dict)
        doc.dep_children_dict = dep_children_dict
        fields["metadata"] = MetadataField(doc)

        return Instance(fields)

    @overrides
    def _instances_from_cache_file(self, cache_filename):
        with open(cache_filename, "rb") as f:
            for entry in pkl.load(f):
                yield entry

    @overrides
    def _instances_to_cache_file(self, cache_filename, instances):
        with open(cache_filename, "wb") as f:
            pkl.dump(instances, f, protocol=pkl.HIGHEST_PROTOCOL)

    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word