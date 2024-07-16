import json
import os
import re
from typing import Iterator, List

import numpy as np

from dataset_reader.base_reader import Query
from dataset_reader.json_reader import JSONReader


class AnnCompoundReader(JSONReader):
    """
    A reader created specifically to read the format used in
    https://github.com/qdrant/ann-filtering-benchmark-datasets, in which vectors
    and their metadata are stored in separate files.
    """

    VECTORS_FILE = "vectors.npy"
    VECTORS_FILE_RE = re.compile("^vectors.*.npy$")
    QUERIES_FILE = "tests.jsonl"

    def read_vectors(self) -> Iterator[List[float]]:
        file_names = []
        for file_name in os.listdir(self.path):
            if not self.VECTORS_FILE_RE.match(file_name):
                continue
            file_names.append(file_name)
        file_names.sort()
        n = len(file_names)
        print(f"Uploading vectors from {n} files: {file_names}")
        for xx, file_name in enumerate(file_names):
            print(f"Uploading vectors from file {xx + 1}/{n}: {file_name}")
            vectors = np.load(self.path / file_name)
            for vector in vectors:
                if self.normalize:
                    vector = vector / np.linalg.norm(vector)
                yield vector.tolist()


    def read_queries(self) -> Iterator[Query]:
        with open(self.path / self.QUERIES_FILE) as payloads_fp:
            for idx, row in enumerate(payloads_fp):
                row_json = json.loads(row)
                vector = np.array(row_json["query"])
                if self.normalize:
                    vector /= np.linalg.norm(vector)
                yield Query(
                    vector=vector.tolist(),
                    sparse_vector=None,
                    meta_conditions=row_json["conditions"],
                    expected_result=row_json["closest_ids"],
                    expected_scores=row_json["closest_scores"],
                )
