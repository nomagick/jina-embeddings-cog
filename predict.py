# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os

# Define the folder prefix
FOLDER_PREFIX = "jina-embedding"
MODELS = []
# Get the current working directory
current_dir = os.getcwd()
# Iterate through the entries in the current directory
for entry in os.scandir(current_dir):
    # Check if the entry is a directory and starts with the prefix
    if entry.is_dir() and entry.name.startswith(FOLDER_PREFIX):
        # Do something with the folder
        MODELS.append(entry.name)

from cog import BasePredictor, Input, Path, ConcatenateIterator

from typing import List, Union
from sentence_transformers import SentenceTransformer
import json
import base64


def map_to_b64(ndarray2d):
    return [base64.b64encode(x) for x in ndarray2d]


def map_to_list(ndarray2d):
    return ndarray2d.tolist()


FORMATS = [
    ("base64", map_to_b64),
    ("array", map_to_list),
]
FORMATS_MAP = dict(FORMATS)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.models = dict()
        for mdl in MODELS:
            print(f"Found model: {mdl}")
            self.models[mdl] = SentenceTransformer(mdl).cuda().eval()

    def predict(
        self,
        model: str = Input(
            description="Embedding model", choices=MODELS, default=MODELS[-1]
        ),
        text: str = Input(
            description="Text content to embed",
            default="",
        ),
        json_text: str = Input(
            description="Text content(s) to embed in JSON format",
            default="",
        ),
        output_format: str = Input(
            description="Format to use in outputs",
            default=FORMATS[0][0],
            choices=[k for (k, _v) in FORMATS],
        ),
    ) -> List[Union[str, List[float]]]:
        """Run a single prediction on the model"""

        map_func = FORMATS_MAP[output_format]

        if json_text:
            parsed = json.loads(json_text)
            if type(parsed) == str:
                return map_func(self.models[model].encode([parsed]))
            return map_func(self.models[model].encode(parsed))

        return map_func(self.models[model].encode([text]))
