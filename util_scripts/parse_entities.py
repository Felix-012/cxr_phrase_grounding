import argparse
import glob
import os

import spacy
import scispacy
from scispacy.linking import EntityLinker
import json
from pathlib import Path
from spacy.tokens import Doc

# Load the spaCy model and add the EntityLinker pipe
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

# Define the base path and target directories
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True,
                        help='base path of report files')
parser.add_argument('--output_path', type=str, required=True,
                        help='path of directory where to store the entity json files')
args = parser.parse_args()

target_dirs = [f"p{num}" for num in range(10, 20)]  # Creates a list of directories from p10 to p19

# Process each directory
for directory in target_dirs:
    path = Path(args.input_path) / directory
    file_list = [item for item in glob.glob(f"{path}/**/*.txt", recursive=True)]

    docs = []
    for file_path in file_list:
        with open(file_path, 'r') as file:
            data = file.read()
            docs.append(nlp(data))

    # Combine documents into one and convert to JSON
    if docs:
        doc = Doc.from_docs(docs)
        json_file = doc.to_json()
        output_path = f"{os.path.join(args.output_path, directory)}_umls.json"
        with open(output_path, 'w') as file:
            json.dump(json_file, file)

    print(f"Finished processing {directory}")
