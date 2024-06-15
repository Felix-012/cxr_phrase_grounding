import json
import torch

TYPE_TO_INT = {"located_at": 0, "modify": 1}


# Example JSON data as a string
json_data = '''
{
  "data_source": "MIMIC-CXR",
  "data_split": "train",
  "entities": {
    "1": {
      "end_ix": 2,
      "label": "ANAT-DP",
      "relations": [],
      "start_ix": 2,
      "tokens": "lungs"
    },
    "2": {
      "end_ix": 4,
      "label": "OBS-DP",
      "relations": [
        [
          "located_at",
          "1"
        ]
      ],
      "start_ix": 4,
      "tokens": "clear"
    },
    "3": {
      "end_ix": 6,
      "label": "ANAT-DP",
      "relations": [],
      "start_ix": 6,
      "tokens": "Cardiomediastinal"
    },
    "4": {
      "end_ix": 8,
      "label": "ANAT-DP",
      "relations": [],
      "start_ix": 8,
      "tokens": "hilar"
    },
    "5": {
      "end_ix": 9,
      "label": "ANAT-DP",
      "relations": [
        [
          "modify",
          "3"
        ],
        [
          "modify",
          "4"
        ]
      ],
      "start_ix": 9,
      "tokens": "contours"
    },
    "6": {
      "end_ix": 11,
      "label": "OBS-DP",
      "relations": [
        [
          "located_at",
          "3"
        ],
        [
          "located_at",
          "4"
        ]
      ],
      "start_ix": 11,
      "tokens": "normal"
    },
    "7": {
      "end_ix": 16,
      "label": "ANAT-DP",
      "relations": [],
      "start_ix": 16,
      "tokens": "pleural"
    },
    "8": {
      "end_ix": 17,
      "label": "OBS-DA",
      "relations": [
        [
          "located_at",
          "7"
        ]
      ],
      "start_ix": 17,
      "tokens": "effusions"
    },
    "9": {
      "end_ix": 19,
      "label": "OBS-DA",
      "relations": [],
      "start_ix": 19,
      "tokens": "pneumothorax"
    }
  },
  "text": "The lungs are clear . Cardiomediastinal and hilar contours are normal . There are no pleural effusions or pneumothorax ."
}
'''

# Initialize lists for nodes and edges
nodes = []
edges = []
edge_types = []
node_to_index = {}
index = 0

data = json.loads(json_data)

# Map each entity to a node index
for entity_id, entity_info in data['entities'].items():
    node_to_index[entity_id] = index
    nodes.append([index, entity_info['label'], entity_info['tokens']])
    index += 1

in_degree_count = {index: 0 for index in range(len(nodes))}
out_degree_count = {index: 0 for index in range(len(nodes))}

# Create edges based on relations
for entity_id, entity_info in data['entities'].items():
    for relation in entity_info['relations']:
        edge_type, related_entity_id = relation
        source_index = node_to_index[entity_id]
        target_index = node_to_index[related_entity_id]
        edges.append([source_index, target_index])
        edge_types.append(TYPE_TO_INT[edge_type])
        out_degree_count[source_index] += 1
        in_degree_count[target_index] += 1

# Convert lists to tensors
input_nodes = torch.LongTensor([n[0] for n in nodes])  # Node indices
input_edges = torch.LongTensor(edges)
attn_edge_type = torch.LongTensor(edge_types)

# Example tensor for attention bias (simplified, normally more complex)
attn_bias = torch.zeros(len(nodes), len(nodes))  # A simple example, typically more complex logic needed

in_degree = torch.LongTensor([in_degree_count[i] for i in range(len(nodes))])
out_degree = torch.LongTensor([out_degree_count[i] for i in range(len(nodes))])

# Spatial positions are more complex to calculate; here we simply use indices
spatial_pos = torch.LongTensor([n[0] for n in nodes])
