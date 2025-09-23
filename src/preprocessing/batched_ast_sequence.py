# from dataclasses import dataclass
#
# from torch_geometric.data import Data
#
# from preprocessing.vocab import VOCAB
# from preprocessing.assignment_ast import *
# import torch
# import numpy as np
#
#
# @dataclass
# class BatchedSequences:
#     lens: torch.tensor
#     batched: torch.tensor
#     graph_indices: torch.tensor
#     batched_graph: Data
#
#
# @dataclass
# class ProcessedSequences:
#     lens: torch.tensor  # (batch_size,) -> contains the length of each sequence
#     batched: torch.tensor  # (batch_size, max_len) -> contains the vocab_ids of non-graph elements
#     graph_indices: torch.tensor  # (batch_size, max_len) -> contains 1 if the element is a graph node
#     graphs: list[list[ASTNode]]  # contains the graphs of each sequence
#     device: str
#
#     def __post_init__(self):
#         if len(VOCAB) == 0:
#             raise ValueError("VOCAB not initialized")
#
#     @classmethod
#     def from_seqs(cls, seqs: list[list[tuple[ASTNode, ASTNode]]], device=None):
#         lens = torch.tensor([len(seq) for seq in seqs], dtype=torch.long)  # needs to be on CPU
#         max_len = lens.max().item()
#         batch_size = len(seqs)
#         return cls(
#             lens,
#             torch.zeros((batch_size, max_len), dtype=torch.long).to(device),
#             torch.zeros((batch_size, max_len), dtype=torch.bool).to(device),
#             [],
#             device
#         )
#
#     def process_node(self, node: ASTNode, i: int, j: int):
#         if len(self.graphs) <= i:
#             self.graphs.append([])
#         if isinstance(node, EpsilonNode):
#             self.batched[i, j] = VOCAB['EPSILON']
#         elif isinstance(node, NullNode):
#             self.batched[i, j] = VOCAB['NULL']
#         elif isinstance(node, EmptyNode):
#             self.batched[i, j] = VOCAB['EMPTY']
#         elif isinstance(node, PropositionNode):
#             self.batched[i, j] = VOCAB[node.proposition]
#         else:
#             self.graph_indices[i, j] = 1
#             self.graphs[i].append(node)
#
#     def __getitem__(self, index):
#         graphs = [g for i in index for g in self.graphs[i]]
#         batched_graph = ProcessedSequences.asts_to_batched_graph(graphs, self.device)
#         return BatchedSequences(
#             self.lens[index],
#             self.batched[index],
#             self.graph_indices[index],
#             batched_graph
#         )
#
#     def all(self):
#         return self[range(len(self.batched))]
#
#     @staticmethod
#     def asts_to_batched_graph(asts: list[ASTNode], device=None) -> Data:
#         nodes = []
#         edges = []
#         root_indices = []
#         start_index = 0
#         for ast in asts:
#             assert isinstance(ast, AndNode) or isinstance(ast, OrNode)
#             n, e = ProcessedSequences.ast_to_graph(ast, start_index)
#             nodes += n
#             edges += e
#             start_index += len(n)
#             root_indices.append(start_index - 1)
#         return Data(
#             x=torch.tensor(nodes, dtype=torch.long),
#             edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
#             root_indices=torch.tensor(root_indices, dtype=torch.long)
#         ).to(device)
#
#     @staticmethod
#     def ast_to_graph(ast: ASTNode, start_index: int = 0) -> tuple[list, list]:
#         if isinstance(ast, EmptyNode):
#             return [VOCAB['EMPTY']], []
#         if isinstance(ast, PropositionNode):
#             return [VOCAB[ast.proposition]], []
#         nodes = []
#         edges = []
#         graph_indices = []
#         for c in ast.children:
#             n, e = ProcessedSequences.ast_to_graph(c, start_index)
#             nodes += n
#             edges += e
#             graph_indices.append(start_index)
#             start_index += len(n)
#
#         new_node = VOCAB['AND'] if isinstance(ast, AndNode) else VOCAB['OR']
#         nodes.append(new_node)
#         for i in graph_indices:
#             edges.append([i, start_index])
#         return nodes, edges
#
#
# class BatchedASTSequence:
#     def __init__(self, seqs: list[list[tuple[ASTNode, ASTNode]]], device=None):
#         self.device = device
#         self.reach_seqs, self.avoid_seqs = self.batch(seqs, device)
#
#     @staticmethod
#     def batch(seqs: list[list[tuple[ASTNode, ASTNode]]], device=None) -> tuple[ProcessedSequences, ProcessedSequences]:
#         reach_seqs = ProcessedSequences.from_seqs(seqs, device)
#         avoid_seqs = ProcessedSequences.from_seqs(seqs, device)
#         for i, seq in enumerate(seqs):
#             for j, (reach, avoid) in enumerate(seq):
#                 reach_seqs.process_node(reach, i, j)
#                 avoid_seqs.process_node(avoid, i, j)
#         return reach_seqs, avoid_seqs
#
#     def __getitem__(self, index: np.ndarray):
#         """
#         Returns a sub-batch of the given sequences.
#         """
#         return self.reach_seqs[index], self.avoid_seqs[index]
#
#     def all(self):
#         return self.reach_seqs.all(), self.avoid_seqs.all()
