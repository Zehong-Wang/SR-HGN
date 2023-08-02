import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax


class SRHGNLayer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 node_dict, 
                 edge_dict, 
                 num_node_heads=4,
                 num_type_heads=4,
                 dropout=0.2, 
                 alpha=0.5, 
        ):
        super(SRHGNLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)

        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.num_node_heads = num_node_heads
        self.num_type_heads = num_type_heads

        self.node_linear = nn.ModuleList()
        self.edge_linear = nn.ModuleList()
        
        self.src_attn = nn.ModuleList()
        self.dst_attn = nn.ModuleList()

        self.sem_attn_src = nn.ModuleList()
        self.sem_attn_dst = nn.ModuleList()
        self.rel_attn = nn.ModuleList()

        for _ in range(self.num_types):
            self.node_linear.append(nn.Linear(input_dim, output_dim))

        for _ in range(self.num_relations):
            self.edge_linear.append(nn.Linear(input_dim, output_dim))
            self.src_attn.append(nn.Linear(input_dim, num_node_heads))
            self.dst_attn.append(nn.Linear(input_dim, num_node_heads))

            self.sem_attn_src.append(nn.Linear(output_dim, num_type_heads))
            self.sem_attn_dst.append(nn.Linear(output_dim, num_type_heads))
            self.rel_attn.append(nn.Linear(output_dim, num_type_heads))

        # Assign learnable relation embedding
        self.rel_emb = nn.Parameter(torch.randn(self.num_relations, output_dim), requires_grad=True)
        nn.init.xavier_normal_(self.rel_emb, gain=1.414)

        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float), requires_grad=False)
        self.epsilon = nn.Parameter(torch.FloatTensor([1e-9]), requires_grad=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, G, h):
        with G.local_scope():

            node_dict, edge_dict = self.node_dict, self.edge_dict

            for src, e, dst in G.canonical_etypes:
                # Extract subgraph for each relation
                sub_graph = G[src, e, dst]
                h_src = h[src]
                h_dst = h[dst]

                e_id = edge_dict[e]
                src_id = node_dict[src]
                dst_id = node_dict[dst]

                # Map source nodes into the space of target nodes, based on the relation type
                h_src = self.drop(self.edge_linear[e_id](h_src))
                
                # Map target nodes in advanced space, based on the target node type
                h_dst = self.drop(self.node_linear[dst_id](h_dst))

                # Calculate attention score similar to GAT
                src_attn = self.drop(self.src_attn[src_id](h_src)).unsqueeze(-1)
                dst_attn = self.drop(self.dst_attn[dst_id](h_dst)).unsqueeze(-1)

                # Combine attention score on the subgraph
                sub_graph.srcdata.update({'attn_src': src_attn})
                sub_graph.dstdata.update({'attn_dst': dst_attn})
                sub_graph.apply_edges(fn.u_add_v('attn_src', 'attn_dst', 'a'))
                a = F.leaky_relu(sub_graph.edata['a'])

                # Store node embedding and normalized attention score based on the relation type
                sub_graph.srcdata[f'v_{e_id}'] = h_src.view(
                    -1, self.num_node_heads, self.output_dim // self.num_node_heads) # Multi-head attention
                sub_graph.edata[f'a_{e_id}'] = self.drop(edge_softmax(sub_graph, a))

            # Aggregate type-level embedding like GAT
            # z: # nodes x # relations x # heads x dim [N x R x H x (D // H)]
            G.multi_update_all({etype: (fn.u_mul_e(f'v_{e_id}', f'a_{e_id}', 'm'), fn.sum('m', 'z')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer='stack')

            z = {}
            attns = {}
            rel_idx_start = 0

            for ntype in G.ntypes:
                dst_id = node_dict[ntype]
                h_dst = h[ntype]

                z_src = G.nodes[ntype].data['z'] # [N x R x H x (D // H)]
                num_nodes = z_src.shape[0]
                num_rel = z_src.shape[1]

                z_src = z_src.view(num_nodes, num_rel, self.output_dim) # [N x R x D]
                z_dst = self.drop(self.node_linear[dst_id](h_dst)) # [N x D]

                sem_attn = torch.zeros(num_nodes, num_rel, self.num_type_heads, device=z_src.device) # [N x R x H]
                rel_attn = torch.zeros(num_nodes, num_rel, self.num_type_heads, device=z_src.device) # [N x R x H]

                # Compute semantic-aware and relation-aware attention scores
                for rel_idx in range(num_rel):
                    normalize = lambda x: x / (torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon))

                    attn_idx = rel_idx_start + rel_idx
                    z_src_rel = z_src[:, rel_idx]

                    sem_attn_src = self.sem_attn_src[attn_idx](normalize(z_src_rel))
                    sem_attn_dst = self.sem_attn_dst[attn_idx](normalize(z_dst))

                    sem_attn[:, rel_idx] = sem_attn_src + sem_attn_dst
                    rel_attn[:, rel_idx] = self.rel_attn[attn_idx](self.rel_emb[attn_idx].unsqueeze(0)).repeat(num_nodes, 1)

                rel_idx_start += num_rel

                sem_attn = self.drop(F.softmax(F.leaky_relu(sem_attn), dim=1))
                rel_attn = self.drop(F.softmax(F.leaky_relu(rel_attn), dim=1))

                attn = self.alpha * sem_attn + (1 - self.alpha) * rel_attn

                # Multiple multi-head attention and node embedding
                z_dst = torch.mul(z_src.view(num_nodes, num_rel, self.num_type_heads, -1), attn.unsqueeze(-1)) # [N x R x H x (D // H)]
                
                # Concatenate all heads
                z_dst = z_dst.view(num_nodes, num_rel, self.output_dim) # [N x R x D]

                # Aggregate all relations and add skip-connection
                z_dst = F.gelu(z_dst.sum(1) + h[ntype])

                z[ntype] = normalize(z_dst)

                attns[ntype] = {'full': attn.detach().cpu().numpy(),
                                'semantic': sem_attn.detach().cpu().numpy(),
                                'relation': rel_attn.detach().cpu().numpy()}
            return z, attns



class SRHGN(nn.Module):
    def __init__(self, 
                 G, 
                 node_dict, 
                 edge_dict, 
                 input_dims, 
                 hidden_dim, 
                 output_dim,
                 num_layers=2, 
                 num_node_heads=4, 
                 num_type_heads=4,
                 alpha=0.5
        ):
        super(SRHGN, self).__init__()

        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.pre_transform = nn.ModuleList()
        for ntype, idx in node_dict.items():
            self.pre_transform.append(nn.Linear(input_dims[ntype], hidden_dim))

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                SRHGNLayer(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    node_dict=node_dict,
                    edge_dict=edge_dict,
                    num_node_heads=num_node_heads,
                    num_type_heads=num_type_heads,
                    alpha=alpha
                ))

        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, G, target):
        h = {}
        attns = []

        # Pre-transformation
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = self.pre_transform[n_id](G.nodes[ntype].data['x'])
            h[ntype] = F.gelu(h[ntype])

        for conv in self.convs:
            h, attn = conv(G, h)
            attns.append(attn)

        logits = self.out(h[target])

        return logits, h[target], attns
