import utils as u
import torch
import os
import pandas as pd
import numpy as np


class Wikipedia_Dataset():
    def __init__(self,args):
        self.ecols = u.Namespace({'FromNodeId': 0,
                                  'ToNodeId': 1,
                                  'Weight': 2,
                                  'TimeStep': 3,
                                  'feats': 4
                                })
        args.wikipedia_args = u.Namespace(args.wikipedia_args)
        
        edges = self.load_edges(args.wikipedia_args)
        edges = self.make_contigous_node_ids(edges)
        num_nodes = edges[:,[self.ecols.FromNodeId,
                            self.ecols.ToNodeId]].unique().size(0)

        timesteps = u.aggregate_by_time(edges[:,self.ecols.TimeStep],args.wikipedia_args.aggr_time)
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        edges[:,self.ecols.TimeStep] = timesteps

        edges[:,self.ecols.Weight] = self.cluster_negs_and_positives(edges[:,self.ecols.Weight])


        #add the reversed link to make the graph undirected
        edges = torch.cat([edges,edges[:,[self.ecols.ToNodeId,
                                          self.ecols.FromNodeId,
                                          self.ecols.Weight,
                                          self.ecols.TimeStep]]])
        
        
        

        #separate classes
        sp_indices = edges[:,[self.ecols.FromNodeId,
                              self.ecols.ToNodeId,
                              self.ecols.TimeStep]].t()
        sp_values = edges[:,self.ecols.Weight]


        neg_mask = sp_values == 0

        neg_sp_indices = sp_indices[:,neg_mask]
        neg_sp_values = sp_values[neg_mask]
        neg_sp_edges = torch.sparse.LongTensor(neg_sp_indices
                                              ,neg_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        pos_mask = sp_values == 1

        pos_sp_indices = sp_indices[:,pos_mask]
        pos_sp_values = sp_values[pos_mask]

        pos_sp_edges = torch.sparse.LongTensor(pos_sp_indices
                                              ,pos_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        #scale positive class to separate after adding
        pos_sp_edges *= 1000

        #we substract the neg_sp_edges to make the values positive
        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()

        #separating negs and positive edges per edge/timestamp
        vals = sp_edges._values()
        neg_vals = vals%1000
        pos_vals = vals//1000
        #We add the negative and positive scores and do majority voting
        vals = pos_vals - neg_vals
        #creating labels new_vals -> the label of the edges
        new_vals = torch.zeros(vals.size(0),dtype=torch.long)
        new_vals[vals>0] = 1
        new_vals[vals<=0] = 0
        indices_labels = torch.cat([sp_edges._indices().t(),new_vals.view(-1,1)],dim=1)

        #the weight of the edges (vals), is simply the number of edges between two entities at each time_step
        vals = pos_vals + neg_vals
        
        
        # print(indices_labels)
        self.nodes_labels_times = self.load_node_labels(indices_labels)
        self.edges = {'idx': indices_labels, 'vals': vals}
        self.num_nodes = num_nodes
        self.num_classes = 2
        self.feats_per_node = 172
        
    def load_node_labels(self, edges):
        nodes_labels_times = edges[:,[0,3,2]]

        return nodes_labels_times
        
    def load_edges(self,args):
        file = os.path.join(args.folder,args.edges_file)
        with open(file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = torch.tensor(edges,dtype = torch.long)
        return edges
    
    def make_contigous_node_ids(self,edges):
        new_edges = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]] = new_edges
        return edges
    
    def cluster_negs_and_positives(self,ratings):
        pos_indices = ratings > 0
        neg_indices = ratings <= 0
        ratings[pos_indices] = 1
        ratings[neg_indices] = 0
        return ratings



class Reddit_Dataset():
    def __init__(self,args):
        self.ecols = u.Namespace({'FromNodeId': 0,
                                  'ToNodeId': 1,
                                  'Weight': 2,
                                  'TimeStep': 3
                                })
        args.reddit_args = u.Namespace(args.reddit_args)
        
        edges = self.load_edges(args.reddit_args)
        edges = self.make_contigous_node_ids(edges)
        num_nodes = edges[:,[self.ecols.FromNodeId,
                            self.ecols.ToNodeId]].unique().size(0)

        timesteps = u.aggregate_by_time(edges[:,self.ecols.TimeStep],args.reddit_args.aggr_time)
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        edges[:,self.ecols.TimeStep] = timesteps

        edges[:,self.ecols.Weight] = self.cluster_negs_and_positives(edges[:,self.ecols.Weight])


        #add the reversed link to make the graph undirected
        edges = torch.cat([edges,edges[:,[self.ecols.ToNodeId,
                                          self.ecols.FromNodeId,
                                          self.ecols.Weight,
                                          self.ecols.TimeStep]]])

        #separate classes
        sp_indices = edges[:,[self.ecols.FromNodeId,
                              self.ecols.ToNodeId,
                              self.ecols.TimeStep]].t()
        sp_values = edges[:,self.ecols.Weight]


        neg_mask = sp_values == -1

        neg_sp_indices = sp_indices[:,neg_mask]
        neg_sp_values = sp_values[neg_mask]
        neg_sp_edges = torch.sparse.LongTensor(neg_sp_indices
                                              ,neg_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        pos_mask = sp_values == 1

        pos_sp_indices = sp_indices[:,pos_mask]
        pos_sp_values = sp_values[pos_mask]

        pos_sp_edges = torch.sparse.LongTensor(pos_sp_indices
                                              ,pos_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        #scale positive class to separate after adding
        pos_sp_edges *= 1000

        #we substract the neg_sp_edges to make the values positive
        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()

        #separating negs and positive edges per edge/timestamp
        vals = sp_edges._values()
        neg_vals = vals%1000
        pos_vals = vals//1000
        #We add the negative and positive scores and do majority voting
        vals = pos_vals - neg_vals
        #creating labels new_vals -> the label of the edges
        new_vals = torch.zeros(vals.size(0),dtype=torch.long)
        new_vals[vals>0] = 1
        new_vals[vals<=0] = 0
        indices_labels = torch.cat([sp_edges._indices().t(),new_vals.view(-1,1)],dim=1)

        #the weight of the edges (vals), is simply the number of edges between two entities at each time_step
        vals = pos_vals + neg_vals


        self.nodes_labels_times = self.load_node_labels(indices_labels)
        self.edges = {'idx': indices_labels, 'vals': vals}
        self.num_nodes = num_nodes
        self.num_classes = 2
        self.feats_per_node = 172
        
    def load_node_labels(self, edges):
        
        nodes_labels_times = edges[:,[0,3,2]]

        return nodes_labels_times



    def load_edges(self,args):
        file = os.path.join(args.folder,args.edges_file)
        with open(file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = torch.tensor(edges,dtype = torch.long)
        return edges
    
    def make_contigous_node_ids(self,edges):
        new_edges = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]] = new_edges
        return edges
    
    def cluster_negs_and_positives(self,ratings):
        pos_indices = ratings > 0
        neg_indices = ratings <= 0
        ratings[pos_indices] = 1
        ratings[neg_indices] = -1
        return ratings
    
    
class Flight_Dataset():
    def __init__(self,args):
        self.ecols = u.Namespace({'FromNodeId': 0,
                                  'ToNodeId': 1,
                                  'Weight': 2,
                                  'TimeStep': 3
                                })
        args.flight_args = u.Namespace(args.flight_args)
        
        edges = self.load_edges(args.flight_args)
        edges = self.make_contigous_node_ids(edges)
        num_nodes = edges[:,[self.ecols.FromNodeId,
                            self.ecols.ToNodeId]].unique().size(0)

        timesteps = u.aggregate_by_time(edges[:,self.ecols.TimeStep],args.flight_args.aggr_time)
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        edges[:,self.ecols.TimeStep] = timesteps

        edges[:,self.ecols.Weight] = self.cluster_negs_and_positives(edges[:,self.ecols.Weight])


        #add the reversed link to make the graph undirected
        edges = torch.cat([edges,edges[:,[self.ecols.ToNodeId,
                                          self.ecols.FromNodeId,
                                          self.ecols.Weight,
                                          self.ecols.TimeStep]]])

        #separate classes
        sp_indices = edges[:,[self.ecols.FromNodeId,
                              self.ecols.ToNodeId,
                              self.ecols.TimeStep]].t()
        sp_values = edges[:,self.ecols.Weight]


        neg_mask = sp_values == -1

        neg_sp_indices = sp_indices[:,neg_mask]
        neg_sp_values = sp_values[neg_mask]
        neg_sp_edges = torch.sparse.LongTensor(neg_sp_indices
                                              ,neg_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        pos_mask = sp_values == 1

        pos_sp_indices = sp_indices[:,pos_mask]
        pos_sp_values = sp_values[pos_mask]

        pos_sp_edges = torch.sparse.LongTensor(pos_sp_indices
                                              ,pos_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        #scale positive class to separate after adding
        pos_sp_edges *= 1000

        #we substract the neg_sp_edges to make the values positive
        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()

        #separating negs and positive edges per edge/timestamp
        vals = sp_edges._values()
        neg_vals = vals%1000
        pos_vals = vals//1000
        #We add the negative and positive scores and do majority voting
        vals = pos_vals - neg_vals
        #creating labels new_vals -> the label of the edges
        new_vals = torch.zeros(vals.size(0),dtype=torch.long)
        new_vals[vals>0] = 1
        new_vals[vals<=0] = 0
        indices_labels = torch.cat([sp_edges._indices().t(),new_vals.view(-1,1)],dim=1)

        #the weight of the edges (vals), is simply the number of edges between two entities at each time_step
        vals = pos_vals + neg_vals


        self.edges = {'idx': indices_labels, 'vals': vals}
        self.num_nodes = num_nodes
        self.num_classes = 2
        self.feats_per_node = 172


    def load_edges(self,args):
        file = os.path.join(args.folder,args.edges_file)
        with open(file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = torch.tensor(edges,dtype = torch.long)
        return edges
    
    def make_contigous_node_ids(self,edges):
        new_edges = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]] = new_edges
        return edges
    
    def cluster_negs_and_positives(self,ratings):
        pos_indices = ratings > 0
        neg_indices = ratings <= 0
        ratings[pos_indices] = 1
        ratings[neg_indices] = -1
        return ratings
    
class Mooc_Dataset():
    def __init__(self,args):
        self.ecols = u.Namespace({'FromNodeId': 0,
                                  'ToNodeId': 1,
                                  'Weight': 2,
                                  'TimeStep': 3
                                })
        args.mooc_args = u.Namespace(args.mooc_args)
        
        edges = self.load_edges(args.mooc_args)
        edges = self.make_contigous_node_ids(edges)
        num_nodes = edges[:,[self.ecols.FromNodeId,
                            self.ecols.ToNodeId]].unique().size(0)

        timesteps = u.aggregate_by_time(edges[:,self.ecols.TimeStep],args.mooc_args.aggr_time)
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        edges[:,self.ecols.TimeStep] = timesteps

        edges[:,self.ecols.Weight] = self.cluster_negs_and_positives(edges[:,self.ecols.Weight])


        #add the reversed link to make the graph undirected
        edges = torch.cat([edges,edges[:,[self.ecols.ToNodeId,
                                          self.ecols.FromNodeId,
                                          self.ecols.Weight,
                                          self.ecols.TimeStep]]])

        #separate classes
        sp_indices = edges[:,[self.ecols.FromNodeId,
                              self.ecols.ToNodeId,
                              self.ecols.TimeStep]].t()
        sp_values = edges[:,self.ecols.Weight]


        neg_mask = sp_values == -1

        neg_sp_indices = sp_indices[:,neg_mask]
        neg_sp_values = sp_values[neg_mask]
        neg_sp_edges = torch.sparse.LongTensor(neg_sp_indices
                                              ,neg_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        pos_mask = sp_values == 1

        pos_sp_indices = sp_indices[:,pos_mask]
        pos_sp_values = sp_values[pos_mask]

        pos_sp_edges = torch.sparse.LongTensor(pos_sp_indices
                                              ,pos_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        #scale positive class to separate after adding
        pos_sp_edges *= 1000

        #we substract the neg_sp_edges to make the values positive
        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()

        #separating negs and positive edges per edge/timestamp
        vals = sp_edges._values()
        neg_vals = vals%1000
        pos_vals = vals//1000
        #We add the negative and positive scores and do majority voting
        vals = pos_vals - neg_vals
        #creating labels new_vals -> the label of the edges
        new_vals = torch.zeros(vals.size(0),dtype=torch.long)
        new_vals[vals>0] = 1
        new_vals[vals<=0] = 0
        indices_labels = torch.cat([sp_edges._indices().t(),new_vals.view(-1,1)],dim=1)

        #the weight of the edges (vals), is simply the number of edges between two entities at each time_step
        vals = pos_vals + neg_vals


        self.nodes_labels_times = self.load_node_labels(indices_labels)
        self.edges = {'idx': indices_labels, 'vals': vals}
        self.num_nodes = num_nodes
        self.num_classes = 2
        self.feats_per_node = 172
        
    def load_node_labels(self, edges):
        
        nodes_labels_times = edges[:,[0,3,2]]

        return nodes_labels_times



    def load_edges(self,args):
        file = os.path.join(args.folder,args.edges_file)
        with open(file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = torch.tensor(edges,dtype = torch.long)
        return edges
    
    def make_contigous_node_ids(self,edges):
        new_edges = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]] = new_edges
        return edges
    
    def cluster_negs_and_positives(self,ratings):
        pos_indices = ratings > 0
        neg_indices = ratings <= 0
        ratings[pos_indices] = 1
        ratings[neg_indices] = -1
        return ratings
    
    
class ML25M_Dataset():
    def __init__(self,args):
        self.ecols = u.Namespace({'FromNodeId': 0,
                                  'ToNodeId': 1,
                                  'Weight': 2,
                                  'TimeStep': 3
                                })
        args.ml25m_args = u.Namespace(args.ml25m_args)
        
        edges = self.load_edges(args.ml25m_args)
        edges = self.make_contigous_node_ids(edges)
        num_nodes = edges[:,[self.ecols.FromNodeId,
                            self.ecols.ToNodeId]].unique().size(0)

        timesteps = u.aggregate_by_time(edges[:,self.ecols.TimeStep],args.ml25m_args.aggr_time)
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        edges[:,self.ecols.TimeStep] = timesteps

        edges[:,self.ecols.Weight] = self.cluster_negs_and_positives(edges[:,self.ecols.Weight])


        #add the reversed link to make the graph undirected
        edges = torch.cat([edges,edges[:,[self.ecols.ToNodeId,
                                          self.ecols.FromNodeId,
                                          self.ecols.Weight,
                                          self.ecols.TimeStep]]])

        #separate classes
        sp_indices = edges[:,[self.ecols.FromNodeId,
                              self.ecols.ToNodeId,
                              self.ecols.TimeStep]].t()
        sp_values = edges[:,self.ecols.Weight]


        neg_mask = sp_values == -1

        neg_sp_indices = sp_indices[:,neg_mask]
        neg_sp_values = sp_values[neg_mask]
        neg_sp_edges = torch.sparse.LongTensor(neg_sp_indices
                                              ,neg_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        pos_mask = sp_values == 1

        pos_sp_indices = sp_indices[:,pos_mask]
        pos_sp_values = sp_values[pos_mask]

        pos_sp_edges = torch.sparse.LongTensor(pos_sp_indices
                                              ,pos_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        #scale positive class to separate after adding
        pos_sp_edges *= 1000

        #we substract the neg_sp_edges to make the values positive
        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()

        #separating negs and positive edges per edge/timestamp
        vals = sp_edges._values()
        neg_vals = vals%1000
        pos_vals = vals//1000
        #We add the negative and positive scores and do majority voting
        vals = pos_vals - neg_vals
        #creating labels new_vals -> the label of the edges
        new_vals = torch.zeros(vals.size(0),dtype=torch.long)
        new_vals[vals>0] = 1
        new_vals[vals<=0] = 0
        indices_labels = torch.cat([sp_edges._indices().t(),new_vals.view(-1,1)],dim=1)

        #the weight of the edges (vals), is simply the number of edges between two entities at each time_step
        vals = pos_vals + neg_vals


        self.nodes_labels_times = self.load_node_labels(indices_labels)
        self.edges = {'idx': indices_labels, 'vals': vals}
        self.num_nodes = num_nodes
        self.num_classes = 2
        self.feats_per_node = 3
        
    def load_node_labels(self, edges):
        
        nodes_labels_times = edges[:,[0,3,2]]

        return nodes_labels_times



    def load_edges(self,args):
        file = os.path.join(args.folder,args.edges_file)
        with open(file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = torch.tensor(edges,dtype = torch.long)
        return edges
    
    def make_contigous_node_ids(self,edges):
        new_edges = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]] = new_edges
        return edges
    
    def cluster_negs_and_positives(self,ratings):
        pos_indices = ratings > 2.5
        neg_indices = ratings <= 2.5
        ratings[pos_indices] = 1
        ratings[neg_indices] = -1
        return ratings
    
    
class dgraphfin_Dataset():
    def __init__(self,args):
        self.ecols = u.Namespace({'FromNodeId': 0,
                                  'ToNodeId': 1,
                                  'Weight': 2,
                                  'TimeStep': 3
                                })
        args.dgraphfin_args = u.Namespace(args.dgraphfin_args)
        
        edges = self.load_edges(args.dgraphfin_args)
        edges = self.make_contigous_node_ids(edges)
        num_nodes = edges[:,[self.ecols.FromNodeId,
                            self.ecols.ToNodeId]].unique().size(0)

        timesteps = u.aggregate_by_time(edges[:,self.ecols.TimeStep],args.dgraphfin_args.aggr_time)
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        edges[:,self.ecols.TimeStep] = timesteps

        edges[:,self.ecols.Weight] = self.cluster_negs_and_positives(edges[:,self.ecols.Weight])


        #add the reversed link to make the graph undirected
        edges = torch.cat([edges,edges[:,[self.ecols.ToNodeId,
                                          self.ecols.FromNodeId,
                                          self.ecols.Weight,
                                          self.ecols.TimeStep]]])

        #separate classes
        sp_indices = edges[:,[self.ecols.FromNodeId,
                              self.ecols.ToNodeId,
                              self.ecols.TimeStep]].t()
        sp_values = edges[:,self.ecols.Weight]


        neg_mask = sp_values == -1

        neg_sp_indices = sp_indices[:,neg_mask]
        neg_sp_values = sp_values[neg_mask]
        neg_sp_edges = torch.sparse.LongTensor(neg_sp_indices
                                              ,neg_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        pos_mask = sp_values == 1

        pos_sp_indices = sp_indices[:,pos_mask]
        pos_sp_values = sp_values[pos_mask]

        pos_sp_edges = torch.sparse.LongTensor(pos_sp_indices
                                              ,pos_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        #scale positive class to separate after adding
        pos_sp_edges *= 1000

        #we substract the neg_sp_edges to make the values positive
        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()

        #separating negs and positive edges per edge/timestamp
        vals = sp_edges._values()
        neg_vals = vals%1000
        pos_vals = vals//1000
        #We add the negative and positive scores and do majority voting
        vals = pos_vals - neg_vals
        #creating labels new_vals -> the label of the edges
        new_vals = torch.zeros(vals.size(0),dtype=torch.long)
        new_vals[vals>0] = 1
        new_vals[vals<=0] = 0
        indices_labels = torch.cat([sp_edges._indices().t(),new_vals.view(-1,1)],dim=1)

        #the weight of the edges (vals), is simply the number of edges between two entities at each time_step
        vals = pos_vals + neg_vals


        self.nodes_labels_times = self.load_node_labels(indices_labels)
        self.edges = {'idx': indices_labels, 'vals': vals}
        self.num_nodes = num_nodes
        self.num_classes = 4
        self.feats_per_node = 17
        
    def load_node_labels(self, edges):
        nodes_labels_times = edges[:,[0,3,2]]

        return nodes_labels_times
        
    def load_edges(self,args):
        file = os.path.join(args.folder,args.edges_file)
        with open(file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = torch.tensor(edges,dtype = torch.long)
        return edges
    
    def make_contigous_node_ids(self,edges):
        new_edges = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]] = new_edges
        return edges
    
    def cluster_negs_and_positives(self,ratings):
        pos_indices = ratings > 0
        neg_indices = ratings <= 0
        ratings[pos_indices] = 1
        ratings[neg_indices] = -1
        return ratings