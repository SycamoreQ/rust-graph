use crate::graph::{
    AttributeValue, Edge, Graph, GraphError, GraphID, GraphOps, LapPE, Node, NodeID, RWSE,
};
use candle_core::{Device, Error, Tensor};
use serde::{Deserialize, Serialize, de::value::UsizeDeserializer, forward_to_deserialize_any};
use std::{
    collections::{HashMap, HashSet},
    ops::Range,
};

pub type TemporalNodeID = usize;
pub type TemporalEdgeID = usize;
pub type TemporalSliceID = usize;
pub type TemporalGraphID = usize;
pub type TemporalGraphBatchID = usize;
pub type TemporalWeight = f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    Incoming,
    Outgoing,
    Temporal,
    Both,
}

#[derive(Debug, Clone)]
pub struct TemporalNode {
    pub id: TemporalNodeID,
    pub node_type: String,
    pub attributes: HashMap<String, AttributeValue>,
    #[serde(skip_serializing, skip_deserializing)]
    pub features: Option<Tensor>,
    pub timestep: TemporalSliceID,
}

impl TemporalNode {
    pub fn new(id: TemporalNodeID, node_type: String, step: TemporalSliceID) -> Self {
        Self {
            id,
            node_type,
            attributes: HashMap::new(),
            features: None,
            timestep: step,
        }
    }

    pub fn with_features(mut self, features: Tensor) -> Self {
        self.features = Some(features);
        self
    }

    pub fn with_attribute<V: Into<AttributeValue>>(mut self, key: String, value: V) -> Self {
        self.attributes.insert(key, value.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEdge {
    pub id: TemporalEdgeID,
    pub src: TemporalNodeID,
    pub dst: TemporalNodeID,
    pub attributes: HashMap<String, AttributeValue>,
    pub weight: f32,
    pub t_weight: TemporalWeight,

    #[serde(skip_serializing, skip_deserializing)]
    pub features: Option<Tensor>,
    pub timestep: TemporalSliceID,
}

impl TemporalEdge {
    pub fn new(id: TemporalEdgeID, src: NodeID, dst: NodeID, edge_type: String) -> Self {
        Self {
            id,
            src,
            dst,
            attributes: HashMap::new(),
            weight: 1.0,
            t_weight: TemporalWeight::new(),
            features: None,
            timestep: TemporalSliceID::new(),
        }
    }

    pub fn with_features(mut self, features: Tensor) -> Self {
        self.features = Some(features);
        self
    }

    pub fn with_attribute<V: Into<AttributeValue>>(mut self, key: String, value: V) -> Self {
        self.attributes.insert(key, value.into());
        self
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    pub fn reverse(&self) -> Self {
        Edge {
            id: self.id,
            src: self.dst,
            dst: self.src,
            edge_type: self.edge_type.clone(),
            features: self.features.clone(),
            weight: self.weight,
            attributes: self.attributes.clone(),
        }
    }

    pub fn with_t_weight(mut self, t_weight: TemporalWeight) -> Self {
        self.t_weight = t_weight;
        self
    }
}

#[derive(Debug, Clone)]
pub struct TemporalGraph {
    pub id: TemporalGraphID,
    pub nodes: HashMap<TemporalNodeID, Node>,
    pub edges: HashMap<TemporalEdgeID, Edge>,
    pub adjacency_list: HashMap<TemporalNodeID, Vec<TemporalEdgeID>>,
    pub reverse_adjacency_list: HashMap<TemporalNodeID, Vec<TemporalEdgeID>>,
    pub edge_type: HashMap<TemporalEdgeID, EdgeType>,
    pub attributes: HashMap<String, String>,
    pub slice_id: TemporalSliceID,
    pub next_slice_id: TemporalSliceID,
    pub prev_slice_id: TemporalSliceID,
}

pub trait TemporalOps {
    fn add_node(&mut self, node: TemporalNode) -> Result<(), TemporalGraphError>;
    fn remove_node(&mut self, node_id: TemporalNodeID) -> Result<TemporalNode, TemporalGraphError>;
    fn get_node(&self, node_id: TemporalNodeID) -> Option<&TemporalNode>;
    fn has_node(&self, node_id: TemporalNodeID) -> bool;
    fn node_count(&self) -> usize;
    fn node_ids(&self) -> Vec<TemporalNodeID>;
    fn add_edge(&mut self, edge: TemporalEdge) -> Result<(), TemporalGraphError>;
    fn remove_edge(&mut self, edge_id: TemporalEdgeID) -> Result<TemporalEdge, TemporalGraphError>;
    fn get_edge(&self, edge_id: TemporalEdgeID) -> Option<&TemporalEdge>;
    fn has_edge(&self, edge_id: TemporalEdgeID) -> bool;
    fn has_edge_between(&self, src: TemporalNodeID, dst: TemporalNodeID) -> bool;
    fn edge_count(&self) -> usize;
    fn edge_ids(&self) -> Vec<TemporalEdgeID>;
    fn neighbors(&self, node_id: TemporalNodeID, direction: EdgeDirection) -> Vec<TemporalNodeID>;
    fn to_adjacency_matrix(&self) -> Vec<Vec<f32>>;
    fn to_edge_list(&self) -> Vec<(TemporalNodeID, TemporalNodeID, f32)>;
    fn to_tensor_format(&self) -> Result<(Tensor, Tensor, Option<Tensor>), TemporalGraphError>;
}

#[derive(Debug, Clone)]
pub enum TemporalGraphError {
    InvalidTimeLabel,
    OutOfTimeBound,
    InvalidSpatialID,
    InvalidTemporalID,
}

#[derive(Debug, Clone)]
pub struct TemporalSlice {
    pub id: TemporalSliceID,
    pub start_time: usize,
    pub end_time: usize,
    pub nodes: HashMap<TemporalNodeID, TemporalNode>,
    pub edges: HashMap<TemporalEdgeID, TemporalEdge>,
    pub adjacency_list: HashMap<TemporalNodeID, Vec<TemporalEdgeID>>,
    pub reverse_adjacency: HashMap<TemporalNodeID, Vec<TemporalEdgeID>>,
    pub metadata: HashMap<String, AttributeValue>,
    pub is_directed: bool,
}

pub trait TemporalSliceOps {
    //Todo : After TemporalGraph impl
}

impl TemporalGraph {
    pub fn new(id: TemporalGraphID , is_directed: bool) -> Self {
        TemporalGraph {
            id ,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            adjacency_list: HashMap::new(),
            reverse_adjacency_list: HashMap::new(),
            edge_type: HashMap::new(),
            attributes: HashMap::new(),
            slice_id: TemporalSliceID::new(),
            next_slice_id: TemporalSliceID::new(),
            prev_slice_id: TemporalSliceID::new(),
        }
    }

    pub fn instantaneous_degree(
        &self,
        node_id: TemporalNodeID,
        timestep: TemporalSliceID,
        direction: EdgeDirection,
    ) -> f32 {
        let mut degree = 0.0;

        for edge in self.edges.values() {
            if edge.timestep != timestep {
                continue;
            }

            match direction {
                EdgeDirection::Outgoing => {
                    if edge.src == node_id {
                        degree += edge.weight;
                    }
                }
                EdgeDirection::Incoming => {
                    if edge.dst == node_id {
                        degree += edge.weight;
                    }
                }
                EdgeDirection::Both => {
                    if edge.src == node_id || edge.dst == node_id {
                        degree += edge.weight;
                    }
                }
                EdgeDirection::Temporal => {
                    if (edge.src == node_id || edge.dst == node_id) && edge.timestep <= timestep {
                        degree += edge.t_weight;
                    }
                }
            }
        }

        degree
    }

    pub fn get_temporal_neighbors(
        &self,
        node_id: TemporalNodeID,
        start_time: TemporalSliceID,
        end_time: TemporalSliceID,
    ) -> Vec<TemporalNodeID> {
        let mut neighbors = HashSet::new();

        for edge in self.edges.values() {
            if edge.timestep >= start_time && edge.timestep <= end_time {
                if edge.src == node_id {
                    neighbors.insert(edge.dst);
                } else if edge.dst == node_id {
                    neighbors.insert(edge.src);
                }
            }
        }

        neighbors.into_iter().collect()
    }

    pub fn temporal_density(&self, timestep: TemporalSliceID) -> Result<f64, GraphError> {
        let n = self.nodes.len() as f64;
        let m = self.edges.len() as f64;

        if n <= 1.0 {
            return 1.0;
        } else {
            if !timestep && self.is_directed() {
                Ok(m / (n * (n - 1.0)))
            } else {
                Ok(2.0 * m / (n * (n - 1.0)))
            }
        }
    }

    pub fn temporal_subgraph(
        &self,
        node_ids: &[TemporalNodeID],
        forward_node_ids: &[TemporalNodeID],
        reverse_node_ids: &[TemporalNodeID],
        timestep: &[TemporalSliceID],
    ) -> Result<TemporalGraph, GraphError> {
        let mut subgraph = TemporalGraph::new(false);
        let prev_node_set: HashSet<TemporalNodeID> = reverse_node_ids.iter().clone().collect();
        let node_set: HashSet<TemporalNodeID> = node_ids.iter().clone().collect();
        let forward_node_set: HashSet<TemporalNodeID> = forward_node_ids.iter().clone().collect();

        ///Takes care of current present subgraph addition
        for &node_id in node_ids {
            if let Ok(node) = self.get_node(node_id) {
                subgraph.add_node(node);
            }
        }

        ///Choose a timeslice to add nodes from the past and future.
        for &time in timestep.iter() {}
    }
}
