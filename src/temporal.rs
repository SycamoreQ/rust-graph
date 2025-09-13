use crate::graph::{
    AttributeValue, Edge, Graph, GraphError, GraphID, GraphOps, LapPE, Node, NodeID, RWSE,
};
use candle_core::{Device, Error, Tensor};
use serde::{Deserialize, Serialize, de::value::UsizeDeserializer};
use std::collections::HashMap;

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

impl Node {
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
    pub id: GraphID,
    pub t_id: TemporalGraphID,
    pub nodes: HashMap<TemporalNodeID, Node>,
    pub edges: HashMap<TemporalEdgeID, Edge>,
    pub adjacency_list: HashMap<TemporalNodeID, Vec<TemporalEdgeID>>,
    pub reverse_adjacency_list: HashMap<TemporalNodeID, Vec<TemporalEdgeID>>,
    pub edge_type: HashMap<TemporalEdgeID, EdgeType>,
    pub attributes: HashMap<String, String>,
    pub slice_id: TemporalSliceID,
    pub batch_id: TemporalGraphBatchID,
    pub next_node_id: TemporalNodeID,
    pub next_edge_id: TemporalEdgeID,
    pub next_slice_id: TemporalSliceID,
    pub next_batch_id: TemporalGraphBatchID,
}

pub trait TemporalOps {
    fn new_timestep(&mut self) -> Self;
    fn from_temporal_graph(graph: TemporalGraph) -> Self;
    fn no_of_spatial_nodes(&self) -> usize;
    fn no_of_temporal_nodes(&self) -> usize;
    fn no_of_spatial_edges(&self) -> usize;
    fn no_of_temporal_edges(&self) -> usize;
    fn no_of_slices(&self) -> usize;
    fn no_of_batches(&self) -> usize;
}

#[derive(Debug, Clone)]
pub enum TemporalGraphError {
    InvalidTimeLabel,
    OutOfTimeBound,
    InvalidSpatialID,
    InvalidTemporalID,
}


impl TemporaGraph{
    pub fn new(is_directed: bool) -> Self {
        TemporaGraph {
            id: GraphID::new(),
            t_id: TemporalGraphID::new(),
            nodes: HashMap::new(),
            edges: HashMap::new(),
            adjacency_list: HashMap::new(),
            reverse_adjacency_list: HashMap::new(),
            edge_type: HashMap::new(),
            attributes: HashMap::new(),
            slice_id: TemporalSliceID::new(),
            batch_id: TemporalGraphBatchID::new(),
            next_node_id: TemporalNodeID::new(),
            next_edge_id: TemporalEdgeID::new(),
            next_slice_id: TemporalSliceID::new(),
            next_batch_id: TemporalGraphBatchID::new(),
        }
    }
    
    pub fn directed() -> Self{
        Self::new(true)
    }
    
    pub fn undirected() -> Self{
        Self::new(false)
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
        
    
        
        
        
        
        
    }
}
