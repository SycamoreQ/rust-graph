use crate::graph::{
    AttributeValue, Edge, Graph, GraphError, GraphID, GraphOps, LapPE, Node, NodeID, RWSE,
};
use candle_core::{Device, Error, Tensor};
use petgraph::visit::Time;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    ops::Range,
};

pub type TemporalNodeID = usize;
pub type TemporalEdgeID = usize;
pub type TemporalSliceID = usize;
pub type TemporalGraphID = usize;
pub type TemporalGraphBatchID = usize;
pub type TemporalWeight = f32;
pub type Timestamp = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    Incoming,
    Outgoing,
    Temporal,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalQueryType {
    Snapshot,        // Graph at specific time
    Interval,        // Graph over time range
    Evolution,       // How graph changes over time
    Aggregated,      // Combined view across time
}

#[derive(Debug, Clone)]
pub struct TemporalNode {
    pub id: TemporalNodeID,
    pub node_type: String,
    pub attributes: HashMap<String, AttributeValue>,
    #[serde(skip_serializing, skip_deserializing)]
    pub features: Option<Tensor>,
    pub birth_time: Timestamp,
    pub death_time: Option<Timestamp>,
    pub activity_periods: Vec<Range<Timestamp>>,
}

impl TemporalNode {
    pub fn new(id: TemporalNodeID, node_type: String, birth_time: Timestamp) -> Self {
        Self {
            id,
            node_type,
            attributes: HashMap::new(),
            features: None,
            birth_time,
            death_time: None,
            activity_periods: vec![birth_time..birth_time + 1],
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

    pub fn is_alive_at(&self, time: Timestamp) -> bool {
        time >= self.birth_time &&
        self.death_time.map_or(true, |death| time < death) &&
        self.activity_periods.iter().any(|range| range.contains(&time))
    }

    pub fn activate(&mut self, start_time: Timestamp, end_time: Option<Timestamp>) {
        let end = end_time.unwrap_or(start_time + 1);
        self.activity_periods.push(start_time..end);
    }

    pub fn deactivate(&mut self, time: Timestamp) {
        self.death_time = Some(time);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEdge {
    pub id: TemporalEdgeID,
    pub src: TemporalNodeID,
    pub dst: TemporalNodeID,
    pub edge_type: String,
    pub attributes: HashMap<String, AttributeValue>,
    pub weight: f32,
    pub temporal_weight: TemporalWeight,
    #[serde(skip_serializing, skip_deserializing)]
    pub features: Option<Tensor>,
    pub start_time: Timestamp,
    pub end_time: Option<Timestamp>, // None means still active
    pub interaction_count: usize,
    pub last_interaction: Timestamp,
}

impl TemporalEdge {
    pub fn new(
        id: TemporalEdgeID,
        src: TemporalNodeID,
        dst: TemporalNodeID,
        edge_type: String,
        start_time: Timestamp,
    ) -> Self {
        Self {
            id,
            src,
            dst,
            edge_type,
            attributes: HashMap::new(),
            weight: 1.0,
            temporal_weight: 1.0,
            features: None,
            start_time,
            end_time: None,
            interaction_count: 1,
            last_interaction: start_time,
        }
    }

    pub fn is_active_at(&self, time: Timestamp) -> bool {
        time >= self.start_time &&
        self.end_time.map_or(true, |end| time < end)
    }

    pub fn update_interaction(&mut self, time: Timestamp, weight_delta: f32) {
        self.interaction_count += 1;
        self.last_interaction = time;
        self.weight += weight_delta;
        let time_diff = time.saturating_sub(self.last_interaction) as f32;
        self.temporal_weight = (self.temporal_weight * (-time_diff / 100.0).exp()).max(0.01);
    }

    pub fn deactivate(&mut self, time: Timestamp) {
        self.end_time = Some(time);
    }
}

#[derive(Debug, Clone)]
pub struct TemporalSlice {
    pub id: TemporalSliceID,
    pub timestamp: Timestamp,
    pub graph_ids: HashMap<GraphID, Graph>,
    pub duration: Timestamp,
    pub nodes: HashMap<TemporalNodeID, TemporalNode>,
    pub edges: HashMap<TemporalEdgeID, TemporalEdge>,
    pub adjacency_list: HashMap<TemporalNodeID, Vec<TemporalEdgeID>>,
    pub reverse_adjacency: HashMap<TemporalNodeID, Vec<TemporalEdgeID>>,
    pub metadata: HashMap<String, AttributeValue>,
    pub is_directed: bool,
    pub change_log: Vec<TemporalChange>,
}

#[derive(Debug, Clone)]
pub enum TemporalChange {
    NodeAdded { node_id: TemporalNodeID, timestamp: Timestamp },
    NodeRemoved { node_id: TemporalNodeID, timestamp: Timestamp },
    EdgeAdded { edge_id: TemporalEdgeID, timestamp: Timestamp },
    EdgeRemoved { edge_id: TemporalEdgeID, timestamp: Timestamp },
    NodeAttributeChanged { node_id: TemporalNodeID, key: String, old_value: AttributeValue, new_value: AttributeValue, timestamp: Timestamp },
    EdgeWeightChanged { edge_id: TemporalEdgeID, old_weight: f32, new_weight: f32, timestamp: Timestamp },
}



pub trait TemporalOps{
    fn add_node_at_time(&mut self, node: TemporalNode, time: Timestamp) -> Result<(), TemporalGraphError>;
    fn add_edge_at_time(&mut self, edge: TemporalEdge, time: Timestamp) -> Result<(), TemporalGraphError>;
    fn advance_time(&mut self, delta: Timestamp) -> Result<(), TemporalGraphError>;
    fn get_snapshot_at(&self, time: Timestamp) -> Result<Graph, TemporalGraphError>;
    fn get_temporal_subgraph(&self, node_ids: &[TemporalNodeID], time_range: Range<Timestamp>) -> Result<TemporalGraph, TemporalGraphError>;
    fn temporal_neighbors(&self, node_id: TemporalNodeID, time_range: Range<Timestamp>) -> Vec<TemporalNodeID>;
    fn temporal_degree(&self, node_id: TemporalNodeID, time: Timestamp, direction: EdgeDirection) -> f32;
    fn get_evolution_pattern(&self, node_id: TemporalNodeID) -> Vec<(Timestamp, String)>; // (time, event_type)
    fn prune_history(&mut self, before_time: Timestamp) -> Result<(), TemporalGraphError>;
    fn merge_slices(&mut self, time_range: Range<Timestamp>) -> Result<TemporalSlice, TemporalGraphError>;
}

#[derive(Debug, Clone)]
pub enum TemporalGraphError {
    InvalidTimeLabel,
    OutOfTimeBound,
    InvalidSpatialID,
    InvalidTemporalID,
    SliceNotFound,
    TimeRangeError,
    NodeNotFound,
    EdgeNotFound,
}


