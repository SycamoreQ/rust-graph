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
        // Decay temporal weight based on time since last interaction
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

impl TemporalSlice{
    pub fn new(id : TemporalSliceID , timestamp: Timestamp , duration: Timestamp , is_directed: bool) -> Self{
        Self{
            id ,
            timestamp,
            duration,
            nodes: Hashmap::new(),
            edges: HashMap::new(),
            adjacency_list: HashMap::new(),
            reverse_adjacency: HashMap::new(),
            metadata: HashMap::new(),
            is_directed,
            change_log: Vec::new(),
        }
    }

    pub fn time_range(&self) -> Range<Timestamp>{
        self.timestamp..self.timestamp + self.duration
    }

    pub fn contains_time(&self , time: Timestamp) -> bool{
        self.time_range().contains(&time)
    }

    pub fn add_change(&mut self , change: TemporalChange){
        self.change_log.push(change);
    }

    pub fn get_active_nodes(&self , time: Timestamp) -> Vec<TemporalNode>{
        self.nodes.values()
            .filter(|n| n.is_alive_at(time))
            .collect()
    }

    pub fn get_active_edges_at(&self, time: Timestamp) -> Vec<&TemporalEdge> {
        self.edges.values()
            .filter(|edge| edge.is_active_at(time))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct TemporalGraph {
    pub id: TemporalGraphID,
    pub slices: BTreeMap<Timestamp, TemporalSlice>, // Ordered by timestamp
    pub global_nodes: HashMap<TemporalNodeID, TemporalNode>,
    pub global_edges: HashMap<TemporalEdgeID, TemporalEdge>,
    pub is_directed: bool,
    pub current_time: Timestamp,
    pub time_granularity: Timestamp, // How fine-grained time steps are
    pub max_history: Option<usize>, // Maximum number of slices to keep
    pub node_evolution: HashMap<TemporalNodeID, Vec<Timestamp>>, // Track when nodes change
    pub edge_evolution: HashMap<TemporalEdgeID, Vec<Timestamp>>, // Track when edges change
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


impl TemporalGraph{

    pub fn new(id: TemporalGraphID, is_directed: bool, time_granularity: Timestamp) -> Self {
        Self {
            id,
            slices: BTreeMap::new(),
            global_nodes: HashMap::new(),
            global_edges: HashMap::new(),
            is_directed,
            current_time: 0,
            time_granularity,
            max_history: None,
            node_evolution: HashMap::new(),
            edge_evolution: HashMap::new(),
        }
    }

    fn get_or_create_slice(&mut self, time: Timestamp) -> &mut TemporalSlice {
        let slice_time = (time / self.time_granularity) * self.time_granularity;
        self.slices.entry(slice_time).or_insert_with(||
            TemporalSlice::new(slice_time, slice_time, self.time_granularity, self.is_directed)
        )
    }

    pub fn temporal_shortest_path(
        &self,
        src: TemporalNodeID,
        dst: TemporalNodeID,
        start_time: Timestamp,
        max_time: Timestamp,
    ) -> Option<Vec<(TemporalNodeID, Timestamp)>> {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;

        let mut distances: HashMap<(TemporalNodeID, Timestamp), f32> = HashMap::new();
        let mut predecessors: HashMap<(TemporalNodeID, Timestamp), (TemporalNodeID, Timestamp)> = HashMap::new();
        let mut heap = BinaryHeap::new();

        heap.push(Reverse((0.0, src, start_time)));
        distances.insert((src, start_time), 0.0);

        while let Some(Reverse((dist, node, time))) = heap.pop() {
            if node == dst {
                // Reconstruct path
                let mut path = Vec::new();
                let mut current = (node, time);
                path.push(current);

                while let Some(&prev) = predecessors.get(&current) {
                    path.push(prev);
                    current = prev;
                }

                path.reverse();
                return Some(path);
            }

            if time >= max_time { continue; }

            // Look for edges at current and future times
            for t in time..=max_time.min(time + self.time_granularity) {
                for edge in self.global_edges.values() {
                    if edge.src == node && edge.is_active_at(t) {
                        let next_node = edge.dst;
                        let next_time = t + 1;
                        let next_dist = dist + edge.weight;

                        let key = (next_node, next_time);
                        if next_dist < *distances.get(&key).unwrap_or(&f32::INFINITY) {
                            distances.insert(key, next_dist);
                            predecessors.insert(key, (node, time));
                            heap.push(Reverse((next_dist, next_node, next_time)));
                        }
                    }
                }
            }
        }
        
        None 
    }
    
    pub fn temporal_similarity(
        &self , 
        other: &TemporalGraph,
        time_range: Timestamp,
    ) -> f32 {
        
        let mut similarity_sum = 0.0
        let mut count = 0; 
        
        for time in time_range.step_by(self.time_granularity){
            let my_nodes = self.get_active_nodes(time);
            let other_nodes = other.get_active_nodes(time);
            
            let intersection = my_nodes.intersection(&other_nodes);
            let union = my_nodes.union(&other_nodes);
            
            if !union.is_empty() {
                similarity_sum += intersection.len() as f32 / union.len() as f32;
                count += 1;
            }
        }
        
        if (count > 0){
            similarity_sum / count as f32
        }
        else{
            0.0 
        }
    }
    
    fn get_active_nodes_at_time(&self, time: Timestamp) -> HashSet<TemporalNodeID> {
        self.global_nodes.values()
            .filter(|node| node.is_alive_at(time))
            .map(|node| node.id)
            .collect()
    }
} 

#[derive(Debug, Clone)]
pub struct TemporalMotif {
    pub edges: Vec<TemporalEdge>,
    pub start_time: Timestamp,
    pub end_time: Timestamp,
    pub signature: String, // Pattern identifier
}

impl TemporalMotif {
    pub fn new(edges: Vec<TemporalEdge>, start_time: Timestamp, end_time: Timestamp) -> Self {
        let signature = Self::compute_signature(&edges);
        Self {
            edges,
            start_time,
            end_time,
            signature,
        }
    }

    fn compute_signature(edges: &[TemporalEdge]) -> String {
        let mut sig = String::new();
        for edge in edges {
            sig.push_str(&format!("{}->", edge.edge_type));
        }
        sig
    }
}


impl TemporalOps for TemporalGraph{
    fn add_node_at_time(&mut self, node: TemporalNode, time: Timestamp) -> Result<(), TemporalGraphError> {
        
        let slice = self.get_or_create_slice(time);
        
        self.global_nodes.insert(node.id , node.clone())?;
        self.node_evolution.entry(node.id).or_default().push(time);        
        
        slice.add_change(TemporalChange::NodeAdded { node_id: node.id, timestamp: time });
        slice.nodes.insert(node.id, node);
        slice.adjacency_list.entry(node.id).or_default();
        slice.reverse_adjacency.entry(node.id).or_default();
        
        self.current_time = self.current_time.max(time);
        
        if let Some(max_history) = self.max_history{
            if max_history < self.node_evolution[&node.id].len() {
                self.node_evolution[&node.id].retain(|&t| t > time - max_history);
            }
        }
    }
    
    fn add_edge_at_time(&mut self, edge: TemporalEdge, time: Timestamp) -> Result<(), TemporalGraphError> {
        
        if !self.global_nodes.contains_key(&edge.src) || !self.global_nodes.contains_key(&edge.dst) {
                Err(TemporalGraphError::NodeNotFound);
        }
        
        let slice = self.get_or_create_slice(time);
        self.global_edges.insert(edge.id, edge.clone());
        self.edge_evolution.entry(edge.id).or_default().push(time);
        
        slice.add_change(TemporalChange::EdgeAdded { edge_id: edge.id, timestamp: time });
        slice.adjacency_list.entry(edge.src).or_default().push(edge.id);
        slice.reverse_adjacency.entry(edge.dst).or_default().push(edge.id);
        
        if !self.is_directed && edge.src != edge.dst {
            slice.adjacency_list.entry(edge.dst).or_default().push(edge.id);
            slice.reverse_adjacency.entry(edge.src).or_default().push(edge.id);
        }
        
        slice.edges.insert(edge.id, edge);
        self.current_time = self.current_time.max(time);
        
        Ok(())
    }
    
    fn advance_time(&mut self, delta: Timestamp) -> Result<(), TemporalGraphError> {
        self.current_time += delta;
        
        // Update temporal weights for existing edges
        for edge in self.global_edges.values_mut() {
            if edge.is_active_at(self.current_time) {
                let time_diff = self.current_time.saturating_sub(edge.last_interaction) as f32;
                edge.temporal_weight = (edge.temporal_weight * (-time_diff / 100.0).exp()).max(0.01);
            }
        }
        
        Ok(())
    }
    
    fn get_snapshot_at(&self, time: Timestamp) -> Result<Graph, TemporalGraphError> {
        
    }
}