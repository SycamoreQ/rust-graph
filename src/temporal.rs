use crate::graph::{Graph, GraphBatch, GraphOps , RWSE , LapPE , Node , Edge , GraphError};
use crate::attn::SpatialTransformer;

pub type TemporalNodeID = usize;
pub type TemporalEdgeID = usize;
pub type TemporalSliceID = usize; 
pub type TemporalGraphID = usize;
pub type TemporalGraphBatchID = usize;
pub type SpatialID = NodeID;

#[derive(Debug , Clone)]
pub enum TemporalGraphError { 
    InvalidTimeLabel, 
    OutOfTimeBound,
    InvalidSpatialID,
    InvalidTemporalID,
}

#[derive(Debug , Clone , PartialEq)]
pub trait GraphBatch{
    fn new() -> Self;
    fn from_graph(graph : Graph) -> Self;
    fn from_temporal_graph(graph : Graph) -> Self;
    fn from_temporal_graph_batch(batch : TemporalGraphBatch) -> Self;
    fn no_of_slices(&self) -> usize; 
    fn assign_time_label(&mut self, time_label : usize) -> Result<(), GraphError>;
} 

pub struct DynamicGraph{
    pub id: TemporalGraphID,
    pub nodes: HashMap<TemporalNodeID , Node> 
    
}

#[derive(Debug , Clone , PartialEq)]
pub struct TemporalGraphBatch{
    pub nodes : Vec
}
