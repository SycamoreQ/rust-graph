use crate::graph::{Graph, GraphBatch, GraphOps , RWSE , LapPE , Node , Edge , GraphError};
use crate::attn::SpatialTransformer;

pub type TemporalNodeID = usize;
pub type TemporalEdgeID = usize;
pub type TemporalSliceID = usize; 
pub type TemporalGraphID = usize;
pub type TemporalGraphBatchID = usize;
pub type SpatialID = NodeID;


#[derive(Debug , Clone)]
pub struct TemporalGraph{
    pub ID: SpatialID,
    pub t_ID: TemporalGraphID,
    pub nodes: HashMap<TemporalNodeID , Node> ,
    pub edges: HashMap<TemporalEdgeID , Edge> ,
    pub adjacency_list: HashMap<TemporalNodeID , Vec<TemporalEdgeID>>,
    pub reverse_adjacency_list: HashMap<TemporalNodeID , Vec<TemporalEdgeID>>,
    pub edge_type: HashMap<TemporalEdgeID , EdgeType>,
    pub attributes: HashMap<String, String>,
    pub slice_id: TemporalSliceID,
    pub batch_id: TemporalGraphBatchID,
    pub next_node_id: TemporalNodeID,
    pub next_edge_id: TemporalEdgeID,
    pub next_slice_id: TemporalSliceID,
    pub next_batch_id: TemporalGraphBatchID,
}




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
