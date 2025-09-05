use std::sync::Arc;
use candle_core::{Tensor, Device, DType};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet, VecDeque}; 
use rayon::prelude::*;

pub type NodeID = u32;
pub type EdgeID = u32;
pub type GraphID = u32; 
pub type SubgraphID = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    Incoming,
    Outgoing,
    Both,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeID,
    pub node_type: String,
    pub attributes: HashMap<String, AttributeValue>,
    pub features: Option<Tensor>,
}

impl Node {
    pub fn new(id: NodeID, node_type: String) -> Self {
        Self {
            id,
            node_type,
            attributes: HashMap::new(),
            features: None,
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

    pub fn feature_dim(&self) -> Option<usize> {
        self.features.as_ref().map(|f| f.shape()[f.shape().len() - 1])
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub id: EdgeID,
    pub src: NodeID,
    pub dst: NodeID,
    pub edge_type: String,
    pub features: Option<Tensor>,
    pub weight: f32,
    pub attributes: HashMap<String, AttributeValue>,
}

impl Edge { 
    pub fn new(id: EdgeID, src: NodeID, dst: NodeID, edge_type: String) -> Self {
        Self {
            id,
            src,
            dst,
            edge_type,
            features: None,
            weight: 1.0,
            attributes: HashMap::new(),
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

    pub fn feature_dim(&self) -> Option<usize> {
        self.features.as_ref().map(|f| f.shape()[f.shape().len() - 1])
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Vector(Vec<f32>),
    Tensor(Tensor),
}

impl From<String> for AttributeValue {
    fn from(s: String) -> Self { AttributeValue::String(s) }
}
impl From<i64> for AttributeValue {
    fn from(i: i64) -> Self { AttributeValue::Integer(i) }
}
impl From<f64> for AttributeValue {
    fn from(f: f64) -> Self { AttributeValue::Float(f) }
}
impl From<bool> for AttributeValue {
    fn from(b: bool) -> Self { AttributeValue::Boolean(b) }
}
impl From<Vec<f32>> for AttributeValue {
    fn from(v: Vec<f32>) -> Self { AttributeValue::Vector(v) }
}

/// Error types for graph operations
#[derive(Debug)]
pub enum GraphError {
    NodeNotFound(NodeID),
    EdgeNotFound(EdgeID),
    InvalidEdge(String),
    FeatureMismatch(String),
    ConversionError(String),
    TensorError(String),
    InvalidOperation(String),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::NodeNotFound(id) => write!(f, "Node not found: {}", id),
            GraphError::EdgeNotFound(id) => write!(f, "Edge not found: {}", id),
            GraphError::InvalidEdge(msg) => write!(f, "Invalid edge: {}", msg),
            GraphError::FeatureMismatch(msg) => write!(f, "Feature mismatch: {}", msg),
            GraphError::ConversionError(msg) => write!(f, "Conversion error: {}", msg),
            GraphError::TensorError(e) => write!(f, "Tensor error: {}", e),
            GraphError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for GraphError {}

pub trait GraphOps {
    fn add_node(&mut self, node: Node) -> Result<(), GraphError>;
    fn add_edge(&mut self, edge: Edge) -> Result<(), GraphError>;
    fn remove_node(&mut self, node_id: NodeID) -> Result<Node, GraphError>;
    fn remove_edge(&mut self, edge_id: EdgeID) -> Result<Edge, GraphError>;
    fn update_node(&mut self, node_id: NodeID, new_node: Node) -> Result<(), GraphError>;
    fn update_edge(&mut self, edge_id: EdgeID, new_edge: Edge) -> Result<(), GraphError>;
    fn get_node(&self, node_id: NodeID) -> Result<&Node, GraphError>;
    fn get_edge(&self, edge_id: EdgeID) -> Result<&Edge, GraphError>;
    fn neighbors(&self, node_id: NodeID, direction: EdgeDirection) -> Result<Vec<NodeID>, GraphError>;
    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;
}

#[derive(Serialize, Debug)]
pub struct Graph {
    pub id: GraphID,
    pub nodes: HashMap<NodeID, Node>,
    pub edges: HashMap<EdgeID, Edge>,
    pub adjacency_list: HashMap<NodeID, Vec<EdgeID>>,
    pub reverse_adjacency: HashMap<NodeID, Vec<EdgeID>>,
    pub node_types: HashMap<String, Vec<NodeID>>,
    pub edge_types: HashMap<String, Vec<EdgeID>>,
    pub attributes: HashMap<String, AttributeValue>,
    pub is_directed: bool,
    pub next_node_id: NodeID,
    pub next_edge_id: EdgeID,
}

impl Graph {
    pub fn new(id: GraphID, is_directed: bool) -> Self {
        Self {
            id,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            adjacency_list: HashMap::new(),
            reverse_adjacency: HashMap::new(),
            node_types: HashMap::new(),
            edge_types: HashMap::new(),
            is_directed,
            attributes: HashMap::new(),
            next_node_id: 1,
            next_edge_id: 1,
        }
    }
    
    pub fn directed() -> Self {
        Self::new(0, true)
    }
    
    pub fn undirected() -> Self {
        Self::new(0, false)
    }
    
    pub fn degree(&self, node_id: NodeID, direction: EdgeDirection) -> usize {
        match direction {
            EdgeDirection::Outgoing => self.adjacency_list.get(&node_id).map_or(0, |v| v.len()),
            EdgeDirection::Incoming => self.reverse_adjacency.get(&node_id).map_or(0, |v| v.len()),
            EdgeDirection::Both => {
                let out_degree = self.adjacency_list.get(&node_id).map_or(0, |v| v.len());
                let in_degree = self.reverse_adjacency.get(&node_id).map_or(0, |v| v.len());
                if self.is_directed {
                    out_degree + in_degree
                } else {
                    out_degree // For undirected graphs, both lists are the same
                }
            }
        }
    }
    
    pub fn density(&self) -> f64 {
        let n = self.nodes.len() as f64;
        let m = self.edges.len() as f64;
        
        if n <= 1.0 {
            return 0.0;
        }
        
        if self.is_directed {
            m / (n * (n - 1.0))
        } else {
            (2.0 * m) / (n * (n - 1.0))
        }
    }
    
    pub fn k_hop(&self, node_id: NodeID, k: usize) -> HashSet<NodeID> {
        let mut visited = HashSet::new();
        let mut current_level = HashSet::new();
        current_level.insert(node_id);
        visited.insert(node_id);
        
        for _ in 0..k {
            let mut next_level = HashSet::new();
            for &node in &current_level {
                if let Some(edge_ids) = self.adjacency_list.get(&node) {
                    for &edge_id in edge_ids {
                        if let Some(edge) = self.edges.get(&edge_id) {
                            if !visited.contains(&edge.dst) {
                                next_level.insert(edge.dst);
                            }
                        }
                    }
                }
            }
            visited.extend(&next_level);
            current_level = next_level;
        }
        
        visited.remove(&node_id); // Remove starting node
        visited
    }
    
    pub fn subgraph(&self, node_ids: &[NodeID]) -> Result<Graph, GraphError> {
        let mut sub_graph = Graph::new(self.id + 1, self.is_directed);
        let node_set: HashSet<NodeID> = node_ids.iter().cloned().collect();

        // Add nodes
        for &node_id in node_ids {
            if let Ok(node) = self.get_node(node_id) {
                sub_graph.add_node(node.clone())?;
            }
        }

        // Add edges between included nodes
        for edge in self.edges.values() {
            if node_set.contains(&edge.src) && node_set.contains(&edge.dst) {
                sub_graph.add_edge(edge.clone())?;
            }
        }

        Ok(sub_graph)
    }
    
    pub fn to_coo(&self) -> Result<(Tensor, Tensor, Option<Tensor>), GraphError> {
        let num_edges = if self.is_directed { 
            self.edges.len() 
        } else { 
            self.edges.len() * 2 // Each undirected edge becomes two directed edges
        };

        let mut edge_indices = Vec::with_capacity(num_edges * 2);
        let mut edge_weights = Vec::with_capacity(num_edges);
        let mut edge_features = Vec::new();
        
        let node_to_idx: HashMap<NodeID, usize> = self.nodes.keys()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        for edge in self.edges.values() {
            let src_idx = *node_to_idx.get(&edge.src)
                .ok_or(GraphError::NodeNotFound(edge.src))?;
            let tgt_idx = *node_to_idx.get(&edge.dst)
                .ok_or(GraphError::NodeNotFound(edge.dst))?;

            edge_indices.extend_from_slice(&[src_idx as f32, tgt_idx as f32]);
            edge_weights.push(edge.weight);

            if let Some(ref features) = edge.features {
                edge_features.push(features.clone());
            }

            // For undirected graphs, add reverse edge
            if !self.is_directed && edge.src != edge.dst {
                edge_indices.extend_from_slice(&[tgt_idx as f32, src_idx as f32]);
                edge_weights.push(edge.weight);
                if let Some(ref features) = edge.features {
                    edge_features.push(features.clone());
                }
            }
        }

        let edge_index = Tensor::from_slice(&edge_indices, (2, num_edges), &Device::Cpu)
            .map_err(|e| GraphError::TensorError(format!("{:?}", e)))?;
        let edge_weight = Tensor::from_slice(&edge_weights, (num_edges,), &Device::Cpu)
            .map_err(|e| GraphError::TensorError(format!("{:?}", e)))?;
        
        let edge_attr = if !edge_features.is_empty() {
            Some(edge_features[0].clone()) // Simplified for demo
        } else {
            None
        };

        Ok((edge_index, edge_weight, edge_attr))
    }

    pub fn laplacian(&self, normalized: bool) -> Result<Tensor, GraphError> {
        let (edge_index, edge_weight, _) = self.to_coo()?;
        let num_nodes = self.nodes.len();
        
        let mut adjacency_matrix = vec![0.0f32; num_nodes * num_nodes];
        let edge_indices = edge_index.to_vec2::<f32>()
            .map_err(|e| GraphError::TensorError(format!("{:?}", e)))?;
        let weights = edge_weight.to_vec1::<f32>()
            .map_err(|e| GraphError::TensorError(format!("{:?}", e)))?;
        
        for (i, edge_pair) in edge_indices.iter().enumerate() {
            if edge_pair.len() >= 2 {
                let src = edge_pair[0] as usize;
                let tgt = edge_pair[1] as usize;
                if src < num_nodes && tgt < num_nodes && i < weights.len() {
                    adjacency_matrix[src * num_nodes + tgt] += weights[i];
                    if !self.is_directed {
                        adjacency_matrix[tgt * num_nodes + src] += weights[i];
                    }
                }
            }
        }
        
        let mut degree = vec![0.0f32; num_nodes];
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                degree[i] += adjacency_matrix[i * num_nodes + j];
            }
        }
        
        let mut laplacian = vec![0.0f32; num_nodes * num_nodes];
        
        for i in 0..num_nodes {
            laplacian[i * num_nodes + i] = degree[i];
            
            for j in 0..num_nodes {
                if i != j {
                    laplacian[i * num_nodes + j] = -adjacency_matrix[i * num_nodes + j];
                }
            }
        }
        
        if normalized {
            for i in 0..num_nodes {
                let deg_sqrt = degree[i].sqrt();
                if deg_sqrt > 0.0 {
                    for j in 0..num_nodes {
                        let deg_j_sqrt = degree[j].sqrt();
                        if deg_j_sqrt > 0.0 {
                            laplacian[i * num_nodes + j] /= deg_sqrt * deg_j_sqrt;
                        }
                    }
                }
            }
        }

        Tensor::from_slice(&laplacian, (num_nodes, num_nodes), &Device::Cpu)
            .map_err(|e| GraphError::TensorError(format!("{:?}", e)))
    }
}

impl GraphOps for Graph {
    fn add_node(&mut self, mut node: Node) -> Result<(), GraphError> {
        if node.id == 0 {
            node.id = self.next_node_id;
            self.next_node_id += 1;
        }

        // Update node type index
        self.node_types.entry(node.node_type.clone())
            .or_default()
            .push(node.id);

        self.adjacency_list.entry(node.id).or_default();
        self.reverse_adjacency.entry(node.id).or_default();
        self.nodes.insert(node.id, node);
        
        Ok(())
    }
    
    fn add_edge(&mut self, mut edge: Edge) -> Result<(), GraphError> {
        if !self.nodes.contains_key(&edge.src) || !self.nodes.contains_key(&edge.dst) {
            return Err(GraphError::NodeNotFound(edge.src));
        }

        if edge.id == 0 {
            edge.id = self.next_edge_id;
            self.next_edge_id += 1;
        }

        self.adjacency_list.entry(edge.src)
            .or_default()
            .push(edge.id);
        self.reverse_adjacency.entry(edge.dst)
            .or_default()
            .push(edge.id);
        
        if !self.is_directed && edge.src != edge.dst {
            self.adjacency_list.entry(edge.dst)
                .or_default()
                .push(edge.id);
            self.reverse_adjacency.entry(edge.src)
                .or_default()
                .push(edge.id);
        }

        // Update edge type index
        self.edge_types.entry(edge.edge_type.clone())
            .or_default()
            .push(edge.id);

        self.edges.insert(edge.id, edge);
        Ok(())
    }

    fn remove_node(&mut self, node_id: NodeID) -> Result<Node, GraphError> {
        let node = self.nodes.remove(&node_id)
            .ok_or(GraphError::NodeNotFound(node_id))?; 
        
        let mut edges_to_remove = Vec::new();
        
        if let Some(outgoing) = self.adjacency_list.remove(&node_id) {
            edges_to_remove.extend(outgoing);
        }
        
        if let Some(incoming) = self.reverse_adjacency.remove(&node_id) {
            edges_to_remove.extend(incoming);
        }
        
        for edge_id in edges_to_remove {
            self.edges.remove(&edge_id);
        }
        
        Ok(node)
    }
    
    fn remove_edge(&mut self, edge_id: EdgeID) -> Result<Edge, GraphError> {
        let edge = self.edges.remove(&edge_id)
            .ok_or(GraphError::EdgeNotFound(edge_id))?;
        
        if let Some(adj_list) = self.adjacency_list.get_mut(&edge.src) {
            adj_list.retain(|&x| x != edge_id);
        }
        if let Some(rev_adj_list) = self.reverse_adjacency.get_mut(&edge.dst) {
            rev_adj_list.retain(|&x| x != edge_id);
        }
        
        if !self.is_directed && edge.src != edge.dst {
            if let Some(adj_list) = self.adjacency_list.get_mut(&edge.dst) {
                adj_list.retain(|&id| id != edge_id);
            }
            if let Some(rev_adj_list) = self.reverse_adjacency.get_mut(&edge.src) {
                rev_adj_list.retain(|&id| id != edge_id);
            }
        }

        // Remove from edge type index
        if let Some(type_list) = self.edge_types.get_mut(&edge.edge_type) {
            type_list.retain(|&id| id != edge_id);
            if type_list.is_empty() {
                self.edge_types.remove(&edge.edge_type);
            }
        }

        Ok(edge)
    }

    fn update_node(&mut self, node_id: NodeID, new_node: Node) -> Result<(), GraphError> {
        if !self.nodes.contains_key(&node_id) {
            return Err(GraphError::NodeNotFound(node_id));
        }
        self.nodes.insert(node_id, new_node);
        Ok(())
    }

    fn update_edge(&mut self, edge_id: EdgeID, new_edge: Edge) -> Result<(), GraphError> {
        if !self.edges.contains_key(&edge_id) {
            return Err(GraphError::EdgeNotFound(edge_id));
        }
        self.edges.insert(edge_id, new_edge);
        Ok(())
    }
    
    fn get_node(&self, node_id: NodeID) -> Result<&Node, GraphError> {
        self.nodes.get(&node_id).ok_or(GraphError::NodeNotFound(node_id))
    }
    
    fn get_edge(&self, edge_id: EdgeID) -> Result<&Edge, GraphError> {
        self.edges.get(&edge_id).ok_or(GraphError::EdgeNotFound(edge_id))
    }
    
    fn neighbors(&self, node_id: NodeID, direction: EdgeDirection) -> Result<Vec<NodeID>, GraphError> {
        let mut neighbors = Vec::new();

        match direction {
            EdgeDirection::Outgoing => {
                if let Some(edge_ids) = self.adjacency_list.get(&node_id) {
                    for &edge_id in edge_ids {
                        if let Some(edge) = self.edges.get(&edge_id) {
                            neighbors.push(edge.dst);
                        }
                    }
                }
            }
            EdgeDirection::Incoming => {
                if let Some(edge_ids) = self.reverse_adjacency.get(&node_id) {
                    for &edge_id in edge_ids {
                        if let Some(edge) = self.edges.get(&edge_id) {
                            neighbors.push(edge.src);
                        }
                    }
                }
            }
            EdgeDirection::Both => {
                let mut neighbor_set = HashSet::new();
                
                if let Some(edge_ids) = self.adjacency_list.get(&node_id) {
                    for &edge_id in edge_ids {
                        if let Some(edge) = self.edges.get(&edge_id) {
                            neighbor_set.insert(edge.dst);
                        }
                    }
                }
                
                if let Some(edge_ids) = self.reverse_adjacency.get(&node_id) {
                    for &edge_id in edge_ids {
                        if let Some(edge) = self.edges.get(&edge_id) {
                            neighbor_set.insert(edge.src);
                        }
                    }
                }
                
                neighbors.extend(neighbor_set);
            }
        }

        Ok(neighbors)
    }

    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

pub fn adjacency_list_to_matrix(adj_list: &[Vec<usize>], n: Option<usize>) -> Vec<Vec<u8>> {
    let size = n.unwrap_or(adj_list.len());
    let mut matrix = vec![vec![0u8; size]; size];
    
    for (i, neighbors) in adj_list.iter().enumerate() {
        for &j in neighbors {
            if j < size {
                matrix[i][j] = 1;
            }
        }
    }
    
    matrix 
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_graph() {
        let directed_graph = Graph::directed();
        assert!(directed_graph.is_directed);
        assert_eq!(directed_graph.node_count(), 0);
        assert_eq!(directed_graph.edge_count(), 0);
        
        let undirected_graph = Graph::undirected();
        assert!(!undirected_graph.is_directed);
        assert_eq!(undirected_graph.node_count(), 0);
        assert_eq!(undirected_graph.edge_count(), 0);
    }
    
    #[test]
    fn test_add_node() {
        let mut graph = Graph::directed();
        let node1 = Node::new(1, "test".to_string());
        let node2 = Node::new(2, "test".to_string());
        
        graph.add_node(node1).unwrap();
        graph.add_node(node2).unwrap();
        
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0);
        assert!(graph.nodes.contains_key(&1));
        assert!(graph.nodes.contains_key(&2));
    }
    
    #[test]
    fn test_add_edge_directed() {
        let mut graph = Graph::directed();
        let node1 = Node::new(1, "test".to_string());
        let node2 = Node::new(2, "test".to_string());
        
        graph.add_node(node1).unwrap();
        graph.add_node(node2).unwrap();

        let edge = Edge::new(1, 1, 2, "connects".to_string());
        graph.add_edge(edge).unwrap();

        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.degree(1, EdgeDirection::Outgoing), 1);
        assert_eq!(graph.degree(1, EdgeDirection::Incoming), 0);
        assert_eq!(graph.degree(2, EdgeDirection::Outgoing), 0);
        assert_eq!(graph.degree(2, EdgeDirection::Incoming), 1);
    }
    
    #[test]
    fn test_add_edge_undirected() {
        let mut graph = Graph::undirected();
        let node1 = Node::new(1, "test".to_string());
        let node2 = Node::new(2, "test".to_string());
        
        graph.add_node(node1).unwrap();
        graph.add_node(node2).unwrap();

        let edge = Edge::new(1, 1, 2, "connects".to_string());
        graph.add_edge(edge).unwrap();

        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.degree(1, EdgeDirection::Outgoing), 1);
        assert_eq!(graph.degree(2, EdgeDirection::Outgoing), 1);
        assert_eq!(graph.degree(1, EdgeDirection::Both), 1);
        assert_eq!(graph.degree(2, EdgeDirection::Both), 1);
    }
    
    #[test]
    fn test_neighbors() {
        let mut graph = Graph::directed();
        
        for i in 1..=3 {
            graph.add_node(Node::new(i, "test".to_string())).unwrap();
        }

        graph.add_edge(Edge::new(1, 1, 2, "edge".to_string())).unwrap();
        graph.add_edge(Edge::new(2, 1, 3, "edge".to_string())).unwrap();
        graph.add_edge(Edge::new(3, 3, 1, "edge".to_string())).unwrap();

        let outgoing_neighbors = graph.neighbors(1, EdgeDirection::Outgoing).unwrap();
        let incoming_neighbors = graph.neighbors(1, EdgeDirection::Incoming).unwrap();
        let all_neighbors = graph.neighbors(1, EdgeDirection::Both).unwrap();

        assert_eq!(outgoing_neighbors.len(), 2);
        assert!(outgoing_neighbors.contains(&2));
        assert!(outgoing_neighbors.contains(&3));

        assert_eq!(incoming_neighbors.len(), 1);
        assert!(incoming_neighbors.contains(&3));

        assert_eq!(all_neighbors.len(), 2);
        assert!(all_neighbors.contains(&2));
        assert!(all_neighbors.contains(&3));
    }
    
    #[test]
    fn test_degree_calculation() {
        let mut graph = Graph::directed();
        
        // Create nodes
        for i in 1..=3 {
            graph.add_node(Node::new(i, "test".to_string())).unwrap();
        }

        // Add edges: 1 -> 2, 1 -> 3, 3 -> 1
        graph.add_edge(Edge::new(1, 1, 2, "edge".to_string())).unwrap();
        graph.add_edge(Edge::new(2, 1, 3, "edge".to_string())).unwrap();
        graph.add_edge(Edge::new(3, 3, 1, "edge".to_string())).unwrap();

        assert_eq!(graph.degree(1, EdgeDirection::Outgoing), 2);
        assert_eq!(graph.degree(1, EdgeDirection::Incoming), 1);
        assert_eq!(graph.degree(1, EdgeDirection::Both), 3);

        assert_eq!(graph.degree(2, EdgeDirection::Outgoing), 0);
        assert_eq!(graph.degree(2, EdgeDirection::Incoming), 1);
        assert_eq!(graph.degree(2, EdgeDirection::Both), 1);

        assert_eq!(graph.degree(3, EdgeDirection::Outgoing), 1);
        assert_eq!(graph.degree(3, EdgeDirection::Incoming), 1);
        assert_eq!(graph.degree(3, EdgeDirection::Both), 2);
    }

    #[test]
    fn test_adjacency_list_to_matrix() {
        let adj_list = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let matrix = adjacency_list_to_matrix(&adj_list, None);
        assert_eq!(matrix, vec![vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0]]);
    }
}