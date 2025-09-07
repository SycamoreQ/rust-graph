use crate::graph::{Edge ,Node, Graph , GraphOps , EdgeDirection , NodeID , EdgeID , GraphError};
use candle_core::{Tensor, Device, DType};
use candle_nn::{Linear , ops::softmax , Module , Dropout , init  }; 
use std::collections::{HashMap}; 


pub struct GATLayer {
    weight_matrix: Tensor,    
    attention_vector: Tensor,   
    input_dim: usize,
    output_dim: usize,
    num_heads: usize,
    dropout_rate: f64,
    alpha: f64,  // LeakyReLU negative slope
    device: Device,
}

impl GATLayer {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        num_heads: usize,
        dropout_rate: f64,
        alpha: f64,
        device: Device,
    ) -> Result<Self> {
        let weight_std = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let weight_matrix = Tensor::randn(0.0f32 , weight_std , &[output_dim, input_dim],  &device)?;
        
        let attention_vector = Tensor::randn(0.0f32 , weight_std , &[2 * output_dim], &device)?;
        
        Ok(Self {
            weight_matrix,
            attention_vector,
            input_dim,
            output_dim,
            num_heads,
            dropout_rate,
            alpha,
            device,
        })
    }
    
    /// Forward pass through GAT layer
    pub fn forward(
        &self,
        node_features: &Tensor,  
        edge_index: &Tensor,     
    ) -> Result<Tensor> {
        let num_nodes = node_features.dim(0)?;
        
        let transformed_features = node_features.matmul(&self.weight_matrix.t())?;
        

        let attention_scores = self.compute_attention_scores(&transformed_features, edge_index)?;
        

        let output = self.aggregate_with_attention(
            &transformed_features, 
            &attention_scores, 
            edge_index
        )?;
        
        output.relu()
    }
    
    /// Compute attention scores between connected nodes
    fn compute_attention_scores(
        &self,
        transformed_features: &Tensor, 
        edge_index: &Tensor,        
    ) -> Result<Tensor> {
        let num_edges = edge_index.dim(1)?;
        let mut attention_scores = Vec::with_capacity(num_edges);
        
        let edge_data = edge_index.to_vec2::<u32>()?;
        let feature_data = transformed_features.to_vec2::<f32>()?;
        let attention_vec = self.attention_vector.to_vec1::<f32>()?;
        
        for edge_idx in 0..num_edges {
            let src_node = edge_data[0][edge_idx] as usize;
            let dst_node = edge_data[1][edge_idx] as usize;
            
            if src_node >= feature_data.len() || dst_node >= feature_data.len() {
                continue;
            }
            
            let src_features = &feature_data[src_node];
            let dst_features = &feature_data[dst_node];
            
            let mut concatenated = Vec::with_capacity(2 * self.output_dim);
            concatenated.extend_from_slice(src_features);
            concatenated.extend_from_slice(dst_features);
            

            let mut score = 0.0f32;
            for (i, &feat) in concatenated.iter().enumerate() {
                if i < attention_vec.len() {
                    score += attention_vec[i] * feat;
                }
            }
            
            // Apply LeakyReLU
            let activated_score = if score > 0.0 { score } else { self.alpha as f32 * score };
            attention_scores.push(activated_score);
        }
        
        Tensor::from_slice(&attention_scores, (num_edges,), &self.device)
    }
    
    /// Applies attention weights and aggregates neighbor features
    fn aggregate_with_attention(
        &self,
        transformed_features: &Tensor, 
        attention_scores: &Tensor,     
        edge_index: &Tensor,         
    ) -> Result<Tensor> {
        let num_nodes = transformed_features.dim(0)?;
        let num_edges = edge_index.dim(1)?;
        
        let mut output = vec![vec![0.0f32; self.output_dim]; num_nodes];
        
        let edge_data = edge_index.to_vec2::<u32>()?;
        let feature_data = transformed_features.to_vec2::<f32>()?;
        let scores = attention_scores.to_vec1::<f32>()?;
        
        let mut node_edges: HashMap<usize, Vec<(usize, usize, f32)>> = HashMap::new();
        
        for edge_idx in 0..num_edges {
            let src = edge_data[0][edge_idx] as usize;
            let dst = edge_data[1][edge_idx] as usize;
            let score = scores[edge_idx];
            
            node_edges.entry(src).or_default().push((dst, edge_idx, score));
        }
        
        // For each node, compute softmax over attention scores and aggregate
        for (src_node, edges) in node_edges {
            if src_node >= num_nodes {
                continue;
            }
            
            let max_score = edges.iter().map(|(_, _, score)| *score).fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = edges.iter()
                .map(|(_, _, score)| (score - max_score).exp())
                .collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            
            for ((dst_node, _, _), &exp_score) in edges.iter().zip(exp_scores.iter()) {
                let weight = exp_score / sum_exp;  // Softmax weight
                
                if *dst_node < feature_data.len() {
                    for (dim, &feature_val) in feature_data[*dst_node].iter().enumerate() {
                        if dim < self.output_dim {
                            output[src_node][dim] += weight * feature_val;
                        }
                    }
                }
            }
        }
        
        // Convert back to tensor
        let flat_output: Vec<f32> = output.into_iter().flatten().collect();
        Tensor::from_slice(&flat_output, (num_nodes, self.output_dim), &self.device)
    }
}


#[derive(Debug , Clone)]
pub struct GCN{
    pub embedding_dim: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub weight_matrix: Tensor,
    pub bias: Tensor,
    pub use_bias: bool,
}

impl GCN{
    pub fn new(input_dim: usize, output_dim: usize, use_bias: bool) -> Self {

    }
    
    pub fn forward(&self , graph: &Graph , node_features : &Vec<Vec<f32>>) -> Vec<Vec<f32>>{
        let nodes = graph.nodes.len();
        let laplacian = Graph::laplacian(graph , true).expect("Laplacian matrix not found");
        
        let mut new_node_feat = vec![vec![0.0; self.output_dim]; nodes];
        
        for i in 0..nodes{
            
            let mut aggregated_features =  vec![0.0; self.input_dim];
            
            for j in 0..nodes{
                let laplacian_weight = laplacian[i][j];
                
                if laplacian_weight != 0.0 && j < node_features.len(){
                    for k in 0..self.input_dim.min(node_features[j].len()){
                        aggregated_features[k] += laplacian_weight * node_features[j][k];
                    }
                }
            }
            for out_idx in 0..self.output_dim {
                let mut sum = 0.0;
                for in_idx in 0..self.input_dim {
                    sum += self.weight_matrix[out_idx][in_idx] * aggregated_features[in_idx];
                }
                
                if self.use_bias {
                    sum += self.bias[out_idx];
                }
                
                output[i][out_idx] = sum;
            }
            
            output
        }
    }
}


pub struct GraphSageLayer{
    pub self_weight: Linear, 
    pub neighbor_weight: Linear, 
    pub activation: fn(&Tensor) -> Tensor,
    pub bias : Linear, 
    pub use_bias: bool,
    pub agg_func: String, 
    
}


impl GraphSageLayer{
    pub fn new(input_dim: usize , output_dim: usize , neighbor_spread: usize ,  device: &Device) -> Result<Self> {
        Ok(Self {
            
            self_weight : Linear::new(
                Tensor::randn(0.0f32 , &[output_dim , input_dim] , DType::F32 , device)?,
                Some(Tensor::zeros(&[output_dim], DType::F32, device)?),
            ),
            neighbor_weight : Linear::new(
                Tensor::randn(0.0f32 , &[output_dim , neighbor_spread * output_dim] , DType::F32 , device)?,
                Some(Tensor::zeros(&[output_dim], DType::F32, device)?),
            ),
            activation: |x| x.relu(),
            bias : Linear::new(
                Tensor::randn(0.0f32 , &[output_dim] , DType::F32 , device)?,
                Some(Tensor::zeros(&[output_dim], DType::F32, device)?),
            ),
            use_bias: true,
            
            agg_func: String::from("mean"),
        })
    }
    
    pub fn mean_aggregation(neighbor_features: &[Tensor]) -> Result<Tensor> {
        if neighbor_features.is_empty() {
            return Err(candle_core::Error::Msg("No neighbors to aggregate".into()));
        }
        
 
        let stacked = Tensor::stack(neighbor_features, 0)?; // [num_neighbors, feature_dim]
        stacked.mean(0) 
    }
    
    
    pub fn forward(&self , node_features: &Tensor , neighbor_features: &Tensor) -> Result<Tensor>{
        let self_transformed = self.self_weight.forward(node_features)?;
        
        // Transform aggregated neighbor features  
        let neighbor_transformed = self.neighbor_weight.forward(neighbor_features)?;
        
        // Combine and activate
        let combined = (&self_transformed + &neighbor_transformed)?;
        (self.activation)(&combined)
    }
}

pub struct GraphSage{
    pub layers: Vec<GraphSageLayer>,
    pub device: Device,
    pub sample_size: usize, 
}

impl GraphSage{
    pub fn new(layer_dims: &[usize], sample_size: Vec<usize> , device: Device) -> Result<Self>{
        let mut layers = Vec::new();
        
        for i in 0..layer_dims.len(){
            layers.push(GraphSageLayer::new(0.0f32 , layer_dims[i], layer_dims[i+1], &device)?)
        }
        
        Ok(Self{
            layers,
            device,
            sample_size,
        })
    }

    
    pub fn forward(&self, graph: &Graph , node_features: &HashMap<NodeID , Tensor> , target_nodes: &[NodeID]) -> Result<HashMap<NodeID, Tensor>>{
        let mut current_features = node_features.clone();
        
        for (layer_idx , layer) in self.layers.iter().enumerate(){
            let sample_size = self.sample_size[layer_idx];
            let mut next_features = HashMap::new();
            
            for &node_id in target_nodes{
                let neighbors = Graph::neighbors(graph, node_id, EdgeDirection::Outgoing)?;
                
                let neighbor_tensor : Vec<Tensor> = neighbors.iter().
                    filter_map(|&n| current_features.get(&n).cloned()).collect();
                
                let aggregated = 
                    if !neighbor_tensor.is_empty(){
                        GraphSageLayer::mean_aggregation(neighbor_tensor).unwrap();
                    }
                    else{
                        Tensor::zeros(&[layer.input_dims] , candle_core::DType::F64 , &self.device).unwrap();
                    };
                
                
                let node_feat = current_features.get(&node_id).unwrap();
                let output = layer.forward(node_feat , &aggregated).unwrap();
                
                next_features.insert(node_id , output);
            }
            current_features = next_features;
        }
        Ok(current_features)
    }
} 


#[cfg(test)]
mod tests {
    use super::*; 
    use crate::graph::{Edge ,Node, Graph , GraphOps , EdgeDirection , NodeID , EdgeID , GraphError};
    
    fn create_test_graph() -> Graph { 
        let mut undirected_graph = Graph::undirected(); 
        
        for i in 0..4 {
            GraphOps::add_node(&mut undirected_graph, Node::new(i, format!("node_{}", i)));
        }
        
        undirected_graph.add_edge(Edge::new(0, 0, 1, "edge".to_string())).unwrap();
        undirected_graph.add_edge(Edge::new(1, 1, 2, "edge".to_string())).unwrap();
        undirected_graph.add_edge(Edge::new(2, 2, 3, "edge".to_string())).unwrap();
        undirected_graph.add_edge(Edge::new(3, 0, 2, "edge".to_string())).unwrap();
        
        undirected_graph
    }
    
    #[test]
    fn test_gat_layer_creation() {
         let device = Device::Cpu;
         let input_dim = 8;
         let output_dim = 4;
         let num_heads = 2;
         let dropout_rate = 0.1;
         let alpha = 0.2;
 
         let gat_layer = GATLayer::new(
             input_dim,
             output_dim,
             num_heads,
             dropout_rate,
             alpha,
             device.clone(),
         );
 
         assert!(gat_layer.is_ok());
         let layer = gat_layer.unwrap();
         assert_eq!(layer.input_dim, input_dim);
         assert_eq!(layer.output_dim, output_dim);
         assert_eq!(layer.num_heads, num_heads);
         assert_eq!(layer.dropout_rate, dropout_rate);
         assert_eq!(layer.alpha, alpha);
     }
 
     #[test]
     fn test_gat_forward_pass() -> Result<() , GraphError> {
         let device = Device::Cpu;
         let input_dim = 4;
         let output_dim = 2;
         let num_nodes = 3;
 
         let gat_layer = GATLayer::new(input_dim, output_dim, 1, 0.0, 0.2, device.clone())?;
 
         
         let node_features = Tensor::new(
             vec![
                 vec![1.0f32, 2.0, 3.0, 4.0],    
                 vec![2.0f32, 1.0, 4.0, 3.0],    
                 vec![3.0f32, 4.0, 1.0, 2.0],    
             ],
             &device,
         )?;
 
         // Create edge index [2, num_edges]
         let edge_index = Tensor::new(
             vec![
                 vec![0u32, 1u32],  
                 vec![1u32, 2u32],  
             ],
             &device,
         )?;
 
         let result = gat_layer.forward(&node_features, &edge_index);
         assert!(result.is_ok());
 
         let output = result?;
         let output_shape = output.dims();
         assert_eq!(output_shape[0], num_nodes);
         assert_eq!(output_shape[1], output_dim);
 
         Ok(())
     }
     
     
     #[test]
     fn test_gat_attention_computation() -> Result<() , GraphError> {
         let device = Device::Cpu;
         let gat_layer = GATLayer::new(3, 2, 1, 0.0, 0.2, device.clone())?;
 
         let transformed_features = Tensor::new(
             vec![
                 vec![1.0f32, 0.5],
                 vec![0.8f32, 1.2],
                 vec![1.5f32, 0.3],
             ],
             &device,
         )?;
 
         let edge_index = Tensor::new(
             vec![
                 vec![0u32, 1u32],
                 vec![1u32, 2u32],
             ],
             &device,
         )?;
 
         let attention_scores = gat_layer.compute_attention_scores(&transformed_features, &edge_index);
         assert!(attention_scores.is_ok());
 
         let scores = attention_scores?;
         assert_eq!(scores.dims()[0], 2); 
 
         Ok(())
     }
     
     #[test]
     fn test_graphsage_layer_creation() -> Result<() , GraphError> {
         let device = Device::Cpu;
         let input_dim = 4;
         let output_dim = 2;
         let neighbor_spread = 10; 
         let sage_layer = GraphSageLayer::new(input_dim, output_dim,neighbor_spread , &device);
         assert!(sage_layer.is_ok());
 
         let layer = sage_layer?;
         assert_eq!(layer.agg_func, "mean");
         assert_eq!(layer.use_bias, true);
 
         Ok(())
     }
 
     #[test]
     fn test_graphsage_mean_aggregation() -> Result<() , GraphError> {
         let device = Device::Cpu;
 
         let neighbor1 = Tensor::new(vec![1.0f32, 2.0, 3.0], &device)?;
         let neighbor2 = Tensor::new(vec![4.0f32, 5.0, 6.0], &device)?;
         let neighbor3 = Tensor::new(vec![7.0f32, 8.0, 9.0], &device)?;
 
         let neighbors = vec![neighbor1, neighbor2, neighbor3];
         
         let result = GraphSageLayer::mean_aggregation(&neighbors);
         assert!(result.is_ok());
 
         let aggregated = result?;
         let expected_mean = vec![4.0f32, 5.0, 6.0]; 
  
         let output_data = aggregated.to_vec1::<f32>()?;
         for (i, &expected) in expected_mean.iter().enumerate() {
             assert!((output_data[i] - expected).abs() < 1e-6);
         }
 
         Ok(())
     }
 
     #[test]
     fn test_graphsage_mean_aggregation_empty() {
         let neighbors: Vec<Tensor> = vec![];
         let result = GraphSageLayer::mean_aggregation(&neighbors);
         assert!(result.is_err());
     }
 
     #[test]
     fn test_graphsage_layer_forward() -> Result<() , GraphError> {
         let device = Device::Cpu;
         let sage_layer = GraphSageLayer::new(3, 2, 10, &device)?;
 
         let node_features = Tensor::new(vec![1.0f32, 2.0, 3.0], &device)?;
         let neighbor_features = Tensor::new(vec![4.0f32, 5.0, 6.0], &device)?;
 
         let result = sage_layer.forward(&node_features, &neighbor_features);
         assert!(result.is_ok());
 
         let output = result?;
         assert_eq!(output.dims()[0], 2); 
 
         Ok(())
     }
 
     #[test]
     fn test_graphsage_model_creation() -> Result<() , GraphError> {
         let device = Device::Cpu;
         let layer_dims = vec![4, 8, 4];
         let sample_sizes = vec![5, 3]; 
 
         let graphsage = GraphSage::new(&layer_dims, sample_sizes.clone(), device.clone());
         assert!(graphsage.is_ok());
 
         let model = graphsage?;
         assert_eq!(model.layers.len(), layer_dims.len() - 1);
         assert_eq!(model.sample_size, sample_sizes);
 
         Ok(())
     }
 
     #[test]
     fn test_edge_index_bounds_checking() -> Result<() , GraphError> {
         let device = Device::Cpu;
         let gat_layer = GATLayer::new(2, 2, 1, 0.0, 0.2, device.clone())?;
 
         // Create node features for 2 nodes
         let node_features = Tensor::new(
             vec![
                 vec![1.0f32, 2.0],
                 vec![3.0f32, 4.0],
             ],
             &device,
         )?;
 
         let edge_index = Tensor::new(
             vec![
                 vec![0u32, 5u32], // node 5 doesn't exist
                 vec![1u32, 1u32],
             ],
             &device,
         )?;
 
         let result = gat_layer.forward(&node_features, &edge_index);
         assert!(result.is_ok());
 
         Ok(())
     }
 
     #[test]
     fn test_gat_self_attention() -> Result<(), GraphError> {
         let device = Device::Cpu;
         let gat_layer = GATLayer::new(2, 2, 1, 0.0, 0.2, device.clone())?;
 
         let node_features = Tensor::new(
             vec![
                 vec![1.0f32, 0.0],
                 vec![0.0f32, 1.0],
             ],
             &device,
         )?;
 
         let edge_index = Tensor::new(
             vec![
                 vec![0u32, 1u32],
                 vec![0u32, 1u32],
             ],
             &device,
         )?;
 
         let result = gat_layer.forward(&node_features, &edge_index)?;
         assert_eq!(result.dims(), &[2, 2]);
 
         Ok(())
     }
 
     #[test]
     fn test_feature_dimension_consistency() {
         let gcn = GCN::new(3, 2, false);
         

         let inconsistent_features = vec![
             vec![1.0, 2.0],        // Wrong dimension (2 instead of 3)
             vec![3.0, 4.0, 5.0],   // Correct dimension
         ];
 
         let graph = create_test_graph();
         
         let result = gcn.forward(&graph, &inconsistent_features);
         assert_eq!(result.len(), graph.nodes.len());
     }
 
     #[test]
     fn test_multi_layer_gat() -> Result<(), GraphError> {
         let device = Device::Cpu;
         
         
         let layer1 = GATLayer::new(4, 8, 2, 0.1, 0.2, device.clone())?;
         let layer2 = GATLayer::new(8, 4, 1, 0.1, 0.2, device.clone())?;
 
         let node_features = Tensor::new(
             vec![
                 vec![1.0f32, 2.0, 3.0, 4.0],
                 vec![5.0f32, 6.0, 7.0, 8.0],
                 vec![9.0f32, 10.0, 11.0, 12.0],
             ],
             &device,
         )?;
 
         let edge_index = Tensor::new(
             vec![
                 vec![0u32, 1u32, 2u32],
                 vec![1u32, 2u32, 0u32],
             ],
             &device,
         )?;
 
         let hidden = layer1.forward(&node_features, &edge_index)?;
         assert_eq!(hidden.dims(), &[3, 8]);
 
         let output = layer2.forward(&hidden, &edge_index)?;
         assert_eq!(output.dims(), &[3, 4]);
 
         Ok(())
     }
}

            