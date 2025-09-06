use crate::Graph::graph::{Node , Edge , Graph , GraphOps , EdgeDirection , NodeID , EdgeID};
use crate::attn::{CrossAttention , BasicTransformerBlock , AttentionBlock};
use candle::core::{Tensor, Device, DType};
use candle::nn::{rnn , lstm}; 
use candle::nn::ops::{softmax , layer_norm , dropout , softmax_last_dim , leaky_relu};
use candle::nn::init::Init;
use candle::candle_transformers as transformers; 


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
        let weight_matrix = Tensor::randn(&[output_dim, input_dim], DType::F32, &device)?
            .mul_scalar(weight_std)?;
        
        let attention_vector = Tensor::randn(&[2 * output_dim], DType::F32, &device)?
            .mul_scalar(0.1)?;
        
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
    pub weight_matrix: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
    pub use_bias: bool,
}

impl GCN{
    pub fn new(input_dim: usize, output_dim: usize, use_bias: bool) -> Self {
        let mut rng = thread_rng();
        
        // Xavier initialization
        let xavier_std = (2.0 / (input_dim + output_dim) as f32).sqrt();
        let weight_matrix = (0..output_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| rng.gen_range(-xavier_std..xavier_std))
                    .collect()
            })
            .collect();
        
        let bias = if use_bias {
            (0..output_dim).map(|_| rng.gen_range(-0.1..0.1)).collect()
        } else {
            vec![0.0; output_dim]
        };
        
        GraphConvolution {
            input_dim,
            output_dim,
            weight_matrix,
            bias,
            use_bias,
        }
    }
    
    pub fn forward(&self , graph: &Graph , node_features : &Vec<Vec<f32>>) -> Vec<Vec<f32>>{
        let nodes = graph.nodes.len();
        let laplacian = graph.Laplacian(normalized = True).expect("Laplacian matrix not found");
        
        let mut new_node_feat = vec![vec![0.0; self.output_dim]; nodes];
        
        for i in 0..nodes{
            
            let mut aggregated_features =  vec![0.0; self.input_dim];
            
            for j in 0..nodes{
                let laplacian_weight = Laplacian[i][j];
                
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
        }
        
        output
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
    pub fn new(input_dim: usize , output_dim: usize , device: &Device) -> Result<Self> {
        Ok(Self {
            
            self_weight : Linear::new(
                Tensor::randn(&[output_dim , input_dim] , DType::F32 , device)?,
                Some(Tensor::zeros(&[output_dim], DType::F32, device)?),
            ),
            neighbor_weight : Linear::new(
                Tensor::randn(&[output_dim , neighbor_spread * embedding_dim] , DType::F32 , device)?,
                Some(Tensor::zeros(&[output_dim], DType::F32, device)?),
            ),
            activation: |x| x.relu(),
            bias : Linear::new(
                Tensor::randn(&[output_dim] , DType::F32 , device)?,
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
            layers.push(GraphSageLayer::new(layer_dims[i], layer_dims[i+1], &device)?)
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
                let neighbors = graph::neighbors(&node_id , Undirected)?;
                
                let neigbor_tensor : Vec<Tensor> = neighbors.iter().
                    filter_map(|&n| current_features.get(&n).cloned()).collect();
                
                let aggregated = 
                    if !neighbor_tensor.is_empty(){
                        GraphSageLayer::mean_aggregation(neighbor_tensor);
                    }
                    else{
                        Tensor::zeros(&[layer.input_dims] , DType = F32 , &self.device)?
                    };
                
                
                let node_feat = current_features.get(&node_id).unwrap();
                let output = layer.forward(node_feat , &aggregated)?;
                
                next_features = next_features.insert(node_id , output);
                
            }
            current_features = next_features;
        }
        Ok(current_features)
    }
} 


            