use crate::graph::{Edge, EdgeDirection, EdgeID, Graph, GraphError, GraphOps, Node, NodeID};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Dropout, Linear, Module, init, ops::softmax};
use transformers::models::components::layers::layer_norm;
use std::collections::HashMap;


pub struct GATMH {
    w: Tensor,      // [H, D_head, D_in]
    a_src: Tensor,  // [H, D_head]
    a_dst: Tensor,  // [H, D_head]
    num_heads: usize,
    d_in: usize,
    d_head: usize,
    dropout: f64,
    alpha: f64, // LeakyReLU negative slope
    device: Device,
}

impl GATMH {
    pub fn new(
        d_in: usize,
        d_out: usize,
        num_heads: usize,
        dropout: f64,
        alpha: f64,
        device: &Device,
    ) -> candle_core::Result<Self> {
        assert!(d_out % num_heads == 0, "d_out must be divisible by num_heads");
        let d_head = d_out / num_heads;
        let w = Tensor::randn(
            0.0f32,
            (2.0 / (d_in + d_head) as f64).sqrt(),
            &[num_heads as i64, d_head as i64, d_in as i64],
            device,
        )?;
        let a_src = Tensor::randn(0.0f32, 0.02, &[num_heads as i64, d_head as i64], device)?;
        let a_dst = Tensor::randn(0.0f32, 0.02, &[num_heads as i64, d_head as i64], device)?;
        Ok(Self {
            w,
            a_src,
            a_dst,
            num_heads,
            d_in,
            d_head,
            dropout,
            alpha,
            device: device.clone(),
        })
    }

    /// x: [N, D_in], edge_index: [2, E] (indices as i64)
    pub fn forward(&self, x: &Tensor, edge_index: &Tensor) -> candle_core::Result<Tensor> {
        let n = x.dim(0)? as usize;
        let e = edge_index.dim(1)? as usize;

        let edges = edge_index.to_vec2::<i64>()?;
        let src_idx = &edges[0];
        let dst_idx = &edges[1];

        let mut head_outputs: Vec<Tensor> = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let w_h = self.w.i(h as i64)?; // [D_head, D_in]
            let a_src_h = self.a_src.i(h as i64)?; // [D_head]
            let a_dst_h = self.a_dst.i(h as i64)?; // [D_head]

            let h_feat = x.matmul(&w_h.t())?; // [N, D_head]
            let h_feat_v = h_feat.to_vec2::<f32>()?;
            let a_src_v = a_src_h.to_vec1::<f32>()?;
            let a_dst_v = a_dst_h.to_vec1::<f32>()?;

            // Compute LeakyReLU attention scores per edge
            let mut scores = vec![0f32; e];
            for k in 0..e {
                let i = src_idx[k] as usize;
                let j = dst_idx[k] as usize;
                if i >= n || j >= n {
                    continue;
                }
                let mut s = 0f32;
                for d in 0..self.d_head {
                    s += a_src_v[d] * h_feat_v[i][d] + a_dst_v[d] * h_feat_v[j][d];
                }
                scores[k] = if s > 0.0 { s } else { (self.alpha as f32) * s };
            }

            // Softmax per source node, then aggregate neighbor features
            let mut by_src: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
            for k in 0..e {
                let i = src_idx[k] as usize;
                let j = dst_idx[k] as usize;
                if i < n && j < n {
                    by_src[i].push((j, scores[k]));
                }
            }
            let mut out = vec![vec![0f32; self.d_head]; n];
            for i in 0..n {
                if by_src[i].is_empty() {
                    continue;
                }
                let max_s = by_src[i]
                    .iter()
                    .map(|(_, s)| *s)
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut exps = Vec::with_capacity(by_src[i].len());
                let mut sum = 0f32;
                for (_, s) in &by_src[i] {
                    let v = (*s - max_s).exp();
                    exps.push(v);
                    sum += v;
                }
                for ((j, _), v) in by_src[i].iter().zip(exps.iter()) {
                    let w = *v / sum;
                    for d in 0..self.d_head {
                        out[i][d] += w * h_feat_v[*j][d];
                    }
                }
            }

            let flat: Vec<f32> = out.into_iter().flatten().collect();
            head_outputs.push(Tensor::from_vec(flat, (n, self.d_head), &self.device)?);
        }

        Tensor::cat(&head_outputs.iter().collect::<Vec<_>>(), 1)
    }
}


pub struct GraphSageLayer {
    pub self_weight: Linear,
    pub neighbor_weight: Linear,
    pub activation: fn(&Tensor) -> Tensor,
    pub bias: Linear,
    pub use_bias: bool,
    pub agg_func: String,
}

impl GraphSageLayer {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        neighbor_spread: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            self_weight: Linear::new(
                Tensor::randn(0.0f32, &[output_dim, input_dim], DType::F32, device)?,
                Some(Tensor::zeros(&[output_dim], DType::F32, device)?),
            ),
            neighbor_weight: Linear::new(
                Tensor::randn(
                    0.0f32,
                    &[output_dim, neighbor_spread * output_dim],
                    DType::F32,
                    device,
                )?,
                Some(Tensor::zeros(&[output_dim], DType::F32, device)?),
            ),
            activation: |x| x.relu(),
            bias: Linear::new(
                Tensor::randn(0.0f32, &[output_dim], DType::F32, device)?,
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

    pub fn forward(&self, node_features: &Tensor, neighbor_features: &Tensor) -> Result<Tensor> {
        let self_transformed = self.self_weight.forward(node_features)?;

        // Transform aggregated neighbor features
        let neighbor_transformed = self.neighbor_weight.forward(neighbor_features)?;

        // Combine and activate
        let combined = (&self_transformed + &neighbor_transformed)?;
        (self.activation)(&combined)
    }
}

pub struct GraphSage {
    pub layers: Vec<GraphSageLayer>,
    pub device: Device,
    pub sample_size: usize,
}

impl GraphSage {
    pub fn new(layer_dims: &[usize], sample_size: Vec<usize>, device: Device) -> Result<Self> {
        let mut layers = Vec::new();

        for i in 0..layer_dims.len() {
            layers.push(GraphSageLayer::new(
                0.0f32,
                layer_dims[i],
                layer_dims[i + 1],
                &device,
            )?)
        }

        Ok(Self {
            layers,
            device,
            sample_size,
        })
    }

    pub fn forward(
        &self,
        graph: &Graph,
        node_features: &HashMap<NodeID, Tensor>,
        target_nodes: &[NodeID],
    ) -> Result<HashMap<NodeID, Tensor>> {
        let mut current_features = node_features.clone();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let sample_size = self.sample_size[layer_idx];
            let mut next_features = HashMap::new();

            for &node_id in target_nodes {
                let neighbors = Graph::neighbors(graph, node_id, EdgeDirection::Outgoing)?;

                let neighbor_tensor: Vec<Tensor> = neighbors
                    .iter()
                    .filter_map(|&n| current_features.get(&n).cloned())
                    .collect();

                let aggregated = if !neighbor_tensor.is_empty() {
                    GraphSageLayer::mean_aggregation(neighbor_tensor).unwrap();
                } else {
                    Tensor::zeros(&[layer.input_dims], candle_core::DType::F64, &self.device)
                        .unwrap();
                };

                let node_feat = current_features.get(&node_id).unwrap();
                let output = layer.forward(node_feat, &aggregated).unwrap();

                next_features.insert(node_id, output);
            }
            current_features = next_features;
        }
        Ok(current_features)
    }
}

pub struct GraphTransformer {
    attn_layers: Vec<GATMH>,
    ff1: Tensor, // [D_out, D_hid]
    ff2: Tensor, // [D_hid, D_out]
    device: Device,
}

impl GraphTransformer {
    pub fn new(
        d_in: usize,
        d_hidden: usize,
        d_out: usize,
        heads: usize,
        layers: usize,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let mut attn_layers = Vec::with_capacity(layers);
        attn_layers.push(GATMH::new(d_in, d_hidden, heads, 0.0, 0.2, device)?);
        for _ in 1..layers {
            attn_layers.push(GATMH::new(d_hidden, d_hidden, heads, 0.0, 0.2, device)?);
        }
        let ff1 = Tensor::randn(
            0.0f32,
            (2.0 / (d_hidden + d_out) as f64).sqrt(),
            &[d_out as i64, d_hidden as i64],
            device,
        )?;
        let ff2 = Tensor::randn(
            0.0f32,
            (2.0 / (d_out + d_hidden) as f64).sqrt(),
            &[d_hidden as i64, d_out as i64],
            device,
        )?;
        Ok(Self {
            attn_layers,
            ff1,
            ff2,
            device: device.clone(),
        })
    }
    
    pub fn forward(&self, x: &Tensor, edge_index: &Tensor) -> candle_core::Result<Tensor> {
        let mut h = x.clone();
        for (li, layer) in self.attn_layers.iter().enumerate() {
            let h_proj = layer.forward(&h, edge_index)?; // [N, D_hid]
            // Residual when dimensions match; otherwise, bypass
            let h_next = if h_proj.dims() == h.dims() {
                (&h_proj + &h)?
            } else {
                h_proj
            };
            // FFN residual: h = h + FF2(ReLU(FF1(h)))
            let f1 = h_next.matmul(&self.ff1.t())?.relu()?;
            let f2 = f1.matmul(&self.ff2.t())?;
            h = (&h_next + &f2)?;
            // Optional: insert normalization/dropout here if needed
            h = layer_norm(h.size(), eps, vb)
            let _ = li; // silence unused if needed
        }
        Ok(h)
    }

    /// Convenience: build edge_index from the graph and run forward.
    pub fn forward_graph(
        &self,
        graph: &Graph,
        x: &Tensor,
    ) -> Result<Tensor, GraphError> {
        let (edge_index, _edge_weight, _edge_attr) = graph.to_coo()?;
        Ok(self.forward(x, &edge_index)?)
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, EdgeDirection, EdgeID, Graph, GraphError, GraphOps, Node, NodeID};

    fn create_test_graph() -> Graph {
        let mut undirected_graph = Graph::undirected();

        for i in 0..4 {
            GraphOps::add_node(&mut undirected_graph, Node::new(i, format!("node_{}", i)));
        }

        undirected_graph
            .add_edge(Edge::new(0, 0, 1, "edge".to_string()))
            .unwrap();
        undirected_graph
            .add_edge(Edge::new(1, 1, 2, "edge".to_string()))
            .unwrap();
        undirected_graph
            .add_edge(Edge::new(2, 2, 3, "edge".to_string()))
            .unwrap();
        undirected_graph
            .add_edge(Edge::new(3, 0, 2, "edge".to_string()))
            .unwrap();

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
    fn test_gat_forward_pass() -> Result<(), GraphError> {
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
        let edge_index = Tensor::new(vec![vec![0u32, 1u32], vec![1u32, 2u32]], &device)?;

        let result = gat_layer.forward(&node_features, &edge_index);
        assert!(result.is_ok());

        let output = result?;
        let output_shape = output.dims();
        assert_eq!(output_shape[0], num_nodes);
        assert_eq!(output_shape[1], output_dim);

        Ok(())
    }

    #[test]
    fn test_gat_attention_computation() -> Result<(), GraphError> {
        let device = Device::Cpu;
        let gat_layer = GATLayer::new(3, 2, 1, 0.0, 0.2, device.clone())?;

        let transformed_features = Tensor::new(
            vec![vec![1.0f32, 0.5], vec![0.8f32, 1.2], vec![1.5f32, 0.3]],
            &device,
        )?;

        let edge_index = Tensor::new(vec![vec![0u32, 1u32], vec![1u32, 2u32]], &device)?;

        let attention_scores =
            gat_layer.compute_attention_scores(&transformed_features, &edge_index);
        assert!(attention_scores.is_ok());

        let scores = attention_scores?;
        assert_eq!(scores.dims()[0], 2);

        Ok(())
    }

    #[test]
    fn test_graphsage_layer_creation() -> Result<(), GraphError> {
        let device = Device::Cpu;
        let input_dim = 4;
        let output_dim = 2;
        let neighbor_spread = 10;
        let sage_layer = GraphSageLayer::new(input_dim, output_dim, neighbor_spread, &device);
        assert!(sage_layer.is_ok());

        let layer = sage_layer?;
        assert_eq!(layer.agg_func, "mean");
        assert_eq!(layer.use_bias, true);

        Ok(())
    }

    #[test]
    fn test_graphsage_mean_aggregation() -> Result<(), GraphError> {
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
    fn test_graphsage_layer_forward() -> Result<(), GraphError> {
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
    fn test_graphsage_model_creation() -> Result<(), GraphError> {
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
    fn test_edge_index_bounds_checking() -> Result<(), GraphError> {
        let device = Device::Cpu;
        let gat_layer = GATLayer::new(2, 2, 1, 0.0, 0.2, device.clone())?;

        // Create node features for 2 nodes
        let node_features = Tensor::new(vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0]], &device)?;

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

        let node_features = Tensor::new(vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]], &device)?;

        let edge_index = Tensor::new(vec![vec![0u32, 1u32], vec![0u32, 1u32]], &device)?;

        let result = gat_layer.forward(&node_features, &edge_index)?;
        assert_eq!(result.dims(), &[2, 2]);

        Ok(())
    }

    #[test]
    fn test_feature_dimension_consistency() {
        let gcn = GCN::new(3, 2, false);

        let inconsistent_features = vec![
            vec![1.0, 2.0],      // Wrong dimension (2 instead of 3)
            vec![3.0, 4.0, 5.0], // Correct dimension
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
            vec![vec![0u32, 1u32, 2u32], vec![1u32, 2u32, 0u32]],
            &device,
        )?;

        let hidden = layer1.forward(&node_features, &edge_index)?;
        assert_eq!(hidden.dims(), &[3, 8]);

        let output = layer2.forward(&hidden, &edge_index)?;
        assert_eq!(output.dims(), &[3, 4]);

        Ok(())
    }
}
