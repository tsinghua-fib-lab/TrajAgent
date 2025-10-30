PARAMS_DESCRIPTION = {
"DeepMove": """
1.lr_step: Learning step. You can choose between [2,3,4,5].
2.learning_rate: Learning rate.Refers to the step size for updating model parameters, influencing the model's convergence speed and stability.You can choose betwen 0.05-0.0001.
3. dropout_p: Dropout probility,prevents overfitting, improves generalization, robustness.You can choose between 0.2-0.5.
""",
"DPLink": """
1.hidden_size: A larger embedding size can better capture the complex relationships between position IDs, but if too large, it may lead to overfitting. You can choose betwen [50,100,200,300].
2. loc_emb_size: Refers to the dimensionality of the embedding of location features, influencing the model's ability to represent the input information. You can choose betwen [150,200,300].
3. tim_emb_size: Refers to the dimensionality of the embedding of time features, influencing the model's ability to represent the input information. You can choose betwen [5,10,20].
4. dropout_p: Dropout probility,prevents overfitting, improves generalization, robustness.You can choose between 0.2-0.5.
5. l2: Weight decay (L2 loss on parameters).You can choose between 1e-4 to 1e-7.
""",
"FPMC":"""
1.Batch Size:
Function: Controls the amount of data used in each training iteration.Larger batch size means faster convergence, but requires more memory and computation.
Larger batch size may require higher learning rate, but may also cause oscillations.Smaller batch size may require lower learning rate to prevent overfitting.
Common values: 32, 64, 128, 256, 512
Type: int
2.Learning Rate:
Function: Controls the speed of model parameter updates.Higher learning rate means faster convergence, but may cause overfitting
Common values: 0.01-0.001
Type: float
3.Embedding Size:
Function: Captures semantic info and relationships,improves model understanding and processing.Larger embedding size improves model accuracy, but may lead to overfitting.
Common values: 64, 128, 256
Type: int
""",
"LSTPM":"""
1.Batch Size:
Function: Controls the amount of data used in each training iteration.Larger batch size means faster convergence, but requires more memory and computation.
Larger batch size may require higher learning rate, but may also cause oscillations.Smaller batch size may require lower learning rate to prevent overfitting.
Common values: 32, 64, 128, 256, 512
Type: int
2.Learning Rate:
Function: Controls the speed of model parameter updates.Higher learning rate means faster convergence, but may cause overfitting
Common values: 0.01-0.001
Type: float
3.Embedding Size:
Function: Captures semantic info and relationships,improves model understanding and processing.Larger embedding size improves model accuracy, but may lead to overfitting.
Common values: 64, 128, 256
Type: int
""",
"RNN":"""
1.Batch Size:
Function: Controls the amount of data used in each training iteration.Larger batch size means faster convergence, but requires more memory and computation.
Larger batch size may require higher learning rate, but may also cause oscillations.Smaller batch size may require lower learning rate to prevent overfitting.
Common values: 32, 64, 128, 256, 512
Type: int
2.Learning Rate:
Function: Controls the speed of model parameter updates.Higher learning rate: means faster convergence, but may cause overfitting
Common values: 0.01-0.001
Type: float
3. Dropout Probility:
Function: Prevents overfitting, improves generalization, robustness
Common values: 0.2-0.5
Type: float
""",
"GETNext":"""
1.poi-embed-dim:POI embedding dimensions.The default value is 128.
2.user-embed-dim:User embedding dimensions.The default value is 128.
3.time-embed-dim:Time embedding dimensions.The default value is 32.
4.cat-embed-dim:Category embedding dimensions.The default value is 32.
5.batch:Batch size.The default value is 20.
6.lr:Initial learning rate.The default value is 0.001
7.transformer-dropout":Dropout rate for transformer.The default value is 0.3.
8.weight_decay":Weight decay (L2 loss on parameters).The default value is 5e-4
""",
"TrajBERT":"""
1.hidden_size: A larger embedding size can better capture the complex relationships between position IDs, but if too large, it may lead to overfitting.
2.num_attention_heads: For simple human movement patterns, two layers of Transformer encoders and two attention heads are sufficient. Too many layers and heads may reduce the model's generalization ability.
3.num_hidden_layers: The number of Transformer encoder layers in the model needs to balance the model's complexity and generalization ability.
""",
"CACSR":"""
1.loc_emb_size: Refers to the dimensionality of the embedding of location features, influencing the model's ability to represent the input information.The default value is 256.
2.tim_emb_size: Refers to the dimensionality of the embedding of time features, influencing the model's ability to represent the input information.The default value is 256.
3.user_emb_size: Refers to the dimensionality of the embedding of user features, influencing the model's ability to represent the input information.The default value is 256.
4.num_layers: The number layer of Bi-LSTM in CACSR model.The default value is 3. 
5.hidden_size:A larger embedding size can better capture the complex relationships between position IDs, but if too large, it may lead to overfitting. The default value is 256. 
""",
"ActSTD":"""
1.hidden_size: A larger embedding size can better capture the complex relationships between position IDs, but if too large, it may lead to overfitting.
2.embedding_size: Refers to the dimensionality of the embedding of input features, influencing the model's ability to represent the input information.The default value is 256.
3.batch_size: Refers to the number of samples used in each training iteration, affecting the model's convergence speed and generalization performance.The default value is 32.
4.learning_rate: Refers to the step size for updating model parameters, influencing the model's convergence speed and stability.
""",
"S2TUL":"""
1.temporal: Measure the distance between two locations from the aspect of travelling distance, where locations below the distance are considered nearby.You can choose between [3,4,5,6].
2.spatio: Measure the distance between two locations from the aspect of travelling time, where locations below the distance are considered nearby.You can choose between [300,400,500,600].
3.hidden_size: A larger embedding size can better capture the complex relationships between position IDs, but if too large, it may lead to overfitting.The default value is [8,16,32,64,128]. 
4.learning_rate(lr): Refers to the step size for updating model parameters, influencing the model's convergence speed and stability.The default value is set to [1e-4, 1e-3, 1e-2].
5.batch_size: Refers to the number of samples used in each training iteration, affecting the model's convergence speed and generalization performance.The default [16,32,64,128].
6.dropout: Dropout probility,prevents overfitting, improves generalization, robustness.You can choose between [0.001,0.01,0.1,0.15].
""",
"GraphMM":"""
1.tf_ratio: It controls the proportion of using the actual target values as inputs during training, affecting how the model learns.The default value is 0.5. 
2.emb_dim: Embedding dimensions.The default value is 256. 
3.lr: Learning rate.Refers to the step size for updating model parameters, influencing the model's convergence speed and stability.The default value is set to 1e-4.
4.batch_size: Refers to the number of samples used in each training iteration, affecting the model's convergence speed and generalization performance.The default value is 32.
""",
"LLMZS":"""
1.context_len: Number of recent trajectory points used as context for prediction. Higher values provide more context but may increase noise. You can choose between 3-10.
2.history_len: Number of historical trajectory points used for training. Higher values provide more historical information but may increase computation time. You can choose between 20-60.
3.temperature: Controls the randomness of the model's output. Lower values (0.01-0.1) make the output more deterministic and focused, while higher values (0.5-1.0) make it more creative and diverse. You can choose between 0.01-1.0.
4.max_new_tokens: Maximum number of tokens the model can generate in response. Higher values allow longer responses but may increase computation time. You can choose between 100-500.
5.top_p: Nucleus sampling parameter that controls diversity by considering only the most likely tokens whose cumulative probability exceeds this threshold. You can choose between 0.8-1.0.
6.top_k: Limits the number of tokens considered for each generation step to the top k most likely tokens. You can choose between 50-500.
7.length_penalty: Penalizes longer sequences during generation. Values > 1 favor shorter outputs, values < 1 favor longer outputs. You can choose between 0.5-2.0.
8.presence_penalty: Penalizes tokens based on their frequency in the generated text. Higher values reduce repetition. You can choose between 0.0-2.0.
9.context_len: Number of context trajectory points to consider for prediction. You can choose between [3, 4, 5, 6, 7]. Default is 6.
10.history_len: Number of historical trajectory points to consider for prediction. You can choose between [10, 20, 30, 40, 50]. Default is 40.
""",
"DeepMM":"""
1.lrate: Learning rate.Refers to the step size for updating model parameters, influencing the model's convergence speed and stability.The default value is set to 0.001.
2.dim_loc_src: Refers to the dimensionality of the embedding of location features, influencing the model's ability to represent the input information.The default value is 256.
3.n_layers_src: Number of encoder layers.Increasing layers enhances feature extraction and long-range dependency capture but also increases computational cost and risk of overfitting.The default value is 2. 
4.n_layers_trg: Number of decoder layers.Adding layers improves generation quality and context memory but also increases computational burden and may lead to overfitting. The default value is 1.
5.batch_size: Refers to the number of samples used in each training iteration, affecting the model's convergence speed and generalization performance.The default value is 128.  
""",
"DSTPP":"""
1.batch_size: Refers to the number of samples used in each training iteration, affecting the model's convergence speed and generalization performance.You can choose between [32,64,128,256].
2.d_inner: This is the size of the inner feed-forward layers in the Transformer model. Larger values provide more expressiveness but can lead to overfitting if the dataset is small.You can choose between [32,64,128,256].
3.n_layers: The number of transformer layers (or encoder/decoder blocks).Increasing layers enhances feature extraction and long-range dependency capture but also increases computational cost and risk of overfitting.You can choose between [2,3,4,5].
""",
"MulTTTE":"""
1. input_dim:Input embedding dimensions.It allows the model to handle more complex features, but it also increases the computational burden. You can choose between [90,100,110,120,130].
2. seq_input_dim: The dimensionality of the sequence input features.You can choose between [90,100,110,120,130].
3. seq_hidden_dim": Defines the hidden layer size of the sequence model.Larger seq_hidden_dim enables the model to capture more complex temporal patterns, but it also increases the model's complexity and computational cost.You can choose between [32,64,128,256].
4. seq_layer: Defines the number of layers in the sequence model.More layers allow the model to learn more complex temporal dependencies, capturing higher-order patterns in the data. However, more layers also introduce higher computational cost and risk of overfitting.You can choose between [1,2,3,4].
5. bert_hiden_size: Defines the hidden size of the BERT model. larger hidden size allows BERT to learn more complex patterns from the input data, but it also increases the model's memory and computation requirements.You can choose between [32,64,128,256].
6. decoder_layer: Defines the number of layers in the decoder part of the model.You can choose between [1,2,3,4].
7. decode_head: Defines the number of attention heads in the decoder’s multi-head attention.You can choose between [1,2,3]
8. bert_hidden_layers: Defines the number of hidden layers in the BERT model.You can choose between [1,2,3,4].
9. bert_attention_heads: Defines the number of attention heads in each BERT layer.You can choose between [6,7,8,9].
""",
"DeepTTE":"""
1. kernel_size:The size of the convolutional filter, which determines the receptive field for feature extraction.You can choose between [6,7,8,9].
2. num_filter: The number of filters in a convolutional layer, defining the number of output channels.You can choose between [64,128,256].
3. num_final_fcs": The number of fully connected (FC) layers at the end of the model.You can choose between [2,3,4,5].
4. final_fc_size: The number of neurons in each fully connected layer.You can choose between [256, 512, 1024].
""",
"TrajCL":""" 
1. trans_attention_head: The number of attention heads.You can choose between [2,3,4].
2. trans_attention_dropout: Dropout rate for transformer.You can choose between [0.05,0.1,0.15].
3. trans_attention_layer: The number of encoder layers.You can choose between choose between [1,2,3,4].
4. embedding_dim": Embedding dimensions of structural feature and the spatial feature .It allows the model to handle more complex structural and spatial features of trajectories, but it also increases the computational burden. You can choose between [128,64,32,256].
""",
"MainTUL":"""
1. batch_size: Refers to the number of samples used in each training iteration, affecting the model's convergence speed and generalization performance.You can choose between [32,64,128,256].
2. learning_rate": Learning rate.Refers to the step size for updating model parameters, influencing the model's convergence speed and stability. You can choose between [0.0005,0.001,0.015].
3. embed_size": The dimensionality of the embedding of input features, influencing the model's ability to represent the input information. You can choose between [16,32,64,128].
4. num_layers": The number of encoder layers.You can choose between [2,3,4,5].
""",
"DutyTTE":"""
1. C: Number of experts in MoEUQ model. Controls the model's capacity and expressiveness. You can choose between [4, 8, 16]. Default is 8.
2. k: Number of selected experts in MoEUQ model. Controls the sparsity and efficiency. You can choose between [2, 4, 6]. Default is 4.
3. E_U: Embedding dimension in MoEUQ model. Controls the feature representation capacity. You can choose between [128, 256, 512]. Default is 256.
4. lr: Learning rate for training. Controls the convergence speed and stability. You can choose between [0.0001, 0.001, 0.01]. Default is 0.001.
5. load_balancing_weight: Weight for load balancing loss in MoEUQ. You can choose between [0.01, 0.05, 0.1]. Default is 0.05.
"""
}
