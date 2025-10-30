QUESTION = """
<TASK>
Please select proper combination of hyperparameters of model in CONFIG HYPERPARAMETERS.Adjust the selection, the combination of hyperparameters based on the main function of hyperparameters, the characteristics of the input data, the tuning principles, and memory to get a high score. You should solve the task with interleaving Thought, Action and Observation steps. 
<CHARACTERISTICS OF INPUT DATA>
The input temporal data contains a time dictionary(key is the user ID,the value is a list containing all time points when the user is active in chronological order) , the input user data contains a user dictionary(key is the user ID,the value is a list containing all items that the user interacts with in chronological order).
<CONFIG HYPERPARAMETERS>
{config_params}
<HYPARAMETER MEANINGS>
{hyparameter_meanings}
"""
THOUGHT = """
In Thought step,you should reason how to choose the combination of hyperparameters to get a higher score.Please consider following aspects:
1.Observe the hyperparameters with high scores in MEMORY, to determine the optimal hyperparameters.
2.Use a grid search: Perform a grid search over a range of hyperparameters with high scores to find the optimal combination with higher scores.
3.Increase batch size and learning rate, and use dropout to avoid overfitting.
4. Stop or reverse the adjusting trend if the score is decreasing.
According to above aspects,please first learn experiences from MEMORY, then make plan for the action step.Please use the sentence structure 'Firstly... Then... Lastly'.Let's think step by step.
Thought:\n
"""
ACTION = """
In Action step, you should consider the Thought step in SCRATCHPAD, and give a dict:{hypermeter name:hypermeter value}.The hypermeter name should be the same with the \
    raw config hypermeter names in CONFIG HYPERPARAMETERS,and hypermeter values should be the same type as the hypermeter values in CONFIG HYPERPARAMETERS.Please do not add any comments to each value.
Action:\n
"""

XR_QUESTION = """
<TASK>
Please select proper combination of hyperparameters of model in CONFIG HYPERPARAMETERS.Adjust the selection, the combination of hyperparameters based on the main function of hyperparameters, the characteristics of the input data, the tuning principles to get a high score. You should solve the task with interleaving Thought, Action and Observation steps. 
<CHARACTERISTICS OF INPUT DATA>
The input temporal data contains a time dictionary(key is the user ID,the value is a list containing all time points when the user is active in chronological order) , the input user data contains a user dictionary(key is the user ID,the value is a list containing all items that the user interacts with in chronological order).
<CONFIG HYPERPARAMETERS>
{config_params}
<TUNING PRINCIPLES>
1.Start with a small batch size (32-64) and a small learning rate (0.001-0.01): This will help prevent overshooting and overfitting.
2.Increase batch size and learning rate: If the model is not overfitting, increasing the batch size and learning rate can help improve convergence.
3.Add dropout (0.2-0.5) to prevent overfitting: If the model is overfitting, adding dropout can help regularize the model.
4.Increase embedding size: If the model is not capturing enough information, increasing the embedding size can help improve representational power.
5.Decrease learning rate and increase batch size: If the model is not converging, decreasing the learning rate and increasing the batch size can help improve stability.
"""
XR_THOUGHT = """
In Thought step,you should reason how to choose the combination of hyperparameters to get a higher score.Please consider following aspects:
1.Use a grid search: Perform a grid search over a range of hyperparameters with high scores to find the optimal combination with higher scores.
2.Increase batch size and learning rate, and use dropout to avoid overfitting.
3. Stop or reverse the adjusting trend if the score is decreasing.
According to above aspects,please make plan for the action step.Please use the sentence structure 'Firstly... Then... Lastly'.Let's think step by step.
Thought:\n
"""
XR_ACTION = """
In Action step, you should consider the Thought step in SCRATCHPAD, and give a dict:{hypermeter name:hypermeter value}.The hypermeter name should be the same with the \
    raw config hypermeter names in CONFIG HYPERPARAMETERS,and hypermeter values should be the same type as the hypermeter values in CONFIG HYPERPARAMETERS.Please do not add any comments to each value.
Action:\n
"""
PARAM_MEANING = {
"TrajBERT":"""
hidden_size: A larger embedding size can better capture the complex relationships between position IDs, but if too large, it may lead to overfitting.
num_attention_heads: For simple human movement patterns, two layers of Transformer encoders and two attention heads are sufficient. Too many layers and heads may reduce the model's generalization ability.
num_hidden_layers: The number of Transformer encoder layers in the model needs to balance the model's complexity and generalization ability.
"""
}
# 1.Start with a small batch size (32-64) and a small learning rate (0.001-0.01): This will help prevent overshooting and overfitting.
# 2.Increase batch size and learning rate: If the model is not overfitting, increasing the batch size and learning rate can help improve convergence.
# 3.Add dropout (0.2-0.5) to prevent overfitting: If the model is overfitting, adding dropout can help regularize the model.
# 4.Increase embedding size: If the model is not capturing enough information, increasing the embedding size can help improve representational power.
# 5.Decrease learning rate and increase batch size: If the model is not converging, decreasing the learning rate and increasing the batch size can help improve stability.