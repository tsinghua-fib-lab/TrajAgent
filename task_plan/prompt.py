TASK_DESCRIPTION = {
    "Map_Matching":"""
    Map Matching means aligning raw trajectory data (e.g., GPS points) to a digital map or road network to infer the most accurate route and correct discrepancies caused by noise or errors.""",
    "Trajectory_Generation": """
    Trajectory Generation means creating synthetic or simulated trajectory data, often for use in predictive modeling, testing, or training machine learning algorithms.""",
    "Trajectory_Representation": """
    Trajectory Representation is a task of defining and structuring trajectory data in a meaningful and computationally efficient way, often involving techniques like feature extraction, segmentation, or transformation into alternative formats (e.g., grids, graphs, or symbolic sequences).""",
    "Trajectory_Recovery": """
    Trajectory Recovery is the task of reconstructing missing or incomplete trajectory data, often using interpolation, inference, or machine learning methods to fill gaps and restore accurate movement patterns.""",
    "Next_Location_Prediction":"""
    Trajectory next location prediction is a complex predictive modeling technique focused on forecasting an individual's 
    future geographical position based on their historical trajectory data. This method primarily revolves around understanding 
    and modeling human mobility patterns, which are influenced by various spatial and temporal factors.
    """,
    "Trajectory_User_Linkage":"""
    Trajectory user linking is a data analysis technique primarily utilized to identify and associate movement trajectory data 
    from various sources or different times to determine if they belong to the same user. This technique is especially useful in
    applications such as traffic management, location services, and social behavior research.
    """,
    "Travel_Time_Estimation": """
    Travel time estimation involves predicting the time required for a user to reach a specific destination based on 
    historical trajectory data and real-time contextual information. This task considers multiple influencing factors 
    such as  traffic conditions, road types, and departure time. This approach is crucial in applications like 
    navigation systems, traffic management, and logistics planning.
    """,
    "Trajectory_Anomaly_Detection":
    """
    Trajectory Anomaly Detection is a technique used to identify movement trajectories that deviate significantly from standard 
    or expected paths. This type of analysis is crucial in various fields, including traffic monitoring, security surveillance, 
    and mobile behavior analysis.
    """
}
TASK_INDEX = {
    "Next_Location_Prediction":1, "Trajectory_User_Linkage":2,"Estimated_Time_of_Arrival":3,"Trajectory_completion":4,"Trajectory_Anomaly_Detection":5,"Map_Matching":6
}

MODULE_DESCRIPTION = """
data transformation module: Transforming the format of a dataset to match the target dataset's format.
model selection module: Choose proper model to do the {task}.It make the choice based on the characteristics of each model, the characteristics of the input data, and historical experiment records. \
data augmentation module: Select proper augmentation methods and use them in proper order to jointly augment the input spatial-temporal data.It adjust the selection and combination sequence of operators based on the meaning of each operator, the characteristics of the input data, and historical experiment records. \
hypermeter optimization module: Select proper combination of hyperparameters of model.It adjust the selection, the combination of hyperparameters based on the main function of hyperparameters, the characteristics of the input data, the tuning principles, and memory to get a high score.\
result optimization module: Selecting the experiment record with the best result among all historical records. It can analyse the records and propose a better solution."
"""