class Model_Config:
    def __init__(self, window_size, use_day, user_freq, loc_freq, threshold):
        self.window_size = window_size
        self.use_day = use_day
        self.user_freq = user_freq
        self.threshold = threshold
        self.loc_freq = loc_freq
        
class Data_Config:
    def __init__(self, max_session_len, min_session_len, min_sessions, max_sessions, min_checkins):
        self.max_session_len = max_session_len
        self.min_session_len = min_session_len
        self.min_sessions = min_sessions
        self.max_sessions = max_sessions
        self.min_checkins = min_checkins
        
class GPS_Data_Config:
    def __init__(self, max_len, min_len):
        self.max_len = max_len
        self.min_len = min_len
        
class GPS_Model_Config:
    def __init__(self, data_size, threshold):
        self.data_size = data_size
        self.threshold = threshold
              