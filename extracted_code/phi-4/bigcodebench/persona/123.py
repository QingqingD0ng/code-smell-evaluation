def task_func(my_list, file_dir='./data_files/', file_ext='.csv'):
    if not isinstance(my_list, list):
        raise TypeError("my_list must be of type list")
    
    my_list.append(12)
    total_files = sum(my_list)
    
    files = glob.glob(os.path.join(file_dir, f"*{file_ext}"))
    if not files or len(files) < total_files:
        raise FileNotFoundError("Not enough files found in the specified directory")
    
    data_frames = []
    for file in files[:total_files]:
        df = pd.read_csv(file)
        data_frames.append(df)
    
    result_df = pd.concat(data_frames, ignore_index=True)
    return result_df