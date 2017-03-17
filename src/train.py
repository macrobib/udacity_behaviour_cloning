from imageHandler import imageHandler

obj = imageHandler('../data/track-1')
obj.print_file_info()
obj.process_data()
obj.load_and_pickle_data()