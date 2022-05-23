import pickle



def save_metrics(history, pickle_file_name, path_to_save):
  
  f = open(""+path_to_save+"/"+pickle_file_name, 'wb')
  pickle.dump(history.history, f)
  f.close()
  return


def load_metrics(pickle_path):

  f = open(pickle_path, 'rb')
  history = pickle.load(f)
  f.close()
  return history