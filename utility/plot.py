import matplotlib.pyplot as plt

def plot(history):

  # Plot the graph for training accuracy
  plt.plot(history['accuracy'])
  plt.plot(history['val_accuracy'])
  plt.title('Digit Recognition Model - Accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['Training', 'Testing'], loc='lower right')
  plt.show()

  # Plot the graph for training loss
  plt.plot(history['loss'])
  plt.plot(history['val_loss'])
  plt.title('Digit Recognition Model - Loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['Training', 'Testing'], loc='upper right')
  plt.show()