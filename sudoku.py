import copy
import numpy as np

def solve(puzzle, sudoku_model):

  # deep copy of the puzzle sample
  X = copy.copy(puzzle)
    
  while(True):
    
        # Predict using the trained model

        probabilities = sudoku_model.predict(X.reshape((1,9,9,1))).squeeze() 

        # Take the maximum probability
        prediction = np.argmax(probabilities, axis=1).reshape((9,9))+1 
        probability = np.around(np.max(probabilities, axis=1).reshape((9,9)), 2) 
        
        #Process
        X = (X+.5)*9
        X = X.reshape((9,9))
        # Output invalid
        masking = (X==0)
     
        if(masking.sum()==0):
            break
            
        probability_masked = probability*masking
    
        max_index = np.argmax(probability_masked)
        x, y = (max_index//9), (max_index%9)
        val = prediction[x][y]
        X[x][y] = val
        X = (X/9)-.5
    
  return prediction


def solve_sudoku(game, sudoku_shape,sudoku_model ):
  output=''
  for x in range(len(game)):
    for y in range(len(game[x])):
      # print(str(game[x][y]))
      var = str(game[x][y])
      output = output + var
    # print(output)
  puzzle = np.array([int(j) for j in output]).reshape(sudoku_shape)
  puzzle = (puzzle/9)-.5
  answer = solve(puzzle, sudoku_model)
  return answer