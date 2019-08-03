# Student_Admission_Prediction
A 2-layer Neural network which Predicts whether the student will get admitted to the university or not on the basis of data of previous years.
  
  The dataset originally came from here: http://www.ats.ucla.edu/
  
  NN predicts student admissions to graduate school at UCLA based on three pieces of data:
- GRE Scores (Test)
- GPA Scores (Grades)
- Class rank (1-4)
  
  Neural network starts off with random weights and then decreases the error and corrects the weights using the Backpropagation Algorithm.
  
  Despite having a very small and unpredictable data-set of just 400 entries the Neural network achieves an accuracy of 75%.
  
  
  # Here is how the data looks like
  
     ![howdatalooks](https://github.com/Tiwarim386/Student_Admission_Prediction/blob/master/howdatalooks.PNG)
      
      
      
   # It seems that the lower the rank, the higher the acceptance rate. So we'll use the rank as one of our inputs.
   
    Here is how the data looks when rank is fixed.
    
   
    ![rank1](https://github.com/Tiwarim386/Student_Admission_Prediction/blob/master/rank1.PNG)
     
    ![rank2](https://github.com/Tiwarim386/Student_Admission_Prediction/blob/master/rank2.PNG)

    ![rank3](https://github.com/Tiwarim386/Student_Admission_Prediction/blob/master/rank3.PNG)

    ![rank4](https://github.com/Tiwarim386/Student_Admission_Prediction/blob/master/rank4.PNG)
