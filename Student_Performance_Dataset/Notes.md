

# Multi Linear Regression - Student Performance Dataset



Initially, the model yielded L1Loss of 18.55. The model has hardcoded parameters and uses L1Loss and SGD with learning parameter set to 0.1. 



The problems:

* The features in the dataset are not normalized. The number of hours studied is ranging from 0-24 and the previous scores range from 0 - 100. The need for standardization is that, the optimizer when it computes the gradient descent for each feature, the feature with the higher scale, would have higher gradient hence the feature with the lesser scale wouldn't be represented leading to higher errors. How do you normalize? z = (x - ʯ) / σ
* L1Loss and SGD do not go well together. Gradient is discontinuous at 0 (L1Loss uses mod func). So we use MSELoss. 
* Instead of hardcoding the parameters, we can use nn.Linear(in\_features=5, out\_features=1) (Note: use of nn.Linear itself doesn't mean it is a neural net) 







