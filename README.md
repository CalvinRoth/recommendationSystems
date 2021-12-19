# recommendationSystems
Class project for optimization for machine learning. 


# dependencies
This code uses the Pandas and Numpy packages 

In addition to run the code you must have access to MovieLens dataset. The version of the data I used is found on the educational data at 
https://grouplens.org/datasets/movielens/

If you need to download the data yourself you will have to change the variable "path" found toward the end of generalFuns.py to point to the dataset "ratings.csv". 

# Organizsation of code
This code is organized into several parts. The main script is called generalFuns.py. This includes generally needed functions like loading in the data but is also here I call the tests that I ran. Beyond that each other file is an algorithm the functions needed to run it. For example ALS.py includes code for the alternating least squares algorithm, PUREsvd has code for PUREsvd test, and slim.py has code for the SLIM matrix factorization algorithm.  
