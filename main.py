"""
University of Waterloo Fall 2023 ECE-358 LAB-1  
József IVÁN GAFO (21111635) jivangaf@uwaterloo.ca
Sonia NAVAS RUTETE (21111397) 
V 1:0
In this module we will write the main code
"""

#Imports
import numpy as np

class Lab1():
    """
    This class is in charge of containing all the 
    methods required for the lab 1
    """

    #We write the main code
    def question1 (self):
        """
        This method is in charge of generating 1000 random variables 
        with lambda 75 with numpy library and it prints the mean and variance
        @param : None
        @return : None
        """
        #For this exercise we will use the library numpy
        # because it makes the operations easier

        # Set lambda
        lambda_param = 75

        # Calculate the expected mean and variance for an exponential distribution with λ=75
        expected_mean = 1 / lambda_param
        expected_variance = 1 / (lambda_param ** 2)

        # We generate 1000 exponential random variables
        random_variables = np.random.exponential(scale=expected_mean, size=1000)
        # Calculate the mean and variance of the generated random variables
        mean = np.mean(random_variables)
        variance = np.var(random_variables)

        # Print the results
        print(f"Mean of generated random variables: {mean}")
        print(f"Expected mean for λ=75: {expected_mean}")
        print(f"Variance of generated random variables: {variance}")
        print(f"Expected variance for λ=75: {expected_variance}")
a=Lab1()
a.question1()
