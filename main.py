"""
University of Waterloo Fall 2023 ECE-358 LAB-1  Group 151
József IVÁN GAFO (21111635) jivangaf@uwaterloo.ca
Sonia NAVAS RUTETE (21111397) srutete@uwaterloo.ca
V 1:0
In this module we will write the main code
"""

#Imports
import math
import random
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
        generated_numbers=[]
        for _ in range(1000):
            #Generate a number between [0,1)
            random_number=random.random()
            #We add the new generated number to the list
            generated_numbers.append(-expected_mean*math.log(1-random_number))

        mean=np.mean(generated_numbers)
        variance=np.var(generated_numbers)

        # Print the results
        print(f"Mean of generated random variables: {mean}")
        print(f"Expected mean for λ=75: {expected_mean}")
        print(f"Variance of generated random variables: {variance}")
        print(f"Expected variance for λ=75: {expected_variance}")
a=Lab1()
a.question1()
