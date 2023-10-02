"""
University of Waterloo Fall 2023 ECE-358 LAB-1  Group 151
József IVÁN GAFO (21111635) jivangaf@uwaterloo.ca
Sonia NAVAS RUTETE (21111397) srutete@uwaterloo.ca
V 1:0
In this module we will write the main code
"""

# Imports
import math
import random
import numpy as np

#We define the type of events
ARRIVAL="A"
DEPARTURE="D"
OBSERVER="O"

class Lab1():
    """
    This class is in charge of containing all the 
    methods required for the lab 1
    """

    def __init__(self) -> None:
        self.__event_list=[]
        self.__num_arrival = 0
        self.__num_departed = 0
        self.__num_observers = 0

    # We write the main code
    def question1(self, lambda_param):
        """
        This method is in charge of generating 1000 random variables 
        with lambda 75 with numpy library and it prints the mean and variance
        @param : None
        @return : None
        """
        # For this exercise we will use the library numpy
        # because it makes the operations easier

        # Calculate the expected mean and variance for an exponential distribution with λ=75
        expected_mean = 1 / lambda_param
        expected_variance = 1 / (lambda_param ** 2)

        # We generate 1000 exponential random variables
        generated_numbers = [self.__generate_exp_distribution(lambda_param) for _ in range(1000)]
        mean = np.mean(generated_numbers)
        variance = np.var(generated_numbers)

        # Print the results
        print(f"Mean of generated random variables: {mean}")
        print(f"Expected mean for λ=75: {expected_mean}")
        print(f"Variance of generated random variables: {variance}")
        print(f"Expected variance for λ=75: {expected_variance}")

    def m_m_1_queue(self, avg_len: int, trans_rate: int, lambda_par: int, T: int) -> None:
        """
        Build your simulator for this queue and explain in words what you have done. Show your code in the report. In 
        particular, define your variables. Should there be a need, draw diagrams to show your program structure. Explain how you 
        compute the performance metrics. Type of queue M/M/ (infinite queue)
        @param avg_len: Is the average length of a packet in bits (L)
        @param trans_rate: Is the transmission rate of the output link in bits/sec (C)
        @return: None
        """
        # Reset event_list values
        self.__event_list=[]
        self.__arrival_list=[]
        self.__departure_list=[]
        
        #Variable for iterations
        queue=[-1]
        simulation_time = 0
        delay_time = 0
        i=0

        while simulation_time < T:
            arrival_timestamp = self.__generate_exp_distribution(lambda_par)+simulation_time
            self.__arrival_list.append(arrival_timestamp)
            simulation_time = arrival_timestamp
        self.__arrival_list.sort()

        


        for i in range(len(self.__arrival_list)):
            self.__departure_list.append(self.__generate_exp_distribution(lambda_par))



        #Now we add the observers event
        simulation_time=0
        while simulation_time<T:
            #The lambda is 5* lambda parameter
            observation_timestamp=self.__generate_exp_distribution(5*lambda_par)+simulation_time
            #We check if we didn't exceed the time
            if simulation_time<observation_timestamp:
                self.__add_event(OBSERVER,observation_timestamp)
                simulation_time=observation_timestamp
                #We sum into de delay
                #TODO: delay_time+=departure_timestamp-arrival_timestamp

        # We sort the event list
        self.__event_list = sorted(self.__event_list, key=lambda x:x[1])

        
        

        total_num_packs_queue = 0
        total_observer_idles = 0
        while self.__event_list:
            event_type, time_event =self.__event_list.pop(0)
            #Arrival
            if event_type=="A":
                self.__num_arrival+=1
            elif event_type=="D":
                self.__num_departed+=1
            else:
                self.__num_observers +=1
                #We record the num of packets that are currently on the uque
                total_num_packs_queue+=(self.__num_arrival-self.__num_departed)
                if self.__num_arrival==self.__num_departed:
                    total_observer_idles += 1
        
        return total_num_packs_queue/self.__num_observers, total_observer_idles/self.__num_observers
    

        #Now we wil calculate the departure time 

    def __generate_exp_distribution(self, lambda_param: int) -> list:
        """
        This method is in charge of generating exponential random variables
        @param lambda_param: An integer that contains the average number of packets arrived
        @param size: An integer that defines how many numbers we generate
        @return list: We return a list with the numbers generated
        """
        expected_mean = 1/lambda_param
        return -expected_mean*math.log(1-random.random())
    
    def m_m_1_k_queue(self):
        ...
        # TODO: FUNCION QUEUE LLENA
        # TODO: CHECK SI ARRIVAL LOSS OR NOT 
        


    def __add_event(self, type, timestamp):
        
        self.__event_list.append((type, timestamp))

    def __queue_is_empty(self):
        return self.__num_arrival==self.__num_departed

def generate_graph_points(avg_packet_length, trans_rate, t):
    step = 0.1
    start = 0.25
    end = 0.95

    result = []

    i = start
    while i < end:
        lambda_para = trans_rate * i / avg_packet_length
        
        list_m_m_1 = a.m_m_1_queue(avg_packet_length, trans_rate, lambda_para, t)
        
        result.append([i, list_m_m_1[0]])
        print(list_m_m_1[0]);
        i += step

    for e in result:
        print(e)



if __name__=="__main__":
    a = Lab1()
    num_packets = 1_000
    lambda_par=75
    trans_rate = 1_000_000
    avg_packet_length = 2_000
    #a.question1(lambda_par)

    # print(a.m_m_1_queue(avg_packet_length,trans_rate,lambda_par,1000))

    generate_graph_points(avg_packet_length, trans_rate, 1000)
