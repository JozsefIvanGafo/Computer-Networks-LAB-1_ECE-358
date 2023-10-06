"""
University of Waterloo Fall 2023 ECE-358 LAB-1  Group 151
József IVÁN GAFO (21111635) jivangaf@uwaterloo.ca
Sonia NAVAS RUTETE (21111397) srutete@uwaterloo.ca
V 1:0
In this module we will write the main code
"""

# Imports
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import random
import time

# We define the type of events
ARRIVAL = "A"
DEPARTURE = "D"
OBSERVER = "O"


class Lab1():
    """
    This class is in charge of containing all the 
    methods required for the lab 1
    """

    def __init__(self) -> None:
        pass
    def __generate_exp_distribution(self, lambda_param: int) -> list:
        """
        This method is in charge of generating exponential random variables
        @param lambda_param: An integer that contains the average number of packets arrived
        @param size: An integer that defines how many numbers we generate
        @return list: We return a list with the numbers generated
        """
        expected_mean = 1/lambda_param
        return -expected_mean*math.log(1-random.random())
    # We write the main code
    def __generate_mm1_arr_obs(self, lambda_par, T, steps=1):
        """
        This method is in charge of generating the list of arrivals and observers
        @param lambda_param: An integer that contains the average number of packets arrived
        @param T: Duration of the simulation
        @param steps: It is 1 for default, will change for observers as the param is different
        @return list: We return a list with the events generated
        """
        # Iteration variables
        aux_list = []
        simulation_time = 0
        #Loop we iterate until we reach the simulation time
        while simulation_time < T:
            #We generate an exponential distribution
            arrival_timestamp = self.__generate_exp_distribution(lambda_par*steps)+simulation_time
            #We add to the aux list
            aux_list.append(arrival_timestamp)
            #We update the simulation time with the new arrival
            simulation_time = arrival_timestamp
        #We return the aux list
        return aux_list

    def m_m_1_k_queue(self, avg_len:int, trans_rate:int,lambda_par:int,T:int,K_num:int)->[float,float,float]:
        """
        This method is in charge of simulating the m_m_1_k queue
        @param avg_len: This integer represent the average length packet in bits
        @param trans_rate: This integer represent the transmission rate of a packet.
        @param lambda_par: This integer represents the parameter lambda od the poisson distribution
        @param T: This integer represent the simulation time
        @param K: This integer represent the max number of packets that a queue can hold
        @return a list: It returns a list of floats where the first element represent E[n],p_idle and p_loss
        """
        list_arrivals = []
        list_arrivals = self.__generate_mm1_arr_obs(lambda_par, T)

        list_observers = []
        list_observers  = self.__generate_mm1_arr_obs(lambda_par*5, T)

        list_events = []
        for e in list_arrivals:
            list_events.append(["A",e])

        for e in list_observers:
            list_events.append(["O",e])
        
        list_events = sorted(list_events, key=lambda x: x[1])
        

        num_elem_queue = 0
        n_arrivals = 0
        n_observers = 0
        n_departures = 0

        total_idles = 0
        total_packs_queue = 0

        last_departure = 0
        departure_list = []


        i = 0
        while i < len(list_events) or departure_list != []:
            if(i == len(list_events)):
                for e in departure_list:
                    departure_list.pop(0)
                    n_departures += 1
                    num_elem_queue -= 1
            else:
                if(departure_list != []):
                    
                    if(departure_list[0][1] < list_events[i][1]):
                        event = departure_list.pop(0)
                    else:
                        event = list_events[i]
                else:
                    event = list_events[i]
                if(event[0] == "A"):
                    if num_elem_queue < K_num:
                        n_arrivals += 1
                        arrival_time = event[1]
                        length_packet = self.__generate_exp_distribution(1/avg_len)
                        service_time = length_packet/trans_rate
                        
                        if num_elem_queue == 0:
                            departure_time = arrival_time + service_time
                        elif num_elem_queue < K_num:
                            departure_time = last_departure + service_time
                        departure_list.append(["D", departure_time])
                        last_departure = departure_time
                        num_elem_queue += 1
                    i += 1

                elif(event[0] == "O"):
                    n_observers += 1
                    total_packs_queue += (n_arrivals - n_departures)
                    if n_arrivals == n_departures:
                        total_idles += 1
                    i+= 1

                elif(event[0] == "D"):
                    n_departures += 1
                    num_elem_queue -= 1
            
        return total_packs_queue/n_observers, total_idles/n_observers, 1 - (n_departures/n_arrivals)
        


    def create_graph_for_m_m_1_k_queue(self, avg_len, trans_rate, T):
        #Define iteration variables
        step = 0.1
        start = 0.25
        end = 1.6

        #list for the different K
        K_list=[10,25,50]

        #where we store the y results
        y=[]
        for _,k in enumerate(K_list):
            print("generating points for graph mm1k queue for k= %i"%(k))
            #we create a result list to store all the point for a given k
            result=[]
            #We run a function per core in cpu (this is so the code run faster)
            with Pool() as pool:
                #we create the inputs for the function mm1k queue
                input_data = [(i, avg_len, trans_rate, trans_rate * i / avg_len, T, k)
                            for i in np.arange(start, end, step)]
                #We run the function on the core
                pool_list = pool.starmap(self.generate_points2, input_data)
                result.append(pool_list)

            #we save append it to y (to save the result and later create the graphs)
            #pos 0 is for E[n]  ,pos 1 is for p_idle and pos 2 is p_loss
            y.append([[point[1][0] for point in result[0]],
                [point[1][1] for point in result[0]],[point[1][2]for point in result[0]] ])
            
            print("Finished generating points for graph mm1 queue for k= %i"%(k))

        print(y)
        # We save the x points ( theya re the same for every k)
        x = [point[0] for point in result[0]]
        

        #We create the graph
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        graph_list=[ax1,ax2,ax3]
        #variables for creating the graph
        text=[['Average number in system E[n]',"average #packets as a function of p"],
              ["Average number in system p_idle","The proportion of time the system is idle"],
              ["Average number in system p_loss", "The proportion of packets the system has lost"]]
        colours=["green","blue","black"]
        for i,graph in enumerate(graph_list):
            # We initialize the graph
            for k,k_number in enumerate(K_list):
                label_name= "K= %i"%(k_number)
                graph.scatter(x, y[k][i],color="red",marker='x')
                graph.plot(x, y[k][i],label=label_name,color=colours[k])
            graph.set_xlabel('Traffic intensity p')
            graph.set_ylabel(text[i][0])
            graph.set_title(text[i][1])
            #we write the legend
            graph.legend()
        #We save it
        # Save the figure as an image in the "graphs" folder
        script_directory = os.path.dirname(__file__)

        # Save the figure as an image named "exercise_3.png" in the same folder as the script
        image_path = os.path.join(script_directory, 'exercise_6.png')
        plt.savefig(image_path)
        



    def generate_points2(self,i,avg_len,trans_rate,lambda_par,T,K):
        # Calculate data point for a specific 'i'
        list_m_m_1 = self.m_m_1_k_queue(avg_len, trans_rate, lambda_par, T,K)
        # If we want E[n] then type_info is 0 if is p_idle then type_info is 1
        return [i, list_m_m_1]



    
if __name__ == "__main__":
    a = Lab1()
    lambda_par = 75
    trans_rate = 1_000_000
    avg_packet_length = 2_000
    T = 1_000

    # RUNNING THE LAB
    # QUESTION 1
    #a.question1(lambda_par)
#
    # INFINITE QUEUE
    #print("QUESTION 2:\n")
    #X = check_T(avg_packet_length, trans_rate, lambda_par, T)
    #print("\tThe Final T will be: " + str(T*X))

    #print("QUESTION3:\n")
    #a.create_graph_for_m_m_1_queue(avg_packet_length,trans_rate,2*T)
#
    ## For p=1.2
    #E, pidle = a.m_m_1_queue(avg_packet_length, trans_rate, trans_rate * 1.2 / avg_packet_length, 2*T)
    #print("QUESTION 4:\n")
    #print("\tFor p = 1.2, the value of E[N] = "+ str(E) + " and the value of pidle =" + str(pidle))


    # FINITE
    #a.m_m_1_k_queue(avg_packet_length,trans_rate,lambda_par,T,10)
    """k = [10, 25, 50]
    for element in k:
        print(generate_graph_points2(avg_packet_length, trans_rate, T, element))"""
    a.create_graph_for_m_m_1_k_queue(avg_packet_length,trans_rate,T)
    
   #""" print(a.m_m_1_k_queue(avg_packet_length, trans_rate, lambda_par, T, 10))
    #print(a.m_m_1_k_queue(avg_packet_length, trans_rate, lambda_par, 2*T, 10))
    #print(a.m_m_1_k_queue(avg_packet_length, trans_rate, lambda_par, 3*T, 10))"""
