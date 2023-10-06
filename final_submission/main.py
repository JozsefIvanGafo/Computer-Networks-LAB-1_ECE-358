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
import math
import random
import os


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

    # We write the main code
    def question1(self, lambda_param):
        """
        This method is in charge of generating 1000 random variables 
        with lambda 75 with numpy library and it prints the mean and variance
        @param lambda_param: Value for lambda 
        @return : None
        """
        # For this exercise we will use the library numpy
        # as it makes the operations easier

        # Calculate the expected mean and variance for an exponential distribution with λ=75
        expected_mean = 1 / lambda_param
        expected_variance = 1 / (lambda_param ** 2)

        # We generate 1000 exponential random variables
        generated_numbers = [self.__generate_exp_distribution(
            lambda_param) for _ in range(1000)]

        # We do the mean and the variance
        mean = np.mean(generated_numbers)
        variance = np.var(generated_numbers)

        # Print the results
        print("\nQUESTION 1:")
        print(f"\tMean of generated random variables: {mean}")
        print(f"\tExpected mean for λ=75: {expected_mean}")
        print(f"\tVariance of generated random variables: {variance}")
        print(f"\tExpected variance for λ=75: {expected_variance}\n")

    def m_m_1_queue(self, avg_len: int, trans_rate: int, lambda_par: int, T: int) -> list:
        """
        Build your simulator for this queue and explain in words what you have done. Show your code in the report. In 
        particular, define your variables. Should there be a need, draw diagrams to show your program structure. Explain how you 
        compute the performance metrics. Type of queue M/M/ (infinite queue)
        @param avg_len: Is the average length of a packet in bits (L)
        @param trans_rate: Is the transmission rate of the output link in bits/sec (C)
        @return: None
        """
        # ! Declaration of variables for the m_m_1 queue
        num_arrival = 0
        num_departed = 0
        num_observers = 0
        transmission_times = []
        arrival_list = []
        departure_list = []
        observer_list = []
        event_list = []
        result_list = []

        # * Arrival

        # we generate the arrivals
        arrival_list = self.__generate_mm1_arr_obs(lambda_par, T)
        

        # * observers
        # Now we add the observers event
        observer_list = self.__generate_mm1_arr_obs(lambda_par*5, T)


        
        

        # * Departure
        # * generate packet lengths for each arrival
        length_packets = []
        # We create the packet size for each arrival
        length_arrival = len(arrival_list)
        length_packets = [self.__generate_exp_distribution(1/avg_len) for _ in range(length_arrival)]

        # * Calculate how much time takes to process all the packet

        for packet in length_packets:
            transmission_times.append(packet / trans_rate)

        # *We calculate the departure time
        queue_time = 0
        departure_time = 0
        # Loop that calculates how the time of departure for each packet
        for count, arrival_packet_time in enumerate(arrival_list):
            # If the queue is idle we just sum the arrival time +transmission [count] and we add it to the list
            if queue_time < arrival_packet_time:
                departure_time = arrival_packet_time+transmission_times[count]    
            # Else there is a queue, and we add the last package  departure time (count-1)  + the transmission[count]
            else:
                departure_time = departure_list[count-1] + transmission_times[count]
            departure_list.append(departure_time)
            # We update the queue time of the queue, with the las departure time
            queue_time = departure_time

        
        # * Order all packets by time with its type
        for arrival_time in arrival_list:
            result_list.append(["A", arrival_time])
        for departure_time in departure_list:
            result_list.append(["D", departure_time])
        for observer_time in observer_list:
            result_list.append(["O", observer_time])
        # We sort all the time events by time
        event_list = sorted(result_list, key=lambda x: x[1])



        # * We calculate E[n] and p_idle
        # Declaration of variables
        total_num_packs_queue = 0
        total_observer_idles = 0

        for _, event in enumerate(event_list):

            event_type = event[0]
            # Arrival
            if event_type == 'A':
                num_arrival += 1
            elif event_type == 'D':
                num_departed += 1
            else:
                num_observers += 1
                # We record the num of packets that are currently on the queue
                total_num_packs_queue += (num_arrival-num_departed)
                if num_arrival == num_departed:
                    total_observer_idles += 1

        return total_num_packs_queue/num_observers, total_observer_idles/num_observers
    
    def __generate_mm1_arr_obs(self, lambda_par,T):
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
            arrival_timestamp = self.__generate_exp_distribution(lambda_par)+simulation_time
            #We add to the aux list
            aux_list.append(arrival_timestamp)
            #We update the simulation time with the new arrival
            simulation_time = arrival_timestamp
        #We return the aux list
        return aux_list

    def __generate_exp_distribution(self, lambda_param: int) -> list:
        """
        This method is in charge of generating exponential random variables
        @param lambda_param: An integer that contains the average number of packets arrived
        @param size: An integer that defines how many numbers we generate
        @return list: We return a list with the numbers generated
        """
        expected_mean = 1/lambda_param
        return -expected_mean*math.log(1-random.random())

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
        #We declare variables 
        num_elem_queue = 0
        n_arrivals = 0
        n_observers = 0
        n_departures = 0

        total_packs_queue = 0
        lost_packets = 0

        last_departure = 0
        departure_list = []

        

        # Generating arrivals
        list_arrivals = []
        list_arrivals = self.__generate_mm1_arr_obs(lambda_par, T)

        #Generating the observers
        list_observers = []
        list_observers  = self.__generate_mm1_arr_obs(lambda_par*5, T)
        
        #We add event on event_list  where in pos 0 we define the type ("A"=arrival, "O"=observers)
        #in pos 1 we have the time_stamp of the event
        list_events = []
        for e in list_arrivals:
            list_events.append(["A",e])

        for e in list_observers:
            list_events.append(["O",e])
        
        #we sort the evnt list by event arrival time
        list_events = sorted(list_events, key=lambda x: x[1])
        
        

        #Loop where we will calculate the departure of the arrivals 
        #calculate the observers stadistics
        i = 0
        while i < len(list_events) or departure_list != []:
            if(i == len(list_events)):
                # If list_events has finished but we still have departures
                for x in range(len(departure_list)):
                    departure_list.pop(0)
                    n_departures += 1
                    num_elem_queue -= 1
            else:
                # Assign current event
                if(departure_list != []):
                    # If there are departures
                    if(departure_list[0][1] < list_events[i][1]):
                        # Check if it goes an observer, arrival or departure
                        event = departure_list.pop(0)
                    else:
                        event = list_events[i]
                else:
                    event = list_events[i]

                if(event[0] == "A"):
                    # ARRIVAL
                    if num_elem_queue < K_num:
                        # QUEUE NOT FULL
                        n_arrivals += 1
                        # Generate service time
                        arrival_time = event[1]
                        length_packet = self.__generate_exp_distribution(1/avg_len)
                        service_time = length_packet/trans_rate
                        
                        if num_elem_queue == 0:
                            # QUEUE EMPTY
                            departure_time = arrival_time + service_time
                        elif num_elem_queue < K_num:
                            # QUEUE WITH ELEMENTS
                            departure_time = last_departure + service_time
                        
                        # Adds the new departure time
                        departure_list.append(["D", departure_time])
                        # Reset the last departure time
                        last_departure = departure_time
                        num_elem_queue += 1
                    else:
                        # QUEUE NOT FULL
                        lost_packets += 1
                    i+= 1
                        

                elif(event[0] == "O"):
                    # OBSERVERS
                    n_observers += 1
                    total_packs_queue += (n_arrivals - n_departures)
                    i+= 1

                elif(event[0] == "D"):
                    # DEPARTURE
                    n_departures += 1
                    num_elem_queue -= 1

        return total_packs_queue/n_observers, lost_packets/n_arrivals


    
    #Generate graphs
    
    def generate_point(self, i, avg_len, trans_rate, lambda_par, T):
        # Calculate data point for a specific 'i'
        list_m_m_1 = self.m_m_1_queue(avg_len, trans_rate, lambda_par, T)
        # If we want E[n] then type_info is 0 if is p_idle then type_info is 1
        return [i, list_m_m_1]

    def create_graph_for_m_m_1_queue(self, avg_len, trans_rate, T):
        step = 0.1
        start = 0.25
        end = 1.05
        # Graph for E[N]
        result=[]
        print("Generating points for graph mm1 queue:")
        #cores=4
        with Pool() as pool:
            input_data = [(i, avg_len, trans_rate, trans_rate * i / avg_len, T)
                        for i in np.arange(start, end, step)]
            pool_list = pool.starmap(self.generate_point, input_data)
            print(pool_list)
        print("\nFinished generating points for graph mm1 queue.\n")
        result.append(pool_list)


        # We save the points
        #pos 0 is for E[n] and pos 1 is for p_idle
        x = [point[0] for point in result[0]]
        y = [[point[1][0] for point in result[0]],
        [point[1][1] for point in result[0]]]

        #We create the graph
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        graph_list=[ax1,ax2]
        text=[['Average number in system E[n]',"average #packets as a function of p"],
              ["Average number in system p_idle","The proportion of time the system is idle as a function of p"]]
        for i,graph in enumerate(graph_list):
            # We initialize the graph
            graph.scatter(x, y[i],color="red",marker='x')
            graph.plot(x, y[i],label="K is infinite")
            graph.set_xlabel('Traffic intensity p')
            graph.set_ylabel(text[i][0])
            graph.set_title(text[i][1])
            graph.legend()
        #We save it
        # Save the figure as an image in the "graphs" folder
        script_directory = os.path.dirname(__file__)

        # Save the figure as an image named "exercise_3.png" in the same folder as the script
        image_path = os.path.join(script_directory, 'exercise_3.png')
        plt.savefig(image_path)
        plt.close()

    def create_graph_for_m_m_1_k_queue(self, avg_len, trans_rate, T):
        #Define iteration variables
        step = 0.1
        start = 0.5
        end = 1.6

        #list for the different K
        K_list=[10,25,50]

        #where we store the y results
        y=[]
        for _,k in enumerate(K_list):
            print("Generating points for graph mm1k queue for k= %i\n"%(k))
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
            #pos 0 is for E[n]  ,pos 1  is p_loss
            print(result)
            y.append([[point[1][0] for point in result[0]],[point[1][1]for point in result[0]] ])
            
            print("\nFinished generating points for graph mm1 queue for k= %i\n"%(k))

        # We save the x points ( theya re the same for every k)
        x = [point[0] for point in result[0]]
        

        #We create the graph
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        graph_list=[ax1,ax2]
        #variables for creating the graph
        text=[['Average number in system E[n]',"average #packets as a function of p"],
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
        plt.close()
        

    def generate_points2(self,i,avg_len,trans_rate,lambda_par,T,K):
        # Calculate data point for a specific 'i'
        list_m_m_1 = self.m_m_1_k_queue(avg_len, trans_rate, lambda_par, T,K)
        # If we want E[n] then type_info is 0 if is p_idle then type_info is 1
        return [i, list_m_m_1]

def check_T(avg_len, trans_rate, lambda_par, T):
    a = Lab1()
    T_counter = 1
    percentage = 0.05
    dif_count_E = 100
    dif_count_pidle = 100
    list_T = []
    gate = True
    while gate:
        E, pidle = a.m_m_1_queue(avg_len, trans_rate, lambda_par, T_counter*T)
        E2, pidle2 = a.m_m_1_queue(avg_len, trans_rate, lambda_par, (T_counter+1)*T)
        difference_E = abs(E-E2)
        difference_pidle = abs(pidle-pidle2)
        if(len(list_T) == 2):
            return min(list_T[0], list_T[1]) * T
        if(difference_E <= E*percentage and difference_pidle <= pidle*percentage):
            if dif_count_E > difference_E and dif_count_pidle > difference_pidle:
                list_T.append(T_counter+1)
                dif_count_E = difference_E
                dif_count_pidle = difference_pidle
            else:
                gate = False
        else:
            gate = False 
        T_counter += 1
    return T
    
def check_T2(avg_len, trans_rate, lambda_par, T, K):
    a = Lab1()
    T_counter = 1
    percentage = 0.05
    dif_count_E = 100
    dif_count_ploss = 100
    list_T = []
    gate = True
    while gate:
        E, ploss = a.m_m_1_k_queue(avg_len, trans_rate, lambda_par, T_counter*T, K)
        E2, ploss2 = a.m_m_1_k_queue(avg_len, trans_rate, lambda_par, (T_counter+1)*T, K)
        difference_E = abs(E-E2)
        difference_ploss = abs(ploss-ploss2)
        if(len(list_T) == 2):
            return min(list_T[0], list_T[1]) * T
        if(difference_E <= E*percentage and difference_ploss <= ploss*percentage):
            if dif_count_E > difference_E and dif_count_ploss > difference_ploss:
                list_T.append(T_counter+1)
                dif_count_E = difference_E
                dif_count_ploss = difference_ploss
            else:
                gate = False
        else:
            gate = False 
        T_counter += 1
    return T

if __name__ == "__main__":
    a = Lab1()
    lambda_par = 75
    trans_rate = 1_000_000
    avg_packet_length = 2_000
    T = 1_000

    # RUNNING THE LAB
    # QUESTION 1
    a.question1(lambda_par)

    # INFINITE QUEUE
    # QUESTION 2
    print("QUESTION 2:\n")
    X = check_T(avg_packet_length, trans_rate, lambda_par, T)
    print("\tThe Final T will be: " + str(X))

    # QUESTION 3
    print("QUESTION3:\n")
    print("The graph will be generated in exercise3.png\n")
    a.create_graph_for_m_m_1_queue(avg_packet_length,trans_rate,X)


    # QUESTION 4
    # For p=1.2
    print("QUESTION 4:\n")
    E, pidle = a.m_m_1_queue(avg_packet_length, trans_rate, trans_rate * 1.2 / avg_packet_length, 2*T)
    print("\tFor p = 1.2, the value of E[N] = "+ str(E) + " and the value of pidle =" + str(pidle))
    

    # FINITE QUEUE
    # QUESTION 5
    print("QUESTION 5:\n")
    trans_rate = 1_000_000
    avg_packet_length = 2_000
    T = 1000
    k = [10, 25, 50]
    X = check_T2(avg_packet_length, trans_rate, trans_rate * 0.5 / avg_packet_length, T, 10)
    print("\tThe Final T will be: " + str(X))
    
    #Question 6
    print ("QUESTION 6:\n")
    print("The graph will be generated in exercise_6.png \n")
    b = Lab1()
    b.create_graph_for_m_m_1_k_queue(avg_packet_length,trans_rate,X)

