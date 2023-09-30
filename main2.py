import random
import math

# Initialize variables
T = 1000  # Simulation time in seconds
λ = 10   # Arrival rate (packets per second)
μ = 500  # Service rate (packets per second)

# Data structures to store events (arrival, departure, observer)
event_list = []

# Initialize counters and statistics
Na = 0   # Number of arrivals
Nd = 0   # Number of departures
No = 0   # Number of observations
total_delay = 0.0  # Total delay time

# Define event types
ARRIVAL = 0
DEPARTURE = 1
OBSERVER = 2

def generate_exponential_random_variable(rate):
    return -math.log(1 - random.random()) / rate

def add_event(event_type, timestamp):
    event_list.append((event_type, timestamp))

def queue_is_empty():
    return Nd == Na

def serve_packet():
    global Nd, total_delay
    Nd += 1
    total_delay += simulation_time

def add_packet_to_queue():
    global Na
    Na += 1

def generate_departure_event():
    service_time = generate_exponential_random_variable(μ)
    departure_time = simulation_time + service_time
    add_event(DEPARTURE, departure_time)

def record_system_state():
    global No
    No += 1

# Main simulation loop
simulation_time = 0.0
while simulation_time < T:
    arrival_timestamp = generate_exponential_random_variable(λ)
    add_event(ARRIVAL, arrival_timestamp)
    simulation_time = arrival_timestamp

while event_list:
    event_list.sort(key=lambda x: x[1])  # Sort events by timestamp
    event_type, event_time = event_list.pop(0)
    
    if event_type == ARRIVAL:
        if queue_is_empty():
            serve_packet()
        else:
            add_packet_to_queue()
            generate_departure_event()
    elif event_type == DEPARTURE:
        serve_packet()
    elif event_type == OBSERVER:
        record_system_state()

# Calculate performance metrics
E[N] = total_delay / Nd
PIDLE = (T - sum([event[1] for event in event_list])) / T

# Print results
print(f"Average number of packets in the queue (E[N]): {E[N]}")
print(f"Proportion of time the server is idle (PIDLE): {PIDLE}")
