import random
import sys
import threading
import time
import csv
import pygame
import os
import config
from config import defaultRed, defaultYellow, signals, noOfSignals
from vehicle import vehicles, defaultStop, simulation, Vehicle
from menuUI import show_menu  # UI for choosing spawn rate
import pandas as pd  # Import pandas to count rows

# Coordinates of signals and timers
signalCoods = [(510, 230), (815, 230), (815, 570), (510, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]

# Vehicle types
vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

# Allowed vehicle types
allowedVehicleTypes = {'car': True, 'bus': True, 'truck': True, 'bike': True}
allowedVehicleTypesList = [i for i, v in enumerate(allowedVehicleTypes) if allowedVehicleTypes[v]]

# Pygame setup
pygame.init()

# Spawn rate mapping
spawn_rates = {"Low": 4, "Medium": 2, "High": 1}

chosen_spawn_rate = None


class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""


# Initialize traffic signals
def initialize():
    minTime, maxTime = 10, 20
    for _ in range(4):
        green_time = random.randint(minTime, maxTime)
        signals.append(TrafficSignal(defaultRed, defaultYellow, green_time))
    repeat()


# Signal timing logic
def repeat():
    while signals[config.currentGreen].green > 0:
        updateValues()
        time.sleep(1)

    config.currentYellow = 1
    for i in range(3):
        for vehicle in vehicles[directionNumbers[config.currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[config.currentGreen]]

    while signals[config.currentGreen].yellow > 0:
        updateValues()
        time.sleep(1)

    config.currentYellow = 0
    signals[config.currentGreen].green = random.randint(10, 20)
    signals[config.currentGreen].yellow = defaultYellow
    signals[config.currentGreen].red = defaultRed

    config.currentGreen = config.nextGreen
    config.nextGreen = (config.currentGreen + 1) % noOfSignals
    signals[config.nextGreen].red = signals[config.currentGreen].yellow + signals[config.currentGreen].green
    repeat()


def updateValues():
    for i in range(noOfSignals):
        if i == config.currentGreen:
            if config.currentYellow == 0:
                signals[i].green -= 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1


# Generate vehicles with selected spawn rate
def generateVehicles():
    if chosen_spawn_rate is None:
        print("Waiting for spawn rate selection...")
        return  # Exit if spawn rate is not selected

    while True:
        vehicle_type = random.choice(allowedVehicleTypesList)
        lane_number = random.randint(1, 2)
        will_turn = random.choice([0, 1])
        direction_number = random.randint(0, 3)

        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number],
                will_turn)
        time.sleep(chosen_spawn_rate)  # Use selected spawn rate


# Collect Traffic Data for DQN
def count_vehicles():
    """Counts the number of vehicles at the current green signal."""
    return sum(len(vehicles[directionNumbers[config.currentGreen]][i]) for i in range(3))


def calculate_waiting_time():
    """Gets waiting time from the signal's red duration."""
    return signals[config.currentGreen].red


def calculate_flow_rate():
    """Counts the number of vehicles that have crossed the intersection."""
    return sum(1 for vehicle in simulation if vehicle.crossed)


def initialize_csv():
    """Creates a CSV file with headers for all four signals, excluding unnecessary data."""
    folder_name = "traffic_logs"
    os.makedirs(folder_name, exist_ok=True)
    filename = os.path.join(folder_name, f"traffic_data_{chosen_spawn_rate}.csv")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Queue_Length_0", "Queue_Length_1", "Queue_Length_2", "Queue_Length_3",
            "Waiting_Time_0", "Waiting_Time_1", "Waiting_Time_2", "Waiting_Time_3",
            "Flow_Rate_0", "Flow_Rate_1", "Flow_Rate_2", "Flow_Rate_3",
            "Spawn_Rate"
        ])

    print(f"Initialized CSV: {filename}")


def log_traffic_data():
    """Logs traffic data for all signals into a CSV file, excluding unnecessary data."""
    folder_name = "traffic_logs"
    os.makedirs(folder_name, exist_ok=True)
    filename = os.path.join(folder_name, f"traffic_data_{chosen_spawn_rate}.csv")

    # Collect data for each signal
    queue_lengths = [sum(len(vehicles[directionNumbers[i]][j]) for j in range(3)) for i in range(4)]
    waiting_times = [signals[i].red for i in range(4)]
    flow_rates = [sum(1 for vehicle in simulation if vehicle.crossed and vehicle.direction_number == i) for i in range(4)]

    # Write the data to the CSV file
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            *queue_lengths,  # Queue lengths
            *waiting_times,  # Waiting times
            *flow_rates,  # Flow rates
            chosen_spawn_rate  # Spawn rate
        ])

    # Count rows and print
    df = pd.read_csv(filename)
    print(f"Logged data - Total Rows: {len(df)}")

# Main simulation
class Main:
    global allowedVehicleTypesList
    global chosen_spawn_rate

    chosen_spawn_rate = show_menu()  # Get spawn rate from menu

    if chosen_spawn_rate is None:
        print("Error: No spawn rate selected")
        sys.exit(1)

    initialize_csv()  # Start data logging

    # Start traffic signal control thread
    thread1 = threading.Thread(target=initialize)
    thread1.daemon = True
    thread1.start()

    # Setup Pygame
    screenWidth, screenHeight = 1400, 800
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    pygame.display.set_caption("SIMULATION")

    background = pygame.image.load('images/intersection.png')
    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)

    # Start vehicle generation thread
    thread2 = threading.Thread(target=generateVehicles)
    thread2.daemon = True
    thread2.start()

    time_step = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.blit(background, (-260, -50))

        for i in range(noOfSignals):
            if i == config.currentGreen:
                if config.currentYellow == 1:
                    signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                signals[i].signalText = signals[i].red if signals[i].red <= 10 else "---"
                screen.blit(redSignal, signalCoods[i])

        for i in range(noOfSignals):
            signalText = font.render(str(signals[i].signalText), True, (255, 255, 255), (0, 0, 0))
            screen.blit(signalText, signalTimerCoods[i])

        for vehicle in simulation:
            screen.blit(vehicle.image, [vehicle.x, vehicle.y])
            vehicle.move()

        # Log traffic data
        log_traffic_data()

        pygame.display.update()


Main()