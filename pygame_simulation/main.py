import os
import random
import sys
import threading
import time

import matplotlib.pyplot as plt
import pygame

import config
from config import defaultRed, defaultYellow, signals, noOfSignals
from menuUI import show_menu  # UI for choosing spawn rate
from vehicle import vehicles, defaultStop, simulation, Vehicle

# Coordinates of signals and timers
signalCoods = [(510, 230), (815, 230), (815, 570), (510, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]

# Vehicle types
vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

# Allowed vehicle types
allowedVehicleTypes = {'car': True, 'bus': True, 'truck': True, 'bike': True}
allowedVehicleTypesList = [i for i, v in enumerate(allowedVehicleTypes) if allowedVehicleTypes[v]]
# Lists to store metrics for plotting
queue_lengths_log = []
waiting_times_log = []
flow_rates_log = []

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


def log_traffic_data():
    """Logs traffic data into lists for plotting."""
    queue_lengths = [sum(len(vehicles[directionNumbers[i]][j]) for j in range(3)) for i in range(4)]
    waiting_times = [signals[i].red for i in range(4)]
    flow_rates = [sum(1 for vehicle in simulation if vehicle.crossed and vehicle.direction_number == i) for i in
                  range(4)]

    queue_lengths_log.append(sum(queue_lengths) / 4)  # Average queue length
    waiting_times_log.append(sum(waiting_times) / 4)  # Average waiting time
    flow_rates_log.append(sum(flow_rates))  # Total throughput


def plot_graphs():
    """Plots graphs for Fixed-Time Traffic Signals and saves them to a folder based on spawn rate."""

    # Create a folder for saving graphs if it doesn't exist
    save_folder = "simulation_graphs"
    os.makedirs(save_folder, exist_ok=True)

    time_steps = list(range(len(queue_lengths_log)))

    plt.figure(figsize=(12, 6))

    # Plot Average Waiting Time
    plt.subplot(1, 3, 1)
    plt.plot(time_steps, waiting_times_log, label="Fixed-Time Signal", color='b')
    plt.xlabel("Time Steps")
    plt.ylabel("Avg. Wait Time (s)")
    plt.title("Average Wait Time at Intersections")
    plt.legend()

    # Plot Traffic Flow Efficiency
    plt.subplot(1, 3, 2)
    plt.plot(time_steps, flow_rates_log, label="Fixed-Time Signal", color='g')
    plt.xlabel("Time Steps")
    plt.ylabel("Total Vehicle Throughput")
    plt.title("Traffic Flow Efficiency")
    plt.legend()

    # Plot Emissions Reduction (Idle Time)
    plt.subplot(1, 3, 3)
    plt.plot(time_steps, queue_lengths_log, label="Fixed-Time Signal", color='r')
    plt.xlabel("Time Steps")
    plt.ylabel("Queue Length")
    plt.title("Reduction in Emissions (Idle Time)")
    plt.legend()

    plt.tight_layout()

    # Save the plot using the chosen spawn rate
    if chosen_spawn_rate is not None:
        save_filename = f"traffic_simulation_{chosen_spawn_rate}.png"
    else:
        save_filename = "traffic_simulation_unknown.png"

    save_path = os.path.join(save_folder, save_filename)
    plt.savefig(save_path)

    print(f"Graph saved successfully at: {save_path}")

    # Show the plot
    plt.show()


class Main:
    global allowedVehicleTypesList
    global chosen_spawn_rate

    chosen_spawn_rate = show_menu()  # Get spawn rate from menu

    if chosen_spawn_rate is None:
        print("Error: No spawn rate selected")
        sys.exit(1)

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

    start_time = time.time()  # Record simulation start time
    simulation_duration = 60  # Run for 60 seconds

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Check if 60 seconds have passed
        if time.time() - start_time >= simulation_duration:
            running = False

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

        # Log traffic data for analysis
        log_traffic_data()

        pygame.display.update()

    pygame.quit()  # Quit Pygame before plotting graphs
    plot_graphs()  # Generate graphs after simulation ends
