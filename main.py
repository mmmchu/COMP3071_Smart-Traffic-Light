import random
import time
import threading
import pygame
import sys

import config
from config import defaultGreen, defaultRed, defaultYellow, signals, noOfSignals
from vehicle import vehicles, defaultStop, simulation, Vehicle

# Coordinates of vehicles' start

vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

# Coordinates of signal image, timer, and vehicle count
signalCoods = [(510, 230), (815, 230), (815, 570), (510, 570)]
signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]

# set allowed vehicle types here
allowedVehicleTypes = {'car': True, 'bus': True, 'truck': True, 'bike': True}
allowedVehicleTypesList = []
randomGreenSignalTimer = True
randomGreenSignalTimerRange = [10, 20]

pygame.init()


class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""


# Initialization of signals with default values
def initialize():
    minTime = randomGreenSignalTimerRange[0]
    maxTime = randomGreenSignalTimerRange[1]
    if randomGreenSignalTimer:
        ts1 = TrafficSignal(0, defaultYellow, random.randint(minTime, maxTime))
        signals.append(ts1)
        ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green, defaultYellow, random.randint(minTime, maxTime))
        signals.append(ts2)
        ts3 = TrafficSignal(defaultRed, defaultYellow, random.randint(minTime, maxTime))
        signals.append(ts3)
        ts4 = TrafficSignal(defaultRed, defaultYellow, random.randint(minTime, maxTime))
        signals.append(ts4)
    else:
        ts1 = TrafficSignal(0, defaultYellow, defaultGreen[0])
        signals.append(ts1)
        ts2 = TrafficSignal(ts1.yellow + ts1.green, defaultYellow, defaultGreen[1])
        signals.append(ts2)
        ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[2])
        signals.append(ts3)
        ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[3])
        signals.append(ts4)
    repeat()


def repeat():
    while signals[config.currentGreen].green > 0:  # while the timer of current green signal is not zero
        updateValues()
        time.sleep(1)
    config.currentYellow = 1  # set yellow signal on
    # reset stop coordinates of lanes and vehicles
    for i in range(0, 3):
        for vehicle in vehicles[directionNumbers[config.currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[config.currentGreen]]
    while signals[config.currentGreen].yellow > 0:  # while the timer of current yellow signal is not zero
        updateValues()
        time.sleep(1)
    config.currentYellow = 0  # set yellow signal off

    # reset all signal times of current signal to default/random times
    if randomGreenSignalTimer:
        signals[config.currentGreen].green = random.randint(randomGreenSignalTimerRange[0],
                                                            randomGreenSignalTimerRange[1])
    else:
        signals[config.currentGreen].green = defaultGreen[config.currentGreen]
    signals[config.currentGreen].yellow = defaultYellow
    signals[config.currentGreen].red = defaultRed

    config.currentGreen = config.nextGreen  # set next signal as green signal
    config.nextGreen = (config.currentGreen + 1) % noOfSignals  # set next green signal
    signals[config.nextGreen].red = signals[config.currentGreen].yellow + signals[
        config.currentGreen].green  # set the red time of next to next signal as (yellow time + green time) of next
    # signal
    repeat()


# Update values of the signal timers after every second
def updateValues():
    for i in range(0, noOfSignals):
        if i == config.currentGreen:
            if config.currentYellow == 0:
                signals[i].green -= 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1


# Generating vehicles in the simulation
def generateVehicles():
    while True:
        vehicle_type = random.choice(allowedVehicleTypesList)
        lane_number = random.randint(1, 2)
        will_turn = 0
        if lane_number == 1:
            temp = random.randint(0, 99)
            if temp < 40:
                will_turn = 1
        elif lane_number == 2:
            temp = random.randint(0, 99)
            if temp < 40:
                will_turn = 1
        temp = random.randint(0, 99)
        direction_number = 0
        dist = [25, 50, 75, 100]
        if temp < dist[0]:
            direction_number = 0
        elif temp < dist[1]:
            direction_number = 1
        elif temp < dist[2]:
            direction_number = 2
        elif temp < dist[3]:
            direction_number = 3
        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number],
                will_turn)
        time.sleep(1)


class Main:
    global allowedVehicleTypesList
    i = 0
    for vehicleType in allowedVehicleTypes:
        if allowedVehicleTypes[vehicleType]:
            allowedVehicleTypesList.append(i)
        i += 1
    thread1 = threading.Thread(name="initialization", target=initialize, args=())  # initialization
    thread1.daemon = True
    thread1.start()

    # Colours
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Screensize
    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)

    # Setting background image i.e. image of intersection
    background = pygame.image.load('images/intersection.png')

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("SIMULATION")

    # Loading signal images and font
    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)
    thread2 = threading.Thread(name="generateVehicles", target=generateVehicles, args=())  # Generating vehicles
    thread2.daemon = True
    thread2.start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.blit(background, (-260, -50))  # display background in simulation
        for i in range(0,
                       noOfSignals):  # display signal and set timer according to current status: green, yello, or red
            if i == config.currentGreen:
                if config.currentYellow == 1:
                    signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                if signals[i].red <= 10:
                    signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])
        signalTexts = ["", "", "", ""]

        # display signal timer
        for i in range(0, noOfSignals):
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i], signalTimerCoods[i])

        # display the vehicles
        for vehicle in simulation:
            screen.blit(vehicle.image, [vehicle.x, vehicle.y])
            vehicle.move()
        pygame.display.update()


Main()
