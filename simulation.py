import threading
import pygame
import sys

import generatevehicle
from controls import signals, noOfSignals, currentGreen, currentYellow, initialize
from generatevehicle import generateVehicles
from movement import simulation

class Main:
    # Coordinates of signal image, timer, and vehicle count
    signalCoods = [(510, 230), (815, 230), (815, 570), (510, 570)]
    signalTimerCoods = [(530, 210), (810, 210), (810, 550), (530, 550)]

    # set allowed vehicle types here
    allowedVehicleTypes = {'car': True, 'bus': True, 'truck': True, 'bike': True}

    pygame.init()

    i = 0
    for vehicleType in allowedVehicleTypes:
        if(allowedVehicleTypes[vehicleType]):
            generatevehicle.allowedVehicleTypesList.append(i)
        i += 1
    thread1 = threading.Thread(name="initialization", target=initialize, args=())    # initialization
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
    thread2 = threading.Thread(name="generateVehicles", target=generateVehicles, args=())    # Generating vehicles
    thread2.daemon = True
    thread2.start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.blit(background, (-260, -50))  # display background in simulation
        for i in range(0, noOfSignals):  # display signal and set timer according to current status: green, yello, or red
            if(i== currentGreen):
                if(currentYellow ==1):
                    signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                if(signals[i].red<=10):
                    signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])
        signalTexts = ["","","",""]

        # display signal timer
        for i in range(0, noOfSignals):
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i],signalTimerCoods[i])

        # display the vehicles
        for vehicle in simulation:  
            screen.blit(vehicle.image, [vehicle.x, vehicle.y])
            vehicle.move()
        pygame.display.update()


Main()