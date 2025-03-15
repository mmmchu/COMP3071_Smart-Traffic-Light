import pygame

import config

speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'bike': 2.5}  # average speeds of vehicles
x = {'right': [1, 0, 1], 'down': [615, 635, 576], 'left': [1400, 1400, 1400], 'up': [790, 750, 709]}
y = {'right': [446, 517, 476], 'down': [1, 0, 1], 'left': [323, 358, 399], 'up': [801, 803, 802]}
vehicles = {'right': {0: [], 1: [], 2: [], 'crossed': 0}, 'down': {0: [], 1: [], 2: [], 'crossed': 0},
            'left': {0: [], 1: [], 2: [], 'crossed': 0}, 'up': {0: [], 1: [], 2: [], 'crossed': 0}}
stopLines = {'right': 450, 'down': 230, 'left': 900, 'up': 635}
defaultStop = {'right': 440, 'down': 220, 'left': 910, 'up': 650}
stoppingGap = 25  # stopping gap
movingGap = 25  # moving gap
vehiclesTurned = {'right': {1: [], 2: []}, 'down': {1: [], 2: []}, 'left': {1: [], 2: []}, 'up': {1: [], 2: []}}
vehiclesNotTurned = {'right': {1: [], 2: []}, 'down': {1: [], 2: []}, 'left': {1: [], 2: []}, 'up': {1: [], 2: []}}
rotationAngle = 3
mid = {'right': {'x': 705, 'y': 445}, 'down': {'x': 695, 'y': 450}, 'left': {'x': 695, 'y': 425},
       'up': {'x': 695, 'y': 400}}
simulation = pygame.sprite.Group()


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        self.crossedIndex = 0
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.originalImage = pygame.image.load(path)
        self.image = pygame.image.load(path)

        if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index - 1].crossed == 0:
            if direction == 'right':
                self.stop = vehicles[direction][lane][self.index - 1].stop
            elif direction == 'left':
                self.stop = vehicles[direction][lane][self.index - 1].stop
            elif direction == 'down':
                self.stop = vehicles[direction][lane][self.index - 1].stop
            elif direction == 'up':
                self.stop = vehicles[direction][lane][self.index - 1].stop
        else:
            self.stop = defaultStop[direction]

        # Set new starting and stopping coordinate
        if direction == 'right':
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] -= temp
        elif direction == 'left':
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] += temp
        elif direction == 'down':
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] -= temp
        elif direction == 'up':
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        currentGreen = config.currentGreen  # Fetch updated values
        currentYellow = config.currentYellow

        if self.direction == 'right':
            if self.crossed == 0 and self.x + self.image.get_rect().width > stopLines[self.direction]:
                self.crossed = 1

                vehicles[self.direction]['crossed'] += 1
                if self.willTurn == 0:
                    vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesNotTurned[self.direction][self.lane]) - 1
            if self.willTurn == 1:
                if self.lane == 2:
                    if self.crossed == 0 or self.x + self.image.get_rect().width < mid[self.direction]['x'] + 10:

                        if ((self.x + self.image.get_rect().width <= self.stop or (
                                currentGreen == 0 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.x + self.image.get_rect().width < (
                                vehicles[self.direction][self.lane][self.index - 1].x - movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.x += self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x += 2.4
                            self.y -= 2.8
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or (self.y > (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].y +
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1].image.get_rect().height + movingGap))):
                                self.y -= self.speed
                elif self.lane == 1:
                    if self.crossed == 0 or self.x + self.image.get_rect().width < stopLines[self.direction] + 150:

                        if ((self.x + self.image.get_rect().width <= self.stop or (
                                currentGreen == 0 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.x + self.image.get_rect().width < (
                                vehicles[self.direction][self.lane][self.index - 1].x - movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.x += self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x += 2
                            self.y += 1.8
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or ((self.y + self.image.get_rect().height) < (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].y - movingGap))):
                                self.y += self.speed
            else:
                if self.crossed == 0:
                    if ((self.x + self.image.get_rect().width <= self.stop or (
                            currentGreen == 0 and currentYellow == 0)) and (
                            self.index == 0 or self.x + self.image.get_rect().width < (
                            vehicles[self.direction][self.lane][self.index - 1].x - movingGap))):
                        self.x += self.speed
                else:
                    if ((self.crossedIndex == 0) or (self.x + self.image.get_rect().width < (
                            vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].x - movingGap))):
                        self.x += self.speed
        elif self.direction == 'down':
            if self.crossed == 0 and self.y + self.image.get_rect().height > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if self.willTurn == 0:
                    vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesNotTurned[self.direction][self.lane]) - 1
            if self.willTurn == 1:
                if self.lane == 1:
                    if self.crossed == 0 or self.y + self.image.get_rect().height < mid[self.direction]['y'] + 50:

                        if ((self.y + self.image.get_rect().height <= self.stop or (
                                currentGreen == 1 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.y + self.image.get_rect().height < (
                                vehicles[self.direction][self.lane][self.index - 1].y - movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.y += self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x += 1.2
                            self.y += 1.8
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or ((self.x + self.image.get_rect().width) < (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].x - movingGap))):
                                self.x += self.speed
                elif self.lane == 2:
                    if self.crossed == 0 or self.y + self.image.get_rect().height < stopLines[self.direction] + 150:

                        if ((self.y + self.image.get_rect().height <= self.stop or (
                                currentGreen == 1 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.y + self.image.get_rect().height < (
                                vehicles[self.direction][self.lane][self.index - 1].y - movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.y += self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x -= 2.5
                            self.y += 2
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or (self.x > (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].x +
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1].image.get_rect().width + movingGap))):
                                self.x -= self.speed
            else:
                if self.crossed == 0:
                    if ((self.y + self.image.get_rect().height <= self.stop or (
                            currentGreen == 1 and currentYellow == 0)) and (
                            self.index == 0 or self.y + self.image.get_rect().height < (
                            vehicles[self.direction][self.lane][self.index - 1].y - movingGap))):
                        self.y += self.speed
                else:
                    if ((self.crossedIndex == 0) or (self.y + self.image.get_rect().height < (
                            vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].y - movingGap))):
                        self.y += self.speed
        elif self.direction == 'left':
            if self.crossed == 0 and self.x < stopLines[self.direction]:

                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if self.willTurn == 0:
                    vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesNotTurned[self.direction][self.lane]) - 1
            if self.willTurn == 1:
                if self.lane == 2:
                    if self.crossed == 0 or self.x > mid[self.direction]['x'] - 40:

                        if ((self.x >= self.stop or (
                                currentGreen == 2 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x +
                                                             vehicles[self.direction][self.lane][
                                                                 self.index - 1].image.get_rect().width + movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.x -= self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x -= 1
                            self.y += 1.2
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or ((self.y + self.image.get_rect().height) < (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].y - movingGap))):
                                self.y += self.speed
                elif self.lane == 1:
                    if self.crossed == 0 or self.x > stopLines[self.direction] - 130:
                        if ((self.x >= self.stop or (
                                currentGreen == 2 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index - 1].x +
                                                             vehicles[self.direction][self.lane][
                                                                 self.index - 1].image.get_rect().width + movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.x -= self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x -= 1.8
                            self.y -= 2.5
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or (self.y > (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].y +
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1].image.get_rect().height + movingGap))):
                                self.y -= self.speed
            else:
                if self.crossed == 0:
                    if ((self.x >= self.stop or (currentGreen == 2 and currentYellow == 0)) and (
                            self.index == 0 or self.x > (
                            vehicles[self.direction][self.lane][self.index - 1].x + vehicles[self.direction][self.lane]
                            [self.index - 1].image.get_rect().width + movingGap))):
                        self.x -= self.speed
                else:
                    if ((self.crossedIndex == 0) or (self.x > (
                            vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].x +
                            vehiclesNotTurned[self.direction][self.lane][
                                self.crossedIndex - 1].image.get_rect().width + movingGap))):
                        self.x -= self.speed
        elif self.direction == 'up':
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
                if self.willTurn == 0:
                    vehiclesNotTurned[self.direction][self.lane].append(self)
                    self.crossedIndex = len(vehiclesNotTurned[self.direction][self.lane]) - 1
            if self.willTurn == 1:
                if self.lane == 2:
                    # if self.crossed == 0 or self.y > stopLines[self.direction] - 60:
                    if self.crossed == 0 or self.y > mid[self.direction]['y'] - 20:

                        if ((self.y >= self.stop or (
                                currentGreen == 3 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y +
                                                             vehicles[self.direction][self.lane][
                                                                 self.index - 1].image.get_rect().height + movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.y -= self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x -= 2
                            self.y -= 1.2
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or (self.x > (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].x +
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1].image.get_rect().width + movingGap))):
                                self.x -= self.speed
                elif self.lane == 1:
                    # if self.crossed == 0 or self.y > mid[self.direction]['y']:
                    if self.crossed == 0 or self.y > stopLines[self.direction] - 100:
                        if ((self.y >= self.stop or (
                                currentGreen == 3 and currentYellow == 0) or self.crossed == 1) and (
                                self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index - 1].y +
                                                             vehicles[self.direction][self.lane][
                                                                 self.index - 1].image.get_rect().height + movingGap) or
                                vehicles[self.direction][self.lane][self.index - 1].turned == 1)):
                            self.y -= self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += rotationAngle
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x += 1
                            self.y -= 1
                            if self.rotateAngle == 90:
                                self.turned = 1
                                vehiclesTurned[self.direction][self.lane].append(self)
                                self.crossedIndex = len(vehiclesTurned[self.direction][self.lane]) - 1
                        else:
                            if (self.crossedIndex == 0 or (self.x < (
                                    vehiclesTurned[self.direction][self.lane][self.crossedIndex - 1].x -
                                    vehiclesTurned[self.direction][self.lane][
                                        self.crossedIndex - 1].image.get_rect().width - movingGap))):
                                self.x += self.speed
            else:
                if self.crossed == 0:
                    if ((self.y >= self.stop or (currentGreen == 3 and currentYellow == 0)) and (
                            self.index == 0 or self.y > (
                            vehicles[self.direction][self.lane][self.index - 1].y + vehicles[self.direction][self.lane]
                            [self.index - 1].image.get_rect().height + movingGap))):
                        self.y -= self.speed
                else:
                    if ((self.crossedIndex == 0) or (self.y > (
                            vehiclesNotTurned[self.direction][self.lane][self.crossedIndex - 1].y +
                            vehiclesNotTurned[self.direction][self.lane][
                                self.crossedIndex - 1].image.get_rect().height + movingGap))):
                        self.y -= self.speed

        return self.y, self.x