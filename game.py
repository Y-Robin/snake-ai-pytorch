import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN = (0, 255, 0)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 60

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.numBlockX = int(self.w/BLOCK_SIZE)
        self.numBlockY = int(self.h/BLOCK_SIZE)
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                        Point(self.head.x-BLOCK_SIZE, self.head.y),
                        Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    
    def getArray(self):
        Matrix = [0]*self.numBlockX *self.numBlockY
        lv = 3
        for i in self.snake:
            try:
                Matrix[int(i[0]/BLOCK_SIZE+self.numBlockX*i[1]/BLOCK_SIZE)] = lv
                if lv >1:
                    lv -= 1 
            except:
                s=1
        Matrix[int(self.food[0]/BLOCK_SIZE+self.numBlockX*self.food[1]/BLOCK_SIZE)] = 4
        return Matrix
    
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def distFood(self):
        xFood = self.food[0]
        yFood = self.food[1]
        distX = (self.head[0]-xFood)**2
        distY = (self.head[1]-yFood)**2
        return distX+distY
        
    def reachFood(self):
        MatrixTest = [[0 for x in range(self.numBlockX)] for y in range(self.numBlockY)] 
        for i in self.snake:
            MatrixTest[int(i[1]/BLOCK_SIZE)][int(i[0]/BLOCK_SIZE)] = 1
        MatrixTest[int(self.head[1]/BLOCK_SIZE)][int(self.head[0]/BLOCK_SIZE)]=1
        MatrixTest = self.RecFood(MatrixTest,int(self.head[0]/BLOCK_SIZE)+1,int(self.head[1]/BLOCK_SIZE))
        MatrixTest = self.RecFood(MatrixTest,int(self.head[0]/BLOCK_SIZE)-1,int(self.head[1]/BLOCK_SIZE))
        MatrixTest = self.RecFood(MatrixTest,int(self.head[0]/BLOCK_SIZE),int(self.head[1]/BLOCK_SIZE)+1)
        MatrixTest = self.RecFood(MatrixTest,int(self.head[0]/BLOCK_SIZE),int(self.head[1]/BLOCK_SIZE)-1)
        #print(MatrixTest)
        if MatrixTest[int(self.food[1]/BLOCK_SIZE)][int(self.food[0]/BLOCK_SIZE)] >= 1:
            
            return True
        else:
            return False
        
    def RecFood(self,MatrixTest,PosX,PosY):
        x = PosX
        y = PosY
        #print([PosX,PosY])
        if PosX<0 or PosX>=len(MatrixTest[0]) or PosY < 0 or PosY>=len(MatrixTest):
            #print("Out")
            #print([PosX,PosY])
            return MatrixTest
        if MatrixTest[PosY][PosX] > 0:
            return MatrixTest
        else:
            #print([PosX,PosY])
            MatrixTest[PosY][PosX] = 1
            MatrixTest = self.RecFood(MatrixTest,x+1,y)
            MatrixTest = self.RecFood(MatrixTest,x-1,y)
            MatrixTest = self.RecFood(MatrixTest,x,y+1)
            MatrixTest = self.RecFood(MatrixTest,x,y-1)
        return MatrixTest
        
        
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        distBefore = self.distFood()
        reachFoodOld = self.reachFood()
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0#1/10*self.score
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            #print(reward)
            return reward, game_over, self.score
        distAfter = self.distFood()
        reachFoodNew = self.reachFood()
        if distAfter < distBefore:
            reward = reward+1
        if (reachFoodOld == True) and not(reachFoodNew):
            reward = -10
            print(reward)
        if (reachFoodOld == False) and (reachFoodNew):
            reward = raward+2
            print(reward)
        
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = reward+10
            self._place_food()
        else:
            self.snake.pop()        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        #print(reward)
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)
        i = 0
        for pt in self.snake:
            if i == 0:
                pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
                i=i+1
            else:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]
        #print(action)
        #print(self.direction)
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir
        #print(self.direction)
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)