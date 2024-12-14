#  filename: main.py
#  author: Yonah Aviv
#  date created: 2020-11-10 6:21 p.m.
#  last modified: 2020-11-18
#  Pydash: Similar to Geometry Dash, a rhythm based platform game, but programmed using the pygame library in Python


"""CONTROLS
Anywhere -> ESC: exit
Main menu -> 1: go to previous level. 2: go to next level. SPACE: start game.
Game -> SPACE/UP: jump, and activate orb
    orb: jump in midair when activated
If you die or beat the level, press SPACE to restart or go to the next level

"""

import csv
import os
import random
import numpy as np

# import the pygame module
import pygame
from fontTools.merge.util import current_time

# will make it easier to use pygame functions
from pygame.math import Vector2
from pygame.draw import rect

# initializes the pygame module
pygame.init()

# creates a screen variable of size 800 x 600
screen = pygame.display.set_mode([800, 600])

# controls the main game while loop
done = False

# controls whether or not to start the game from the main menu
start = False

# sets the frame rate of the program
clock = pygame.time.Clock()
jump_memory = []
jump_time = 0

"""
CONSTANTS
"""
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

"""lambda functions are anonymous functions that you can assign to a variable.
e.g.
1. x = lambda x: x + 2  # takes a parameter x and adds 2 to it
2. print(x(4))
>>6
"""
color = lambda: tuple([random.randint(0, 255) for i in range(3)])  # lambda function for random color, not a constant.
GRAVITY = Vector2(0, 0.86)  # Vector2 is a pygame


"""
Reinforcement Learning
"""

import numpy as np
import random
import time



class GameTimer:
    def __init__(self):
        self.start_time = None

    def start(self):
        """Start or reset the timer."""
        self.start_time = time.time()

    def get_elapsed_time(self):
        """Get the elapsed time since the timer started."""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time




# GLOBAL VARIABLES
global_timer = GameTimer()
death_counter = 0




class RLAgent:
    def __init__(self, state_size, action_size, timer, jump_memory):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))  # Q-table for state-action values
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.gamma = 0.95  # Discount factor
        self.timer = timer
        self.last_death_time = None  # Tracks the time of the last death
        self.experiment_offset = 0.1  # Fraction of a second before death to try jumping
        self.success_updated = False  # Flag to track if success has been updated
        self.jump_time = jump_time


    def should_jump(self):
        """Check if it's time to jump based on the last death time."""
        current_time = self.timer.get_elapsed_time()

        if self.last_death_time is not None:
            if (current_time + self.experiment_offset) > self.last_death_time:
                print(f"Jump time: {self.last_death_time}")
                return True
        return False

    def update_jump_memory(self, success):
        """Update memory with successful jumps or adjust experiment offset."""
        current_time = self.timer.get_elapsed_time()

        if success:
            # If the jump was successful, save it in memory and reset experimentation
             # Get the largest timestamp in memory
            if len(jump_memory) == 0:
                print(f"Adding successful jump timestamp: {current_time}")
                jump_memory.append(current_time)
                print(f"Size of Jump Memory: {len(jump_memory)}")
                self.success_updated = False

            else:
                print("Entered else")
                max_timestamp = max(jump_memory)
                # Only add the new timestamp if it's greater than the largest one
                print(current_time)
                print(max_timestamp)
                if current_time > max_timestamp:
                    print(f"Adding successful jump timestamp: {current_time}")
                    jump_memory.append(current_time)
                    self.success_updated = False
            self.last_death_time = None  # Reset death tracking after success
            self.experiment_offset = 0.1  # Reset experiment offset to default
        else:
            # If the jump failed, set the last death time and adjust offset for next attempt
            print(f"Player died at {current_time}. Adjusting next jump timing.")
            self.last_death_time = current_time

    def check_and_update_for_success(self):
        """Check if current time is greater than last death time and update success."""
        current_time = self.timer.get_elapsed_time()


        if self.last_death_time is None:
            return
        elif self.last_death_time is not None and current_time > self.last_death_time + 0.5:
            # Update memory as success once when passing last death time
            self.update_jump_memory(success=True)
            self.success_updated = True  # Set the flag to prevent further updates





"""
Main player class
"""


class Player(pygame.sprite.Sprite):
    """Class for player. Holds update method, win and die variables, collisions and more."""
    win: bool
    died: bool

    def __init__(self, image, platforms, pos, *groups):
        """
        :param image: block face avatar
        :param platforms: obstacles such as coins, blocks, spikes, and orbs
        :param pos: starting position
        :param groups: takes any number of sprite groups.
        """
        super().__init__(*groups)
        self.onGround = False  # player on ground?
        self.platforms = platforms  # obstacles but create a class variable for it
        self.died = False  # player died?
        self.win = False  # player beat level?

        self.image = pygame.transform.smoothscale(image, (32, 32))
        self.rect = self.image.get_rect(center=pos)  # get rect gets a Rect object from the image
        self.jump_amount = 11  # jump strength
        self.particles = []  # player trail
        self.isjump = False  # is the player jumping?
        self.vel = Vector2(0, 0)  # velocity starts at zero
        self.jump_memory = jump_memory
        self.agent = RLAgent(32, 32, global_timer, self.jump_memory)

    def draw_particle_trail(self, x, y, color=(255, 255, 255)):
        """draws a trail of particle-rects in a line at random positions behind the player"""

        self.particles.append(
                [[x - 5, y - 8], [random.randint(0, 25) / 10 - 1, random.choice([0, 0])],
                 random.randint(5, 8)])

        for particle in self.particles:
            particle[0][0] += particle[1][0]
            particle[0][1] += particle[1][1]
            particle[2] -= 0.5
            particle[1][0] -= 0.4
            rect(alpha_surf, color,
                 ([int(particle[0][0]), int(particle[0][1])], [int(particle[2]) for i in range(2)]))
            if particle[2] <= 0:
                self.particles.remove(particle)

    def collide(self, yvel, platforms):
        global coins

        for p in platforms:
            if pygame.sprite.collide_rect(self, p):
                """pygame sprite builtin collision method,
                sees if player is colliding with any obstacles"""
                if isinstance(p, Orb) and (keys[pygame.K_UP] or keys[pygame.K_SPACE]):
                    pygame.draw.circle(alpha_surf, (255, 255, 0), p.rect.center, 18)
                    screen.blit(pygame.image.load("images/editor-0.9s-47px.gif"), p.rect.center)
                    self.jump_amount = 12  # gives a little boost when hit orb
                    self.jump()
                    self.jump_amount = 11  # return jump_amount to normal

                if isinstance(p, End):
                    self.win = True

                if isinstance(p, Spike):
                    self.died = True  # die on spike

                if isinstance(p, Coin):
                    # keeps track of all coins throughout the whole game(total of 6 is possible)
                    coins += 1

                    # erases a coin
                    p.rect.x = 0
                    p.rect.y = 0

                if isinstance(p, Platform):  # these are the blocks (may be confusing due to self.platforms)

                    if yvel > 0:
                        """if player is going down(yvel is +)"""
                        self.rect.bottom = p.rect.top  # dont let the player go through the ground
                        self.vel.y = 0  # rest y velocity because player is on ground

                        # set self.onGround to true because player collided with the ground
                        self.onGround = True

                        # reset jump
                        self.isjump = False
                    elif yvel < 0:
                        """if yvel is (-),player collided while jumping"""
                        self.rect.top = p.rect.bottom  # player top is set the bottom of block like it hits it head
                    else:
                        """otherwise, if player collides with a block, he/she dies."""
                        self.vel.x = 0
                        self.rect.right = p.rect.left  # dont let player go through walls
                        self.died = True

    def jump(self):
        self.vel.y = -self.jump_amount  # players vertical velocity is negative so ^

    def update(self):
        """update player"""
        if self.isjump:
            if self.onGround:
                """if player wants to jump and player is on the ground: only then is jump allowed"""
                self.jump()

        if not self.onGround:  # only accelerate with gravity if in the air
            self.vel += GRAVITY  # Gravity falls

            # max falling speed
            if self.vel.y > 100: self.vel.y = 100

        # do x-axis collisions
        self.collide(0, self.platforms)

        # increment in y direction
        self.rect.top += self.vel.y

        # assuming player in the air, and if not it will be set to inversed after collide
        self.onGround = False

        # do y-axis collisions
        self.collide(self.vel.y, self.platforms)

        # check if we won or if player won
        eval_outcome(self.win, self.died, self.agent)


"""
Obstacle classes
"""


# Parent class
class Draw(pygame.sprite.Sprite):
    """parent class to all obstacle classes; Sprite class"""

    def __init__(self, image, pos, *groups):
        super().__init__(*groups)
        self.image = image
        self.rect = self.image.get_rect(topleft=pos)









#  ====================================================================================================================#
#  classes of all obstacles. this may seem repetitive but it is useful(to my knowledge)
#  ====================================================================================================================#
# children
class Platform(Draw):
    """block"""

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


class Spike(Draw):
    """spike"""

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


class Coin(Draw):
    """coin. get 6 and you win the game"""

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


class Orb(Draw):
    """orb. click space or up arrow while on it to jump in midair"""

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


class Trick(Draw):
    """block, but its a trick because you can go through it"""

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


class End(Draw):
    "place this at the end of the level"

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


"""
Functions
"""


def init_level(map):
    """this is similar to 2d lists. it goes through a list of lists, and creates instances of certain obstacles
    depending on the item in the list"""
    x = 0
    y = 0

    for row in map:
        for col in row:

            if col == "0":
                Platform(block, (x, y), elements)

            if col == "Coin":
                Coin(coin, (x, y), elements)

            if col == "Spike":
                Spike(spike, (x, y), elements)
            if col == "Orb":
                orbs.append([x, y])

                Orb(orb, (x, y), elements)

            if col == "T":
                Trick(trick, (x, y), elements)

            if col == "End":
                End(avatar, (x, y), elements)
            x += 32
        y += 32
        x = 0


def blitRotate(surf, image, pos, originpos: tuple, angle: float):
    """
    rotate the player
    :param surf: Surface
    :param image: image to rotate
    :param pos: position of image
    :param originpos: x, y of the origin to rotate about
    :param angle: angle to rotate
    """
    # calcaulate the axis aligned bounding box of the rotated image
    w, h = image.get_size()
    box = [Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    box_rotate = [p.rotate(angle) for p in box]

    # make sure the player does not overlap, uses a few lambda functions(new things that we did not learn about number1)
    min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])
    # calculate the translation of the pivot
    pivot = Vector2(originpos[0], -originpos[1])
    pivot_rotate = pivot.rotate(angle)
    pivot_move = pivot_rotate - pivot

    # calculate the upper left origin of the rotated image
    origin = (pos[0] - originpos[0] + min_box[0] - pivot_move[0], pos[1] - originpos[1] - max_box[1] + pivot_move[1])

    # get a rotated image
    rotated_image = pygame.transform.rotozoom(image, angle, 1)

    # rotate and blit the image
    surf.blit(rotated_image, origin)


def won_screen():
    """show this screen when beating a level"""
    global attempts, level, fill
    attempts = 0
    player_sprite.clear(player.image, screen)
    screen.fill(pygame.Color("yellow"))
    txt_win1 = txt_win2 = "Nothing"
    if level == 1:
        if coins == 6:
            txt_win1 = f"Coin{coins}/6! "
            txt_win2 = "the game, Congratulations"
    else:
        txt_win1 = f"level{level}"
        txt_win2 = f"Coins: {coins}/6. "
    txt_win = f"{txt_win1} You beat {txt_win2}! Press SPACE to restart, or ESC to exit"

    won_game = font.render(txt_win, True, BLUE)

    screen.blit(won_game, (200, 300))
    level += 1

    wait_for_key()
    reset()



def eval_outcome(won: bool, died: bool, agent_input: RLAgent):
    current_time = agent_input.timer.get_elapsed_time()

    """Handle win or death and reset timer."""
    if won:
        won_screen()
    if died:
        reset()
        player.agent.last_death_time = current_time
        agent_input.update_jump_memory(success=False)
        print("Player Died")
        global_timer.start()


def block_map(level_num):
    """
    :type level_num: rect(screen, BLACK, (0, 0, 32, 32))
    open a csv file that contains the right level map
    """
    lvl = []
    with open(level_num, newline='') as csvfile:
        trash = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in trash:
            lvl.append(row)
    return lvl


def start_screen():
    """main menu. option to switch level, and controls guide, and game overview."""
    global level
    if not start:
        screen.fill(BLACK)
        if pygame.key.get_pressed()[pygame.K_1]:
            level = 0
        if pygame.key.get_pressed()[pygame.K_2]:
            level = 1

        welcome = font.render(f"Welcome to Pydash. choose level({level + 1}) by keypad", True, WHITE)

        controls = font.render("Controls: jump: Space/Up exit: Esc", True, GREEN)

        screen.blits([[welcome, (100, 100)], [controls, (100, 400)], [tip, (100, 500)]])

        level_memo = font.render(f"Level {level + 1}.", True, (255, 255, 0))
        screen.blit(level_memo, (100, 200))


def reset():
    """Resets the sprite groups, music, etc., for death and new level."""
    global player, elements, player_sprite, level
    if level == 1:
        pygame.mixer.music.load(os.path.join("music", "castle-town.mp3"))
        pygame.mixer_music.play()
    player_sprite = pygame.sprite.Group()
    elements = pygame.sprite.Group()
    player = Player(avatar, elements, (150, 150), player_sprite)
    init_level(block_map(level_num=levels[level]))




def move_map():
    """moves obstacles along the screen"""
    for sprite in elements:
        sprite.rect.x -= CameraX


def draw_stats(surf, money=0):
    """
    draws progress bar for level, number of attempts, displays coins collected, and progressively changes progress bar
    colors
    """
    global fill
    progress_colors = [pygame.Color("red"), pygame.Color("orange"), pygame.Color("yellow"), pygame.Color("lightgreen"),
                       pygame.Color("green")]

    tries = font.render(f" Attempt {str(attempts)}", True, WHITE)
    BAR_LENGTH = 600
    BAR_HEIGHT = 10
    for i in range(1, money):
        screen.blit(coin, (BAR_LENGTH, 25))
    fill += 0.5
    outline_rect = pygame.Rect(0, 0, BAR_LENGTH, BAR_HEIGHT)
    fill_rect = pygame.Rect(0, 0, fill, BAR_HEIGHT)
    col = progress_colors[int(fill / 100)]
    rect(surf, col, fill_rect, 0, 4)
    rect(surf, WHITE, outline_rect, 3, 4)
    screen.blit(tries, (BAR_LENGTH, 0))


def wait_for_key():
    """separate game loop for waiting for a key press while still running game loop
    """
    global level, start
    waiting = True
    while waiting:
        clock.tick(60)
        pygame.display.flip()

        if not start:
            start_screen()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    start = True
                    waiting = False
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()


def coin_count(coins):
    """counts coins"""
    if coins >= 3:
        coins = 3
    coins += 1
    return coins


def resize(img, size=(32, 32)):
    """resize images
    :param img: image to resize
    :type img: im not sure, probably an object
    :param size: default is 32 because that is the tile size
    :type size: tuple
    :return: resized img

    :rtype:object?
    """
    resized = pygame.transform.smoothscale(img, size)
    return resized

def display_timer(screen, font, timer):
    """Display the elapsed time on the screen."""
    elapsed_time = timer.get_elapsed_time()
    timer_text = font.render(f"Time: {elapsed_time:.2f} seconds", True, (255, 255, 255))  # White color
    screen.blit(timer_text, (10, 10))  # Display at top-left corner of screen


"""
Global variables
"""
font = pygame.font.SysFont("lucidaconsole", 20)

# square block face is main character the icon of the window is the block face
avatar = pygame.image.load(os.path.join("images", "avatar.png"))  # load the main character
pygame.display.set_icon(avatar)
#  this surface has an alpha value with the colors, so the player trail will fade away using opacity
alpha_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

# sprite groups
player_sprite = pygame.sprite.Group()
elements = pygame.sprite.Group()

# images
spike = pygame.image.load(os.path.join("images", "obj-spike.png"))
spike = resize(spike)
coin = pygame.image.load(os.path.join("images", "coin.png"))
coin = pygame.transform.smoothscale(coin, (32, 32))
block = pygame.image.load(os.path.join("images", "block_1.png"))
block = pygame.transform.smoothscale(block, (32, 32))
orb = pygame.image.load((os.path.join("images", "orb-yellow.png")))
orb = pygame.transform.smoothscale(orb, (32, 32))
trick = pygame.image.load((os.path.join("images", "obj-breakable.png")))
trick = pygame.transform.smoothscale(trick, (32, 32))

#  ints
fill = 0
num = 0
CameraX = 0
attempts = 0
coins = 0
angle = 0
level = 0

# list
particles = []
orbs = []
win_cubes = []

# initialize level with
levels = ["level_0.csv", "level_1.csv", "level_2.csv"]
level_list = block_map(levels[level])
level_width = (len(level_list[0]) * 32)
level_height = len(level_list) * 32
init_level(level_list)

# set window title suitable for game
pygame.display.set_caption('Pydash: Geometry Dash in Python')

# initialize the font variable to draw text later
text = font.render('image', False, (255, 255, 0))

# music
music = pygame.mixer_music.load(os.path.join("music", "bossfight-Vextron.mp3"))
pygame.mixer_music.play()

# bg image
bg = pygame.image.load(os.path.join("images", "bg.png"))

# create object of player class
player = Player(avatar, elements, (150, 150), player_sprite)

# show tip on start and on death
tip = font.render("tip: tap and hold for the first few seconds of the level", True, BLUE)

state_size = 1000  # Example: discretize player position and obstacle distances into states
action_size = 2  # Actions: [0: do nothing, 1: jump]

player.agent.last_death_time = 0

while not done:
    keys = pygame.key.get_pressed()
    if not start:
        wait_for_key()
        reset()
        start = True


    player.vel.x = 6

    if player.agent.should_jump():
        player.isjump = True

    # If player dies near an obstacle, attempt to learn from it by jumping earlier next time
    # Update jump memory based on success or failure of actions
    if player.died:
        death_counter += 1
        print(death_counter)




    # Get current state (discretized player position and obstacle distance)
    player_x = player.rect.x // 50  # Example: discretize position
    obstacle_x = min([sprite.rect.x for sprite in elements if isinstance(sprite, Spike)], default=800) // 50
    current_state = min(player_x + obstacle_x, state_size - 1)

    # Decide whether to jump based on memory and timing


    if player.agent.success_updated == False:
        player.agent.check_and_update_for_success()


    # MANUAL INPUT
    eval_outcome(player.win, player.died, player.agent)


    if keys[pygame.K_UP] or keys[pygame.K_SPACE]:
        player.isjump = True



    # Reduce the alpha of all pixels on this surface each frame.
    # Control the fade2 speed with the alpha value.
    alpha_surf.fill((255, 255, 255, 1), special_flags=pygame.BLEND_RGBA_MULT)

    player_sprite.update()


    CameraX = player.vel.x  # for moving obstacles
    move_map()  # apply CameraX to all elements

    screen.blit(bg, (0, 0))  # Clear the screen(with the bg)

    player.draw_particle_trail(player.rect.left - 1, player.rect.bottom + 2,
                               WHITE)
    screen.blit(alpha_surf, (0, 0))  # Blit the alpha_surf onto the screen.
    #draw_stats(screen, coin_count(coins))

    if player.isjump:
        """rotate the player by an angle and blit it if player is jumping"""
        angle -= 8.1712  # this may be the angle needed to do a 360 deg turn in the length covered in one jump by player
        blitRotate(screen, player.image, player.rect.center, (16, 16), angle)
    else:
        """if player.isjump is false, then just blit it normally(by using Group().draw() for sprites"""
        player_sprite.draw(screen)  # draw player sprite group
    elements.draw(screen)  # draw all other obstacles


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                """User friendly exit"""
                done = True
            if event.key == pygame.K_2:
                """change level by keypad"""
                player.jump_amount += 1

            if event.key == pygame.K_1:
                """change level by keypad"""

                player.jump_amount -= 1

    display_timer(screen, font, global_timer)
    pygame.display.flip()
    clock.tick(60)
pygame.quit()
