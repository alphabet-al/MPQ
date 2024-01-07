import numpy as np
import pyautogui
import cv2
import matplotlib.pyplot as plt
import random

# res = pyautogui.locateOnScreen('red_tile.png')

# print(res)
# tile = pyautogui.center(res)

# pyautogui.moveTo(tile)
# # pyautogui.click(tile)

# cv2.imshow('screen', screen_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# cv2.imshow('tile_img', tile_img)
# cv2.waitKey()
# cv2.destroyAllWindows()


screen_img = cv2.imread('screen.png', cv2.IMREAD_UNCHANGED)
tile_names = ['red', 'green', 'blue', 'yellow', 'purple', 'black', 'team']
tiles = {name: [cv2.imread(f'{name}_tile.png', cv2.IMREAD_UNCHANGED)] for name in tile_names}

for k,v in tiles.items():
    tile_data = v[0]
    h, w, _ = tile_data.shape

    res = cv2.matchTemplate(screen_img, tile_data, cv2.TM_CCOEFF_NORMED)
    
    # cv2.imshow('result', res)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    threshold = 0.55
    blk_thresh = .35
    if k == 'black':
        yloc, xloc = np.where(res >= blk_thresh)
    else:
        yloc, xloc = np.where(res >= threshold)


    # double up rectangles around objects recognized and then filter out those
    # that are duplicates
    rectangles = []
    for (x, y) in zip(xloc, yloc):
        rectangles.append([int(x), int(y), int(w), int(h)])
        rectangles.append([int(x), int(y), int(w), int(h)])

    rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)
    # print number of tiles for each type recognized to terminal for debug
    print(len(rectangles))

    for (x, y, w, h) in rectangles:
        cv2.rectangle(screen_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        xc = int(x + w/2)
        yc = int(y + h/2)
        tiles[k].append((xc, yc))


    # visualize how well object recognition works
    # plt.imshow(cv2.cvtColor(screen_img, cv2.COLOR_BGR2RGB))
    # plt.title("Detected Matches")
    # plt.show()

# Initialize variables to store min and max coordinates
min_x, min_y = float('inf'), float('inf')
max_x, max_y = float('-inf'), float('-inf')

# Assuming 'tiles' is your dictionary with detected tiles
for tile_type, data in tiles.items():
    for coord in data[1:]:  # Skip the first element which is the image
        x, y = coord
        min_x, min_y = min(min_x, x), min(min_y, y)
        max_x, max_y = max(max_x, x), max(max_y, y)

# Now you have the extreme coordinates of your grid
# Calculate grid cell size
grid_width = max_x - min_x
grid_height = max_y - min_y
num_rows, num_cols = 8, 8  # Example for an 8x8 grid
cell_width = 88
cell_height = 88

# Initialize grid representation
grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
grid_coord = {}

# Map tiles to grid
for tile_type, data in tiles.items():
    for x, y in data[1:]:
        col = int((x - min_x) / cell_width)
        row = int((y - min_y) / cell_height)
        # print(f"Row: {row}, Col: {col}")

        grid[row][col] = tile_type  # Map the tile type to its grid position
        grid_coord[(row,col)] = (x, y)

def is_matchable(r, c, tile):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    moves = []

    def check_dir(dr, dc):
        count = 0
        y, x = r + dr, c + dc
        while 0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] == tile:
            count += 1
            y += dr
            x += dc
        return count

    def add_valid_move(dr, dc):
        y, x = r + dr, c + dc
        if 0 <= y < len(grid) and 0 <= x < len(grid[0]) and grid[y][x] == tile:
            moves.append(((y, x), (r, c)))

    for dr, dc in directions:
        opposite_dir = (-dr, -dc)
        if check_dir(dr, dc) + check_dir(*opposite_dir) >= 2:
            # Add valid moves from perpendicular directions
            for perp_dr, perp_dc in directions:
                if perp_dr != dr and perp_dc != dc:  # Check perpendicular directions
                    add_valid_move(perp_dr, perp_dc)

    return bool(moves), moves



def moves():
    directions = {(0, 1), (0, -1), (1, 0), (-1, 0)}
    moves = []        

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            for tile in tile_names:
                valid = is_matchable(r, c, tile)
                if valid[0]:
                    # print(f"Placing a '{tile}' tile at ({r}, {c}) can form a match-3.")
                    moves.append(valid[1])
    return moves

valid_moves = moves()
flattened_moves = [move for sublist in valid_moves for move in sublist]
chosen_move = random.choice(flattened_moves) 
click_from, click_to = grid_coord[chosen_move[0]], grid_coord[chosen_move[1]]

pyautogui.moveTo(click_from)
pyautogui.click(click_from)

pyautogui.moveTo(click_to)
pyautogui.click(click_to)

print(chosen_move)