import numpy as np
import pyautogui
import cv2
import matplotlib.pyplot as plt

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

# initialize tiles
screen_img = cv2.imread('screen.png', cv2.IMREAD_UNCHANGED)
screen_gray = cv2.cvtColor(screen_img, cv2.COLOR_BGR2GRAY)
red = cv2.imread('red_tile.png', cv2.IMREAD_GRAYSCALE)
blue = cv2.imread('blue_tile.png', 0)
gren = cv2.imread('g_tile.png', 0)
yellow = cv2.imread('y_tile.png', 0)
purple = cv2.imread('pur_tile.png', 0)
black = cv2.imread('blk_tile.png', 0)
team = cv2.imread('tu_tile.png', 0)

# blur = cv2.GaussianBlur(red, (5,5), 0)
# edges = cv2.Canny(blur, 25, 100)
edges = cv2.Canny(red, 25, 200)
# contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# contour_image = np.zeros_like(red)
# cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

plt.subplot(121),plt.imshow(red,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# h, w, _ = tile_img.shape

# res = cv2.matchTemplate(screen_img, tile_img, cv2.TM_CCOEFF_NORMED)
# threshold = 0.7
# yloc, xloc = np.where(res >= threshold)


# rectangles = []
# for (x, y) in zip(xloc, yloc):
#     rectangles.append([int(x), int(y), int(w), int(h)])
#     rectangles.append([int(x), int(y), int(w), int(h)])

# rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)
# print(len(rectangles))

# for (x, y, w, h) in rectangles:
#     cv2.rectangle(screen_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
#     xc = int(x + w/2)
#     yc = int(y + h/2)
#     print(xc,yc)

# plt.imshow(cv2.cvtColor(screen_img, cv2.COLOR_BGR2RGB))
# plt.title("Detected Matches")
# plt.show()

