import random

import cv2
import numpy as np
from typing import List, Tuple, Optional

IntPoint = Tuple[int, int]
Edge = List[IntPoint]

START_THRESHOLD = 180
STOP_THRESHOLD = 20
MIN_LENGTH = 25
LINE_CONNECTION_THRESHOLD = 4


def load_smooth_and_grey_image() -> None:
    """
    Load in an image and build a smoothed, grayscale image of it.
    This generates the following three global variables, each is a np.ndarray with
             the corresponding dimensions, type and range.
    • source_image (H, W, 3) integers 0-255
    • smoothed_image (H, W, 3) integers 0-255
    • grey_image (H, W, 1) integers 0-255


    :return: None
    """
    global source_image, smoothed_image, grey_image
    source_image = cv2.imread("Tables.png")  # load image rom disk

    # create a smoothed out version of this image to de-noise it.
    smoothed_image = cv2.GaussianBlur(source_image, (5, 5), 0)

    # convert this to black-and-white range 0 to 255 (ints)
    grey_image = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2GRAY)


def measure_change_in_image(img: np.ndarray) -> None:
    """
    analyzes the given grayscale image shape (H, W, 1), composed of integers 0-255 to find the change in pixel values
    in the x-direction, the y-direction, the change magnitude and the angle of change. Creates the following four global
    variables, with the corresponding shapes, types and ranges:

    • dx_image, dy_image, (H, W, 1) floats -255.0 - +255.0
    • change_magnitude_image (H, W, 1) floats 0-255.0
    • change_angle_image (H, W, 1) floats -π - +π
    :param img:  the grayscale source image.
    :return: None
    """
    global dx_image, dy_image, change_magnitude_image, change_angle_image

    dx_image = find_horizontal_differences(img)   # range will be -255.0 to +255.0
    dy_image = find_vertical_differences(img)

    # TODO: generate the images, "change_magnitude_image" and "change_angle_image" - the former is the pythagorean
    #    magnitude of dx and dy at each pixel. The latter is the angle of the triangle formed by dx and dy at each
    #    pixel.
    #    Because numpy is soooo handy, it is possible to do these calculations in one line apiece. Make use of the
    #    following cellwise operators:
    #    *
    #    +
    #    np.sqrt
    #    np.atan2


def find_horizontal_differences(source: np.ndarray) -> np.ndarray:
    """
    Finds the "dx" of the given source... that is the difference between each point and the one to the left of it.
    :param source: an (N x M x 1) ndarray of range 0-255
    :return:  an (N x M x 1) ndarray of type float (range -255.0 to +255.0)
    """
    result: np.ndarray = np.zeros(shape=source.shape, dtype=float)

    # TODO: you write this! (Hint: remember that the ndarrays are (row, col), not (x,y).

    return result


def find_vertical_differences(source: np.ndarray) -> np.ndarray:
    """
    Finds the "dy" of the given source... that is the difference between each point and the one above it.
    :param source: an (N x M x 1) ndarray of range 0-255
    :return:  an (N x M x 1) ndarray of type float (range -255.0 to +255.0)
    """
    result: np.ndarray = np.zeros(shape=source.shape, dtype=float)
    # TODO: you write this! (Hint: remember that the ndarrays are (row, col), not (x,y).

    return result


def build_an_edge(magnitude_map, angle_map) -> Optional[Edge]:
    """
    Assuming at least one point in the magnitude_map is at START_THRESHOLD or higher, get a list of connected integer
    points that represents a section of line that passes through the greatest magnitude point in the magnitude_map,
    consisting only of points that are at or above the STOP_THRESHOLD. If no points were at START_THRESHOLD, return
    None.
    Postcondition: the magnitude map will have the "used up" pixels set to zero, so they don't get reused.
    :param magnitude_map: a grid of floats indicating the amount of change at each pixel
    :param angle_map: a grid of floats indicating the direction of change at each pixel.
    :return: an Edge, or None, if nothing meets the threshold criteria.
    """
    # This finds the (r, c) coordinates of the point in magnitude_map with the highest value.
    start_pos = list(np.unravel_index(np.argmax(magnitude_map), magnitude_map.shape))

    # And here is what that value is.
    start_mag = magnitude_map[start_pos[0], start_pos[1]]

    # This starts our output list off with the starting point.
    points_to_add = [(int(start_pos[0]), int(start_pos[1])), ]

    # Bail out now, if the highest point wasn't up to the START_THRESHOLD.
    if start_mag < START_THRESHOLD:
        return None

    # It would be a good idea to find the angle from angle_map at the starting pos, as well.

    # You're going to need to have a moving point that starts where start_pos is, but holds floats.

    # Here's a loop... do this over and over again...
    # make your floating point take a small step at 90° to the right from the direction at the current point.
    # calculate a rounded off version of that point.
    # is this point in bounds, and is it a point we haven't been to yet?
    #    if so, is the magnitude above the STOP_THRESHOLD?
    #         if so, add this location to the output list, update what we consider current with the new information
    #         (including the current angle) and repeat. Set this current (integer) point to zero in the magnitude_map.

    # Repeat the process above, only this time go 90° to the left. But reverse the list of points before you do!
    #   (So the first and last points of the list are the endpoints of this edge...)

    # Return the list.

    return points_to_add


"""
#Optional -- this little bit of code will respond to the user clicking in the change_magnitude map by drawing a line in
destination map at that location, pointing in the direction of the angle map at that location.
def mouse_click(event, x, y,
                flags, param):
    # to check if left mouse
    # button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"{x: }\t{y: }")
        ang = change_angle_image[y, x]
        x2 = int( x+20*np.cos(ang))
        y2 = int( y+20*np.sin(ang))

        cv2.line(destination, (x,y), (x2,y2), (0,0,255), 1)
        cv2.imshow("Destination",destination)
"""


def add_edge_to_edge_list(edge_to_add: Edge) -> None:
    """
    checks whether this edge is a continuation of one of the existing edges. If so, it appends this to that edge.
    Otherwise, it just adds this edge to the list.
    postcondition: the list of edges may have
    • gotten bigger by one (adding an unconnected edge), or
    • kept the same number of edges (added the new edge to an existing edge - e.g., list had edge A and adding new edge
          N made combined into a new edge, AN), or
    • gotten shorter by one edge (the new edge connected two existing edges together - e.g, list had edges A and B
          already, and new edge N made for one compound edge, ANB.)
    :param edge_to_add: an edge of IntPoints
    :return: None.
    """
    global num_combines

    # TODO: Optional.... compare the edge_to_add to all the edges in edge_list already. If any of them have endpoints
    #  within the criteria of is_adjacent(pt1, pt2) of the edge_to_add, then take that edge out of the list, combine
    #  them into one edge and continue searching to see whether there are any other edges to join before adding.

    edge_list.append(edge_to_add)


def clear_short_lines() -> None:
    """
    goes through the global edge_list and removes any lines shorter than MIN_LENGTH
    :return: None
    """
    global num_short_lines_removed
    num_short_lines_removed = 0
    # TODO: go through the edge_list and remove any edges that have fewer than MIN_LENGTH points in them.


def is_adjacent(p1: IntPoint, p2: IntPoint) -> bool:
    """
    are these two points close enough to be considered connected? (In this case, are they within a square of
    size = 2 * LINE_CONNECTION_THRESHOLD - 1 of each other?)
    :param p1: (x1, y1)
    :param p2: (x2, y2)
    :return: whether p1 and p2 are within just a few pixels of each other.
    """
    return abs(p1[0] - p2[0]) < LINE_CONNECTION_THRESHOLD and abs(p1[1] - p2[1]) < LINE_CONNECTION_THRESHOLD


def draw_all_edges(canvas: np.ndarray, edges_to_draw: List[Edge], line_thickness=3) -> None:
    """
    draws all the lines in edge_list onto destination, with each line a unique color.
    :param canvas: the nd.nparray in which to draw
    :param edges_to_draw: the list of Lines (i.e., lists of points)
    :param line_thickness: the width of the lines to draw.
    :return: None
    """

    for found_edge in edges_to_draw:
        # pick a random (bright) color for this line
        r = random.randrange(64, 255)
        g = random.randrange(64, 255)
        b = random.randrange(64, 255)
        for pt in found_edge:
            # draw points in line_thickness x line_thickness square around this point
            for i in range(-int(line_thickness / 2), int(line_thickness / 2 + 0.5)):
                for j in range(-int(line_thickness / 2), int(line_thickness / 2 + 0.5)):
                    if 0 <= pt[0] + i < source_image.shape[0] and 0 <= pt[1] + j < source_image.shape[1]:
                        # only color worthwhile lines, but black out any we find (or we'll keep finding the old ones).
                        # if len(edge) > MIN_LENGTH:
                        canvas[pt[0] + i, pt[1] + j] = (b, g, r)


if __name__ == '__main__':
    global num_combines
    global source_image, smoothed_image, grey_image
    load_smooth_and_grey_image()
    measure_change_in_image(grey_image)
    edge_list = []
    num_combines = 0

    # Comment out one of the following:
    # --------------------
    # Either start the destination as all black....
    # destination = np.zeros(source_image.shape,dtype=np.uint8)

    # Or as a dark grey version of the greyscale map.
    destination = cv2.cvtColor(grey_image, cv2.COLOR_GRAY2BGR)
    destination = destination * 0.25
    destination = destination.astype(np.uint8)
    # --------------------

    # shows the image (0-1.0) in a window called "Sobel_mag original."
    cv2.imshow("Sobel_mag original", change_magnitude_image / 255)

    line_count = 1  # number of lines we've found
    edge = build_an_edge(change_magnitude_image, change_angle_image)
    while edge is not None:
        # if len(edge) > MIN_LENGTH:
        add_edge_to_edge_list(edge)

        print(f"{line_count}\t{edge=}")

        # draw this line in destination and black it out in change_magnitude
        for p in edge:
            # draw points in 3 x 3 around this point
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if 0 <= p[0] + i < source_image.shape[0] and 0 <= p[1] + j < source_image.shape[1]:
                        change_magnitude_image[p[0] + i, p[1] + j] = 0
        edge = build_an_edge(change_magnitude_image, change_angle_image)
        line_count += 1
    clear_short_lines()
    
    draw_all_edges(destination, edge_list)

    print(f"{line_count=}\t{num_combines=}\t{len(edge_list)=}\t{num_short_lines_removed=}")

    # cv2.imshow("Source",source_image)
    # cv2.imshow("Grey",grey_image)
    # cv2.imshow("Sobel_X",dx_image/512+0.5)
    # cv2.imshow("Sobel_Y",dy_image/512+0.5)
    cv2.imshow("Sobel_mag", change_magnitude_image/255)
    cv2.imshow("Destination", destination)
    # cv2.setMouseCallback('Sobel_mag', mouse_click)

    # wait for a keyboard click, then close all windows and quit.
    cv2.waitKey(0)
    cv2.destroyAllWindows()
