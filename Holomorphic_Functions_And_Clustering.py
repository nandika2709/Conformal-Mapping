#!/usr/bin/env python2
"""
Holomorphic Functions and Clustering
"""

import cmath
import numpy as np
import Foreground_Detection_Vases as foreground
from skimage import io
import matplotlib.pyplot as plt
import math
import random


def find_foreground(filename):
    """Finds the foreground of an image."""
    
    image_to_parse = io.imread(filename)
    scaled_image_list = foreground.create_scaled_image(image_to_parse)
    found_foreground = foreground.smoothed_foreground(scaled_image_list, 0.9)
    height = len(found_foreground)
    width = len(found_foreground[0])
    
    return np.ones((height, width)) - found_foreground
    

def place_in_disk_bw(image):
    """Adds a border to an image so the new image contains the bounding disk of the old one. 
    Uses a one parameter (e.g. black and white) image."""
    
    height = len(image)
    width = len(image[0])
    radius = math.sqrt(height**2 + width**2)/2
    top_buffer = int(math.ceil(radius - height/2))
    side_buffer = int(math.ceil(radius - width/2))
    top_strip = np.zeros((top_buffer, width))
    side_strip = np.zeros((height + 2*top_buffer, side_buffer))
    taller_image = np.concatenate((top_strip, image, top_strip), 0)
    wider_image = np.concatenate((side_strip, taller_image, side_strip), 1)
    
    return [wider_image, radius]


def place_in_disk_color(image):
    """Adds a border to an image so the new image contains the bounding disk of the old one. 
    Uses a three parameter (color) image."""
    
    height = len(image)
    width = len(image[0])
    depth = len(image[0][0])
    radius = math.sqrt(height**2 + width**2)/2
    top_buffer = int(math.ceil(radius - height/2))
    side_buffer = int(math.ceil(radius - width/2))
    top_strip = np.ones((top_buffer, width, depth)) * 255
    side_strip = np.ones((height + 2*top_buffer, side_buffer, depth)) * 255
    taller_image = np.concatenate((top_strip, image, top_strip), 0)
    wider_image = np.concatenate((side_strip, taller_image, side_strip), 1)
    
    return [wider_image, radius]


def current_coordinates(foreground_image):
    """Creates an array showing the complex coordinates of all pixels in the foreground."""
    
    height = len(foreground_image)
    width = len(foreground_image[0])
    row_indices = np.transpose([np.arange(height)]) * np.ones((1, width))
    col_indices = np.ones((height, 1)) * np.arange(width)
    row_indices = np.ones((height, width)) - (row_indices * 2.0 / float(height))
    col_indices = col_indices * 2.0 / float(width) - np.ones((height, width))
    
    return np.multiply(row_indices*1j + col_indices, foreground_image)
    
    
def alpha_slide(coordinates, foreground_image, alpha):
    """Creates an array of coordinates obtained by sliding the complex number alpha to 0."""
    
    height = len(foreground_image)
    width = len(foreground_image[0])
    alpha_array = np.ones((height, width)) * alpha
    new_coords = np.divide(coordinates - alpha_array, np.ones((height, width)) - np.multiply(np.conjugate(alpha_array), coordinates))
    
    return np.multiply(new_coords, foreground_image)
    

def unsquare(coordinates, foreground_image, alpha):
    """Creates an array of coordinates obtained by taking the square root with a branch cut through alpha."""
    
    height = len(foreground_image)
    width = len(foreground_image[0])
    unit_cut = alpha/abs(alpha)
    cut_array = np.ones((height, width)) * unit_cut
    rotated = np.multiply(coordinates, -np.conjugate(cut_array))
    unsquared = rotated ** 0.5
    rotated_back = np.multiply(unsquared, -cut_array)
    
    return np.multiply(rotated_back, foreground_image)
    

def find_radius(coordinates, foreground_image, min_angle, max_angle):
    """Finds the largest distance from a foreground pixel to the center, using the given coordinates, 
    restricting to the slice between the min and max angles."""
    
    height = len(foreground_image)
    width = len(foreground_image[0])
    radii = np.multiply(np.absolute(coordinates), foreground_image)
    angles = np.angle(coordinates)
    min_array = np.ones((height, width))*min_angle
    max_array = np.ones((height, width))*max_angle
    angle_slice = np.logical_and(angles >= min_array, angles < max_array)
    radii_slice = np.multiply(angle_slice, radii)
    
    return np.amax(radii_slice)
    

def find_smallest_radius(coordinates, foreground_image, num_slices):
    """Finds the slice of the foreground image, using the given coordinates, which has the smallest radius."""
    
    height = len(foreground_image)
    width = len(foreground_image[0])
    smallest_radius = float(height + width)
    best_angle = 0.0
    epsilon = random.random()
    
    for index in range(num_slices + 1):
        
        min_angle = (index - epsilon)*2*math.pi/float(num_slices) - math.pi
        max_angle = (index - epsilon + 1)*2*math.pi/float(num_slices) - math.pi
        radius = find_radius(coordinates, foreground_image, min_angle, max_angle)
        
        if radius < smallest_radius and radius > 0.0:
            smallest_radius = radius
            best_angle = (min_angle + max_angle)/2.0
            
    return [smallest_radius, best_angle]


def find_smallest_radius_fast(coordinates, boundary_image):
    """Finds the boundary pixel with the smallest distance from the center."""
    
    height = len(boundary_image)
    width = len(boundary_image[0])
    radii = np.maximum(np.absolute(coordinates), np.ones((height, width)) - boundary_image)
    closest_indices = np.argmin(radii)
    closest_index_row = closest_indices // width
    closest_index_col = closest_indices % width
    closest_coord = coordinates[closest_index_row][closest_index_col]
    radius = abs(closest_coord)
    angle = cmath.phase(closest_coord)
    
    return [radius, angle]


def circular_pixel_neighbors(pixel, height, width, radius):
    """Returns a list of all pixels within a certain radius of the given pixel."""
    
    neighbors = []
    current_row = int((1.0 - pixel.imag) * float(height - 1) / 2.0)
    current_col = int((1.0 + pixel.real) * float(width - 1) / 2.0)
    
    for row in range(int(max(current_row - math.ceil(radius), 0)), int(min(current_row + math.ceil(radius) + 1, height))):
        for col in range(int(max(current_col - math.ceil(radius), 0)), int(min(current_col + math.ceil(radius) + 1, width))):
            
            if [row, col] != [current_row, current_col] and math.sqrt((row - current_row)**2 + (col - current_col)**2) <= radius:
                neighbors.append([row, col])
                
    return neighbors


def pixel_neighbors(pixel, height, width):
    """Returns a list of the neighbors of a pixel, including corners."""
    
    neighbors = []
    current_row = pixel[0]
    current_col = pixel[1]
    
    for row in range(max(current_row-1, 0), min(current_row+2, height)):
        for col in range(max(current_col-1, 0), min(current_col+2, width)):
            
            if [row, col] != [current_row, current_col]:
                neighbors.append([row, col])
                
    return neighbors


def find_boundary(foreground_image):
    """Returns an image marking only the foreground pixels which have background pixels as neighbors."""
    
    height = len(foreground_image)
    width = len(foreground_image[0])
    boundary_image = np.zeros((height, width))
    
    for row in range(height):
        for col in range(width):
            
            if foreground_image[row][col] == 1.0:
                neighbors = pixel_neighbors([row, col], height, width)
                boundary = False
                
                for neighbor in neighbors:
                    
                    if foreground_image[neighbor[0]][neighbor[1]] == 0.0:
                        boundary = True
                        
                if boundary == True:
                    boundary_image[row][col] = 1.0
                    
    return boundary_image


def closest_color_faster(current_coord, color_coords, color_foreground, color_image):
    """Finds the color of the closest colored pixel to the current coordinate."""
    
    height = len(color_coords)
    width = len(color_coords[0])
    current_coord_array = current_coord * np.ones((height, width))
    distances = np.maximum(abs(color_coords - current_coord_array), np.ones((height, width)) - color_foreground)
    
    closest_indices = np.argmin(distances)
    closest_index_row = closest_indices // width
    closest_index_col = closest_indices % width
    
    closest_red = color_image[closest_index_row][closest_index_col][0]
    closest_green = color_image[closest_index_row][closest_index_col][1]
    closest_blue = color_image[closest_index_row][closest_index_col][2]
    closest_color = [closest_red, closest_green, closest_blue]
    
    if color_foreground[closest_index_row][closest_index_col] == 0.0:
        print("Not foreground")
        
    return closest_color


def find_top_and_bottom(foreground_image):
    """Finds the top and bottom foreground pixels on the center line of the image."""
    
    height = len(foreground_image)
    width = len(foreground_image[0])
    width_center = int(width/2)
    found_top = False
    top_row = 0
    
    while found_top == False and top_row < height:
        
        current_pixel = foreground_image[top_row][width_center]
        
        if current_pixel == 1.0:
            found_top = True
        else: 
            top_row += 1
            
    found_bottom = False
    bottom_row = height - 1
    
    while found_bottom == False and bottom_row >= 0:
        
        current_pixel = foreground_image[bottom_row][width_center]
        
        if current_pixel == 1.0:
            found_bottom = True
        else: 
            bottom_row -= 1
            
    return [[top_row, width_center], [bottom_row, width_center]]


def turn_rightside_up(coordinates, foreground_image):
    """Transforms coordinates so that the top and bottom of the image map to 
    the top and bottom of the disk, respectively."""
    
    top_and_bottom = find_top_and_bottom(foreground_image)
    top = top_and_bottom[0]
    bottom = top_and_bottom[1]
    top_coord = coordinates[top[0]][top[1]]
    bottom_coord = coordinates[bottom[0]][bottom[1]]
    
    angle_diff = top_coord/bottom_coord
    mid_angle = angle_diff**0.5
    mid_coord = bottom_coord*mid_angle
    mid_coord = mid_coord/abs(mid_coord)
    
    if mid_coord.real < 0.0:
        
        mid_coord = -mid_coord
        
    new_coordinates = coordinates/mid_coord
    new_top_coord = new_coordinates[top[0]][top[1]]
    #new_bottom_coord = new_coordinates[bottom[0]][bottom[1]]
    chi = cmath.polar(new_top_coord/1j)[1]
    
    if chi > math.pi:
        chi -= 2*math.pi
        
    alpha = -math.atan(chi/2.0)
    
    return alpha_slide(new_coordinates, foreground_image, alpha)
    

def weigh_halves(coordinates):
    """Determines how many foreground pixels are being mapped to the upper and lower 
    halves of the unit disk."""
    
    top_array = coordinates.imag > 0.0
    bottom_array = coordinates.imag < 0.0
    top_count = np.sum(top_array)
    bottom_count = np.sum(bottom_array)
    
    return [top_count, bottom_count]


def balance_halves(coordinates, foreground_image):
    """Performs an alpha slide so that pixels are approximately balanced between the 
    upper and lower halves of the unit disk."""
    
    current_balance = weigh_halves(coordinates)
    current_top = current_balance[0]
    current_bottom = current_balance[1]
    
    if current_top == current_bottom:
        return coordinates
    elif current_top > current_bottom:
        scale = 1.0
    else:
        scale = -1.0
        
    new_scale = scale
    new_coords = coordinates
    
    while new_scale == scale:
        
        alpha = 0.001j * scale
        new_coords = alpha_slide(new_coords, foreground_image, alpha)
        current_balance = weigh_halves(new_coords)
        current_top = current_balance[0]
        current_bottom = current_balance[1]
        
        if current_top == current_bottom:
            return new_coords
        elif current_top > current_bottom:
            new_scale = 1.0
        else:
            new_scale = -1.0
            
    return new_coords


def fill_disk(coordinates, foreground_image, num_slices, num_iterations):
    """Transforms the coordinates to make the foreground image fill the unit disk."""
    
    boundary = find_boundary(foreground_image)
    current_coords = coordinates
    iteration = 1
    first_step = True
    done = False
    
    while iteration <= num_iterations and done == False:
        
        print("Iteration:")
        print(iteration)
        iteration += 1
        
        if first_step:
            alpha_info = find_smallest_radius(current_coords, boundary, num_slices)
        else: 
            alpha_info = find_smallest_radius_fast(current_coords, boundary)
            
        radius = alpha_info[0]
        print("Smallest radius:")
        print(radius)
        if radius == 0.0:
            radius = 0.01
        if radius > 0.9:
            first_step = False
        if radius > 0.999:
            done = True
            
        angle = alpha_info[1]
        alpha = cmath.rect(radius, angle)
        larger_radius = math.sqrt(radius)
        beta = cmath.rect(larger_radius, angle)
        
        if alpha_info[0] > 0.0:
            new_coords = alpha_slide(current_coords, foreground_image, alpha)
        newer_coords = unsquare(new_coords, foreground_image, alpha)
        
        if alpha_info[0] > 0.0:
            current_coords = alpha_slide(newer_coords, foreground_image, -beta)
            
    return current_coords


def render_transformed_image(coordinates, foreground_image, image, height, width):
    """Renders a transformed image on the unit disk."""
    
    initial_height = len(image)
    initial_width = len(image[0])
    depth = len(image[0][0])
    new_image = np.ones((height, width, depth)) * 255
    
    for row in range(initial_height):
        for col in range(initial_width):
            
            if foreground_image[row][col] == 1.0:
                color = image[row][col]
                red = color[0]
                green = color[1]
                blue = color[2]
                new_coords = coordinates[row][col]
                
                if abs(new_coords) <= 1.0:
                    new_row = int((1.0 - new_coords.imag) * float(height - 1) / 2.0)
                    new_col = int((1.0 + new_coords.real) * float(width - 1) / 2.0)
                    
                    new_image[new_row][new_col][0] = red * 255
                    new_image[new_row][new_col][1] = green * 255
                    new_image[new_row][new_col][2] = blue * 255
                    
    for row in range(height):
        for col in range(width):
            
            radius = math.sqrt((height/2 - row)**2 + (width/2 - col)**2)
            
            if radius > height/2:
                new_image[row][col][0] = 0.0
                new_image[row][col][1] = 0.0
                new_image[row][col][2] = 0.0
                
    return new_image


def show_stretched_vase(vase_number, num_iterations):
    """Creates an image of a vase stretched to fill a disk."""
    
    written_foreground = read_array("Foreground_Vase_" + str(vase_number) + ".txt")
    original_image = io.imread("Original_Vase_" + str(vase_number) + ".jpg")
    scaled_image = foreground.create_scaled_image(original_image)[0]
    image_in_disk = place_in_disk_color(scaled_image)[0]
    im_height = len(written_foreground)
    im_width = len(written_foreground[0])

    starting_coords = current_coordinates(written_foreground)
    stretched_coords = fill_disk(starting_coords, written_foreground, 40, num_iterations)
    righted_coords = turn_rightside_up(stretched_coords, written_foreground)
    balanced_coords = balance_halves(righted_coords, written_foreground)
    stretched_image = render_transformed_image(balanced_coords, written_foreground, image_in_disk, im_height, im_width)

    plt.imshow(stretched_image.astype(int))
    plt.show()
    return


def repaint_image(shape_foreground, shape_coords, color_coords, color_foreground, color_image):
    """Repaints the shape foreground to have new colors, based on the shape coordinates."""
    
    height = len(shape_foreground)
    width = len(shape_foreground[0])
    new_image = np.ones((height, width, 3))
    
    for row in range(height):
        for col in range(width):
            
            if shape_foreground[row][col] == 1.0:
                current_coord = shape_coords[row][col]
                new_color = closest_color_faster(current_coord, color_coords, color_foreground, color_image)
                
                red = new_color[0]
                green = new_color[1]
                blue = new_color[2]
                
                new_image[row][col][0] = red
                new_image[row][col][1] = green
                new_image[row][col][2] = blue
                
    return new_image


def shapeshift_and_repaint(shape_coords, shape_foreground, color_coords, color_foreground, color_image, num_slices, num_iterations):
    """Given two images, shapeshifts one to the other."""
    
    new_shape_coords = fill_disk(shape_coords, shape_foreground, num_slices, num_iterations)
    newer_shape_coords = turn_rightside_up(new_shape_coords, shape_foreground)
    new_color_coords = fill_disk(color_coords, color_foreground, num_slices, num_iterations)
    newer_color_coords = turn_rightside_up(new_color_coords, color_foreground)
    
    height = len(shape_foreground)
    width = len(shape_foreground[0])
    rendered_color_image = render_transformed_image(newer_color_coords, color_foreground, color_image, height, width)
    
    return repaint_image(shape_foreground, newer_shape_coords, rendered_color_image)
    

def find_local_stretching(coords, foreground, mesh_size):
    "Finds the local stretching factor on each mesh square within the unit disk."""
    
    height = len(coords)
    width = len(coords[0])
    num_pixels = np.sum(foreground)
    disk_data = np.zeros((mesh_size, mesh_size))
    truncated_coords = coords[:height-1, :width-1]
    right_coords = coords[:height-1, 1:]
    down_coords = coords[1:, :width-1]
    truncated_foreground = foreground[:height-1, :width-1]
    right_foreground = foreground[:height-1, 1:]
    down_foreground = foreground[1:, :width-1]
    
    right_change = np.nan_to_num(np.log(math.sqrt(num_pixels) * np.absolute(right_coords - truncated_coords)))
    down_change = np.nan_to_num(np.log(math.sqrt(num_pixels) * np.absolute(down_coords - truncated_coords)))
    move_right_foreground = np.multiply(truncated_foreground, right_foreground)
    move_down_foreground = np.multiply(truncated_foreground, down_foreground)
    right_change_foreground = np.multiply(right_change, move_right_foreground)
    down_change_foreground = np.multiply(down_change, move_down_foreground)
    change_foreground = np.multiply(move_right_foreground, move_down_foreground)
    
    smallest_stretch = 1000.0
    biggest_stretch = -1000.0
    
    for row in range(mesh_size):
        for col in range(mesh_size):
            
            if math.sqrt((2*row - mesh_size)**2 + (2*col - mesh_size)**2) <= mesh_size:
                fine_row_start = -1.0 + 2.0*float(row)/float(mesh_size)
                fine_row_stop = -1.0 + 2.0*float(row + 1)/float(mesh_size)
                fine_col_start = -1.0 + 2.0*float(col)/float(mesh_size)
                fine_col_stop = -1.0 + 2.0*float(col + 1)/float(mesh_size)
                
                fine_pixel_location = np.logical_and(np.logical_and(np.real(truncated_coords) >= fine_col_start, np.real(truncated_coords) <= fine_col_stop), np.logical_and(-np.imag(truncated_coords) >= fine_row_start, -np.imag(truncated_coords) <= fine_row_stop))
                fine_pixel_right = np.sum(np.multiply(right_change_foreground, fine_pixel_location))
                fine_pixel_down = np.sum(np.multiply(down_change_foreground, fine_pixel_location))
                
                total_fine_pixel = np.sum(np.multiply(change_foreground, fine_pixel_location))
                
                if total_fine_pixel != 0.0:
                    fine_pixel_stretch = (fine_pixel_right + fine_pixel_down)/total_fine_pixel
                    
                disk_data[row][col] = fine_pixel_stretch
                
                if fine_pixel_stretch < smallest_stretch:
                    smallest_stretch = fine_pixel_stretch
                    
                if fine_pixel_stretch > biggest_stretch:
                    biggest_stretch = fine_pixel_stretch
                    
    return disk_data
            

def write_array(filename, array):
    """Writes the contents of a 2D array to a text file."""
    
    text_file  = open(filename, 'w')
    height = len(array)
    width = len(array[0])
    
    for row in range(height):
        for col in range(width):
            
            text_file.write(str(array[row][col]))
            
            if col < width - 1:
                text_file.write(" ")
                
        if row < height - 1:
            text_file.write('\n')
            
    text_file.close()
    return
    

def read_array(filename):
    """Creates a 2D ndarray of zeros and ones based on the content of a text file."""
    
    text_file  = open(filename, 'r')
    rows = text_file.readlines()
    height = len(rows)
    width = len(rows[0].split(" "))
    built_image = np.zeros((height, width))
    
    for row in range(height):
        read_row = rows[row].split(" ")
        
        for col in range(width):
            
            if read_row[col] == str(1.0):
                built_image[row][col] = 1.0
                
    text_file.close()
    return built_image


def read_float_array(filename):
    """Creates a 2D ndarray of floats based on the content of a text file."""
    
    text_file  = open(filename, 'r')
    rows = text_file.readlines()
    height = len(rows)
    width = len(rows[0].split(" "))
    built_image = np.zeros((height, width))
    
    for row in range(height):
        read_row = rows[row].split(" ")
        
        for col in range(width):
            built_image[row][col] = float(read_row[col])
            
    text_file.close()
    return built_image


def read_complex_array(filename):
    """Creates a 2D ndarray of complex numbers based on the content of a text file."""
    
    text_file  = open(filename, 'r')
    rows = text_file.readlines()
    height = len(rows)
    width = len(rows[0].split(" "))
    built_image = np.zeros((height, width), dtype = complex)
    
    for row in range(height):
        read_row = rows[row].split(" ")
        
        for col in range(width):
            built_image[row][col] = complex(read_row[col])
            
    text_file.close()
    return built_image


def find_distance(array1, array2):
    """Finds the Pythagorean distance between two arrays of the same size."""
    
    return math.sqrt(np.sum(np.multiply(array1 - array2, array1 - array2)))


def find_all_distances(distances_filename):
    """Finds all distances between pairs of images in a list, and writes an array of these distances to the file."""
    
    foregrounds = []
    coordinates = []
    stretched_coordinates = []
    
    for index in range(12):
        
        print("Index:")
        print(index)
        foreground_filename = "Foreground_Vase_" + str(index + 1) + ".txt"
        coords_filename = "Stretched_Vase_" + str(index + 1) + ".txt"
        new_foreground = read_array(foreground_filename)
        coords = read_complex_array(coords_filename)
        local_stretching = find_local_stretching(coords, new_foreground, 20)
        foregrounds.append(new_foreground)
        coordinates.append(coords)
        stretched_coordinates.append(local_stretching)
        
    distances = np.zeros((12,12))
    
    for first_index in range(12):
        for second_index in range(12):
            
            local_stretching1 = stretched_coordinates[first_index]
            local_stretching2 = stretched_coordinates[second_index]
            distance = find_distance(local_stretching1, local_stretching2)
            distances[first_index][second_index] = distance
            
    write_array(distances_filename, distances)
    return


def hierarchical_clustering(distance_array, num_clusters):
    """Performs hierarchical clustering on data given an array of distances and a desired number of clusters."""
    
    clusters = []
    width = len(distance_array)
    distances = np.copy(distance_array)
    
    for index in range(width):
        clusters.append(set({index}))
        distances[index][index] = float("inf")
        
    while len(clusters) > num_clusters:
        
        closest_pair = np.argmin(distances)
        closest_pair_row = closest_pair // width
        closest_pair_col = closest_pair % width
        distances[closest_pair_row][closest_pair_col] = float("inf")
        distances[closest_pair_col][closest_pair_row] = float("inf")
        new_clusters = clusters[:]
        new_cluster = set({})
        
        for cluster in clusters:
            
            if closest_pair_col in cluster or closest_pair_row in cluster:
                new_clusters.remove(cluster)
                new_cluster = new_cluster.union(cluster)
                
        new_clusters.append(new_cluster)
        clusters = new_clusters
        
    return clusters


"""The following code finds the foreground of an image. Uncomment the code to run it.
Change the vase number (to an integer in [1, 12]) to find a different vase's foreground."""


#found_foreground = find_foreground("Original_Vase_6.jpg")
#found_foreground = place_in_disk_bw(found_foreground)[0]
#plt.imshow(found_foreground.astype(int))
#plt.show()


"""The following code uses an image and its foreground to find a version of the image
which has been stretched to fill a disk. Uncomment the code to run it. 
Change the first argument (to an integer in [1, 12]) to stretch a different vase.
Change the second argument to change the number of iterations."""


#show_stretched_vase(1, 100)


"""The following code uses the stretched coordinates for all vase images to create an array of 
the distances between their shapes, and writes this array to a file. Uncomment the code to run it."""


#find_all_distances("Blank_Distances_File.txt")


"""The following code uses the array of distances between images of vases to cluster them by shape."""


distances_filename = "Distances.txt"
distances = read_float_array(distances_filename)[:10, :10]
print("The clustering is:")
print(hierarchical_clustering(distances, 3))
