

import sys
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.graphics.instructions import RenderContext
from kivy.graphics.transformation import Matrix
from kivy.graphics import *
from kivy.graphics.opengl import *
from kivy.clock import Clock
from kivy.utils import platform
import logging
import os
import threading
from math import *

logger = logging.getLogger(__name__)

import datetime
start_time = 0
def get_elapsed(str):
    global start_time
    if str == "start":
        start_time = datetime.datetime.now()
    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    start_time = end_time
    print(f"{str} -> {elapsed_time}")

from .Objloader import ObjFile
#arc camera
import math
from .arcball_from_cpp import *
#input
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent

#calculate the 3d distance
def len_3d(pos1,pos2):
    return math.sqrt((pos1[0] - pos2[0])*(pos1[0] - pos2[0])+(pos1[1]-pos2[1])*(pos1[1]-pos2[1])+(pos1[2]-pos2[2])*(pos1[2]-pos2[2]))

def len_2d(pos1,pos2):
    return math.sqrt((pos1[0] - pos2[0])*(pos1[0] - pos2[0])+(pos1[1]-pos2[1])*(pos1[1]-pos2[1]))

def normalize(dir):
    length = len_3d(dir,[0,0,0])
    if(length < 0.0001):
        print('normalize failed')
        return [1,0,0]
    inv_length = 1.0 / length
    return [dir[0]*inv_length,dir[1]*inv_length,dir[2]*inv_length]

def normalize_angle(angle):
    while (angle < 0): angle += 360
    while (angle > 360): angle -= 360
    return angle

ZOOMSTEP = 1.1
DEFAULT_ZOOM = 0.65
M_PI = 3.141592653
MESH_LINE_CHUNK = 65500 # Max G-code lines per line_strip mesh (65500 vertices)

#binary search left key
def binary_find_left(array,key):
    length=len(array)
    ans=length
    l=0
    r=length-1
    while(l<=r):
        mid=(l+r)>>1
        if(array[mid]>=key):
            ans=mid
            r=mid-1
        else: l=mid+1
    return ans-1

#rotate point around axis & angle
#https://stackoverflow.com/questions/6721544/circular-rotation-around-an-arbitrary-axis
#https://kivy.org/doc/stable/api-kivy.graphics.transformation.html
def rotate_pt_by_x_axis_angle(pt_x,pt_y,pt_z,angle_in_degree):
    axis = [1,0,0]
    mat_rot_x = Matrix()
    angle_in_radian = angle_in_degree * 3.1415926 / 180.0
    mat_rot_x.rotate(angle_in_radian,axis[0],axis[1],axis[2])
    rot_pt = mat_rot_x.transform_point(pt_x,pt_y,pt_z)
    return rot_pt
def rotate_mat_by_x_axis_angle(angle_in_degree):
    axis = [1,0,0]
    mat_rot_x = Matrix()
    angle_in_radian = angle_in_degree * 3.1415926 / 180.0
    mat_rot_x.rotate(angle_in_radian,axis[0],axis[1],axis[2])
    return mat_rot_x


#####function
def vec3_add(v1, v2):
    return [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]


def vec3_sub(v1, v2):
    return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]


def vec3_mul_float(v1, f):
    return [v1[0] * f, v1[1] * f, v1[2] * f]


def vec3_divide(v1, ff):
    f = 1.0 / ff
    return [v1[0] * f, v1[1] * f, v1[2] * f]


def vec3_len(v1):
    return sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])


def vec3_max(v1, v2):
    return [max(v1[0], v2[0]), max(v1[1], v2[1]), max(v1[2], v2[2])]


def vec3_min(v1, v2):
    return [min(v1[0], v2[0]), min(v1[1], v2[1]), min(v1[2], v2[2])]


def bbox_max_side_length(min_pt, max_pt):
    """Retrieve the largest axis span from the bounding box"""
    ex = max_pt[0] - min_pt[0]
    ey = max_pt[1] - min_pt[1]
    ez = max_pt[2] - min_pt[2]
    m = max(ex, ey, ez)
    if m <= 0.0 or not isfinite(m):
        return 0.0
    return m


def vec3_distance(v1, v2):
    v3 = vec3_sub(v1, v2)
    return vec3_len(v3)


GRID_STEP_MM = 10.0
GRID_MAJOR_STEP_MM = 100.0
GRID_COLOR_MINOR = [0.35, 0.35, 0.35]
GRID_COLOR_MAJOR = [0.55, 0.55, 0.55]
# axis.obj points along +Y; rotations map meshes to world axes (see axis arrow setup)
AXIS_COLOR_X = [1.0, 0.0, 0.0]
AXIS_COLOR_Y = [0.0, 1.0, 0.0]
AXIS_COLOR_Z = [0.0, 0.0, 1.0]

# T1–T10 tool colors (matches toolpath.glsl tool_palette_color).
TOOL_PALETTE = (
    (0.406684, 0.735902, 0.235489, 1),
    (0.000000, 0.459774, 0.840728, 1),
    (0.779915, 0.319537, 0.130857, 1),
    (0.740127, 0.236840, 0.700182, 1),
    (0.000000, 0.755849, 0.602221, 1),
    (0.825216, 0.043248, 0.043248, 1),
    (0.894806, 0.717161, 0.000000, 1),
    (0.128923, 0.578319, 0.877916, 1),
    (0.431518, 0.268501, 0.839063, 1),
    (0.248716, 0.777237, 0.402157, 1),
)

DEFAULT_FEED_MM_MIN = 3000.0
VERTEX_FLOAT_NUM = 11

COLOR_SCHEME_BY_TYPE = 0
COLOR_SCHEME_BY_TOOL = 1
COLOR_SCHEME_BY_SPEED = 2
COLOR_SCHEME_UI_BY_TYPE = 'Move type'
COLOR_SCHEME_UI_BY_TOOL = 'Tool'
COLOR_SCHEME_UI_BY_SPEED = 'Speed'


def feed_mm_min_for_move(is_rapid, feed_value=None):
    """Feed rate (mm/min) stored per vertex; 0 for rapid moves."""
    if is_rapid:
        return 0.0
    if feed_value is not None:
        try:
            feed = float(feed_value)
            if feed > 0.0:
                return feed
        except (TypeError, ValueError):
            pass
    return DEFAULT_FEED_MM_MIN


def tool_palette_rgb(tool_number):
    """RGB for a 1-based tool number (wraps palette like the shader)."""
    idx = (max(int(tool_number), 1) - 1) % len(TOOL_PALETTE)
    return TOOL_PALETTE[idx][:3]


def speed_colormap_rgb(t):
    """Match toolpath.glsl speed_colormap."""
    t = max(0.0, min(1.0, float(t)))
    if t < 0.33:
        a = (0.2, 0.4, 0.9)
        b = (0.1, 0.7, 0.5)
        u = t / 0.33
    elif t < 0.66:
        a = (0.1, 0.7, 0.5)
        b = (0.95, 0.85, 0.15)
        u = (t - 0.33) / 0.33
    else:
        a = (0.95, 0.85, 0.15)
        b = (0.9, 0.25, 0.2)
        u = (t - 0.66) / 0.34
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u)


GRID_QUAD_MIN_SIZE = 10.0


class MeshManager():

    def __init__(self):

        ##data container

        self.positions = []
        # raw positions (unrotated G-code coordinates)
        self.raw_positions = []
        # all lengths
        self.lengths = []
        # vertex type
        self.vertex_types = []
        # raw numbers
        self.raw_linenumbers = []
        # feed rate (mm/min) per vertex, from CNC parser
        self.raw_feed_rates = []
        # angles of vertices [4 axis]
        self.angles_of_vertices = []

        # mesh container
        self.meshes = []

        # vertices
        self.vertices = []
        ##  bounding area

        # record the max size of area
        self.area_size = 0.0
        # bounding box (min/max per axis)
        self.min_pt = [float('inf'), float('inf'), float('inf')]
        self.max_pt = [float('-inf'), float('-inf'), float('-inf')]
        # cetner of meshes
        self.area_center_sum = [0, 0, 0]
        self.area_center_sum_index = 0
        self.position_scale = 1.0 #same to scale_invert

        ## attributes
        self.is_4_axis = None

    def clear(self):
        self.positions.clear()
        self.raw_positions.clear()
        # all lengths
        self.lengths.clear()
        # vertex type
        self.vertex_types.clear()
        # raw numbers
        self.raw_linenumbers.clear()
        self.raw_feed_rates.clear()
        # angles of vertices [4 axis]
        self.angles_of_vertices.clear()
        # mesh container
        self.meshes.clear()
        # vertices
        self.vertices.clear()

        #move to origin
        self.area_size = 0.0
        self.min_pt = [float('inf'), float('inf'), float('inf')]
        self.max_pt = [float('-inf'), float('-inf'), float('-inf')]
        self.area_center_sum = [0,0,0]
        self.area_center_sum_index = 0
        self.position_scale = 1.0  # same to scale_invert
        self.is_4_axis = None

    def get_pt_count(self):
        return len(self.positions)

    def map_color(self, color_str):
        if color_str == 'Green':
            return [0., 1., 0.]
        elif color_str == 'Red':
            return [1., 0., 0.]
        return [1., 1., 1.]

    # get center of meshes
    def get_center(self):
        if self.area_center_sum_index == 0:
            return [0, 0, 0]

        return vec3_divide(self.area_center_sum, self.area_center_sum_index)

    def get_center_of_view(self):
        return vec3_mul_float(self.get_center(), self.position_scale)

    def get_vertex_position(self,idx):
        base = idx * VERTEX_FLOAT_NUM
        return [self.vertices[base], self.vertices[base + 1], self.vertices[base + 2]]
    # parse single line
    def parse_line(self, line):
        arr_pt = line.split(' ')

        # position (raw G-code coordinates)
        raw_pos = [float(arr_pt[1]), float(arr_pt[3]), float(arr_pt[5])]
        
        # Store raw positions before rotation
        self.raw_positions.append(raw_pos[0])
        self.raw_positions.append(raw_pos[1])
        self.raw_positions.append(raw_pos[2])
        
        pos = raw_pos
        if self.is_4_axis:
            angle = float(arr_pt[7])
            pos = rotate_pt_by_x_axis_angle(pos[0], pos[1], pos[2], angle)

        self.positions.append(pos[0])
        self.positions.append(pos[1])
        self.positions.append(pos[2])
        self.min_pt = vec3_min(self.min_pt, pos)
        self.max_pt = vec3_max(self.max_pt, pos)

        # for center calculating
        self.area_center_sum = vec3_add(self.area_center_sum, pos)
        self.area_center_sum_index += 1

        # get attributes of this point
        vertex = [0] * VERTEX_FLOAT_NUM
        if self.is_4_axis:
            # 1 position
            vertex[0] = pos[0]
            vertex[1] = pos[1]
            vertex[2] = pos[2]

            #angle
            angle = float(arr_pt[7])

            # 2 color
            color = self.map_color(arr_pt[9])
            vertex[3] = color[0]
            vertex[4] = color[1]
            vertex[5] = color[2]

            # 3 line number in gcode
            vertex[6] = float(arr_pt[11])


            # 4 type id
            vertex[7] = len(self.positions) - 1

            # 5 distance attribute
            vertex[8] = 0  # set after length is calculated

            # 6 set tool knife id
            vertex[9] = float(arr_pt[13])

            # 7 feed rate (mm/min)
            is_rapid = arr_pt[9] == "Red"
            feed = feed_mm_min_for_move(is_rapid)
            vertex[10] = feed

            # push this vertex to container
            self.vertices.extend(vertex)
            self.vertex_types.append(1.0 if arr_pt[9] == "Green" else 2.0)  # line type[red | green]
            self.raw_linenumbers.append(vertex[6])
            self.raw_feed_rates.append(feed)
            self.angles_of_vertices.append(angle)
        else:
            # 1 position
            vertex[0] = pos[0]
            vertex[1] = pos[1]
            vertex[2] = pos[2]

            # 2 color
            color = self.map_color(arr_pt[7])
            vertex[3] = color[0]
            vertex[4] = color[1]
            vertex[5] = color[2]

            # 3 line number in gcode
            vertex[6] = float(arr_pt[9])

            # 4 type id
            vertex[7] = len(self.positions) - 1

            # 5 distance attribute
            vertex[8] = 0  # set after length is calculated

            # 6 set tool knife id
            vertex[9] = float(arr_pt[11])

            # 7 feed rate (mm/min)
            is_rapid = arr_pt[7] == "Red"
            feed = feed_mm_min_for_move(is_rapid)
            vertex[10] = feed

            # push this vertex to container
            self.vertices.extend(vertex)

            self.vertex_types.append(1.0 if arr_pt[7] == "Green" else 2.0)  # line type[red | green]
            self.raw_linenumbers.append(vertex[6])
            self.raw_feed_rates.append(feed)

    def parse_line_data(self,linedata):

        # position (raw G-code coordinates)
        raw_pos = [linedata[0],linedata[1],linedata[2]]
        
        # Store raw positions before rotation
        self.raw_positions.extend(raw_pos)

        #angle
        angle = linedata[3]
        pos = rotate_pt_by_x_axis_angle(raw_pos[0], raw_pos[1], raw_pos[2], angle)

        self.positions.extend(pos)
        self.min_pt = vec3_min(self.min_pt, pos)
        self.max_pt = vec3_max(self.max_pt, pos)

        # for center calculating
        self.area_center_sum = vec3_add(self.area_center_sum, pos)
        self.area_center_sum_index += 1

        # get attributes of this point
        vertex = [0] * VERTEX_FLOAT_NUM
        # 1 position
        vertex[0] = pos[0]
        vertex[1] = pos[1]
        vertex[2] = pos[2]

        # angle

        # 2 color
        color = [1.0,0.0,0.0] if linedata[4] == 0.0 else [0.0,1.0,0.0]
        vertex[3] = color[0]
        vertex[4] = color[1]
        vertex[5] = color[2]

        # 3 line number in gcode
        vertex[6] = linedata[5]

        # 4 type id
        vertex[7] = len(self.positions) - 1

        # 5 distance attribute
        vertex[8] = 0  # set after length is calculated

        # 6 set tool knife id
        vertex[9] = linedata[6]

        # 7 feed rate (mm/min)
        is_rapid = linedata[4] == 0.0 or linedata[4] < 0.5
        feed_value = linedata[7] if len(linedata) > 7 else None
        feed = feed_mm_min_for_move(is_rapid, feed_value)
        vertex[10] = feed

        # push this vertex to container
        self.vertices.extend(vertex)
        self.vertex_types.append(1.0 if linedata[4] > 0.5 else 2.0)  # line type[red | green]
        self.raw_linenumbers.append(vertex[6])
        self.raw_feed_rates.append(feed)
        self.angles_of_vertices.append(angle)

    def generate_meshes(self):
        # 0 scale all points to fit largest bbox side in ~2 units
        vertex_count = len(self.positions) // 3
        vertex_float_num = VERTEX_FLOAT_NUM
        if vertex_count == 0:
            self.meshes.clear()
            return
        max_extent = bbox_max_side_length(self.min_pt, self.max_pt)
        self.position_scale = (2.0) if max_extent == 0 else (2.0 / max_extent)
        for i in range(vertex_count):
            self.vertices[vertex_float_num * i + 0] = self.positions[3 * i + 0] * self.position_scale
            self.vertices[vertex_float_num * i + 1] = self.positions[3 * i + 1] * self.position_scale
            self.vertices[vertex_float_num * i + 2] = self.positions[3 * i + 2] * self.position_scale


        # 1 calculate lengths
        self.lengths = [0] * vertex_count
        for i in range(1, vertex_count):
            pos1 = [self.vertices[vertex_float_num * (i - 1) + 0], self.vertices[vertex_float_num * (i - 1) + 1],
                    self.vertices[vertex_float_num * (i - 1) + 2]]
            pos2 = [self.vertices[vertex_float_num * (i) + 0], self.vertices[vertex_float_num * (i) + 1],
                    self.vertices[vertex_float_num * (i) + 2]]

            cur_line_len = vec3_distance(pos1, pos2)
            self.lengths[i] = self.lengths[i - 1] + cur_line_len

        # 2 set distance id
        for i in range(vertex_count):
            self.vertices[vertex_float_num * i + 8] = self.lengths[i]

        self.seg_mesh_vertex_count = 65500
        # 3 construct meshes
        self.meshes.clear()
        mesh_start_id = 0
        mesh_end_id = min(self.seg_mesh_vertex_count, vertex_count)  # not included

        while (True):
            # process each mesh
            indices = []
            for i in range(mesh_end_id - mesh_start_id):
                indices.append(i)
            mesh = [self.vertices[vertex_float_num * mesh_start_id:vertex_float_num * mesh_end_id], indices]

            self.meshes.append(mesh)


            # skip to next mesh
            if mesh_end_id == vertex_count:
                break  # run to end

            # resuse the last mesh vertex to make sure continous lines
            mesh_start_id = mesh_end_id - 1
            mesh_end_id = min(mesh_start_id + self.seg_mesh_vertex_count, vertex_count)

    def add_lines(self, rawlines):
        # parse line

        # 1 check gcode type
        is_4_axis = False
        if (len(rawlines) > 0 and 'A:' in rawlines[0]):
            is_4_axis = True

        if self.is_4_axis is None:
            self.is_4_axis = is_4_axis
        elif self.is_4_axis != is_4_axis:
            print("conflict line type!")

        # 2 parse single line
        for line in rawlines:
            self.parse_line(line.strip())

        self.generate_meshes()

    def add_data_arrs(self, rawdata,is_end=True):
        # parse line

        # 1 check gcode type
        self.is_4_axis = True

        # 2 parse single line
        for linedata in rawdata:
            self.parse_line_data(linedata)

        if is_end:
            self.generate_meshes()

def load_data(lines):

    #X: 0.0 Y: 0.0 Z: 0.0 Color: Green Tool: 1
    def map_color(color_str):
        if color_str == 'Green':
            return [0.,1.,0.]
        elif color_str == 'Red':
            return [1.,0.,0.]
        return [1.,1.,1.]


    vert_center = [0,0,0]

    meshes = []

    is_4_axis = False
    if(len(lines)>0 and 'A:' in lines[0]):
        is_4_axis = True

    print(f"is_4_axis:{is_4_axis}")

    how_many_meshes = len(lines) // MESH_LINE_CHUNK + 1
    line_start = 0
    line_end = min(MESH_LINE_CHUNK, len(lines))
    vertex_id = 0
    original_vertex_id = 0
    scale = 1.0
    positions = []
    max_pt = [float('-inf'), float('-inf'), float('-inf')]
    min_pt = [float('inf'), float('inf'), float('inf')]
    for mesh_id in range(how_many_meshes):
        for line in lines[line_start:line_end]:
            arr_pt = line.split(' ')

            pos = [scale*float(arr_pt[1]),scale*float(arr_pt[3]),scale*float(arr_pt[5])]
            
            #angle
            if(is_4_axis):
                angle = float(arr_pt[7])

                rot_pos = rotate_pt_by_x_axis_angle(pos[0],pos[1],pos[2],angle)

                positions.append(rot_pos[0])
                positions.append(rot_pos[1])
                positions.append(rot_pos[2])
                
                min_pt[0] = min(min_pt[0], rot_pos[0])
                min_pt[1] = min(min_pt[1], rot_pos[1])
                min_pt[2] = min(min_pt[2], rot_pos[2])
                max_pt[0] = max(max_pt[0],rot_pos[0])
                max_pt[1] = max(max_pt[1],rot_pos[1])
                max_pt[2] = max(max_pt[2],rot_pos[2])
            else:
                positions.append(pos[0])
                positions.append(pos[1])
                positions.append(pos[2])

                    
                min_pt[0] = min(min_pt[0], pos[0])
                min_pt[1] = min(min_pt[1], pos[1])
                min_pt[2] = min(min_pt[2], pos[2])
                max_pt[0] = max(max_pt[0],pos[0])
                max_pt[1] = max(max_pt[1],pos[1])
                max_pt[2] = max(max_pt[2],pos[2])

        line_start = line_end - 1  # overlap by one line between consecutive meshes
        line_end = min(line_start + MESH_LINE_CHUNK, len(lines))

    vertices_count = int(len(positions)/3)
    print(vertices_count)
    #apply scale to 2
    max_point = bbox_max_side_length(min_pt, max_pt)
    scale_invert = (2.0) if max_point == 0 else (2.0 / max_point)
    for i in range(len(positions)):
        positions[i] *= scale_invert

    for i in range(vertices_count):
        vert_center[0] += positions[3*i+0]
        vert_center[1] += positions[3*i+1]
        vert_center[2] += positions[3*i+2]
    #calculate the center of target
    vert_center[0] /= vertices_count
    vert_center[1] /= vertices_count
    vert_center[2] /= vertices_count

    #calculate the distance field
    lengths = [0]*vertices_count
    for i in range(1,vertices_count):
        pos1 = [positions[3*i-3],positions[3*i-2],positions[3*i-1]]
        pos2 = [positions[3*i],positions[3*i+1],positions[3*i+2]]
        cur_line_len = len_3d(pos1,pos2)
        lengths[i] = lengths[i-1] + cur_line_len

    line_start = 0
    line_end = min(MESH_LINE_CHUNK, len(lines))
    angles_of_vertices = []
    raw_linenumbers = []
    raw_feed_rates = []
    vertex_types = []
    for mesh_id in range(how_many_meshes):

        vertices = []
        indices = []
        has_last_vertex = False
        last_vertex = []
        #start indices
        index = 0
        for line in lines[line_start:line_end]:


            arr_pt = line.split(' ')

            #pos
            pos = [scale_invert*float(arr_pt[1]),scale_invert*float(arr_pt[3]),scale_invert*float(arr_pt[5])]
            
            #angle
            if(is_4_axis):
                angle = float(arr_pt[7])

                rot_pos = rotate_pt_by_x_axis_angle(pos[0],pos[1],pos[2],angle)

                this_vertex = []
                this_vertex.append(rot_pos[0])
                this_vertex.append(rot_pos[1])
                this_vertex.append(rot_pos[2])

                #color
                color = map_color(arr_pt[9])
                this_vertex.append(color[0]) #3
                this_vertex.append(color[1]) #4
                this_vertex.append(color[2]) #5
                #line number
                this_vertex.append(float(arr_pt[11])) #6
                #type id
                this_vertex.append(float(vertex_id)) #7

                #distance id
                raw_linenumbers.append(float(arr_pt[11]))
                this_vertex.append(lengths[original_vertex_id]) #8

                #store data in memery
                vertex_types.append(1.0 if arr_pt[9]=="Green" else 2.0) #line type[red | green]
                angles_of_vertices.append(angle)

                
                #vertex tool
                this_vertex.append(float(arr_pt[13])) #9
                feed = feed_mm_min_for_move(arr_pt[9] == "Red")
                this_vertex.append(feed) #10
                raw_feed_rates.append(feed)

                vertices += this_vertex
            else:
                vertices.append(pos[0])
                vertices.append(pos[1])
                vertices.append(pos[2])

                #color
                color = map_color(arr_pt[7])
                vertices.append(color[0])
                vertices.append(color[1])
                vertices.append(color[2])
                #line number
                vertices.append(float(arr_pt[9]))
                #type id
                vertices.append(float(vertex_id))

                #distance id
                raw_linenumbers.append(float(arr_pt[9]))
                vertices.append(lengths[vertex_id])

                #store data in memery
                vertex_types.append(1.0 if arr_pt[7]=="Green" else 2.0) #line type[red | green]

                
                #vertextool
                vertices.append(float(arr_pt[11]))
                feed = feed_mm_min_for_move(arr_pt[7] == "Red")
                vertices.append(feed)
                raw_feed_rates.append(feed)

            indices.append(index)
            index = index + 1
            vertex_id = vertex_id + 1
            original_vertex_id += 1

        line_start = line_end - 1  # overlap by one line between consecutive meshes
        line_end = min(line_start + MESH_LINE_CHUNK, len(lines))

        meshes.append([vertices,indices])
    print("mesh count: %d"%len(meshes))

    return [is_4_axis,meshes,vert_center,len(lines),lengths,vertex_types,raw_linenumbers,raw_feed_rates,positions,scale_invert,angles_of_vertices,min_pt,max_pt]



def frame_call_back_test(distance,num):
    print(f'Current line: {num}')

class GCodeViewer(Widget):
    axis = (0,0,1)
    angle = 0

    three_axis_mode = True
        
    g_old_curosr = [0,0]
    g_cursor = [0,0]
    left_button_down = False
    middle_button_down = False
    right_button_down = False
    g_wheel_data = 0
    lines_center = [0,0,0]

    display_count = 0
    total_line_count = 0
    add_dir = 1
    dynamic_display = True
    move_speed = 0.8
    move_scale = 1.0
    move_scale_by_positon = 1.0

    # Clear cached mesh data before loading a new file
    clear_before_new_load = False

    # When True, compute segment-based time estimates (distance/feed); when False, skip extra parsing.
    high_precision_time_estimate = True

    line_times = []
    total_time = 0.0
    lengths = []
    raw_linenumbers = []
    raw_positions = []
    raw_feed_rates = []
    frame_callback = None
    time_estimate_progress_callback = None
    log_callback = None
    error_popup_callback = None

    #camera
    m_xRot = 30
    m_yRot = 180

    m_xRotTarget = 90
    m_yRotTarget = 0

    m_zoom = DEFAULT_ZOOM

    m_xPan = 0
    m_yPan = 0
    m_xLastRot = 30
    m_yLastRot = 180
    m_xLastPan = 0
    m_yLastPan = 0
    m_lastPos = [0, 0]
    m_distance = 10

    m_xLookAt = 0
    m_yLookAt = 0
    m_zLookAt = 0

    m_xMin = 0
    m_xMax = 0
    m_yMin = 0
    m_yMax = 0
    m_zMin = 0
    m_zMax = 0
    m_xSize = 0
    m_ySize = 0
    m_zSize = 0

    off_x = 0
    off_y = 0

    orbit = True
    _grid_visible = True
    color_scheme = COLOR_SCHEME_BY_TYPE
    feed_min = 0.0
    feed_max = DEFAULT_FEED_MM_MIN

    def __init__(self):
        super().__init__()
        self.canvas = RenderContext()
        shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')

        self.gridmesh = RenderContext()
        self.gridmesh.shader.source = os.path.join(shader_dir, 'grid.glsl')
        self._setup_grid_quad()

        self.linemesh = RenderContext()
        self.linemesh.shader.source = os.path.join(shader_dir, 'toolpath.glsl')

        self.pointermesh = RenderContext()
        self.pointermesh.shader.source = os.path.join(shader_dir, 'tool_pointer.glsl')

        axis_shader = os.path.join(shader_dir, 'axis_helper.glsl')
        self.axisxmesh = RenderContext()
        self.axisxmesh.shader.source = axis_shader
        self.axisymesh = RenderContext()
        self.axisymesh.shader.source = axis_shader
        self.axiszmesh = RenderContext()
        self.axiszmesh.shader.source = axis_shader


        self.meshmanager = MeshManager()

        # Dirty flags: set True whenever the scene must be re-rendered.
        # _scene_dirty covers view/pointer/axis uniform changes; _proj_dirty
        # covers the projection matrix (zoom, pan, resize).
        self._scene_dirty = True
        self._proj_dirty = True

        # Pre-computed constant matrices reused every frame to avoid per-frame
        self._identity_mat = Matrix()
        self._axis_y_rot = Matrix().rotate(0.5 * math.pi, 1, 0, 0)
        self._axis_z_rot = Matrix().rotate(-0.5 * math.pi, 0, 0, 1)
        self._proj_matrix = Matrix()
        self.m_viewMatrix = Matrix()
        self._grid_visible = True
        self._viewer_meshes_active = False

        self.bind(size=self._on_size_change, pos=self._on_size_change)

        self._apply_color_scheme_uniform()
        self._update_feed_range_uniforms()
        Clock.schedule_interval(self._on_frame_tick, 1/60)


    def _on_size_change(self, *args):
        self._proj_dirty = True
        self._scene_dirty = True

    def _get_line_vertex_fmt(self):
        return [
            (b'position', 3, 'float'),
            (b'color_att', 3, 'float'),
            (b'type', 1, 'float'),
            (b'vertex_id', 1, 'float'),
            (b'distance_id', 1, 'float'),
            (b'vertex_tool', 1, 'float'),
            (b'vertex_feed', 1, 'float'),
        ]

    def _get_grid_vertex_fmt(self):
        return [(b'position', 3, 'float')]

    def _setup_grid_quad(self):
        """Single quad on the XY plane; grid lines are drawn in the fragment shader."""
        verts = [-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, -0.5, 0.5, 0.0, 0.5, 0.5, 0.0]
        indices = [0, 1, 2, 2, 1, 3]
        self.gridmesh.clear()
        with self.gridmesh:
            self.cb = Callback(self.setup_gl_context)
            Mesh(
                fmt=self._get_grid_vertex_fmt(),
                vertices=verts,
                indices=indices,
                mode='triangles',
            )
            self.cb = Callback(None)

    def _add_canvas_children(self):
        self.canvas.add(self.gridmesh)
        self.canvas.add(self.linemesh)
        self.canvas.add(self.pointermesh)
        self.canvas.add(self.axisxmesh)
        self.canvas.add(self.axisymesh)
        self.canvas.add(self.axiszmesh)
        self._viewer_meshes_active = True

    def _grid_quad_extent(self):
        """World-space quad width so the plane covers the viewport when orbiting."""
        asp = self.size[0] / max(self.size[1], 1.0)
        return max(self.m_distance * self.m_zoom * max(asp, 1.0) * 4.0, GRID_QUAD_MIN_SIZE)

    def _update_grid_uniforms(self):
        scale = self.move_scale_by_positon if self.move_scale_by_positon else 1.0
        center = getattr(self, 'lines_center', [0.0, 0.0, 0.0])
        self.gridmesh['center_offset'] = Matrix().translate(-center[0], -center[1], -center[2])
        self.gridmesh['view_mat'] = self.m_viewMatrix
        self.gridmesh['grid_visible'] = 1.0 if self._grid_visible else 0.0
        self.gridmesh['grid_size'] = float(self._grid_quad_extent())
        self.gridmesh['subcell_size'] = float(GRID_STEP_MM * scale)
        self.gridmesh['cell_size'] = float(GRID_MAJOR_STEP_MM * scale)
        self.gridmesh['color_minor'] = GRID_COLOR_MINOR
        self.gridmesh['color_major'] = GRID_COLOR_MAJOR
        self.gridmesh['color_axis_x'] = AXIS_COLOR_X
        self.gridmesh['color_axis_y'] = AXIS_COLOR_Y

    def clearDisplay(self):
        self.lengths = []
        self._cannot_visualise = False
        self.vertex_types = []
        self.positions = []
        self.line_times = []
        self.total_time = 0.0
        self.raw_feed_rates = []
        self.linemesh.clear()
        self.canvas.remove(self.linemesh)
        self.canvas.remove(self.gridmesh)
        self.canvas.remove(self.pointermesh)
        self.pointermesh.clear()
        self.canvas.remove(self.axisxmesh)
        self.axisxmesh.clear()
        self.canvas.remove(self.axisymesh)
        self.axisymesh.clear()
        self.canvas.remove(self.axiszmesh)
        self.axiszmesh.clear()
        self.display_count = 0
        self._viewer_meshes_active = False

    def set_frame_callback(self, framecallback):
        self.frame_callback = framecallback


    def set_error_popup_callback(self, callback):
        """Set callback(message) to show error in UI (e.g. load_error popup). Called when gcode cannot be visualised."""
        self.error_popup_callback = callback

    def set_play_over_callback(self, playovercallback):
        self.play_over_callback = playovercallback

    def load_mesh_manager(self,lines):
        self.clearDisplay()
        self._add_canvas_children()

        self.meshmanager.add_lines(lines)

        ff = self._get_line_vertex_fmt()

        self.lengths = self.meshmanager.lengths
        self.vertex_types = self.meshmanager.vertex_types
        self.positions = self.meshmanager.positions
        self.raw_positions = self.meshmanager.raw_positions
        self.raw_linenumbers = self.meshmanager.raw_linenumbers
        self.angles_of_vertices = self.meshmanager.angles_of_vertices

        self.total_line_count = self.meshmanager.get_pt_count()
        self.total_distance = self.meshmanager.lengths[-1]
        self.move_scale_by_positon = self.meshmanager.position_scale


        lines.clear()

        self.is_4_axis = self.meshmanager.is_4_axis

        obj1 = 'pointer.obj'
        obj2 = 'axis.obj'
        if not os.path.exists(obj1):
            obj1 = os.path.join(os.path.dirname(__file__), obj1)
            obj2 = os.path.join(os.path.dirname(__file__), obj2)

        self.pointer = ObjFile(obj1)
        self.axis_obj = ObjFile(obj2)


        # 4-axis: rotate toolhead mesh instead of the toolpath
        self.rotate_line_or_knife = False
        if (self.is_4_axis):
            self.rotate_line_or_knife = True


        with self.canvas:
            with self.linemesh:
                self.cb = Callback(self.setup_gl_context)
                for mesh in self.meshmanager.meshes:
                    Mesh(fmt=ff, vertices=mesh[0], indices=mesh[1], mode='line_strip')

                self.cb = Callback(None)

            with self.pointermesh:
                self.cb = Callback(None)
                m = list(self.pointer.objects.values())[0]
                self.mesh = Mesh(
                    vertices=m.vertices,
                    indices=m.indices,
                    fmt=m.vertex_format,
                    mode='triangles',
                )
                self.cb = Callback(None)

            # axis
            with self.axisxmesh:
                self.cb = Callback(None)
                m = list(self.axis_obj.objects.values())[0]
                self.mesh = Mesh(
                    vertices=m.vertices,
                    indices=m.indices,
                    fmt=m.vertex_format,
                    mode='triangles',
                )
                self.cb = Callback(None)
            with self.axisymesh:
                self.cb = Callback(None)
                m = list(self.axis_obj.objects.values())[0]
                self.mesh = Mesh(
                    vertices=m.vertices,
                    indices=m.indices,
                    fmt=m.vertex_format,
                    mode='triangles',
                )
                self.cb = Callback(None)
            with self.axiszmesh:
                self.cb = Callback(None)
                m = list(self.axis_obj.objects.values())[0]
                self.mesh = Mesh(
                    vertices=m.vertices,
                    indices=m.indices,
                    fmt=m.vertex_format,
                    mode='triangles',
                )
                self.cb = Callback(self.reset_gl_context)

        self.lines_center = self.meshmanager.get_center_of_view()
        self.linemesh['center_offset'] = Matrix().translate(-self.lines_center[0], -self.lines_center[1],
                                                              -self.lines_center[2])

        # rendering line meshes
        self.linemesh['display_count'] = -1.0
        # 0 means display all thing
        self.linemesh['vertex_type_display'] = 0.0

        self.pointermesh['offset'] = (-self.lines_center[0], -self.lines_center[1], -self.lines_center[2])

        self.m_zoom = DEFAULT_ZOOM
        self.update_proj()
        self.update_view()
        self._scene_dirty = True
        #force update
        self.canvas.ask_update()


    def clear_loaded_memery(self):
        if self.clear_before_new_load:
            self.clear_before_new_load = False

            self.meshmanager.clear()


    def load_array(self, tmpdataarrs, is_end=True):
        self.clear_loaded_memery()

        dataarrs = []
        # Insert bridging segments when feed/rapid colors change
        last_color = -1
        last_line = -1
        for line in tmpdataarrs:

            color = line[4]

            need_regenerate = False
            if (color >= 0 and last_color >= 0):
                if (color != last_color):
                    need_regenerate = True

            if (need_regenerate):
                replace_str = last_color
                copyline = line.copy()
                copyline[4] = last_color

                dataarrs.append(copyline)
                dataarrs.append(line)

            else:
                dataarrs.append(line)

            last_line = line
            last_color = color


        if is_end:
            self.clear_before_new_load = True
            self.clearDisplay()

            self._add_canvas_children()


        self.meshmanager.add_data_arrs(dataarrs,is_end)

        if is_end:
            ff = self._get_line_vertex_fmt()

            self.lengths = self.meshmanager.lengths
            self.vertex_types = self.meshmanager.vertex_types
            self.positions = self.meshmanager.positions
            self.raw_positions = self.meshmanager.raw_positions
            self.raw_linenumbers = self.meshmanager.raw_linenumbers
            self.raw_feed_rates = self.meshmanager.raw_feed_rates
            self.angles_of_vertices = self.meshmanager.angles_of_vertices

            self.total_line_count = self.meshmanager.get_pt_count()
            self.total_distance = self.meshmanager.lengths[-1]
            self.move_scale_by_positon = self.meshmanager.position_scale

            self.is_4_axis = self.meshmanager.is_4_axis

            # Compute per-segment durations from travel distance and feed rate (for time estimate)
            if self.high_precision_time_estimate and len(self.raw_feed_rates) >= len(self.raw_linenumbers or []):
                self._compute_line_times_async()

            self._update_feed_range_uniforms()

            obj1 = 'pointer.obj'
            obj2 = 'axis.obj'
            if not os.path.exists(obj1):
                obj1 = os.path.join(os.path.dirname(__file__), obj1)
                obj2 = os.path.join(os.path.dirname(__file__), obj2)

            self.pointer = ObjFile(obj1)
            self.axis_obj = ObjFile(obj2)


            # 4-axis: rotate toolhead mesh instead of the toolpath
            self.rotate_line_or_knife = False
            if (self.is_4_axis):
                self.rotate_line_or_knife = True


            with self.canvas:
                with self.linemesh:
                    self.cb = Callback(self.setup_gl_context)
                    for mesh in self.meshmanager.meshes:
                        Mesh(fmt=ff, vertices=mesh[0], indices=mesh[1], mode='line_strip')

                    self.cb = Callback(None)

                with self.pointermesh:
                    self.cb = Callback(None)
                    m = list(self.pointer.objects.values())[0]
                    self.mesh = Mesh(
                        vertices=m.vertices,
                        indices=m.indices,
                        fmt=m.vertex_format,
                        mode='triangles',
                    )
                    self.cb = Callback(None)

                # axis
                with self.axisxmesh:
                    self.cb = Callback(None)
                    m = list(self.axis_obj.objects.values())[0]
                    self.mesh = Mesh(
                        vertices=m.vertices,
                        indices=m.indices,
                        fmt=m.vertex_format,
                        mode='triangles',
                    )
                    self.cb = Callback(None)
                with self.axisymesh:
                    self.cb = Callback(None)
                    m = list(self.axis_obj.objects.values())[0]
                    self.mesh = Mesh(
                        vertices=m.vertices,
                        indices=m.indices,
                        fmt=m.vertex_format,
                        mode='triangles',
                    )
                    self.cb = Callback(None)
                with self.axiszmesh:
                    self.cb = Callback(None)
                    m = list(self.axis_obj.objects.values())[0]
                    self.mesh = Mesh(
                        vertices=m.vertices,
                        indices=m.indices,
                        fmt=m.vertex_format,
                        mode='triangles',
                    )
                    self.cb = Callback(self.reset_gl_context)


            self.lines_center = self.meshmanager.get_center_of_view()
            self.linemesh['center_offset'] = Matrix().translate(-self.lines_center[0], -self.lines_center[1],
                                                                  -self.lines_center[2])

            # rendering line meshes
            self.linemesh['display_count'] = -1.0
            # 0 means display all thing
            self.linemesh['vertex_type_display'] = 0.0

            self.pointermesh['offset'] = (-self.lines_center[0], -self.lines_center[1], -self.lines_center[2])

            self.m_zoom = DEFAULT_ZOOM
            self.update_proj()
            self.update_view()
            self._scene_dirty = True
            #force update
            self.canvas.ask_update()



    def load_1data_display(self,lines):

        #X: 0.0 Y: 0.0 Z: 0.0 Color: Green Tool: 1
        def map_color(color_str):
            if color_str == 'Green':
                return [0.,1.,0.]
            elif color_str == 'Red':
                return [1.,0.,0.]
            return [1.,1.,1.]


        vert_center = [0,0,0]

        meshes = []

        is_4_axis = False
        if(len(lines)>0 and 'A:' in lines[0]):
            is_4_axis = True

        print(f"is_4_axis:{is_4_axis}")

        how_many_meshes = len(lines) // MESH_LINE_CHUNK + 1
        line_start = 0
        line_end = min(MESH_LINE_CHUNK, len(lines))
        original_vertex_id = 0
        scale = 1.0
        positions = []
        max_pt = [float('-inf'), float('-inf'), float('-inf')]
        min_pt = [float('inf'), float('inf'), float('inf')]
        for mesh_id in range(how_many_meshes):
            for line in lines[line_start:line_end]:
                arr_pt = line.split(' ')

                pos = [scale*float(arr_pt[1]),scale*float(arr_pt[3]),scale*float(arr_pt[5])]
                
                #angle
                if(is_4_axis):
                    angle = float(arr_pt[7])

                    rot_pos = rotate_pt_by_x_axis_angle(pos[0],pos[1],pos[2],angle)

                    positions.append(rot_pos[0])
                    positions.append(rot_pos[1])
                    positions.append(rot_pos[2])
                    
                    min_pt[0] = min(min_pt[0], rot_pos[0])
                    min_pt[1] = min(min_pt[1], rot_pos[1])
                    min_pt[2] = min(min_pt[2], rot_pos[2])
                    max_pt[0] = max(max_pt[0],rot_pos[0])
                    max_pt[1] = max(max_pt[1],rot_pos[1])
                    max_pt[2] = max(max_pt[2],rot_pos[2])
                else:
                    positions.append(pos[0])
                    positions.append(pos[1])
                    positions.append(pos[2])

                        
                    min_pt[0] = min(min_pt[0], pos[0])
                    min_pt[1] = min(min_pt[1], pos[1])
                    min_pt[2] = min(min_pt[2], pos[2])
                    max_pt[0] = max(max_pt[0],pos[0])
                    max_pt[1] = max(max_pt[1],pos[1])
                    max_pt[2] = max(max_pt[2],pos[2])

            line_start = line_end - 1  # overlap by one line between consecutive meshes
            line_end = min(line_start + MESH_LINE_CHUNK, len(lines))

        vertices_count = int(len(positions)/3)
        print(vertices_count)
        #apply scale to 2
        max_point = bbox_max_side_length(min_pt, max_pt)
        scale_invert = (2.0) if max_point == 0 else (2.0 / max_point)
        for i in range(len(positions)):
            positions[i] *= scale_invert

        for i in range(vertices_count):
            vert_center[0] += positions[3*i+0]
            vert_center[1] += positions[3*i+1]
            vert_center[2] += positions[3*i+2]
        #calculate the center of target
        vert_center[0] /= vertices_count
        vert_center[1] /= vertices_count
        vert_center[2] /= vertices_count

        #calculate the distance field

        lengths = [0]*vertices_count
        for i in range(0,vertices_count):
            if i == 0:
                if len(self.positions) > 0:
                    pos1 = [self.positions[-3],self.positions[-2],self.positions[-1]]
                    pos2 = [self.positions[3*i],self.positions[3*i+1],self.positions[3*i+2]]
                    cur_line_len = len_3d(pos1, pos2)
                    lengths[i] = self.lengths[-1] + cur_line_len

            else:
                pos1 = [positions[3*i-3],positions[3*i-2],positions[3*i-1]]
                pos2 = [positions[3*i],positions[3*i+1],positions[3*i+2]]
                cur_line_len = len_3d(pos1,pos2)
                lengths[i] = lengths[i-1] + cur_line_len


        

        line_start = 0
        line_end = min(MESH_LINE_CHUNK, len(lines))
        angles_of_vertices = []
        raw_linenumbers = []
        raw_feed_rates = []
        vertex_types = []
        for mesh_id in range(how_many_meshes):

            vertices = []
            indices = []
            has_last_vertex = False
            last_vertex = []
            #start indices
            index = 0
            for line in lines[line_start:line_end]:


                arr_pt = line.split(' ')

                #pos
                pos = [scale_invert*float(arr_pt[1]),scale_invert*float(arr_pt[3]),scale_invert*float(arr_pt[5])]
                
                #angle
                if(is_4_axis):
                    angle = float(arr_pt[7])

                    rot_pos = rotate_pt_by_x_axis_angle(pos[0],pos[1],pos[2],angle)

                    this_vertex = []
                    this_vertex.append(rot_pos[0])
                    this_vertex.append(rot_pos[1])
                    this_vertex.append(rot_pos[2])

                    #color
                    color = map_color(arr_pt[9])
                    this_vertex.append(color[0]) #3
                    this_vertex.append(color[1]) #4
                    this_vertex.append(color[2]) #5
                    #line number
                    this_vertex.append(float(arr_pt[11])) #6
                    #type id
                    this_vertex.append(float(self.vertex_id)) #7

                    #distance id
                    raw_linenumbers.append(float(arr_pt[11]))
                    this_vertex.append(lengths[original_vertex_id]) #8

                    #store data in memery
                    vertex_types.append(1.0 if arr_pt[9]=="Green" else 2.0) #line type[red | green]
                    angles_of_vertices.append(angle)

                    
                    #vertex tool
                    this_vertex.append(float(arr_pt[13])) #9
                    feed = feed_mm_min_for_move(arr_pt[9] == "Red")
                    this_vertex.append(feed)
                    raw_feed_rates.append(feed)

                    vertices += this_vertex
                else:
                    vertices.append(pos[0])
                    vertices.append(pos[1])
                    vertices.append(pos[2])

                    #color
                    color = map_color(arr_pt[7])
                    vertices.append(color[0])
                    vertices.append(color[1])
                    vertices.append(color[2])
                    #line number
                    vertices.append(float(arr_pt[9]))
                    #type id
                    vertices.append(float(self.vertex_id))

                    #distance id
                    raw_linenumbers.append(float(arr_pt[9]))
                    vertices.append(lengths[original_vertex_id])

                    #store data in memery
                    vertex_types.append(1.0 if arr_pt[7]=="Green" else 2.0) #line type[red | green]

                    
                    #vertextool
                    vertices.append(float(arr_pt[11]))
                    feed = feed_mm_min_for_move(arr_pt[7] == "Red")
                    vertices.append(feed)
                    raw_feed_rates.append(feed)

                indices.append(index)
                index = index + 1
                self.vertex_id = self.vertex_id + 1
                original_vertex_id += 1

            line_start = line_end - 1  # overlap by one line between consecutive meshes
            line_end = min(line_start + MESH_LINE_CHUNK, len(lines))

            meshes.append([vertices,indices])
        print("mesh count: %d"%len(meshes))

        return [is_4_axis,meshes,vert_center,len(lines),lengths,vertex_types,raw_linenumbers,raw_feed_rates,positions,scale_invert,angles_of_vertices]


    def load_with_display(self,tmplines):
        self.clearDisplay()

        self._add_canvas_children()
        self.raw_feed_rates = []

        last_color = ''
        last_line = ''
        ff = self._get_line_vertex_fmt()
        #every 65535
        meshes = []
        lines = []
        current_processed_count = 0
        self.vertex_id = 0
        for i in range(len(tmplines)):

            #process line varibles
            thisline = tmplines[i]
            color = ''
            if 'Green' in thisline:
                color = 'Green'
            elif 'Red' in thisline:
                color = 'Red'

            #process
            need_regenerate = False
            if(len(color) > 0 and len(last_color)>0):
                if(color != last_color):
                    need_regenerate = True

                

            # else:

            lines.append(thisline)

            if len(lines) == MESH_LINE_CHUNK:
                #create mesh and refresh display
                [is_4_axis,mmeshes,vert_center,total_line_count,lengths,vertex_types,raw_linenumbers,raw_feed_rates,positions,position_scale,angles_of_vertices] = \
                    self.load_1data_display(lines)


                if len(self.positions) == 0:
                    self.lines_center = vert_center
                else:
                    total_pt_count = len(self.positions)
                    this__pt_count = len(positions)
                    total_sum_position = [ self.lines_center[0] * total_pt_count + vert_center[0] * this__pt_count,\
                                        self.lines_center[1]  * total_pt_count + vert_center[1] * this__pt_count, \
                                        self.lines_center[2]  * total_pt_count + vert_center[2] * this__pt_count]
                    
                    total_pt_count += this__pt_count
                    self.lines_center = [total_sum_position[0]/total_pt_count,total_sum_position[1]/total_pt_count,total_sum_position[2]/total_pt_count]
            

                print(vert_center)
                print(self.lines_center)
                self.lengths += lengths
                self.vertex_types += vertex_types
                self.positions += positions
                self.raw_feed_rates += raw_feed_rates
                lines.clear()

                self.is_4_axis = is_4_axis
                with self.canvas:
                    with self.linemesh:
                        for mesh in mmeshes:
                            Mesh(fmt=ff, vertices=mesh[0], indices=mesh[1], mode='line_strip')
                        
                    
                self.linemesh['center_offset'] = Matrix().translate(-self.lines_center[0],-self.lines_center[1],-self.lines_center[2])


                #rendering line meshes
                self.linemesh['display_count'] = -1.0
                #0 means display all thing
                self.linemesh['vertex_type_display'] = 0.0

                #rendering pointer
                self.pointermesh['offset'] = (-self.lines_center[0],-self.lines_center[1],-self.lines_center[2])

                self.update_proj()
                self.update_view()
                self._scene_dirty = True

        # Process any lines left in the final partial batch
        if len(lines)>0:
            #create mesh and refresh display
            [is_4_axis,mmeshes,vert_center,total_line_count,lengths,vertex_types,raw_linenumbers,raw_feed_rates,positions,position_scale,angles_of_vertices] = \
                self.load_1data_display(lines)

            
            if len(self.positions) == 0:
                self.lines_center = vert_center
            else:
                total_pt_count = len(self.positions)
                this__pt_count = len(positions)
                total_sum_position = [ self.lines_center[0] * total_pt_count + vert_center[0] * this__pt_count,\
                                      self.lines_center[1]  * total_pt_count + vert_center[1] * this__pt_count, \
                                      self.lines_center[2]  * total_pt_count + vert_center[2] * this__pt_count]
                
                total_pt_count += this__pt_count
                self.lines_center = [total_sum_position[0]/total_pt_count,total_sum_position[1]/total_pt_count,total_sum_position[2]/total_pt_count]
            
            self.lengths += lengths

            self.vertex_types += vertex_types
            self.positions += positions
            self.raw_feed_rates += raw_feed_rates
            lines.clear()

            self.is_4_axis = is_4_axis
            with self.canvas:
                
                with self.linemesh:
                    self.cb = Callback(self.setup_gl_context)
                    for mesh in mmeshes:
                        Mesh(fmt=ff, vertices=mesh[0], indices=mesh[1], mode='line_strip')
                    
                    self.cb = Callback(None)
                
            self.linemesh['center_offset'] = Matrix().translate(-self.lines_center[0],-self.lines_center[1],-self.lines_center[2])


            #rendering line meshes
            self.linemesh['display_count'] = -1.0
            #0 means display all thing
            self.linemesh['vertex_type_display'] = 0.0

            #rendering pointer
            self.pointermesh['offset'] = (-self.lines_center[0],-self.lines_center[1],-self.lines_center[2])

            self.update_proj()
            self.update_view()
            self._scene_dirty = True

        self._update_feed_range_uniforms()

    def load(self, tmplines):
        self.clearDisplay()


        # Insert bridging segments when feed/rapid colors change
        lines = []
        last_color = ''
        last_line = ''
        for line in tmplines:
            color = ''
            if 'Green' in line:
                color = 'Green'
            elif 'Red' in line:
                color = 'Red'
            
            need_regenerate = False
            if(len(color) > 0 and len(last_color)>0):
                if(color != last_color):
                    need_regenerate = True
            
            if(need_regenerate):
                replace_str = last_color
                copyline = line.replace(color,last_color)
                
                lines.append(copyline)
                lines.append(line)

            else:
                lines.append(line)

            last_line = line
            last_color = color

        self._add_canvas_children()
        ff = self._get_line_vertex_fmt()
        
        [is_4_axis,meshes,vert_center,total_line_count,lengths,vertex_types,raw_linenumbers,raw_feed_rates,positions,position_scale,angles_of_vertices,min_pt,max_pt] = load_data(lines)
        print("how many meshes:",len(meshes))

        self.positions = positions
        self.lengths = lengths
        self._cannot_visualise = False
        self.raw_linenumbers = raw_linenumbers
        self.raw_feed_rates = raw_feed_rates
        
        self.vertex_types = vertex_types
        self.move_scale_by_positon = position_scale
        self.angles_of_vertices = angles_of_vertices

        self.is_4_axis = is_4_axis
        self.total_line_count = total_line_count
        self.total_distance = lengths[len(lengths) - 1]
        # No raw gcode file in this path; time estimate will use machine value
        self.line_times = []
        self.total_time = 0.0
        self._update_feed_range_uniforms()

        obj1 = 'pointer.obj'
        obj2 = 'axis.obj'
        if not os.path.exists(obj1):
            obj1 = os.path.join(os.path.dirname(__file__), obj1)
            obj2 = os.path.join(os.path.dirname(__file__), obj2)

        self.pointer = ObjFile(obj1)

        self.axis_obj = ObjFile(obj2)

        # 4-axis: rotate toolhead mesh instead of the toolpath
        self.rotate_line_or_knife = False
        if(self.is_4_axis):
            self.rotate_line_or_knife = True

        with self.canvas:
            
            with self.linemesh:
                self.cb = Callback(self.setup_gl_context)
                for mesh in meshes:
                    Mesh(fmt=ff, vertices=mesh[0], indices=mesh[1], mode='line_strip')
                
                self.cb = Callback(None)

            with self.pointermesh:

                self.cb = Callback(None)
                m = list(self.pointer.objects.values())[0]
                self.mesh = Mesh(
                    vertices=m.vertices,
                    indices=m.indices,
                    fmt=m.vertex_format,
                    mode='triangles',
                )
                self.cb = Callback(None)
            
            #axis
            with self.axisxmesh:
                self.cb = Callback(None)
                m = list(self.axis_obj.objects.values())[0]
                self.mesh = Mesh(
                    vertices=m.vertices,
                    indices=m.indices,
                    fmt=m.vertex_format,
                    mode='triangles',
                )
                self.cb = Callback(None)
            with self.axisymesh:
                self.cb = Callback(None)
                m = list(self.axis_obj.objects.values())[0]
                self.mesh = Mesh(
                    vertices=m.vertices,
                    indices=m.indices,
                    fmt=m.vertex_format,
                    mode='triangles',
                )
                self.cb = Callback(None)
            with self.axiszmesh:
                self.cb = Callback(None)
                m = list(self.axis_obj.objects.values())[0]
                self.mesh = Mesh(
                    vertices=m.vertices,
                    indices=m.indices,
                    fmt=m.vertex_format,
                    mode='triangles',
                )
                self.cb = Callback(self.reset_gl_context)
        #move to center
        self.linemesh['center_offset'] = Matrix().translate(-vert_center[0],-vert_center[1],-vert_center[2])


        #rendering line meshes
        self.linemesh['display_count'] = -1.0
        #0 means display all thing
        self.linemesh['vertex_type_display'] = 0.0

        #rendering pointer
        self.pointermesh['offset'] = (-vert_center[0],-vert_center[1],-vert_center[2])
        self.lines_center = vert_center

        self.m_zoom = DEFAULT_ZOOM
        self.update_proj()
        self.update_view()
        self._scene_dirty = True

    def update_proj(self):
        asp = self.size[0] / max(self.size[1], 1.0)
        proj = Matrix()
        zoomidx = self.m_zoom
        proj.view_clip((-0.5 + self.m_xPan) * asp * zoomidx, (0.5 + self.m_xPan) * asp * zoomidx, (-0.5 + self.m_yPan)*zoomidx, (0.5 + self.m_yPan)*zoomidx, 2, self.m_distance * 2,1)
        self._proj_matrix = proj
        self.linemesh['proj_mat'] = proj
        self.gridmesh['proj_mat'] = proj
        self._update_grid_uniforms()
        self.pointermesh['projection_mat'] = proj
        self.axisxmesh['projection_mat'] = proj
        self.axisymesh['projection_mat'] = proj
        self.axiszmesh['projection_mat'] = proj

    def update_view(self):
        r = self.m_distance
        angY = -M_PI / 180.0 * self.m_yRot
        angX = M_PI / 180.0 * self.m_xRot

        eye = (r * math.cos(angX) * math.sin(angY) + self.m_xLookAt, r * math.cos(angX) * math.cos(angY) + self.m_yLookAt, r * math.sin(angX) + self.m_zLookAt)
        
        center = (self.m_xLookAt, self.m_yLookAt, self.m_zLookAt)
        up = (-math.sin(angY + (M_PI if self.m_xRot < 0 else 0)) if abs(self.m_xRot) == 90 else 0, 
            -math.cos(angY + (M_PI if self.m_xRot < 0 else 0)) if abs(self.m_xRot) == 90 else 0,
            math.cos(angX))
        up = normalize(up)
        self.m_viewMatrix=Matrix().look_at(eye[0],eye[1],eye[2], center[0],center[1],center[2],up[0],up[1],up[2])
        self._update_grid_uniforms()


    def setup_gl_context(self, *args):
        glViewport(self.pos[0]+self.off_x,self.pos[1]+self.off_y,self.size[0],self.size[1])
        glEnable(GL_DEPTH_TEST)

    def reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)
        glViewport(0,0,Window.size[0],Window.size[1])
        pass

    #get total segment count
    def get_total_seg_count(self):
        return self.total_line_count

    #get max distance
    def get_total_distance(self):
        return self.lengths[len(self.lengths)-1]

    #set display offset
    def set_display_offset(self,offx,offy):
        self.off_x = offx
        self.off_y = offy
        self._scene_dirty = True

    #set displaying limit
    def set_pos_by_distance(self, distance):
        if distance > self.get_total_distance():
            print("distance is out of bounds")
            return
        self.display_count = float(distance)
        self._scene_dirty = True
        # Sync cur_line_index to display_count so get_cur_pos_index() returns the correct line
        if self.lengths:
            cur_display_distance = float(self.display_count)
            line_index = binary_find_left(self.lengths, cur_display_distance)
            line_ratio = 0.0
            if line_index < len(self.lengths) - 1 and self.lengths[line_index + 1] > self.lengths[line_index]:
                line_ratio = (cur_display_distance - self.lengths[line_index]) / (self.lengths[line_index + 1] - self.lengths[line_index])
            self.cur_line_index = line_index + line_ratio
        # Trigger frame callback to update line highlighting
        if self.frame_callback is not None:
            cur_distance, linenumber = self.get_cur_pos_index()
            self.frame_callback(cur_distance, linenumber)

    def _report_time_estimate_progress(self, state, percent):
        """Call the progress callback on the main thread (call from worker via Clock.schedule_once)."""
        if self.time_estimate_progress_callback is not None:
            self.time_estimate_progress_callback(state, percent)

    def _apply_line_times_result(self, line_times):
        """Apply worker result on main thread."""
        self.line_times = line_times if line_times else []
        self.total_time = self.line_times[-1] if self.line_times else 0.0
        self._report_time_estimate_progress('done', 100)

    def _compute_line_times_async(self):
        """
        Compute cumulative time (seconds) in a background thread so the UI stays responsive.
        Uses raw_feed_rates from the CNC parser (no file I/O). Shows progress via
        time_estimate_progress_callback if set ('start', 'progress', 'done').
        """
        self.line_times = []
        self.total_time = 0.0
        n = len(self.raw_linenumbers) if self.raw_linenumbers else 0
        if n < 2 or not self.raw_positions or len(self.raw_positions) < n * 3:
            return
        if not self.raw_feed_rates or len(self.raw_feed_rates) < n:
            return
        raw_positions = list(self.raw_positions)
        raw_linenumbers = list(self.raw_linenumbers)
        raw_feed_rates = list(self.raw_feed_rates)
        viewer = self
        PROGRESS_INTERVAL = 100

        def report(state, percent):
            Clock.schedule_once(lambda dt: viewer._report_time_estimate_progress(state, percent), 0)

        def worker():
            line_times = _compute_line_times_worker(
                raw_positions, raw_linenumbers, raw_feed_rates,
                lambda pct: report('progress', pct), PROGRESS_INTERVAL)
            Clock.schedule_once(lambda dt: viewer._apply_line_times_result(line_times), 0)

        report('start', 0)
        threading.Thread(target=worker, daemon=True).start()

    def _compute_line_times(self):
        """
        Compute cumulative time (seconds) at each vertex from segment distance and
        feed rate from CNC parser (raw_feed_rates). Sets self.line_times and self.total_time.
        (Synchronous fallback; normal path uses _compute_line_times_async.)
        """
        self.line_times = []
        self.total_time = 0.0
        n = len(self.raw_linenumbers) if self.raw_linenumbers else 0
        if n < 2 or not self.raw_positions or len(self.raw_positions) < n * 3:
            return
        if not self.raw_feed_rates or len(self.raw_feed_rates) < n:
            return
        result = _compute_line_times_worker(
            self.raw_positions, self.raw_linenumbers, self.raw_feed_rates, None, 0)
        self.line_times = result
        self.total_time = self.line_times[-1] if self.line_times else 0.0

    def get_elapsed_time_by_distance(self, distance):
        """
        Return elapsed time (seconds) at the given display distance.
        Returns None if line_times are not available.
        """
        if not self.line_times or not self.lengths:
            return None
        if distance <= 0:
            return 0.0
        total_dist = self.lengths[-1]
        if total_dist <= 0 or distance >= total_dist:
            return self.total_time
        n = len(self.lengths)
        for i in range(n - 1):
            if self.lengths[i] <= distance <= self.lengths[i + 1]:
                seg_len = self.lengths[i + 1] - self.lengths[i]
                if seg_len <= 0:
                    return self.line_times[i] if i < len(self.line_times) else None
                fraction = (distance - self.lengths[i]) / seg_len
                t0 = self.line_times[i] if i < len(self.line_times) else 0.0
                t1 = self.line_times[i + 1] if i + 1 < len(self.line_times) else t0
                return t0 + fraction * (t1 - t0)
        return None

    def get_remaining_time_by_lineidx(self, line_number, ratio=0.5):
        """
        Return estimated remaining time (seconds) from the given line number.
        Uses distance/feed-based estimate when line_times are available.
        Returns None to fall back to machine estimate.
        """
        if not self.line_times or self.total_time <= 0:
            return None
        distance = self.get_distance_by_lineidx(line_number, ratio)
        if distance is None:
            return None
        elapsed = self.get_elapsed_time_by_distance(distance)
        if elapsed is None:
            return None
        return max(0.0, self.total_time - elapsed)

    def get_distance_by_lineidx(self,lineidx,ratio):
        # Validate that we have the necessary data
        if not self.raw_linenumbers or not self.lengths:
            return None

        left_pos = binary_find_left(self.raw_linenumbers,lineidx)
        while(left_pos>0 and self.raw_linenumbers[left_pos-1] == lineidx):
            left_pos = left_pos - 1

        right_pos = left_pos
        while (right_pos<len(self.raw_linenumbers)-1 and self.raw_linenumbers[right_pos+1] == lineidx):
            right_pos = right_pos + 1
        #skip to next pos(lineidx+1)
        right_pos = right_pos + 1
        
        # Ensure bounds are valid since not all lines are movements
        if left_pos >= len(self.lengths):
            left_pos = len(self.lengths) - 1
        if right_pos >= len(self.lengths):
            right_pos = len(self.lengths) - 1
        if left_pos < 0:
            left_pos = 0
        if right_pos < 0:
            right_pos = 0
        
        # Ensure we have valid indices
        if left_pos >= len(self.lengths) or right_pos >= len(self.lengths):
            return None
            
        #start point
        start_distance = self.lengths[left_pos]
        end_distance = self.lengths[right_pos]

        return start_distance*(1.0 - ratio) + end_distance * ratio

    def set_distance_by_lineidx(self,lineidx,ratio):
        # Validate that we have the necessary data
        if not self.raw_linenumbers or not self.lengths:
            return

        left_pos = binary_find_left(self.raw_linenumbers,lineidx)
        while(left_pos>0 and self.raw_linenumbers[left_pos-1] == lineidx):
            left_pos = left_pos - 1

        right_pos = left_pos
        while(right_pos<len(self.raw_linenumbers)-1 and self.raw_linenumbers[right_pos+1] == lineidx):
            right_pos = right_pos + 1
        #skip to next pos(lineidx+1)
        right_pos = right_pos + 1

        # Ensure bounds are valid since not all lines are movements
        if left_pos >= len(self.lengths):
            left_pos = len(self.lengths) - 1
        if right_pos >= len(self.lengths):
            right_pos = len(self.lengths) - 1
        if left_pos < 0:
            left_pos = 0
        if right_pos < 0:
            right_pos = 0
        
        # Ensure we have valid indices
        if left_pos >= len(self.lengths) or right_pos >= len(self.lengths):
            return
            
        #start point
        start_distance = self.lengths[left_pos]
        end_distance = self.lengths[right_pos]

        cur_distance = start_distance*(1.0 - ratio) + end_distance*ratio
        self.set_pos_by_distance(cur_distance)

    def get_cur_pos_index(self):
        line_number = -1
        
        if self.cur_line_index < len(self.raw_linenumbers):
            line_number = self.raw_linenumbers[int(self.cur_line_index)]
        
        return [self.display_count,line_number]

    def enable_dynamic_displaying(self,dynamic_display):
        self.dynamic_display = dynamic_display
        self._scene_dirty = True

    def show_all(self):
        self.dynamic_display = False
        self.display_count = self.get_total_distance()
        self._scene_dirty = True

    def restore_default_view(self):
        self.m_xLookAt = 0
        self.m_yLookAt = 0
        self.m_zLookAt = 0
        self.m_xRot = 30
        self.m_yRot = 180
        self.m_zoom = DEFAULT_ZOOM
        self.m_xPan = 0
        self.m_yPan = 0
        self.update_proj()
        self.update_view()
        self._scene_dirty = True

    def set_move_speed(self,mov_speed):
        self.move_speed = mov_speed

    def set_display_mask(self, mask_val):
        """Filter visible segment types via decimal-encoded mask (see shaders/toolpath.glsl)."""
        self.linemesh['vertex_type_display'] = mask_val
        self._scene_dirty = True

    def _apply_color_scheme_uniform(self):
        self.linemesh['color_scheme'] = float(self.color_scheme)

    def _update_feed_range_uniforms(self):
        feeds = [float(f) for f in (self.raw_feed_rates or []) if f and float(f) > 0.0]
        if feeds:
            self.feed_min = min(feeds)
            self.feed_max = max(feeds)
        else:
            self.feed_min = 0.0
            self.feed_max = DEFAULT_FEED_MM_MIN
        if self.feed_max <= self.feed_min:
            self.feed_max = self.feed_min + 1.0
        self.linemesh['feed_min'] = float(self.feed_min)
        self.linemesh['feed_max'] = float(self.feed_max)

    def set_color_scheme(self, scheme):
        """Set toolpath color scheme from UI label or internal id ('by_type' / 'by_tool' / 'by_speed')."""
        if scheme in (COLOR_SCHEME_UI_BY_TOOL, 'by_tool', COLOR_SCHEME_BY_TOOL):
            self.color_scheme = COLOR_SCHEME_BY_TOOL
        elif scheme in (COLOR_SCHEME_UI_BY_SPEED, 'by_speed', COLOR_SCHEME_BY_SPEED):
            self.color_scheme = COLOR_SCHEME_BY_SPEED
        else:
            self.color_scheme = COLOR_SCHEME_BY_TYPE
        self._apply_color_scheme_uniform()
        self._scene_dirty = True

    #repeat this function every 1/60 s
    def _on_frame_tick(self, _):

        # Recompute projection only when it is actually stale (resize / zoom / pan).
        if self._proj_dirty:
            self.update_proj()
            self._proj_dirty = False

        if self.lengths is None or len(self.lengths) <= 1:
            #data is not loaded yet
            return
        
        # Skip the entire frame when nothing has changed and playback is paused.
        if not self.dynamic_display and not self._scene_dirty:
            return

        if self.dynamic_display:
            self.add_dir = self.move_speed * self.move_scale * self.move_scale_by_positon

            if (self.display_count >= self.get_total_distance()):
                self.dynamic_display = False
            else:
                self.display_count = self.display_count + self.add_dir

            


        

        self.linemesh['display_count'] = float(self.display_count)

        #which segment we are located
        cur_display_distance = float(self.display_count)
        line_index = binary_find_left(self.lengths,cur_display_distance)
        line_ratio = 0
        if(line_index < len(self.lengths)-1):
            segment_length = self.lengths[int(line_index)+1] - self.lengths[int(line_index)]
            if segment_length == 0:
                if not self._cannot_visualise:
                    msg = "Gcode cannot be visualised due to parser error or gcode complexity.\n\nFeatures of the Controller that depend on visualisations have been disabled.\n\nFile playback can be attempted."
                    logger.error(msg)
                    if self.error_popup_callback is not None:
                        self.error_popup_callback(msg)
                    self._cannot_visualise = True
                self.dynamic_display = False
                return
            line_ratio = (cur_display_distance - self.lengths[int(line_index)]) / segment_length
            
        line_index_withratio = line_index + line_ratio

        self.cur_line_index = line_index_withratio

        # Per-frame callback during toolpath playback only
        if self.frame_callback is not None and self.dynamic_display:
            [cur_distance,linenumber]= self.get_cur_pos_index()
            self.frame_callback(cur_distance,linenumber)

        if(self.vertex_types[line_index] > 1.0):
            self.move_scale = 2.0
        else:
            self.move_scale = 1.0



        self.linemesh['rotation_mat'] = self._identity_mat
        
        self.linemesh['view_mat'] = self.m_viewMatrix
        self._update_grid_uniforms()

        pointer_updated_pos = 3*int(line_index_withratio)
        
        self.pointermesh['rotation'] = self._identity_mat
        if pointer_updated_pos < len(self.positions):
            base_start = int(line_index_withratio)
            ratio = line_index_withratio - base_start
            offset = 0.0

            last_pos = vec3_sub(self.meshmanager.get_vertex_position(int(line_index_withratio)),self.lines_center)

            if(self.is_4_axis):
                last_angle = self.angles_of_vertices[int(pointer_updated_pos/3)]
            
            if(ratio>0.0 and pointer_updated_pos+5 < len(self.positions)):
                next_pos = vec3_sub(self.meshmanager.get_vertex_position(int(line_index_withratio)+1),self.lines_center)
                lerp_pos = [next_pos[0] * ratio + (1.0 - ratio)*last_pos[0],next_pos[1] * ratio + (1.0 - ratio)*last_pos[1],next_pos[2] * ratio + (1.0 - ratio)*last_pos[2]]
                self.pointermesh['offset'] = lerp_pos
                
                if(self.is_4_axis):
                    next_angle = self.angles_of_vertices[int(pointer_updated_pos/3)+1]
                    lerp_angle = next_angle * ratio + (1.0 - ratio)*last_angle
                    if(not self.rotate_line_or_knife):
                        self.pointermesh['rotation'] = rotate_mat_by_x_axis_angle(lerp_angle)
                    else:
                        self.linemesh['rotation_mat'] = rotate_mat_by_x_axis_angle(-lerp_angle)
                        len_to_center = len_2d([lerp_pos[1],lerp_pos[2]],[-self.lines_center[1],-self.lines_center[2]])
                        rot_point = self.linemesh['rotation_mat'].transform_point(lerp_pos[0],lerp_pos[1],lerp_pos[2])


                        self.pointermesh['offset'] = rot_point
            else:
                if(self.is_4_axis):
                    if(not self.rotate_line_or_knife):
                        self.pointermesh['rotation'] = rotate_mat_by_x_axis_angle(last_angle)
                    else:
                        self.linemesh['view_mat']=self.linemesh['view_mat'].multiply(rotate_mat_by_x_axis_angle(-last_angle))
                        
                        len_to_center = len_3d(last_pos,[-self.lines_center[0],-self.lines_center[1],0])
                        self.pointermesh['offset'] = [-self.lines_center[0],-self.lines_center[1],len_to_center -self.lines_center[2]]


        
        
        self.pointermesh['modelview_mat'] = self.m_viewMatrix

        #axis
        axis_offset = (-self.lines_center[0],-self.lines_center[1],-self.lines_center[2])
        self.axisxmesh['offset'] = axis_offset
        self.axisxmesh['rotation'] = self._identity_mat
        self.axisxmesh['diff_color'] = AXIS_COLOR_Y

        self.axisymesh['offset'] = axis_offset
        self.axisymesh['rotation'] = self._axis_y_rot
        self.axisymesh['diff_color'] = AXIS_COLOR_Z

        self.axiszmesh['offset'] = axis_offset
        self.axiszmesh['rotation'] = self._axis_z_rot
        self.axiszmesh['diff_color'] = AXIS_COLOR_X

        self.axisxmesh['modelview_mat'] = self.m_viewMatrix
        self.axisymesh['modelview_mat'] = self.m_viewMatrix
        self.axiszmesh['modelview_mat'] = self.m_viewMatrix

        self.g_old_curosr  = self.g_cursor
        self.g_wheel_data = 0
        self._scene_dirty = False

    #mouse event
    #
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            try:
                touchpos = [touch.pos[0], self.size[1] - touch.pos[1]]
                self.m_lastPos = touchpos.copy()
                self.m_xLastRot = self.m_xRot
                self.m_yLastRot = self.m_yRot
                self.m_xLastPan = self.m_xPan
                self.m_yLastPan = self.m_yPan

                if 'button' in touch.profile:
                    if touch.is_mouse_scrolling:
                        if touch.button == 'scrolldown':
                            self.zoom_out()
                        elif touch.button == 'scrollup':
                            self.zoom_in()

                self.update_proj()
                self.update_view()
                self._scene_dirty = True

                if touch.is_double_tap:
                    self.restore_default_view()

            except:
                print(sys.exc_info()[1])

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            try:
                touchpos = [touch.pos[0], self.size[1] - touch.pos[1]]

                if (not 'button' in touch.profile or touch.button == 'left'):
                    if self.orbit:
                        self.m_yRot = normalize_angle(self.m_yLastRot - (touchpos[0] - self.m_lastPos[0]) * 0.5)
                        self.m_xRot = self.m_xLastRot + (touchpos[1] - self.m_lastPos[1]) * 0.5

                        if (self.m_xRot < -90): self.m_xRot = -90.0
                        if (self.m_xRot > 90): self.m_xRot = 90.0

                        self.update_view()
                    else:
                        self.m_xPan = self.m_xLastPan - (touchpos[0] - self.m_lastPos[0]) * 1 / self.size[0]
                        self.m_yPan = self.m_yLastPan + (touchpos[1] - self.m_lastPos[1]) * 1 / self.size[1]

                        self.update_proj()

                elif ('button' in touch.profile and touch.button == 'right'):
                    self.m_xPan = self.m_xLastPan - (touchpos[0] - self.m_lastPos[0]) * 1 / self.size[0]
                    self.m_yPan = self.m_yLastPan + (touchpos[1] - self.m_lastPos[1]) * 1 / self.size[1]

                    self.update_proj()

                self.g_cursor = [touch.pos[0], touch.pos[1]]
                self._scene_dirty = True
            except:
                print(sys.exc_info()[1])

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            try:
                self.g_old_curosr = self.g_cursor = [touch.pos[0], touch.pos[1]]
            except:
                print(sys.exc_info()[1])

    def zoom_in(self):
        if (self.m_zoom > 0.1):
            self.m_zoom /= ZOOMSTEP
            self.update_proj()
            self.update_view()
            self._scene_dirty = True

    def zoom_out(self):
        if (self.m_zoom < 10):
            self.m_zoom *= ZOOMSTEP
            self.update_proj()
            self.update_view()
            self._scene_dirty = True

    def set_orbit(self, orbit = True):
        self.orbit = orbit

    def set_grid_visible(self, visible=True):
        visible = bool(visible)
        if visible == self._grid_visible:
            return
        self._grid_visible = visible
        self._update_grid_uniforms()
        self._scene_dirty = True

    def is_grid_visible(self):
        return self._grid_visible


def _compute_line_times_worker(raw_positions, raw_linenumbers, raw_feed_rates, progress_callback, progress_interval):
    """
    Core logic for line time computation. Can run in a thread.
    Uses feed rates from raw_feed_rates (from CNC parser); no file I/O.
    progress_callback(percent) is called every progress_interval segments; use 0 to disable.
    Returns list of cumulative times (line_times).
    """
    n = len(raw_linenumbers)
    DEFAULT_FEED_MM_MIN = 3000.0
    MIN_FEED_MM_MIN = 0.001
    line_times = [0.0]
    for i in range(1, n):
        pos1 = [
            raw_positions[3 * (i - 1)],
            raw_positions[3 * (i - 1) + 1],
            raw_positions[3 * (i - 1) + 2],
        ]
        pos2 = [
            raw_positions[3 * i],
            raw_positions[3 * i + 1],
            raw_positions[3 * i + 2],
        ]
        segment_length_mm = len_3d(pos1, pos2)
        feed = DEFAULT_FEED_MM_MIN
        if raw_feed_rates and i < len(raw_feed_rates):
            try:
                f = float(raw_feed_rates[i])
                if f >= MIN_FEED_MM_MIN:
                    feed = f
            except (ValueError, TypeError):
                pass
        duration_sec = (segment_length_mm * 60.0) / feed
        line_times.append(line_times[-1] + duration_sec)
        if progress_callback and progress_interval > 0 and i % progress_interval == 0:
            progress_callback(100.0 * i / n)
    if progress_callback and progress_interval > 0 and n > 1:
        progress_callback(100.0)
    return line_times


if __name__ == '__main__':
    class MyApp(App):
        def build(self):
            viewer = GCodeViewer()
            viewer.set_play_over_callback(frame_call_back_test)
            lines = []
            with open('parsernew/gcodes(1).txt', "r") as file:
                content = file.read()[2:-2]
                for line in content.split('], ['):
                    arr = line.split(',')
                    lines.append([
                        float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]),
                        float(arr[4]), float(arr[5]), float(arr[6]),
                    ])

            get_elapsed("start")
            step = 10000
            for i in range(len(lines) // step + 1):
                start_idx = i * step
                end_idx = min((i + 1) * step, len(lines))
                viewer.load_array(lines[start_idx:end_idx], end_idx == len(lines))
            get_elapsed("loaded")

            viewer.set_distance_by_lineidx(1000, 0.5)
            viewer.show_all()
            return viewer

    MyApp().run()
