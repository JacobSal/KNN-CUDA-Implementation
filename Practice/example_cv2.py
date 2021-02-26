# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:50:41 2020

@author: jsalm
"""

import cv2
import numpy as np

def draw_str(img_dst, coords, s, font_size = 1.0, color_RGB = (255, 255, 255)):
    xc, yc = coords
    cv2.putText(img_dst, s, (xc + 1, yc + 1), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 0), thickness = 2, lineType = cv2.LINE_AA)
    cv2.putText(img_dst, s, (xc, yc), cv2.FONT_HERSHEY_PLAIN, font_size, rgb2bgr_color(color_RGB), lineType = cv2.LINE_AA)

class PointClicker(object):

    def __init__(self,img, win_name, max_clicks = 1, save_path = "", draw_polygon_clicks = False):
        self.WINDOW_NAME = win_name
        self.save_path = save_path
        self.img = img
        self.click_counter = 0
        self.img_save_number = 0
        self.is_new_mouse_click = False
        self.max_number_of_clicks = max_clicks
        self.clicked_points = np.ndarray((self.max_number_of_clicks, 2), dtype = int)
        self.shift_mouse_pos = None
        self.verbose = True
        self.draw_lines = draw_polygon_clicks
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        self.lines = self.max_number_of_clicks * [None]  # To Contain list of line pairs for example: [[(x0,y0),(x1,y1)], [(x1,y1),(x2,y2)],[(x2,y2),(x_curr,y_curr)]]
        cv2.setMouseCallback(self.WINDOW_NAME, self.on_mouse_callback)

    def on_mouse_callback(self, event, xc, yc, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            if flags == cv2.EVENT_FLAG_SHIFTKEY:
#         if flags == (cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_SHIFTKEY):
                self.shift_mouse_pos = (xc, yc)
            if self.draw_lines:
                if self.click_counter > 0 and self.click_counter != self.max_number_of_clicks:
                    self.lines[self.click_counter - 1] = [tuple(self.clicked_points[self.click_counter - 1]), (xc, yc)]
                    self.is_new_mouse_click = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.click_counter += 1
            if self.click_counter > self.max_number_of_clicks:
                self.click_counter = 1  # Reset counter
                self.lines = self.max_number_of_clicks * [None]  # Reset all lines
            self.clicked_points[self.click_counter - 1] = (xc, yc)
            if self.draw_lines:
                if self.click_counter > 1:
                    self.lines[self.click_counter - 2] = [tuple(self.clicked_points[self.click_counter - 2]), tuple(self.clicked_points[self.click_counter - 1])]
                    if self.click_counter == self.max_number_of_clicks:  # Close the loop
                        self.lines[self.click_counter - 1] = [tuple(self.clicked_points[self.click_counter - 1]), tuple(self.clicked_points[0])]
            if self.verbose:
                print("Clicked on (u,v) = ", self.clicked_points[self.click_counter - 1])
            self.is_new_mouse_click = True

    def get_clicks_uv_coords(self, img, verbose = True):
        '''
        @return: the np array of valid points clicked. NOTE: the arrangement is in the (u,v) coordinates
        '''
        self.verbose = verbose
        cv2.imshow(self.WINDOW_NAME, img)

        # while cv2.waitKey(1) == -1:  # While not any key has been pressed
        ch_pressed_waitkey = cv2.waitKey(1)
        while not(ch_pressed_waitkey == 27):  # Pressing the Escape key breaks the loop
            if (ch_pressed_waitkey & 255) == ord('r'):
                self.click_counter = 0  # reset count
                self.is_new_mouse_click = True
                self.lines = self.max_number_of_clicks * [None]

            # Grab a point
            if self.is_new_mouse_click:
                channels = img.ndim
                    # img_copy = img.copy()  # Keep the original image
                if channels < 3:
                    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    vis = img.copy()

                draw_points(vis, self.clicked_points, num_valid_points = self.click_counter)
                draw_lines(vis, self.lines, thickness = 5)
                draw_str(vis, (10, 20), "Keyboard Commands:", color_RGB = (255, 0, 0))
                draw_str(vis, (20, 40), "'R': restart", color_RGB = (255, 0, 0))
                draw_str(vis, (20, 60), "'(1-9)': rewind by #", color_RGB = (255, 0, 0))
                draw_str(vis, (20, 80), "'Esc': finish", color_RGB = (255, 0, 0))
                cv2.imshow(self.WINDOW_NAME, vis)

            # To indicate to go back to the previous case (useful when detecting corner detection errors)
            if 48 < ch_pressed_waitkey < 58:  # Indicate to go back to the previous case (useful when detecting corner detection errors)
                return ch_pressed_waitkey - 48  # Because 48 is mapped to key 0

            self.is_new_mouse_click = False  # Reset indicator
            ch_pressed_waitkey = cv2.waitKey(10)

#         cv2.destroyWindow(self.window_name)

        return self.clicked_points[:self.click_counter]

    def get_clicks_uv_coords_for_stereo(self, stereo_model, show_correspondence_on_circular_img = False, min_disparity = 1, max_disparity = 0, verbose = False):
        '''
        @return: the two np arrays of valid points clicked and its correspondences. NOTE: the arrangement is in the (u,v) coordinates
        '''
        self.verbose = verbose
        target_window_name = 'Target Point Correspondences'
        cv2.namedWindow(target_window_name, cv2.WINDOW_NORMAL)

        target_coords = None
        reference_coords = None
        img_reference = stereo_model.top_model.panorama.panoramic_img  # Acting as the right image
        img_target = stereo_model.bot_model.panorama.panoramic_img
        if show_correspondence_on_circular_img:
            omni_top_coords = None
            omni_bot_coords = None
            img_omni = stereo_model.current_omni_img
            omni_window_name = 'Correspondences on Omni Image'
            cv2.namedWindow(omni_window_name, cv2.WINDOW_NORMAL)

        cv2.imshow(self.WINDOW_NAME, img_reference)
        # while cv2.waitKey(1) == -1:  # While not any key has been pressed
        ch_pressed_waitkey = cv2.waitKey(1)
        while not(ch_pressed_waitkey == 27):  # Pressing the Escape key breaks the loop
            if (ch_pressed_waitkey & 255) == ord('r'):
                self.click_counter = 0  # reset count
                self.is_new_mouse_click = False
                cv2.imshow(self.WINDOW_NAME, img_reference)
                cv2.imshow(target_window_name, img_target)
                if show_correspondence_on_circular_img:
                    cv2.imshow(omni_window_name, img_omni)

            # Grab a point
            if self.is_new_mouse_click:
                channels = img_reference.ndim
                    # img_copy = img_reference.copy()  # Keep the original image
                if channels < 3:
                    vis_ref = cv2.cvtColor(img_reference, cv2.COLOR_GRAY2BGR)
                    vis_target = cv2.cvtColor(img_target, cv2.COLOR_GRAY2BGR)
                    if show_correspondence_on_circular_img:
                        vis_omni = cv2.cvtColor(img_omni, cv2.COLOR_GRAY2BGR)
                else:
                    vis_ref = img_reference.copy()
                    vis_target = img_target.copy()
                    if show_correspondence_on_circular_img:
                        vis_omni = img_omni.copy()

                # Find correspondence
                reference_coords, target_coords, disparities = stereo_model.resolve_pano_correspondences_from_disparity_map(self.clicked_points[:self.click_counter], min_disparity = min_disparity, max_disparity = max_disparity, verbose = verbose)
                # Update clicks
                self.click_counter = int(np.count_nonzero(reference_coords) / 2)
                self.clicked_points[:self.click_counter] = reference_coords

                # Write instructions on image
                draw_str(vis_ref, (10, 20), "Keyboard Commands:")
                draw_str(vis_ref, (20, 40), "'R': to restart")
                draw_str(vis_ref, (20, 60), "'Esc': to finish")
                # Draw points on panoramas
                ref_pts_color = (255, 0, 0)  # RGB = red
                tgt_pts_color = (0, 0, 255)  # RGB = blue
                pt_thickness = 5
                draw_points(vis_ref, reference_coords.reshape(-1, 2), color = ref_pts_color, thickness = pt_thickness)
                cv2.imshow(self.WINDOW_NAME, vis_ref)
                draw_points(vis_target, reference_coords.reshape(-1, 2), color = ref_pts_color, thickness = pt_thickness)
                # Coloring a blue dot at the proper target location:
                draw_points(vis_target, target_coords.reshape(-1, 2), color = tgt_pts_color, thickness = pt_thickness)
                cv2.imshow(target_window_name, vis_target)
                if show_correspondence_on_circular_img and self.click_counter > 0 and self.verbose:
                    _, _, omni_top_coords = stereo_model.top_model.panorama.get_omni_pixel_coords_from_panoramic_pixel(reference_coords)
                    _, _, omni_bot_coords = stereo_model.bot_model.panorama.get_omni_pixel_coords_from_panoramic_pixel(target_coords)
                    print("Omni pixel coords: TOP %s, BOT %s" % (omni_top_coords[0, self.click_counter - 1], omni_bot_coords[0, self.click_counter - 1]))
                    draw_points(vis_omni, omni_top_coords[..., :2].reshape(-1, 2), color = ref_pts_color, thickness = pt_thickness)
                    draw_points(vis_omni, omni_bot_coords[..., :2].reshape(-1, 2), color = tgt_pts_color, thickness = pt_thickness)
                    cv2.imshow(omni_window_name, vis_omni)

            self.is_new_mouse_click = False  # Reset indicator
            ch_pressed_waitkey = cv2.waitKey(1)

        return reference_coords, target_coords, disparities

    def save_image(self, img, img_name = None, num_of_zero_padding = 6):
        if img_name:
            name_prefix = img_name
        else:
            name_prefix = "img"

        n = str(self.img_save_number)
        img_name = '%s-%s.png' % (name_prefix, n.zfill(num_of_zero_padding))

        if self.save_path:
            complete_save_name = self.save_path + img_name
        else:
            complete_save_name = img_name

        print('Saving', complete_save_name)
        cv2.imwrite(complete_save_name, img)

        self.img_save_number += 1  # Increment save counter

def rgb2bgr_color(rgb_color):
    return (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))

def draw_points(img_input, points_uv_coords, num_valid_points = None, color = None, thickness = 1):
    '''
    @param img_input: The image on which points will be drawn to (NOTE: it doesn't preserve the image)
    @param points_uv_coords: FIXME: the uv coordinates list or ndarray must be of shape (n, 2) for n points.
    Note that the coordinates will be expressed as integers while visualizing
    @param color: a 3-tuple of the RGB color for these points
    '''
    if color == None:
        color = (0, 0, 255)  # Red because BGR(B,G,R)
    else:  # Swap the passed color from RGB into BGR
        color = rgb2bgr_color(color)

    if num_valid_points == None:
        num_valid_points = len(points_uv_coords)

    for i in range(num_valid_points):
        pt = points_uv_coords[i]
        if np.isnan(pt[0]) or np.isnan(pt[1]):
            print("nan cannot be drawn!")
        else:
            try:  # TODO: also, out of border points cannot be drawn!
                pt_as_tuple = (int(pt[0]), int(pt[1]))  # Recall: (pt[0],pt[1]) # (x, u or col and y, v or row)
                cv2.circle(img_input, pt_as_tuple, 2, color, thickness, 8, 0)
            except:
                print("Point", pt_as_tuple, "cannot be drawn!")

def draw_lines(img_input, lines_list, color = None, thickness = 2):
    '''
    @param img_input: The image on which points will be drawn to (NOTE: it doesn't preserve the image)
    @param lines_list: A list of point pairs such as [[(x0,y0),(x1,y1)], [(x1,y1),(x2,y2)],[(x2,y2),(x_last,y_Last)], None, None]
    @param color: a 3-tuple of the RGB color for these points
    '''
    if color == None:
        color = (0, 0, 255)  # Red because BGR(B,G,R)
    else:  # Swap the passed color from RGB into BGR
        color = rgb2bgr_color(color)

    for pts in lines_list:
        if pts is not None:
            [pt_beg, pt_end] = pts
            cv2.line(img_input, pt_beg, pt_end, color, thickness = thickness, lineType = cv2.LINE_AA)
            
if __name__== '__main__':
    rootdir = r"C:\Users\jsalm\Documents\Python Scripts\3DRecon\images_5HT\dAIH_20x_Nosectioning_1_CH2.tif"
    image = cv2.imread(rootdir)
    window = PointClicker(image,'panzoomwindow',9,r"C:\Users\jsalm\Documents\Python Scripts\3DRecon\saved_training",True)
    key = -1
    while key != ord('q') and key != 27 and cv2.getWindowProperty(window.WINDOW_NAME,0) >=0:
        key = cv2.waitKey(1)
    'end while'