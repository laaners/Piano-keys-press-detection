import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import mediapipe as mp
import threading
import time
from midiutil.MidiFile import MIDIFile

def threadMultiple(functions, results):
    def threadFunction(callback_and_args, results, index):
        #print("Start {}".format(index))
        callback = callback_and_args["function"]
        callback_args = callback_and_args["args"]
        results[index] = callback(*callback_args)
        #print("End {}".format(index))
    threads = [None]*len(functions)
    for i in range(len(functions)):
        threads[i] = threading.Thread(target=threadFunction, args=(functions[i],results,i))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

def plotImg(img, cmap=None):
    if cmap is None:
        plt.imshow(img)
        plt.show()
        plt.close()
    else:
        plt.imshow(img, cmap='gray')
        plt.show()
        plt.close()

def extractFrames(video, start_time, end_time=None, samples_per_second=None, callback=None, callback_args=None):
    cap = cv2.VideoCapture(video)
    fps = np.ceil(cap.get(cv2.CAP_PROP_FPS))

    if end_time == None:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_time = frame_count/fps

    if samples_per_second is None or samples_per_second > fps:
        samples_per_second = np.ceil(fps)

    ris = []
    frames = start_time*fps #starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames)
    samples_per_second *= 60//fps

    while frames < end_time*fps:
        print("{}/{}".format(frames,end_time*fps))
        print("{}/{}".format(frames/fps,end_time))

        is_read, frame = cap.read()

        if not is_read:
            break

        if callback is not None:
            frame_processed = callback(frame, *callback_args)
            ris.append(frame_processed)
        else:
            ris.append(frame)

        clear_output(wait=True)

        frames += 60/samples_per_second
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames)
    return ris

def playVideo(video, start_time, end_time=None, frames_every=1, callback=None, callback_args=None, save_video=False, video_name="out.mp4", video_size=(1920,1080)):
    cap = cv2.VideoCapture(video)
    fps = np.ceil(cap.get(cv2.CAP_PROP_FPS))

    out = None
    if save_video:
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, video_size)

    if end_time == None:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_time = frame_count/fps

    frames = start_time*fps #starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames)

    while frames < end_time*fps:
        print("{}/{}".format(frames,end_time*fps))
        print("{}/{}".format(frames/fps,end_time))

        for i in range(frames_every):
            is_read, frame = cap.read()

        if not is_read:
            break

        if callback is not None:
            frame_processed = callback(frame, *callback_args)
            cv2.rectangle(frame_processed, (10, 2), (100,20), (255,255,255), -1)
            cv2.putText(frame_processed, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.imshow(video, cv2.resize(frame_processed,(960, 540)))
            cv2.resizeWindow(video, 960, 540)
            if save_video:
                ph, pw = frame_processed.shape[0:2]
                sw, sh = video_size
                print("{} =?= {}, {} =?= {}".format(ph,sh,pw,sw))
                if ph != sh or pw != sw:
                    print("Different sizes")
                    frame_processed = cv2.resize(frame_processed, video_size)
                out.write(frame_processed)
        else:
            cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
            cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.imshow(video, cv2.resize(frame,(960, 540)))
            cv2.resizeWindow(video, 960, 540)
            if save_video:
                ph, pw = frame.shape[0:2]
                sw, sh = video_size
                print("{} =?= {}, {} =?= {}".format(ph,sh,pw,sw))
                if ph != sh or pw != sw:
                    print("Different sizes")
                    frame = cv2.resize(frame, video_size)
                out.write(frame)

        clear_output(wait=True)

        frames += frames_every

        k = cv2.waitKey(50) & 0xFF
        if k == 27: #escape key
            break
    cv2.destroyAllWindows()

def segmentationMaskTuning(frame, roi_x=None, roi_y=None):
    if roi_x is None or roi_y is None:
        roi = frame.copy()
    else:
        print(roi_y[0])
        print(roi_y[1])
        roi = frame.copy()[roi_x[0]:roi_x[1], roi_y[0]:roi_y[1]]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_thresh = np.array([128,128,128], dtype=np.uint8)
    upper_thresh = np.array([255,255,255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_thresh, upper_thresh)

    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO) # make a window with name 'image'    
    cv2.createTrackbar('h_low', 'image',128,255, lambda x:x)
    cv2.createTrackbar('h_high','image',255,255, lambda x:x)
    cv2.createTrackbar('s_low', 'image',128,255, lambda x:x)
    cv2.createTrackbar('s_high','image',255,255, lambda x:x)
    cv2.createTrackbar('v_low', 'image',128,255, lambda x:x)
    cv2.createTrackbar('v_high','image',255,255, lambda x:x)

    def setChannel(low_ratio_track_bar, high_ratio_track_bar):
        low_ratio = cv2.getTrackbarPos(low_ratio_track_bar, 'image')
        high_ratio = cv2.getTrackbarPos(high_ratio_track_bar, 'image')
        return low_ratio, high_ratio

    resize = roi.shape[:2] #(320, int(roi.shape[0]*320/roi.shape[1]))

    while(1):
        resized_gray = cv2.resize(cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY), resize)
        resized_mask = cv2.resize(mask, resize)
        numpy_horizontal_concat = np.concatenate((resized_mask, resized_gray), axis=1) # to display image side by side
        cv2.imshow('image', numpy_horizontal_concat)
        cv2.resizeWindow('image', 1280, 320)
        k = cv2.waitKey(50) & 0xFF
        if k == 27: #escape key
            break
        
        h_low, h_high = setChannel('h_low', 'h_high')
        s_low, s_high = setChannel('s_low', 's_high')
        v_low, v_high = setChannel('v_low', 'v_high')

        lower_thresh = np.array([h_low,s_low,v_low], dtype=np.uint8)
        upper_thresh = np.array([h_high,s_high,v_high], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
    
    cv2.destroyAllWindows()
    return lower_thresh, upper_thresh, mask

def segmentationInRange(frame, lower_thresh, upper_thresh, show=True):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_binary = cv2.inRange(frame_hsv, lower_thresh, upper_thresh)
    if show:
        fig = plt.figure(figsize=(16, 9))
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(frame_binary, cmap='gray')
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(frame, cmap='gray')
        plt.show()
        plt.close()
    return frame_binary

def cannyEdgeDetector(frame):
    return cv2.Canny(frame, 255/3, 255, apertureSize=3)

def getRectifyingMatrix(frame, lower_thresh, upper_thresh, processed=False, pts2=None):
    if not processed:
        frame_segmented = segmentationInRange(frame, lower_thresh, upper_thresh, False)
    else:
        frame_segmented = frame.copy()

    frame_canny = cannyEdgeDetector(frame_segmented)

    # empty canvas
    frame_empty = np.zeros(shape=frame.shape[0:2], dtype=np.uint8)

    # to find max area contour
    frame_max_area = cv2.dilate(frame_canny, np.ones((3,3)), iterations=1)

    #frame_max_area = cv2.erode(frame_max_area, np.ones((3,3)), iterations=1)
    contours, _ = cv2.findContours(frame_max_area, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.eye(3), None, None, (10,5)

    #sort array based on area of contours
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))

    #if we have 2 or more large contours it means we split the keyboard contour in more parts
    #and we have to join them
    new_max = []
    max1 = contours[-1]
    for point in max1:
        new_max.append(point)

    for i in range(len(contours)-1):
        max2 = contours[i]
        if cv2.contourArea(max1) < 2.5*cv2.contourArea(max2): #arbitrary threshold
            for point in max2:
                new_max.append(point)
    max_contour = np.array(new_max, dtype=np.int32)

    #approximate contour to a polygon
    hull = cv2.convexHull(max_contour)

    #draw the keyboard contour on a black frame
    cv2.drawContours(frame_empty, [hull], -1, (255,255,255), thickness=cv2.FILLED)

    # the empty canvas now has the max convex hull, approximate it to a 4-point polygon
    # we find again the contour and then use the proper opencv functions
    #frame_empty = cv2.dilate(frame_empty, np.ones((5,5)), iterations=3)
    contours, _ = cv2.findContours(frame_empty, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key = lambda x: cv2.contourArea(x))
    peri = cv2.arcLength(max_contour,True)
    approx = cv2.approxPolyDP(max_contour, 0.02*peri,True)

    thresh = peri
    approx = cv2.approxPolyDP(max_contour, peri/thresh,True)
    while len(approx) != 4:
        thresh -= 0.1
        approx = cv2.approxPolyDP(max_contour, peri/thresh, True)

    if len(approx) == 4: #if we have 4 points
        # find position of top-left corner
        min_val = approx[0][0][0]+approx[0][0][1] #sum of 2 coordinates of each point
        top_left_pos = 0
        for i in range(len(approx)):
            corner = approx[i][0]
            if corner[0]+corner[1] < min_val: #top-left has both minimum
                top_left_pos = i
                min_val = corner[0]+corner[1]

        # approx always returns the points counter-clock wise
        top_left = approx[top_left_pos][0]
        bottom_left = approx[(top_left_pos+1) % 4][0]
        bottom_right = approx[(top_left_pos+2) % 4][0]
        top_right = approx[(top_left_pos+3) % 4][0]

        keyboard_width = np.max([
            top_right[0]-top_left[0],
            bottom_right[0]-bottom_left[0]
        ])
        keyboard_height = np.max([
            bottom_right[1]-top_right[1],
            bottom_left[1]-top_left[1]
        ])

        pts1 = np.float32(approx)
        if pts2 is None:
            pts2 = np.roll(np.float32([
                [top_left[0], top_left[1]],
                [top_left[0], top_left[1]+keyboard_height],
                [top_left[0]+keyboard_width, top_left[1]+keyboard_height],
                [top_left[0]+keyboard_width, top_left[1]]]), top_left_pos, axis=0)

        #map the approx points to a rectangle to find H
        H_shape = cv2.getPerspectiveTransform(pts1,pts2)
        
        cv2.drawContours(frame_canny, [approx], -1, (255,255,255), 3)
        plotImg(frame_canny)

        return H_shape, pts1, pts2, (keyboard_width, keyboard_height)
    return np.eye(3), None, None, (10,5)

def getRectifyingMatrixManual(frame):
    approx = []
    h, w, _ = frame.shape
    scaling_h = h/720
    scaling_w = w/1280
    def on_click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            approx.append((x*scaling_w//1, y*scaling_h//1))
        
    # register the callback
    cv2.namedWindow("Get 4 points")
    cv2.setMouseCallback("Get 4 points", on_click_event)

    # loop until 'c' key is pressed, or four points have been collected
    while True:
        cv2.imshow("Get 4 points", cv2.resize(frame, (1280,720)))
        k = cv2.waitKey(50) & 0xFF
        if k == 27 or len(approx) == 4:
            break
    approx = np.float32(approx)
    
    cv2.destroyAllWindows()

    # find position of top-left corner
    min_val = approx[0][0]+approx[0][1] #sum of 2 coordinates of each point
    top_left_pos = 0
    for i in range(len(approx)):
        corner = approx[i]
        if corner[0]+corner[1] < min_val: #top-left has both minimum
            top_left_pos = i
            min_val = corner[0]+corner[1]

    # approx always returns the points counter-clock wise
    top_left = approx[top_left_pos]
    bottom_left = approx[(top_left_pos+1) % 4]
    bottom_right = approx[(top_left_pos+2) % 4]
    top_right = approx[(top_left_pos+3) % 4]

    keyboard_width = np.max([
        top_right[0]-top_left[0],
        bottom_right[0]-bottom_left[0]
    ])
    keyboard_height = np.max([
        bottom_right[1]-top_right[1],
        bottom_left[1]-top_left[1]
    ])

    pts1 = np.float32(approx)
    pts2 = np.roll(np.float32([
        [top_left[0], top_left[1]],
        [top_left[0], top_left[1]+keyboard_height],
        [top_left[0]+keyboard_width, top_left[1]+keyboard_height],
        [top_left[0]+keyboard_width, top_left[1]]]), top_left_pos, axis=0)

    #map the approx points to a rectangle to find H
    H_shape = cv2.getPerspectiveTransform(pts1,pts2)
    
    frame_tmp = frame.copy()
    cv2.drawContours(frame_tmp, [approx.astype(int)], -1, (255,255,255), thickness=5)
    plotImg(frame_tmp)

    return H_shape, pts1, pts2, (keyboard_width, keyboard_height)

def rectifyWithMatrix(frame, lower_thresh, upper_thresh, H_shape=None, rotation=np.eye(3), pts2=None):
    if H_shape is None and pts2 is not None:
        H_shape, _, _, _ = getRectifyingMatrix(frame, lower_thresh, upper_thresh, pts2=pts2) #if we need to recompute H
    H_shape = np.dot(rotation, H_shape)
    return cv2.warpPerspective(frame, H_shape, (1920+192,1080+108)) #apply H

def projectPoint(p, H):
    p_ = np.dot(H,[p[0], p[1], 1])
    p_ /= p_[2]
    return p_.astype(int)[0:2]

def getKeyboardContour(frame, lower_thresh, upper_thresh, pts2, H_shape, keyboard_shape, frame_segmented=None):
    if pts2 is None:
        return None

    w, h = keyboard_shape

    # hide everything outside keyboard
    mask = np.zeros((1080+108,1920+192), dtype=np.uint8)
    cv2.drawContours(mask, [pts2.astype(int)], -1, (255, 255, 255), -1) # fill rectangle, fill the mask
    # frame_contour = cv2.bitwise_and(frame_contour, frame_contour, mask = mask)

    # segment
    frame_contour = cv2.warpPerspective(frame, H_shape, (1920+192,1080+108))
    if frame_segmented is None:
        frame_segmented = segmentationInRange(frame_contour, lower_thresh, upper_thresh, False)
    else:
        frame_segmented = cv2.warpPerspective(frame_segmented, H_shape, (1920+192,1080+108))
    frame_contour = cv2.bitwise_and(frame_segmented, mask)

    # line crossing black keys
    min_val = pts2[0][0]+pts2[0][1] 
    top_left_pos = 0
    for i in range(len(pts2)):
        corner = pts2[i]
        if corner[0]+corner[1] < min_val:
            top_left_pos = i
            min_val = corner[0]+corner[1]

    # counter-clock wise
    top_left = tuple(map(int,pts2[top_left_pos]))
    bottom_left = tuple(map(int,pts2[(top_left_pos+1) % 4]))
    bottom_right = tuple(map(int,pts2[(top_left_pos+2) % 4]))
    top_right = tuple(map(int,pts2[(top_left_pos+3) % 4]))

    keys = {"white": [], "black": []}

    w_multiplier = 2.5

    if w < h: # keyboard in vertical
        x = top_right[0]-w//3
        x = int(x)
        y1 = top_right[1]
        y2 = bottom_right[1]

        # find largest black key
        widths = []
        y = y1
        y = int(y)
        while y < y2:
            a_extreme = y
            pixel = frame_contour[y][x]
            while pixel == 0 and y < y2:
                y += 1
                pixel = frame_contour[y][x]
            b_extreme = y
            if np.abs(b_extreme-a_extreme) != 0:
                widths.append(np.abs(b_extreme-a_extreme))
            y += 1
        max_w = np.mean(widths) #max(widths)

        # find the black keys extremes and their middle points
        white_widths = []
        b_extremes = []
        a_extremes = []
        y = y1
        while y < y2:
            a_extreme = y
            pixel = frame_contour[y][x]
            while pixel == 0 and y < y2:
                y += 1
                pixel = frame_contour[y][x]
            b_extreme = y
            cur_w = np.abs(b_extreme-a_extreme)
            if cur_w*w_multiplier >= max_w:
                a_extremes.append(a_extreme)
                b_extremes.append(b_extreme)
                mid = (b_extreme-a_extreme)//2+a_extreme
                white_widths.append(mid)
            y += 1

        # find when one of the extremes meets white
        black_widths_diff = []
        white_widths = []
        for i in range(len(a_extremes)):
            a_extreme = a_extremes[i]
            b_extreme = b_extremes[i]
            black_widths_diff.append(b_extreme-a_extreme)
        min_black_width = min(black_widths_diff)

        for i in range(len(a_extremes)):
            a_extreme = a_extremes[i]
            b_extremes[i] = a_extremes[i]+min_black_width
            b_extreme = b_extremes[i]
            mid = (b_extreme-a_extreme)//2+a_extreme
            white_widths.append(mid)
        '''
        '''

        x_nodes = [] # x coordinates for first points touching white
        y_nodes = [] # y coordinates for first points touching white
        for mid in white_widths:
            for i in range(top_right[0]-2,top_left[0],-1):
                if frame_contour[mid][i] != 0:
                    x_nodes.append(i)
                    y_nodes.append(mid)
                    break
        
        # Black keys rectangles
        mid_line_x = int(np.mean(x_nodes))
        for i in range(len(a_extremes)):
            a_extreme = a_extremes[i]
            b_extreme = b_extremes[i]
            start_point = (top_right[0], a_extreme)
            end_point = (mid_line_x, b_extreme)
            keys["black"].append({ "id": i+1, "pt1": start_point, "pt2": end_point})

        # White keys rectangles
        keys["white"] = white_widths
        
        # find minimum width of w a white key
        white_widths_diff = [j-i for i, j in zip(white_widths[:-1], white_widths[1:])]
        min_w = min(white_widths_diff)
        
        # when a detected white width is unually wide, it means there are two keys
        for i in range(1,len(white_widths)):
            width = white_widths[i]-white_widths[i-1]
            if width > min_w*1.4:
                i = white_widths[i-1]+width//2
                keys["white"].append(i)
        keys["white"].sort()
        
        # append the first white key in a tmp array
        start_point = (top_left[0], top_left[1])
        end_point = (top_right[0], keys["white"][1])
        to_append = {"id": 1, "pt1": start_point, "pt2": end_point}
        tmp_arr = [to_append]

        # map widths to rectangles identifying white keys
        for i in range(len(keys["white"])-1):
            start_point = (top_left[0], keys["white"][i])
            end_point = (top_right[0], keys["white"][i+1])
            to_append = {"id": i+2, "pt1": start_point, "pt2": end_point}
            tmp_arr.append(to_append)
        keys["white"] = tmp_arr

        # fill final undetected white keys
        i = len(keys["white"])+1
        while keys["white"][-1]["pt2"][1] < bottom_left[1]:
            start_point = (top_left[0], keys["white"][-1]["pt2"][1])
            end_point = (top_right[0], 2*keys["white"][-1]["pt2"][1]-keys["white"][-2]["pt2"][1])
            to_append = {"id": i, "pt1": start_point, "pt2": end_point}
            keys["white"].append(to_append)
            i += 1
        keys["white"][-1]["pt2"] = (keys["white"][-1]["pt2"][0], bottom_left[1])

    else: # keyboard in horizontal
        y = top_right[1]+h//3
        y = int(y)
        x1 = top_left[0]
        x2 = top_right[0]

        # find largest black key
        widths = []
        x = x1
        x = int(x)
        while x < x2:
            a_extreme = x
            pixel = frame_contour[y][x]
            while pixel == 0 and x < x2:
                x += 1
                pixel = frame_contour[y][x]
            b_extreme = x
            if np.abs(b_extreme-a_extreme) != 0:
                widths.append(np.abs(b_extreme-a_extreme))
            x += 1
        max_w = np.mean(widths) #max(widths)

        # find the black keys extremes and their middle points
        b_extremes = []
        a_extremes = []
        x = x1
        while x < x2:
            a_extreme = x
            pixel = frame_contour[y][x]
            while pixel == 0 and x < x2:
                x += 1
                pixel = frame_contour[y][x]
            b_extreme = x
            cur_w = np.abs(b_extreme-a_extreme)
            if cur_w*w_multiplier >= max_w: #remove too small widths
                a_extremes.append(a_extreme)
                b_extremes.append(b_extreme)
                mid = (b_extreme-a_extreme)//2+a_extreme
            x += 1

        # find when one of the extremes meets white
        black_widths_diff = []
        white_widths = []
        for i in range(len(a_extremes)):
            a_extreme = a_extremes[i]
            b_extreme = b_extremes[i]
            black_widths_diff.append(b_extreme-a_extreme)
        min_black_width = min(black_widths_diff)

        for i in range(len(a_extremes)):
            a_extreme = a_extremes[i]
            b_extremes[i] = a_extremes[i]+min_black_width
            b_extreme = b_extremes[i]
            mid = (b_extreme-a_extreme)//2+a_extreme
            white_widths.append(mid)
        '''
        '''

        x_nodes = [] # x coordinates for first points touching white
        y_nodes = [] # y coordinates for first points touching white
        for mid in white_widths:
            for i in range(top_right[1],bottom_right[1]):
                if frame_contour[i][mid] != 0:
                    x_nodes.append(mid)
                    y_nodes.append(i)
                    break

        # Black keys rectangles
        mid_line_y = int(np.mean(y_nodes))
        for i in range(len(a_extremes)):
            a_extreme = a_extremes[i]
            b_extreme = b_extremes[i]
            start_point = (a_extreme, top_right[1]) #top-left point of every black key
            end_point = (b_extreme, mid_line_y) #bottom-right point 
            keys["black"].append({ "id": i+1, "pt1": start_point, "pt2": end_point})

        # White keys rectangles
        keys["white"] = white_widths

        # find minimum width of w a white key
        white_widths_diff = [j-i for i, j in zip(white_widths[:-1], white_widths[1:])]
        min_w = min(white_widths_diff)

        # when a detected white width is unually wide, it means there are two keys
        for i in range(1,len(white_widths)):
            width = white_widths[i]-white_widths[i-1]
            if width > min_w*1.4:
                i = white_widths[i-1]+width//2
                keys["white"].append(i)
        keys["white"].sort()

        # append the first white key in a tmp array
        start_point = (top_left[0], top_left[1])
        end_point = (keys["white"][1], bottom_right[1])
        to_append = {"id": 1, "pt1": start_point, "pt2": end_point}
        tmp_arr = [to_append]

        # map widths to rectangles identifying white keys
        for i in range(len(keys["white"])-1):
            start_point = (keys["white"][i], top_left[1])
            end_point = (keys["white"][i+1], bottom_right[1])
            to_append = {"id": i+2, "pt1": start_point, "pt2": end_point}
            tmp_arr.append(to_append)
        keys["white"] = tmp_arr

        # fill final undetected white keys
        i = len(keys["white"])+1
        while keys["white"][-1]["pt2"][0] < bottom_right[0]:
            start_point = (keys["white"][-1]["pt2"][0], top_left[1])
            end_point = (2*keys["white"][-1]["pt2"][0]-keys["white"][-2]["pt2"][0], bottom_right[1])
            to_append = {"id": i, "pt1": start_point, "pt2": end_point}
            keys["white"].append(to_append)
            i += 1
        keys["white"][-1]["pt2"] = (bottom_right[0], keys["white"][-1]["pt2"][1])

    H_shape_inv = np.linalg.inv(H_shape)
    for key in keys["white"]:
        x1, y1 = key["pt1"]
        x2, y2 = key["pt2"]
        key["contour"] = np.array([[
            projectPoint([x1, y1], H_shape_inv),
            projectPoint([x1, y2], H_shape_inv),
            projectPoint([x2, y2], H_shape_inv),
            projectPoint([x2, y1], H_shape_inv)
        ]])

    for key in keys["black"]:
        x1, y1 = key["pt1"]
        x2, y2 = key["pt2"]
        key["contour"] = np.array([[
            projectPoint([x1, y1], H_shape_inv),
            projectPoint([x1, y2], H_shape_inv),
            projectPoint([x2, y2], H_shape_inv),
            projectPoint([x2, y1], H_shape_inv)
        ]])
    return keys

def candidateKeys(frame_segmented, keys,
                  white=True, white_hough_thresh=60, black_hough_thresh=40):
    if not white:
        frame_lines = np.zeros(frame_segmented.shape[0:2])
        contours, _ = cv2.findContours(frame_segmented, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mean_area = np.mean([cv2.contourArea(x) for x in contours])
        for cnt in contours:
            if cv2.contourArea(cnt) > mean_area:
                cv2.drawContours(frame_lines,[cnt],0,(255,255,255),thickness=cv2.FILLED)
        frame_segmented = frame_lines

    y = cv2.Sobel(frame_segmented, cv2.CV_64F, 1,0, ksize=3, scale=1)
    frame_sobel = cv2.convertScaleAbs(y)

    # cv2.imwrite("zdacanc\\soble_white_{}.png".format(white), frame_sobel)

    key = keys["black"][0]
    max_contour_h = max(key["contour"][0][:,1])
    key = keys["black"][-1]
    max_contour_h = (max_contour_h+max(key["contour"][0][:,1]))//2
    
    split_height = max_contour_h

    frame_sobel = np.concatenate((frame_sobel[0:split_height,0:1920],
        np.zeros((1080-split_height,1920), dtype=np.uint8)),
    axis=0)
    frame_lines = frame_segmented.copy() #frame_canny.copy()

    canditate_pressed = []

    if white:
        lines = cv2.HoughLines(frame_sobel, 1, np.pi/180, white_hough_thresh)
    else:
        lines = cv2.HoughLines(frame_sobel, 1, np.pi/180, black_hough_thresh)
    if lines is None:
        lines = []
    theta_within = np.pi/2
    theta_tol = 1

    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr

        if theta > np.pi:
            theta = theta-np.pi
            r_theta[0][1] = theta            

        # Stores the value of cos(theta) in a
        a = np.cos(theta)
    
        # Stores the value of sin(theta) in b
        b = np.sin(theta)
    
        # x0 stores the value rcos(theta)
        x0 = a*r
    
        # y0 stores the value rsin(theta)
        y0 = b*r
    
        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1920*(-b))
    
        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1920*(a))
    
        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1920*(-b))
    
        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1920*(a))

        if np.abs(theta+np.pi/2-theta_within) > np.pi/180*theta_tol:
            continue
        else:
            canditate_pressed.append((x1+x2)//2)

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        cv2.line(frame_lines, (x1, y1), (x2, y2), (255, 255, 255), 2)

    candidate_keys = []
    if white:
        for key in keys["white"]:
            for x in canditate_pressed:
                min_contour_h = min(key["contour"][0][:,1])
                max_contour_h = max(key["contour"][0][:,1])

                min_contour_w = min(key["contour"][0][:,0])
                max_contour_w = max(key["contour"][0][:,0])

                p_x = int((max_contour_w-min_contour_w)//2)+x
                p_y = int((max_contour_h-min_contour_h)//2)+min_contour_h
                p = (x, int(p_y))
                cv2.circle(frame_lines, p, 5, (255, 255, 255), -1)
                inside = cv2.pointPolygonTest(np.array(key["contour"]),p,False)
                if inside == 1:
                    cv2.drawContours(frame_lines,key["contour"],-1,255,thickness=2)
                    candidate_keys.append(key)
    else:
        for key in keys["black"]:
            for x in canditate_pressed:
                min_contour_h = min(key["contour"][0][:,1])
                max_contour_h = max(key["contour"][0][:,1])

                min_contour_w = min(key["contour"][0][:,0])
                max_contour_w = max(key["contour"][0][:,0])

                p_x = int((max_contour_w-min_contour_w)//2)+x
                p_y = int((max_contour_h-min_contour_h)//2)+min_contour_h
                p = (x, int(p_y))
                cv2.circle(frame_lines, p, 5, (255, 255, 255), -1)
                inside = cv2.pointPolygonTest(np.array(key["contour"]),p,True)
                #inside = cv2.pointPolygonTest(np.array(key["contour"]),p,False)
                #if inside == 1:
                if np.abs(inside) < int((max_contour_w-min_contour_w)//2):
                    cv2.drawContours(frame_lines,key["contour"],-1,255,thickness=2)
                    candidate_keys.append(key)
    return candidate_keys, frame_lines

def mpHandTrackLandmarks(frame):
    frame_tracked = frame.copy()
    drawingModule = mp.solutions.drawing_utils
    handsModule = mp.solutions.hands
    frameHeight, frameWidth = frame_tracked.shape[:2]
    ris = {}
    with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0, max_num_hands=2, model_complexity=0) as hands:
        # Print handedness and draw hand landmarks on the image.
        results = hands.process(cv2.cvtColor(frame_tracked, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            ris = {}
        if results.multi_hand_landmarks != None:
            for i in range(len(results.multi_hand_landmarks)):
                handLandmarks = results.multi_hand_landmarks[i]
                for point in handsModule.HandLandmark:
                    normalizedLandmark = handLandmarks.landmark[point]
                    pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, frameWidth, frameHeight)
                    key = str(point).replace("HandLandmark.","")+"_"+str(i)
                    ris[key] = pixelCoordinatesLandmark
    return ris

def pointInKeyContour(p, frame_ris, notes_to_play,
                      candidate_white_keys, candidate_black_keys,
                      white_keys_map, black_keys_map):
    inside_black = False
    for key in candidate_black_keys:
        dist = np.abs(cv2.pointPolygonTest(np.array(key["contour"]),p,True))
        inside = cv2.pointPolygonTest(np.array(key["contour"]),p,False)
        if inside == 1 or dist < 5:
            cv2.drawContours(frame_ris, key["contour"], -1, (0, 0, 255), thickness=2)
            if black_keys_map[str(key["id"])] not in notes_to_play:
                notes_to_play.append(black_keys_map[str(key["id"])])
            inside_black = True
    if not inside_black:
        for key in candidate_white_keys:
            dist = np.abs(cv2.pointPolygonTest(np.array(key["contour"]),p,True))
            inside = cv2.pointPolygonTest(np.array(key["contour"]),p,False)
            if dist < 5  or inside == 1:
                cv2.drawContours(frame_ris, key["contour"], -1, (0, 255, 0), thickness=2)
                if white_keys_map[str(key["id"])] not in notes_to_play:
                    notes_to_play.append(white_keys_map[str(key["id"])])

def detectPressedKeys(frame,
        cur_frame_segmented, mask_hand,
        mask_keyboard,
        keys, white_keys_map, black_keys_map,
        final_notes, stop_point):

    '''
    stop_point =
        None,
        "shadows_white",
        "shadows_black",
        "hough_white",
        "hough_black",
    Different stop_point value indicates intermediate state of the pressed keys detection
    '''
    cv2.imwrite("zdacanc\\frame_a.png", frame)

    # white keys
    frame_shadows_white = cv2.bitwise_not(cv2.bitwise_or(cur_frame_segmented, mask_hand))
    frame_shadows_white = cv2.bitwise_and(frame_shadows_white, mask_keyboard)
    if stop_point != None and stop_point == "shadows_white":
        return frame_shadows_white

    # black keys
    frame_shadows_black = cv2.bitwise_not(mask_hand)
    frame_shadows_black = cv2.bitwise_and(frame_shadows_black, mask_keyboard)
    frame_shadows_black = cv2.bitwise_xor(frame_shadows_black, cur_frame_segmented)
    frame_shadows_black = cv2.bitwise_xor(frame_shadows_white, frame_shadows_black)
    if stop_point != None and stop_point == "shadows_black":
        return frame_shadows_black
    
    functions = [
        {"function": candidateKeys, "args": [frame_shadows_white, keys]},
        {"function": candidateKeys, "args": [frame_shadows_black, keys, False]}
    ]
    results = [None]*len(functions)
    threadMultiple(functions, results)
    candidate_white_keys, frame_hough_white = results[0]
    if stop_point != None and stop_point == "hough_white":
        return frame_hough_white

    candidate_black_keys, frame_hough_black = results[1]
    if stop_point != None and stop_point == "hough_black":
        return frame_hough_black

    # Draw the finger tips landmarks
    hand_landmarks = mpHandTrackLandmarks(frame)
    # get finger tips: https://google.github.io/mediapipe/solutions/hands.html#multi_hand_landmarks
    finger_tips = ["THUMB_TIP","INDEX_FINGER_TIP","MIDDLE_FINGER_TIP","RING_FINGER_TIP","PINKY_TIP", "PINKY_DIP", "THUMB_IP"]
    if bool(hand_landmarks): # not empty dictionary
        notes_to_play = []
        for landmark in finger_tips:
            cv2.circle(frame, hand_landmarks[landmark+"_0"], 5, (255, 0, 0), -1)
            pointInKeyContour(hand_landmarks[landmark+"_0"], frame, notes_to_play,
                              candidate_white_keys, candidate_black_keys,
                              white_keys_map, black_keys_map)
            if landmark+"_1" in hand_landmarks:
                cv2.circle(frame, hand_landmarks[landmark+"_1"], 5, (255, 0, 0), -1)
                pointInKeyContour(hand_landmarks[landmark+"_1"], frame, notes_to_play,
                                  candidate_white_keys, candidate_black_keys,
                                  white_keys_map, black_keys_map)
        final_notes.append({"frame": len(final_notes), "notes": notes_to_play})

    # cv2.imwrite("zdacanc\\frame_shadows_white.png", frame_shadows_white)
    # cv2.imwrite("zdacanc\\frame_shadows_black.png", frame_shadows_black)
    # cv2.imwrite("zdacanc\\frame_hough_white.png", frame_hough_white)
    # cv2.imwrite("zdacanc\\frame_hough_black.png", frame_hough_black)
    cv2.imwrite("zdacanc\\frame_z.png", frame)
    
    return frame

def generateMidiFile(final_notes, title, fps):
    # convert the list of notes found to suitable format for MIDI
    final_notes_by_time = {}
    for i in range(len(final_notes)):
        for note in final_notes[i]["notes"]:
            if not note in final_notes_by_time:
                final_notes_by_time[note] = []
            else:
                final_notes_by_time[note].append(i)

    final_notes_intervals = { k: [] for k in range(len(final_notes))}
    for k in final_notes_by_time:
        frames = final_notes_by_time[k]
        i = 0
        while i < len(frames):
            start = frames[i]
            j = i
            while j+1 < len(frames) and frames[j+1]-frames[j] == 1:
                j += 1
            end = frames[j]
            i = j+1
            final_notes_intervals[start].append({"note": k, "start": start, "end": end})

    # create your MIDI object
    mf = MIDIFile(1)     # only 1 track
    track = 0   # the only track

    time = 0    # start at the beginning
    mf.addTrackName(track, time, "Sample Track")
    # tempo = BPM = 60*fps
    video_duration_sec = (final_notes[-1]["frame"]-final_notes[0]["frame"])/24
    mf.addTempo(track, time, fps*60)

    # add some notes
    channel = 0
    volume = 100

    for k in final_notes_intervals:
        for note_obj in final_notes_intervals[k]:
            pitch = note_obj["note"]
            time = note_obj["start"]
            duration = note_obj["end"]-note_obj["start"]+1
            mf.addNote(track, channel, pitch, time, duration, volume)

    # write it to disk
    with open(title, 'wb') as outf:
        mf.writeFile(outf)