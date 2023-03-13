import cv2 
import numpy as np

imgs = np.load('sample_0_pusht_state_images.npy')    
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_name = 'output.mp4'
frame_width = imgs[0].shape[0]
frame_height = imgs[0].shape[1]
fps = 30

out = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, frame_height))

for img in imgs:
    # frame = cv2.resize(img, (frame_width, frame_height))
    out.write(img)

out.release()