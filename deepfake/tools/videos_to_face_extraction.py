import cv2
import os
import mediapipe as mp
from tqdm import tqdm
import argparse
import random

def parse_arg():
    parser = argparse.ArgumentParser(description="Input the video and output the frames contain faces")
    parser.add_argument('video_folder')
    parser.add_argument('mask_folder')
    parser.add_argument('frame_save_path')
    parser.add_argument('mask_save_path')
    parser.add_argument('--frames_num',default=20)
    args = parser.parse_args()
    return args

def main(args):
    # make save dirs
    if not os.path.exists(args.frame_save_path):
        os.makedirs(args.frame_save_path)
    if not os.path.exists(args.mask_save_path):
        os.makedirs(args.mask_save_path)
    # init the mediapipe
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    # list all the videos & masks
    video_list = os.listdir(args.video_folder)
    mask_list  = os.listdir(args.mask_folder)
    # save number
    save_num = 0

    # start read video_list
    with mp_face_detection.FaceDetection(min_detection_confidence=0.9) as face_detection:
        for video in tqdm(video_list):
            # read video & mask one by one
            # print(f"Detect the face in {video}.")
            video_pth = os.path.join(args.video_folder,video)
            mask_pth  = os.path.join(args.mask_folder,video)
            video   = cv2.VideoCapture(video_pth)
            m_video = cv2.VideoCapture(mask_pth)
            # get the random frames
            video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_list = random.sample(range(video_frame_count),args.frames_num)
            frames_list = sorted(frames_list)
            print(frames_list)
            ret, frame = video.read()
            _,   mask  = m_video.read()
            cnt = 0
            while ret:
                # extract frame in the frames_list. 
                if cnt in frames_list:
                    height, width, _ = frame.shape
                    results = face_detection.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    if not results.detections:
                        break
                    # possible there are more than one bbox
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        # mp_drawing.draw_detection(annotated_img, detection)
                        x = round(bbox.xmin*width*0.8)
                        y = round(bbox.ymin*height*0.4)
                        width_bbox  = round(bbox.width*width*1.8)
                        height_bbox = round(bbox.height*height*1.8)
                    # save the result image
                    frame_save_pth = os.path.join(args.frame_save_path, str(save_num)+'.png')
                    mask_save_path = os.path.join(args.mask_save_path,  str(save_num)+'.png')
                    cv2.imwrite(frame_save_pth, frame[y:y+height_bbox,x:x+width_bbox,:])
                    # for the faceswap, modify the mask into binary imgs
                    mask = 255*(mask[y:y+height_bbox,x:x+width_bbox,:]!=0)
                    cv2.imwrite(mask_save_path, mask)
                    save_num += 1
                ret, frame = video.read()
                _,   mask  = m_video.read()
                cnt += 1
            video.release()
            m_video.release()


if __name__=="__main__":
    args = parse_arg()
    main(args)

'''
command take-away note:
python tools/videos_to_face_extraction.py \
/media/large_storage/Jiyuan_Shen/editable/manipulated_sequences/FaceSwap/c23/videos/ \
/media/large_storage/Jiyuan_Shen/editable/manipulated_sequences/FaceSwap/masks/videos/ \
editable/FaceSwap/imgs/ editable/FaceSwap/masks/
'''