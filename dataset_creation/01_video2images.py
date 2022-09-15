import os
import cv2

video_path = "/home/seb/Datasets/sopra/videos/IMG_1270.MOV"
img_save_location = "/home/seb/Datasets/sopra/source_images"

first_label = 1396     # define the start of the numbering sequence
max_image_count = 0  # define how many images should be extracted (0: no limit)
skip_frames = 10

# Load video
video = cv2.VideoCapture(video_path)
success, image = video.read()

count = 0
skip_count = skip_frames

while success:
  if skip_count == skip_frames:
    print('Saved frame Nr.', count, '  Label:', (first_label+count))
    cv2.imwrite(os.path.join(img_save_location, 'frame_%04d.jpg' % (first_label+count)), image)
    skip_count = 0
    count += 1
  else:
    skip_count += 1

  if max_image_count != 0 and count == max_image_count:
    break

  # Read in next frame
  success, image = video.read()

print("Finished")