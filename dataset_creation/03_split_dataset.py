import os
import shutil

# Input
SOURCE_IMG =   "/home/seb/Datasets/sopra/augmented_images"
SOURCE_MASKS = "/home/seb/Datasets/sopra/augmented_masks"

# Output
TRAIN_IMG =   "/home/seb/Datasets/sopra_train/images"
TRAIN_MASKS = "/home/seb/Datasets/sopra_train/masks"

TEST_IMG =   "/home/seb/Datasets/sopra_test/images"
TEST_MASKS = "/home/seb/Datasets/sopra_test/masks"

n_files = 4724

for i in range(n_files):
    print(i, "/", n_files)
    if i % 10 <= 8:
        # Copy to train
        shutil.copy2(os.path.join(SOURCE_IMG, 'img_{}.png'.format(i)), os.path.join(TRAIN_IMG, 'img_{}.png'.format(i)))
        shutil.copy2(os.path.join(SOURCE_MASKS, 'mask_{}.png'.format(i)), os.path.join(TRAIN_MASKS, 'mask_{}.png'.format(i)))
    if i % 10 == 9:
        # Copy to test
        shutil.copy2(os.path.join(SOURCE_IMG, 'img_{}.png'.format(i)), os.path.join(TEST_IMG, 'img_{}.png'.format(i)))
        shutil.copy2(os.path.join(SOURCE_MASKS, 'mask_{}.png'.format(i)), os.path.join(TEST_MASKS, 'mask_{}.png'.format(i)))
