import cv2
import imgaug as ia
from imgaug import augmenters as iaa

def main():
	im_path = "/home/ml/Pictures/frog.jpg"
	im_data = cv2.imread(im_path)

	aug_seq = iaa.Sequential(
            [
                iaa.Add((-20, 20)),
                iaa.ContrastNormalization((0.8, 1.6)),
                iaa.AddToHueAndSaturation((-21, 21)),
                iaa.SaltAndPepper(p=0.1),
                iaa.Scale({"width":500, "height":"keep-aspect-ratio"}, 1),
                iaa.CropAndPad(
	                percent=(-0.05, 0.1),
	                pad_mode=ia.ALL,
	                pad_cval=(0, 255)
            	)
            ],
            random_order=True)

	for i in range(100):
		im_data_aug = aug_seq.augment_image(im_data)	
		cv2.imshow("frame", im_data_aug)

		if cv2.waitKey(50) & 0xFF == ord('q'):
			break
	
	cv2.destroyAllWindows()


if __name__ == "__main__":
    if __name__ == "__main__":
    	main()
