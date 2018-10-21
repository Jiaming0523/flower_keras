from PIL import Image
import os

def read_and_prep_images(img_paths, img_height, img_width): 
	imgs = [load_img(img_path, target_size=(img_height, img_width)) 	for img_path in img_paths] 
	    return np.array([img_to_array(img) for img in imgs])

test_data = read_and_prep_images(image_names[0:10]) preds = model_1.predict(test_data)

for i, img_path in enumerate(image_names): display(Image(img_path)) print(preds[i])

