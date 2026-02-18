import cv2
import numpy as np
import os

# Crop diamond ore from annotated data
def crop_obb(image_path, label_path, output_dir):
    # Load image, get image width & height
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Read text file containing dimension data for each boudning box in an image
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Each line corresponds to a bounding box
    for i, line in enumerate(lines):
        values = line.strip().split()
        class_id = values[0]
        # OBB outputs bounding boxes as 4 corner coordinates, normalized between 0 and 1, in the below format:
        # class x1 y1 x2 y2 x3 y3 x4 y4 (normalized )
        coords = list(map(float, values[1:]))
        
        # Denormalize coordinates
        points = []
        for j in range(0, len(coords), 2):
            x = int(coords[j] * w)
            y = int(coords[j+1] * h)
            points.append([x, y])
        
        points = np.array(points, dtype=np.float32)
        
        # Get the center point, size and rotation angle of the area inside the bounding box
        rect = cv2.minAreaRect(points)
        # Get the corners of the area inside the bounding box
        box = cv2.boxPoints(rect)
        box = np.float32(box)
        
        # Get width and height of the area inside the bounding box
        width = int(rect[1][0])
        height = int(rect[1][1])
        
        # Set the  4 corner coordinates of the area inside the bounding box so the image is un-rotated (clockwise from bottom left to bottom right)
        output_points = np.array([
            [0, height-1], [0, 0],
            [width-1, 0], [width-1, height-1]
            ], dtype=np.float32)
        
        # produces the transformation of the area inside the bounding box so it matches the un-rotated corners,
        #    as a 3x3 matrix for consumption by warpPerspective
        M = cv2.getPerspectiveTransform(box, output_points)

        # straigheten the image according to the transformation M and crop to the area within the bounding box
        cropped = cv2.warpPerspective(image, M, (width, height))
        
        # Resize to target size
        size = 128
        cropped = cv2.resize(cropped, (size, size))
        
        # name the file using the original filename and the current bounding box iteration
        filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}.png"

        # save the file to the output directory
        cv2.imwrite(os.path.join(output_dir, filename), cropped)
    
def main():
    # Set input and output directories
    image_dir = "Data/prototype_132/images"
    label_dir = "Data/prototype_132/labels/"
    output_dir = "Output/cropped/"
    os.makedirs(output_dir, exist_ok=True)

    # Loop through images in image directory
    for image_file in os.listdir(image_dir):
        if image_file.endswith(".png"):
            # Get path of current image
            image_path = os.path.join(image_dir, image_file)
            # Get path to label file associated with current image
            label_path = os.path.join(label_dir, image_file.replace(".png", ".txt"))
            
            if os.path.exists(label_path):
                # Apply cropping and straightening to obtain each annotated area as cropped, scaled, straightened image
                crop_obb(image_path, label_path, output_dir)

if __name__ == "__main__":
    main()