import gradio as gr
import os


import gradio as gr
from PIL import Image
import os
import cv2
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Contour_processing:

    def generate_curve_points(self, start_point, end_point, control_point):
        t = np.linspace(0, 1, 100)
        curve_points = [(int((1 - x) ** 2 * start_point[0] + 2 * (1 - x) * x * control_point[0] + x ** 2 * end_point[0]),
                        int((1 - x) ** 2 * start_point[1] + 2 * (1 - x) * x * control_point[1] + x ** 2 * end_point[1]))
                        for x in t]

        return curve_points
    
    def filter_contours_by_coordinates(self, contours, top_contours, point1, point2):
        # Extract row numbers from the provided points
        row1, row2 = point1[1], point2[1]

        # Determine the target row as the greater of the two
        target_row = max(row1, row2)

        # Filter contours based on coordinates
        filtered_contours = []

        for contour in contours:
            # Check if the contour contains points within the specified row range
            has_points_in_row_range = any(
                target_row - 100 <= point[0][1] <= target_row + 200
                for point in contour
            )

            # Check if both conditions are met
            if has_points_in_row_range:
                filtered_contours.append(contour)

        # Sort the filtered contours based on length in descending order
        sorted_contours = sorted(filtered_contours, key=len, reverse=True)

        # Select the top contours
        selected_contours = sorted_contours[:top_contours]

        return selected_contours
    
    def is_point_in_contour(self, point, contour):
        pt = tuple([int(round(point[0]) ), int(round( point[1] )) ])
        return cv2.pointPolygonTest(contour, pt, False) >= 0

    def get_filtered_contours(self, image_org, mask_org, point1, point2, threshold=150, top_contours=5,):
        # Read the image and mask
        # image = cv2.resize(image_org, (3008, 1688))
        # image = image[:1288, :, :]
        mask = mask_org[:,:,1]

        # Convert the image to grayscale
        gray = cv2.cvtColor(image_org, cv2.COLOR_BGRA2GRAY)

        # Threshold the image
        thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]

        thresh[mask == 255] = 0

        # Find contours in the image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        point2_gap = 100 - point2[0]
        point2_original_width = 3008 - point2_gap
        point2_new= (point2_original_width,point2[1])

        # Check if point1 and point2 are present in any of the contours
        point1_in_contour = any(self.is_point_in_contour(point1, contour) for contour in contours)
        point2_in_contour = any(self.is_point_in_contour(point2_new, contour) for contour in contours)

        # Filter contours based on coordinates
        filtered_contours = self.filter_contours_by_coordinates(contours, top_contours,point1,point2)

        if not point1_in_contour or not point2_in_contour:
            if threshold<240:
                print("recursed")
                threshold_value = threshold+20
                filtered_contours = self.get_filtered_contours(image_org, mask_org, point1, point2, threshold=threshold_value, top_contours=5)


        cont = cv2.drawContours(image_org, filtered_contours, -1, (0, 255, 0), 3)
        cv2.imwrite("cont.jpg",cont)

        return filtered_contours

    def match_points_with_contours(self, contours, curve_points):
        matched_points = 0

        for curve_point in curve_points:
            curve_point = np.array(curve_point, dtype=np.float32)

            for contour in contours:
                contour_points = np.array(contour, dtype=np.float32)
                distances = cv2.pointPolygonTest(contour_points, tuple(curve_point), True)

                if distances >= 0:
                    matched_points += 1
                    break

        return matched_points

    def find_optimal_height(self, filtered_contours,start_point,end_point):
        max_matched_points = 0
        curve_height = 0.16
        equal_count = 0

        for i in range(11, 29):
            i_value = i / 100.0
            # Generate curve points based on the parameters
            control_point = (1502, int(start_point[1] - i_value * 1688))
            curve_points_to_try = self.generate_curve_points(start_point, end_point, control_point)
            # Match curve points with contours
            matched_points = self.match_points_with_contours(filtered_contours, curve_points_to_try)

            print(f"For i = {i_value}, Number of matched points: {matched_points}")

            # Update optimal_i if the current i gives more matched points
            if matched_points > max_matched_points:
                max_matched_points = matched_points
                curve_height = i_value
            elif matched_points == max_matched_points and matched_points!=0:
                equal_count+=1
                if equal_count>3:
                    curve_height = i_value
                    equal_count=0

        return curve_height

    def find_points_with_extreme_width(self, image,threshold=150):
        try:
            first_5_columns = image[:-400, :50, :]
            last_5_columns = image[:-400, -50:, :]
            new_image = np.concatenate((first_5_columns, last_5_columns), axis=1)
        

            # Convert the image to grayscale
            gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)


            # Threshold the image
            thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]

            # Apply morphological operations (example: dilation)
            kernel = np.ones((5,5),np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)

            # Find contours in the image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            contours = sorted(contours, key=len)
            contours = contours[-2:]

            cont = cv2.drawContours(new_image, contours, -1, (0, 255, 0), 3)
            cv2.imwrite("cont_max.jpg",cont)

            # Initialize variables to store the points with the lowest and highest width
            min_width_point = None
            max_width_point = None
            min_width = float('inf')  # Initialize with a large value
            max_width = -1  # Initialize with a small value

            for contour in contours:
                # Iterate through points within the contour
                for point in contour:
                    x, y = point[0]

                    # Update min_width_point if current width is smaller
                    if x < min_width or (x == min_width and y > min_width_point[1]):
                        min_width = x
                        min_width_point = (x, y)

                    # Update max_width_point if current width is larger
                    if x > max_width or (x == max_width and y > max_width_point[1]):
                        max_width = x
                        max_width_point = (x, y)

            print(min_width_point, max_width_point)

            if(min_width_point is None or max_width_point is None or min_width_point[0] >= 30 or max_width_point[0] <= 60):
                print("entered")
                threshold_value = threshold+10
                min_width_point,max_width_point = self.find_points_with_extreme_width(image,threshold=threshold_value)

            return min_width_point, max_width_point
        except:
            return[0,850],[3007,950]
        

class Background:

    def background_generator(self, white, left_max_row, right_max_row, curve_point,curve_height,thickness):
        height, width = 1688, 3008

        background_color = (253, 252, 252)
        image = np.full((height, width, 3), background_color, dtype=np.uint8)

        line_color = (0, 0, 0)
        line_thickness = thickness

        shadow_color = (white, white, white)
        shadow_thickness = 600

        start_point = (0, left_max_row)
        end_point = (width, right_max_row)
        control_point = (curve_point, int(left_max_row - curve_height * height))

        shadow_start_point = (0, left_max_row+(shadow_thickness//2)+(thickness//2))
        shadow_end_point = (width, right_max_row+(shadow_thickness//2)+(thickness//2))
        shadow_control_point = (curve_point, int(left_max_row - (curve_height/2) * height))

        height, width = image.shape[:2]

        midpoint = left_max_row if left_max_row>right_max_row else right_max_row

        upper_half = image[:midpoint, :]
        lower_half = image[midpoint:, :]

        lower_half[:] = (white,white,white)


        t = np.linspace(0, 1, 100)
        curve_points = [(int((1 - x) ** 2 * start_point[0] + 2 * (1 - x) * x * control_point[0] + x ** 2 * end_point[0]),
                        int((1 - x) ** 2 * start_point[1] + 2 * (1 - x) * x * control_point[1] + x ** 2 * end_point[1]))
                        for x in t]


        p = np.linspace(0, 1, 100)
        curve2_points = [(int((1 - x) ** 2 * shadow_start_point[0] + 2 * (1 - x) * x * shadow_control_point[0] + x ** 2 * shadow_end_point[0]),
                        int((1 - x) ** 2 * shadow_start_point[1] + 2 * (1 - x) * x * shadow_control_point[1] + x ** 2 * shadow_end_point[1]))
                        for x in t]


        for i in range(1, len(curve_points)):
            cv2.line(image, curve2_points[i - 1], curve2_points[i], shadow_color, shadow_thickness, cv2.LINE_AA)

        for i in range(1, len(curve_points)):
            cv2.line(image, curve_points[i - 1], curve_points[i], line_color, line_thickness, cv2.LINE_AA)

        # Apply Gaussian blur to the entire image
        blur_radius = 9
        image = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)

        return image
    

    def merging(self, car_path,background_path):
        car = Image.open(car_path)
        background = Image.open(background_path)
        car = car.resize(background.size)
        merged_image = Image.alpha_composite(background.convert('RGBA'), car.convert('RGBA'))
        return merged_image

    def generate_and_merge(self, white,left,right,control,curve,thickness):
        background = self.background_generator(white,left,right,control,curve,thickness)
        cv2.imwrite("background.jpg", background)

        # Merge the extracted image with the new background
        final_image_path = f"results/extracted/input_image.png"
        merged_image = self.merging(final_image_path, "background.jpg")
        merged_image.save("results/merged_user.png")

        complete = cv2.imread("results/merged_user.png")

        complete = cv2.cvtColor(complete, cv2.COLOR_RGB2BGR)
        
        return complete

    def overlay_images(self, background, opacity):        
        if background=="Auto-Generated":
            png_path = "results/merged_auto.png"
        else:
            png_path = "results/merged_user.png"

        jpg_img = cv2.imread("results/temp.jpg",cv2.IMREAD_UNCHANGED)
        png_img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        png_img = cv2.resize(png_img, (jpg_img.shape[1], jpg_img.shape[0]))
        png_img = png_img[:,:,:3]
        merged = cv2.addWeighted(png_img, opacity, jpg_img, 1 - opacity, 0)
        merged = cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)
        return merged
    
class Prediction_model:
    def __init__(self):
        self.W = 512
        self.H = 512
        model_path = "experiment_alpha_only.h5"
        self.extraction_model = tf.keras.models.load_model(model_path)
        self.background = Background()
        self.image_processor = Contour_processing()

    def read_image(self, path):
        # path = path.decode()
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        first_5_columns = x[500:-400, :50, :]
        last_5_columns = x[500:-400, -50:, :]
        new_image = np.concatenate((first_5_columns, last_5_columns), axis=1)
        x = cv2.resize(new_image, (self.H, self.W))
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        return x

    def reshaping(self, original_image):
        image = cv2.resize(original_image,(512,788))

        height, width = image.shape[:2] 

        top_rows = 500
        bottom_rows = 400

        empty_row = np.zeros((1, width), dtype=np.uint8)

        for _ in range(top_rows):
            image = np.vstack((empty_row, image))

        for _ in range(bottom_rows):
            image = np.vstack((image, empty_row))

        return image
    
    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def prediction(self, image_org):
        """ Directory for storing files """
        for item in ["joint", "mask", "extracted"]:
            self.create_dir(f"results/{item}")
        
        name = "input_image"
        image_path = "results/temp.jpg" 

        image_org = cv2.resize(image_org,(3008,1688))
        image = cv2.cvtColor(image_org, cv2.COLOR_RGB2BGR)
        # image = cv2.resize(image,(3008,1688))
        cv2.imwrite(image_path,image)
        x = cv2.resize(image, (self.W, self.H))
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        pred = self.extraction_model.predict(x, verbose=0)

        """ Save final mask """
        image_h, image_w, _ = image.shape

        y0 = pred[0][0]
        y0 = cv2.resize(y0, (image_w, image_h))
        y0 = np.expand_dims(y0, axis=-1)
        ny = np.where(y0 > 0, 1, y0)

        rgb = image[:, :, 0:3]
        alpha = y0 * 255

        final = np.concatenate((rgb.copy(), alpha), axis=2)
        yy = cv2.merge((ny.copy(), ny.copy(), ny.copy(), y0.copy()))
        mask = yy * 255

        final_image_path = f"results/extracted/{name}.png"
        mask_image_path = f"results/mask/{name}.png"

        cv2.imwrite(final_image_path, final)
        cv2.imwrite(mask_image_path, mask)

        # Read both images with IMREAD_UNCHANGED
        final_image = cv2.imread(final_image_path, cv2.IMREAD_UNCHANGED)
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)

        # Convert to RGB color space
        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGRA2RGBA)

        #gpt suggetion
        image_org = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)

        min_width_idx, max_width_idx = self.image_processor.find_points_with_extreme_width(image_org,threshold=140)
        contours = self.image_processor.get_filtered_contours(image_org,mask_image,min_width_idx,max_width_idx)
        height = self.image_processor.find_optimal_height(contours,min_width_idx,max_width_idx)

        # Generate the background
        background = self.background.background_generator(244,min_width_idx[1],max_width_idx[1],1502,height,27)
        cv2.imwrite("background.jpg", background)

        # Merge the extracted image with the new background
        final_image_path = f"results/extracted/input_image.png"
        merged_image = self.background.merging(final_image_path, "background.jpg")
        merged_image.save("results/merged_auto.png")

        complete = cv2.imread("results/merged_auto.png")

        complete = cv2.cvtColor(complete, cv2.COLOR_RGB2BGR)
        
        return final_image_rgb, mask_image, complete,min_width_idx[1], max_width_idx[1],height
    
predictor = Prediction_model()
background_maker = Background()
image_processor = Contour_processing()
    
with gr.Blocks() as demo:

    gr.Markdown("# Backgroud generation uses Contours")
    with gr.Row():
        im = gr.Image(label="Upload")
        im_2 = gr.Image(label="Extracted Car")
        im_3 = gr.Image(label="White Mask")
    with gr.Row():
        with gr.Column(scale=1):
            btn = gr.Button(value="Extract")
            left_value = gr.Textbox(value="", label="Left Point")
            right_value = gr.Textbox(value="", label="Right Point")
            height = gr.Textbox(value="", label="Radius")
        with gr.Column(scale=1):
            im_4 = gr.Image(label="Auto-Generated Background")

    btn.click(predictor.prediction, inputs=[im], outputs=[im_2,im_3,im_4, left_value, right_value,height])
    
    gr.Markdown("# Create your background here")
    with gr.Row():       
        with gr.Column(scale=1):
            white = gr.Slider(200, 255, value=240, step=1, label="Brightness Of Floor:")
            left_max_count = gr.Slider(1, 1688, value=800, step=1, label="Left point:")
            right_max_count = gr.Slider(1, 1688, value=900, step=1, label="Right Point:")
            curve_height_slider = gr.Slider(0, 1, step=0.01, value=0.12, label="Curve Height")
            control = gr.Slider(1, 3008, value=1502, step=1, label="Highest Curve Point")
            thickness = gr.Slider(1,50, value=27, step=1, label="Black Strip Thickness" )
            
        with gr.Column(scale=1):
            im = gr.Image(label="User-Generated Background")
            btn = gr.Button(value="Create Background")

    btn.click(background_maker.generate_and_merge, inputs=[white,left_max_count, right_max_count,control,curve_height_slider,thickness], outputs=[im])

    gr.Markdown("# Merge New Image with Original")
    with gr.Row():
        with gr.Column(scale=1):
            background = gr.Radio(["Auto-Generated", "User-Generated"], label="Background", info="Select which image to merge with original")
            opacity = gr.Slider(0,1, step=0.1, value=0.8, label="Opacity of Generated Image")
            btn = gr.Button(value="Merge Images")
        with gr.Column(scale=1):
            mer_im = gr.Image(label="Merged Image")
    btn.click(background_maker.overlay_images,inputs=[background,opacity],outputs=[mer_im])

if __name__ == "__main__":
    demo.launch()