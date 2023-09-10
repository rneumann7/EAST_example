import cv2
import os
import numpy as np
from tkinter import Label, Frame, Button, Tk, LEFT
from PIL import ImageTk, Image
import decodeBox as cv2DecodeBoxes
from tkinter import filedialog as fd


class App:
    """This class is the main class of the application. 
    It creates the window and the widgets and handles the button events.
    """

    def __init__(self):
        # set path to temp_image folder set start image for view in gui
        self.temp_im_path = os.getcwd() + "/temp_image/"
        self.preview_image = Image.open(os.getcwd() + "/example_images/start.jpg")
        # create window, set title, set geometry
        self.root = Tk()
        self.root.title("Scene Text Detection")
        self.root.geometry("1920x1080")

        # create widgets, split layout in upper and lower frame
        self.upper_frame = Frame(self.root, width=1920, height=880)
        self.lower_frame = Frame(self.root, width=1920, height=200)
        self.start_im = ImageTk.PhotoImage(self.preview_image)
        self.image = Label(self.upper_frame, image=self.start_im)
        self.image.image = self.start_im
        self.btn_load = Button(self.lower_frame,text="Load Image",command=self.select_file, width=10,height=5,bg="White",fg="Black",)
        self.btn_nms = Button(self.lower_frame, text="NMS",command= lambda: self.detect_text(False,True), width=10,height=5,bg="White",fg="Black",)
        self.btn_bounding = Button(self.lower_frame,text="Bounding Boxes", command= lambda: self.detect_text(False, False), width=10,height=5,bg="White",fg="Black",)
        self.btn_score = Button(self.lower_frame,text="Score Map", command= lambda: self.detect_text(True, False), width=10,height=5,bg="White",fg="Black",)

        # use pack() layout manager, this centers everything which is very easy
        self.upper_frame.pack()
        self.lower_frame.pack()
        self.image.pack()
        self.btn_load.pack(side=LEFT, padx= 50, pady=25)
        self.btn_score.pack(side=LEFT, padx= 50, pady=25)
        self.btn_bounding.pack(side=LEFT, padx= 50, pady=25)
        self.btn_nms.pack(side=LEFT, padx= 50, pady=25)
        
    # downsize image to max_width and max_height for image display in gui
    def downsize_image(self, image, max_width, max_height):
        if len(image.shape) == 2:
            height, width = image.shape
        else:
            height, width, _ = image.shape
        # Calculate the aspect ratio
        aspect_ratio = width / height

        # Determine the new dimensions while preserving the aspect ratio
        if width > height:
            new_width = min(width, max_width)
            new_height = int(new_width / aspect_ratio)
            if new_height > max_height:
                new_height = min(height, max_height)
                new_width = int(new_height * aspect_ratio)
        else:
            new_height = min(height, max_height)
            new_width = int(new_height * aspect_ratio)

        # downsize the image using OpenCV
        downsized_image = cv2.resize(image, (new_width, new_height))
        return downsized_image

    # open file dialog to select image, set temp image and display image in window
    def select_file(self):
        filetypes = (
            ('jpg files', '*.jpg'),
            ('png files', '*.png')
        )

        image_path = fd.askopenfilename(
            title='Open an image',
            initialdir= os.getcwd() + "/example_images/",
            filetypes=filetypes)
        
        self.set_temp_image(image_path)
        self.update_image(cv2.imread(image_path))

    # update image in gui window
    def update_image(self, image):
        downsized_image = self.downsize_image(image, 1900, 800)
        self.preview_image = ImageTk.PhotoImage(self.cv_to_pil(downsized_image))
        self.image.config(image=self.preview_image)
        self.image.image = self.preview_image

    # save temp image in the temp_image folder
    def set_temp_image(self, im_path):
        image = cv2.imread(im_path)
        cv2.imwrite(self.temp_im_path + "temp.jpg", image)

    # convert opencv image to pil image
    def cv_to_pil(self, im):
        # BGR to RGB image
        image_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # numpy array to PIL Image
        pil_image = Image.fromarray(image_rgb)
        return pil_image

    def detect_text(self, score_flag, nms):
        # Settings
        confThreshold = 0.5
        nmsThreshold = 0.4
        inpWidth = 960
        inpHeight = 960
        bbox_line_width = 2

        # Load the detection model and create a list of output names
        outputNames = []
        detection_model = cv2.dnn.readNet("EAST.pb")
        outputNames.extend([
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"])

        # Open the current temp image
        img = cv2.imread(self.temp_im_path + "temp.jpg")

        # Get image height and width
        height_ = img.shape[0]
        width_ = img.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)

        # Create a blob from image and run the model
        blob = cv2.dnn.blobFromImage(img, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)
        detection_model.setInput(blob)
        model_output = detection_model.forward(outputNames)

        # Get scores and geometry
        scores = model_output[0]
        geometry = model_output[1]

        if score_flag:
            # Create score map image
            score_map = np.array(scores[0][0])
            result = np.where(score_map > 0.5, 255, 0)
            score_map_im = result.astype(np.uint8)
            score_map_im = cv2.resize(score_map_im, (width_, height_))
            self.update_image(score_map_im)

        else: 
            # decode bounding boxes to readable opencv format
            [boxes, confidences] = cv2DecodeBoxes.decodeBoundingBoxes(scores, geometry, confThreshold)
            if nms:
                # Apply NMS
                indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)        
            else:
                # Take all boxes
                indices = list(range(len(boxes)))

            for i in indices:
                    # get 4 corners of the rotated rect
                    corner_points = cv2.boxPoints(boxes[i])
                    # scale the bounding box coordinates back
                    for j in range(4):
                        corner_points[j][0] *= rW
                        corner_points[j][1] *= rH
                    # draw lines for the bounding box
                    for j in range(4):
                        p1 = (int(corner_points[j][0]), int(corner_points[j][1]))
                        p2 = (int(corner_points[(j + 1) % 4][0]), int(corner_points[(j + 1) % 4][1]))
                        cv2.line(img, p1, p2, (0, 255, 0), thickness=bbox_line_width)
        
            self.update_image(img)


if __name__ == "__main__":
    app = App()
    app.root.mainloop()