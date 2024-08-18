from tkinter import *
import cv2
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageTk
from resizeimage import resizeimage
from tkinter import messagebox
import matplotlib.pyplot as plt
import os

class App:
    def __init__(self, root):
        self.root = root
        self.output_dir = ''  
        self.input_image_path = ''
        self.root.title('CV Project | Authors: Muhammad Ilyas, Moavia Hassan')
        self.root.geometry('1200x700+80+1')
        self.root.resizable(False, False)
        self.heading = Label(text="Operations", font=("Helvetica", 20, 'bold'), fg="black")
        self.heading.place(x=95, y=10)
        # =========================   Images   ==========================
        self.add_noiseimg1 = PhotoImage(file='buttons/add_noise1.png')

        self.add_noiseimg2 = PhotoImage(file='buttons/add_noise2.png')

        self.blurimg1 = PhotoImage(file='buttons/blur1.png')
        self.blurimg2 = PhotoImage(file='buttons/blur2.png')

        self.cannyimg1 = PhotoImage(file='buttons/canny1.png')
        self.cannyimg2 = PhotoImage(file='buttons/canny2.png')

        self.harrisimg1 = PhotoImage(file='buttons/harris1.png')
        self.harrisimg2 = PhotoImage(file='buttons/harris2.png')

        self.hogimg1 = PhotoImage(file='buttons/hog1.png')
        self.hogimg2 = PhotoImage(file='buttons/hog2.png')

        self.kmeansimg1 = PhotoImage(file='buttons/kmeans1.png')
        self.kmeansimg2 = PhotoImage(file='buttons/kmeans2.png')

        self.laplacianimg1 = PhotoImage(file='buttons/laplacian1.png')
        self.laplacianimg2 = PhotoImage(file='buttons/laplacian2.png')

        self.marr_hildrethimg1 = PhotoImage(file='buttons/marr_hildreth1.png')
        self.marr_hildrethimg2 = PhotoImage(file='buttons/marr_hildreth2.png')

        self.prewittimg1 = PhotoImage(file='buttons/prewitt1.png')
        self.prewittimg2 = PhotoImage(file='buttons/prewitt2.png')

        self.prewittximg1 = PhotoImage(file='buttons/prewittx1.png')
        self.prewittximg2 = PhotoImage(file='buttons/prewittx2.png')

        self.prewittyimg1 = PhotoImage(file='buttons/prewitty1.png')
        self.prewittyimg2 = PhotoImage(file='buttons/prewitty2.png')

        self.remove_blurimg1 = PhotoImage(file='buttons/remove_blur1.png')
        self.remove_blurimg2 = PhotoImage(file='buttons/remove_blur2.png')

        self.remove_noiseimg1 = PhotoImage(file='buttons/remove_noise1.png')
        self.remove_noiseimg2 = PhotoImage(file='buttons/remove_noise2.png')

        self.siftimg1 = PhotoImage(file='buttons/sift1.png')
        self.siftimg2 = PhotoImage(file='buttons/sift2.png')

        self.sobelimg1 = PhotoImage(file='buttons/sobel1.png')
        self.sobelimg2 = PhotoImage(file='buttons/sobel2.png')

        self.sobelximg1 = PhotoImage(file='buttons/sobelx1.png')
        self.sobelximg2 = PhotoImage(file='buttons/sobelx2.png')

        self.sobelyimg1 = PhotoImage(file='buttons/sobely1.png')
        self.sobelyimg2 = PhotoImage(file='buttons/sobely2.png')

        self.browseimg1 = PhotoImage(file='buttons/browse1.png')
        self.browseimg2 = PhotoImage(file='buttons/browse2.png')

        # =========================   Buttons  ==========================
        
        self.noise = Button(root, image=self.add_noiseimg1, borderwidth=0, command=self.add_noise, cursor='hand2')
        self.noise.place(x=50, y=60)
        self.noise.bind('<Enter>', self.noiseEnter)
        self.noise.bind('<Leave>', self.noiseLeave)
        
        self.remove_noise_btn = Button(root, image=self.remove_noiseimg1, borderwidth=0, command=self.remove_noise, cursor='hand2')
        self.remove_noise_btn.place(x=190, y=60)
        self.remove_noise_btn.bind('<Enter>', self.removeNoiseEnter)
        self.remove_noise_btn.bind('<Leave>', self.removeNoiseLeave)
        
        self.blur = Button(root, image=self.blurimg1, borderwidth=0, command=self.add_blur,cursor='hand2')
        self.blur.place(x=50, y=110)
        self.blur.bind('<Enter>', self.blurEnter)
        self.blur.bind('<Leave>', self.blurLeave)
        
        self.remove_blur_btn = Button(root, image=self.remove_blurimg1, borderwidth=0, command=self.remove_blur, cursor='hand2')
        self.remove_blur_btn.place(x=190, y=110)
        self.remove_blur_btn.bind('<Enter>', self.removeBlurEnter)
        self.remove_blur_btn.bind('<Leave>', self.removeBlurLeave)
        
        self.sift = Button(root, image=self.siftimg1, borderwidth=0, command=self.apply_SIFT,cursor='hand2')
        self.sift.place(x=50, y=160)
        self.sift.bind('<Enter>', self.siftEnter)
        self.sift.bind('<Leave>', self.siftLeave)
        
        self.harris = Button(root, image=self.harrisimg1, borderwidth=0, command=self.apply_Harris,cursor='hand2')
        self.harris.place(x=190, y=160)
        self.harris.bind('<Enter>', self.harrisEnter)
        self.harris.bind('<Leave>', self.harrisLeave)
        
        self.canny = Button(root, image=self.cannyimg1, borderwidth=0,command=self.Canny_Edge_Detection, cursor='hand2')
        self.canny.place(x=50, y=210)
        self.canny.bind('<Enter>', self.cannyEnter)
        self.canny.bind('<Leave>', self.cannyLeave)
        
        self.hog = Button(root, image=self.hogimg1, borderwidth=0, command=self.apply_hog, cursor='hand2')
        self.hog.place(x=190, y=210)
        self.hog.bind('<Enter>', self.hogEnter)
        self.hog.bind('<Leave>', self.hogLeave)
        
        self.laplacian = Button(root, image=self.laplacianimg1, borderwidth=0, command=self.apply_Laplacian, cursor='hand2')
        self.laplacian.place(x=50, y=260)
        self.laplacian.bind('<Enter>', self.laplacianEnter)
        self.laplacian.bind('<Leave>', self.laplacianLeave)
        
        self.marr_hildreth = Button(root, image=self.marr_hildrethimg1, borderwidth=0, command=self.apply_Marr_Hildreth, cursor='hand2')
        self.marr_hildreth.place(x=190, y=260)
        self.marr_hildreth.bind('<Enter>', self.marr_hildrethEnter)
        self.marr_hildreth.bind('<Leave>', self.marr_hildrethLeave)
        
        self.prewitt = Button(root, image=self.prewittimg1, borderwidth=0, command=self.apply_prewitt,cursor='hand2')
        self.prewitt.place(x=50, y=310)
        self.prewitt.bind('<Enter>', self.prewittEnter)
        self.prewitt.bind('<Leave>', self.prewittLeave)
        
        self.sobel = Button(root, image=self.sobelimg1, borderwidth=0, command=self.apply_sobel, cursor='hand2')
        self.sobel.place(x=190, y=310)
        self.sobel.bind('<Enter>', self.sobelEnter)
        self.sobel.bind('<Leave>', self.sobelLeave)
        
        self.prewitt_x = Button(root, image=self.prewittximg1, borderwidth=0, command=self.apply_prewitt_x, cursor='hand2')
        self.prewitt_x.place(x=50, y=360)
        self.prewitt_x.bind('<Enter>', self.prewitt_x_Enter)
        self.prewitt_x.bind('<Leave>', self.prewitt_x_Leave)
        
        self.prewitt_y = Button(root, image=self.prewittyimg1, borderwidth=0, command=self.apply_prewitt_y, cursor='hand2')
        self.prewitt_y.place(x=190, y=360)
        self.prewitt_y.bind('<Enter>', self.prewitt_y_Enter)
        self.prewitt_y.bind('<Leave>', self.prewitt_y_Leave)
        
        self.sobel_x = Button(root, image=self.sobelximg1, borderwidth=0, command=self.apply_sobel_x, cursor='hand2')
        self.sobel_x.place(x=50, y=410)
        self.sobel_x.bind('<Enter>', self.sobel_x_Enter)
        self.sobel_x.bind('<Leave>', self.sobel_x_Leave)
        
        self.sobel_y = Button(root, image=self.sobelyimg1, borderwidth=0, command=self.apply_sobel_y, cursor='hand2')
        self.sobel_y.place(x=190, y=410)
        self.sobel_y.bind('<Enter>', self.sobel_y_Enter)
        self.sobel_y.bind('<Leave>', self.sobel_y_Leave)
        
        self.kMeans = Button(root, image=self.kmeansimg1, borderwidth=0, command=self.apply_kMeans, cursor='hand2')
        self.kMeans.place(x=120, y=470)
        self.kMeans.bind('<Enter>', self.KMeans_Enter)
        self.kMeans.bind('<Leave>', self.KMeans_Leave)
        # ==========================================================
        
        # Create a canvas widget
        self.canvas = Canvas(root, width=800, height=500)
        
        # Draw a vertical line
        self.canvas.create_line(50, 0, 50, 700, width=2)
        self.canvas.place(x=300, y=20)
        
        self.label1 = Label(self.canvas, text="Input Image", font=("Helvetica", 20, 'bold'))
        self.label1.place(x=210, y=0)

        self.label2 = Label(self.canvas, text="Output Image", font=("Helvetica", 20, 'bold'))
        self.label2.place(x=600, y=0)
        # ============================================================


        self.upload_img1 = Label(self.root, text="No Image\nAvailable", font=("times new roman", 15), bg="#3f51b5", fg="white", bd=1, relief=RIDGE)
        self.upload_img1.place(x=420, y=90, width=350, height=400)
        
        self.upload_img2 = Label(self.root, text="No Image\nAvailable", font=("times new roman", 15), bg="#3f51b5", fg="white", bd=1, relief=RIDGE)
        self.upload_img2.place(x=820, y=90, width=350, height=400)

        # ============================================================
        

        self.input_label  = Label(self.root, text="Input Image", font=("Helvetica", 22, 'bold'), bg="#023548", fg="white")
        self.input_label.place(x=0, y=550, relwidth=1)
       
        self.input_image_path_text = Text(root, font=("times new roman", 10), wrap=NONE, bd=2, relief=RIDGE, state=DISABLED, bg="lightyellow")
        self.input_image_path_text.place(x=150, y=620, width=750, height=40)

        scroolbar_for_select_input_image = Scrollbar(self.input_image_path_text, orient=HORIZONTAL, cursor="hand2")
        scroolbar_for_select_input_image.pack(side=BOTTOM, fill=X)

        self.input_image_path_text.config(xscrollcommand=scroolbar_for_select_input_image.set)

        scroolbar_for_select_input_image.config(command=self.input_image_path_text.xview)

        self.browse_btn = Button(root, image=self.browseimg1, borderwidth=0, cursor="hand2", command=self.load_image)
        self.browse_btn.place(x=950, y=619)
        self.browse_btn.bind('<Enter>', self.browse_Enter)
        self.browse_btn.bind('<Leave>', self.browse_Leave)
    # ==========================================================

    def noiseEnter(self, e):
        self.noise.config(image=self.add_noiseimg2)

    def noiseLeave(self, e):
        self.noise.config(image=self.add_noiseimg1)
        
    def removeNoiseEnter(self, e):
        self.remove_noise_btn.config(image=self.remove_noiseimg2)

    def removeNoiseLeave(self, e):
        self.remove_noise_btn.config(image=self.remove_noiseimg1)
        
    def blurEnter(self, e):
        self.blur.config(image=self.blurimg2)

    def blurLeave(self, e):
        self.blur.config(image=self.blurimg1)
        
    def removeBlurEnter(self, e):
        self.remove_blur_btn.config(image=self.remove_blurimg2)

    def removeBlurLeave(self, e):
        self.remove_blur_btn.config(image=self.remove_blurimg1)
        
    def siftEnter(self, e):
        self.sift.config(image=self.siftimg2)

    def siftLeave(self, e):
        self.sift.config(image=self.siftimg1)
        
    def harrisEnter(self, e):
        self.harris.config(image=self.harrisimg2)

    def harrisLeave(self, e):
        self.harris.config(image=self.harrisimg1)
        
    def cannyEnter(self, e):
        self.canny.config(image=self.cannyimg2)

    def cannyLeave(self, e):
        self.canny.config(image=self.cannyimg1)
        
    def hogEnter(self, e):
        self.hog.config(image=self.hogimg2)

    def hogLeave(self, e):
        self.hog.config(image=self.hogimg1)
        
    def laplacianEnter(self, e):
        self.laplacian.config(image=self.laplacianimg2)

    def laplacianLeave(self, e):
        self.laplacian.config(image=self.laplacianimg1)
        
    def marr_hildrethEnter(self, e):
        self.marr_hildreth.config(image=self.marr_hildrethimg2)

    def marr_hildrethLeave(self, e):
        self.marr_hildreth.config(image=self.marr_hildrethimg1)
        
    def prewittEnter(self, e):
        self.prewitt.config(image=self.prewittimg2)

    def prewittLeave(self, e):
        self.prewitt.config(image=self.prewittimg1)
        
    def sobelEnter(self, e):
        self.sobel.config(image=self.sobelimg2)

    def sobelLeave(self, e):
        self.sobel.config(image=self.sobelimg1)
        
    def prewitt_x_Enter(self, e):
        self.prewitt_x.config(image=self.prewittximg2)

    def prewitt_x_Leave(self, e):
        self.prewitt_x.config(image=self.prewittximg1)
        
    def prewitt_y_Enter(self, e):
        self.prewitt_y.config(image=self.prewittyimg2)

    def prewitt_y_Leave(self, e):
        self.prewitt_y.config(image=self.prewittyimg1)
        
    def sobel_x_Enter(self, e):
        self.sobel_x.config(image=self.sobelximg2)

    def sobel_x_Leave(self, e):
        self.sobel_x.config(image=self.sobelximg1)
        
    def sobel_y_Enter(self, e):
        self.sobel_y.config(image=self.sobelyimg2)

    def sobel_y_Leave(self, e):
        self.sobel_y.config(image=self.sobelyimg1)
        
    def KMeans_Enter(self, e):
        self.kMeans.config(image=self.kmeansimg2)

    def KMeans_Leave(self, e):
        self.kMeans.config(image=self.kmeansimg1)
        
    def browse_Enter(self, e):
        self.browse_btn.config(image=self.browseimg2)

    def browse_Leave(self, e):
        self.browse_btn.config(image=self.browseimg1)

        # ==========================================================
    def load_image(self):
        self.input_image_path = filedialog.askopenfilename(title="Open Image", filetypes=(
            ("JPG Files", "*.jpg"), ("PNG Files", "*.png"), ("All Files", "*.*")))
        self.input_image_path = self.input_image_path.replace("/", "\\")
        
        self.output_dir = os.path.dirname(self.input_image_path)
                
        if self.input_image_path:
            self.imagecv1 = cv2.imread(self.input_image_path)
            height1, width1 = self.imagecv1.shape[:2]
            self.get_image1 = Image.open(self.input_image_path)
            
            if width1 > 300 or height1 > 400:
                # set the scaling factors
                if width1 < 450 and height1 < 600: 
                    fx1 = 0.65
                    fy1 = 0.65
                else:
                    fx1 = 0.5
                    fy1 = 0.5
                # calculate the new image dimensions
                new_height1 = int(height1 * fy1)
                new_width1 = int(width1 * fx1)
                
                self.resized_image1 = resizeimage.resize_cover(self.get_image1, [new_width1, new_height1])
                self.final_loaded_image1 = ImageTk.PhotoImage(self.resized_image1)
            else:
                self.final_loaded_image1 = ImageTk.PhotoImage(self.get_image1)
                
            self.upload_img2.config(image="", text="No Image\nAvailable", font=(
                "times new roman", 15), bg="#3f51b5", fg="white", bd=1, relief=RIDGE)
            self.upload_img1.config(image=self.final_loaded_image1, bg='#f0f0f0')
            self.input_image_path_text.config(state=NORMAL)
            self.input_image_path_text.delete('1.0', END)
            self.input_image_path_text.insert('1.0', self.input_image_path)
            self.input_image_path_text.config(state=DISABLED)
        else:
            pass
        
    def show_results(self, output_image):
        self.image2 = output_image
        self.imagecv2 = cv2.imread(self.image2)
        height2, width2 = self.imagecv2.shape[:2]
        self.get_image2 = Image.open(self.image2)

        if width2 > 300 or height2 > 400:
            # set the scaling factors
            if width2 < 450 and height2 < 600:
                fx2 = 0.65
                fy2 = 0.65
            else:
                fx2 = 0.5
                fy2 = 0.5
            # calculate the new image dimensions
            new_height2 = int(height2 * fy2)
            new_width2 = int(width2 * fx2)

            self.resized_image2 = resizeimage.resize_cover(self.get_image2, [new_width2, new_height2])
            self.final_loaded_image2 = ImageTk.PhotoImage(self.resized_image2)
        else:
            self.final_loaded_image2 = ImageTk.PhotoImage(self.get_image2)
        self.upload_img2.config(image=self.final_loaded_image2, bg='#f0f0f0')
        
    def add_noise(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        image = cv2.imread(self.input_image_path)
        
        # Convert the image to floating point format
        image1 = image.astype(np.float32)

        noise = np.random.randn(*image1.shape) * 20

        noisy_image = image1 + noise

        # Convert the image back to unsigned 8-bit integer format
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        cv2.imwrite(os.path.join(f"{self.output_dir}", "noisy_image.jpg"), noisy_image)
        self.show_results(os.path.join(f"{self.output_dir}", "noisy_image.jpg"))

    def remove_noise(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        img = cv2.imread(self.input_image_path)

        # Apply the denoising function
        denoised_img = cv2.fastNlMeansDenoising(img, None, 12, 7, 21)
        cv2.imwrite(os.path.join(f"{self.output_dir}", 'denoised.jpg'), denoised_img)
        self.show_results(os.path.join(f"{self.output_dir}", 'denoised.jpg'))
        
    def add_blur(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        image = cv2.imread(self.input_image_path)
        # apply a Gaussian blur to the image with a kernel size of (9, 9)
        blurred = cv2.GaussianBlur(image, (3, 3), 0)

        cv2.imwrite(os.path.join(f"{self.output_dir}", 'blurred.jpg'), blurred)
        self.show_results(os.path.join(f"{self.output_dir}", 'blurred.jpg'))
        
    def remove_blur(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        image= cv2.imread(self.input_image_path)

        # Convert to grayscale for fast processing and good results
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur filter
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply unsharp mask filter
        unsharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

        cv2.imwrite(os.path.join(f"{self.output_dir}", 'Unblurred_image.jpg'), unsharp)
        self.show_results(os.path.join(f"{self.output_dir}", 'Unblurred_image.jpg'))

    def apply_SIFT(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        image = cv2.imread(self.input_image_path)

        # Create SIFT object
        sift = cv2.xfeatures2d.SIFT_create()

        # Detect SIFT features
        keypoints = sift.detect(image, None)

        # Draw SIFT features on image
        img_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
        cv2.imwrite(os.path.join(f"{self.output_dir}", 'sift.jpg'), img_with_keypoints)
        self.show_results(os.path.join(f"{self.output_dir}", 'sift.jpg'))
    def apply_Harris(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        image = cv2.imread(self.input_image_path, 0)

        # Define parameters for Harris Edge Detector
        block_size = 2
        ksize = 3
        k = 0.04

        # Apply Harris Edge Detector
        dst = cv2.cornerHarris(image, block_size, ksize, k)

        # Dilate result for visualization
        dst = cv2.dilate(dst, None)

        # Threshold the result and set pixels above the threshold to white
        thresh = 0.01 * dst.max()
        image[dst > thresh] = 255
        cv2.imwrite(os.path.join(f"{self.output_dir}", 'harris.jpg'), image)
        self.show_results(os.path.join(f"{self.output_dir}", 'harris.jpg'))
    def Canny_Edge_Detection(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        img = cv2.imread(self.input_image_path, cv2.IMREAD_GRAYSCALE)

        # Canny edge detector
        canny = cv2.Canny(img, 50, 100)

        cv2.imwrite(os.path.join(f"{self.output_dir}", 'Canny.jpg'), canny)
        self.show_results(os.path.join(f"{self.output_dir}", 'Canny.jpg'))      


    def apply_hog(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        image = cv2.imread(self.input_image_path, cv2.COLOR_BGR2GRAY)

        # Compute the HOG descriptor
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        (rects1, weights1) = hog.detectMultiScale(
            image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        for (x, y, w, h) in rects1:
            hog_result = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(f"{self.output_dir}", 'hog.jpg'), hog_result)
        self.show_results(os.path.join(f"{self.output_dir}", 'hog.jpg'))

    def apply_Laplacian(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        img = cv2.imread(self.input_image_path, cv2.IMREAD_GRAYSCALE)

        laplacian = cv2.Laplacian(img, cv2.CV_64F)

        cv2.imwrite(os.path.join(f"{self.output_dir}", "Laplacian.jpg"), laplacian)
        self.show_results(os.path.join(f"{self.output_dir}", "Laplacian.jpg"))
        
    def apply_Marr_Hildreth(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        img = cv2.imread(self.input_image_path, cv2.IMREAD_GRAYSCALE)

        sigma = 1
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        gaussian = cv2.GaussianBlur(img, (3, 3), sigma)
        marr_hildreth = laplacian - gaussian

        cv2.imwrite(os.path.join(f"{self.output_dir}", "Marr_Hildreth.jpg"), marr_hildreth)
        self.show_results(os.path.join(f"{self.output_dir}", 'Marr_Hildreth.jpg'))
    
    def apply_prewitt(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        image = cv2.imread(self.input_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Creating the kernels
        Prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        Prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        # Applying the kernels to the grayscale image using convolution
        prewitt_x = cv2.filter2D(gray, -1, Prewitt_kernel_x)
        prewitt_y = cv2.filter2D(gray, -1, Prewitt_kernel_y)

        # Combining the results to get the final image
        prewitt = np.sqrt(np.square(prewitt_x) + np.square(prewitt_y))
        prewitt = (prewitt * 255.0 / prewitt.max()).astype(np.uint8)

        cv2.imwrite(os.path.join(f"{self.output_dir}", "Prewitt.jpg"), prewitt)
        
        self.show_results(os.path.join(f"{self.output_dir}", "Prewitt.jpg"))

    def apply_sobel(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        image = cv2.imread(self.input_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Creating the kernels
        Sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Applying the kernels to the grayscale image using convolution
        sobel_x = cv2.filter2D(gray, -1, Sobel_kernel_x)
        sobel_y = cv2.filter2D(gray, -1, Sobel_kernel_y)

        # Combining the results to get the final image
        sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
        sobel = (sobel * 255.0 / sobel.max()).astype(np.uint8)

        cv2.imwrite(os.path.join(f"{self.output_dir}", "Sobel.jpg"), sobel)
        self.show_results(os.path.join(f"{self.output_dir}", "Sobel.jpg"))
        
    def apply_prewitt_x(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        image = cv2.imread(self.input_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Creating the kernel
        Prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

        # Applying the kernels to the grayscale image using convolution
        prewitt_x = cv2.filter2D(gray, -1, Prewitt_kernel_x)
        cv2.imwrite(os.path.join(f"{self.output_dir}", 'prewitt_x.jpg'), prewitt_x)
        self.show_results(os.path.join(f"{self.output_dir}", 'prewitt_x.jpg'))
        
    def apply_prewitt_y(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        image = cv2.imread(self.input_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Creating the kernels
        Prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        
        # Applying the kernels to the grayscale image using convolution
        prewitt_y = cv2.filter2D(gray, -1, Prewitt_kernel_y)
        cv2.imwrite(os.path.join(f"{self.output_dir}", 'prewitt_y.jpg'), prewitt_y)
        self.show_results(os.path.join(f"{self.output_dir}", 'prewitt_y.jpg'))
        
    def apply_sobel_x(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        image = cv2.imread(self.input_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Creating the kernels
        Sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        # Applying the kernels to the grayscale image using convolution
        sobel_x = cv2.filter2D(gray, -1, Sobel_kernel_x)
        cv2.imwrite(os.path.join(f"{self.output_dir}", 'sobel_x.jpg'), sobel_x)
        self.show_results(os.path.join(f"{self.output_dir}", 'sobel_x.jpg'))
        
    def apply_sobel_y(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        image = cv2.imread(self.input_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Creating the kernels
        Sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Applying the kernels to the grayscale image using convolution
        sobel_y = cv2.filter2D(gray, -1, Sobel_kernel_y)
        cv2.imwrite(os.path.join(f"{self.output_dir}", 'sobel_y.jpg'), sobel_y)
        self.show_results(os.path.join(f"{self.output_dir}", 'sobel_y.jpg'))

    def apply_kMeans(self):
        if self.input_image_path == '':
            messagebox.showwarning('Warning!', "No Image is selected!")
            return
        image = cv2.imread(self.input_image_path)

        # Change color to RGB (from BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
        pixel_vals = image.reshape((-1, 3))

        # Convert to float type
        pixel_vals = np.float32(pixel_vals)

        # the below line of code defines the criteria for the algorithm to stop running,
        # which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
        # becomes 85%
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

        # then perform k-means clustering with number of clusters defined as 6
        # also random centres are initially choosed for k-means clustering
        k = 6
        retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # convert data into 8-bit values
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]

        # reshape data into the original image dimensions
        segmented_image = segmented_data.reshape((image.shape))
        plt.imsave(os.path.join(f"{self.output_dir}", 'k_means.jpg'), segmented_image)
        self.show_results(os.path.join(f"{self.output_dir}", 'k_means.jpg'))
        plt.imshow(segmented_image)
        plt.show()

if __name__ == '__main__':
    root = Tk()
    obj = App(root)
    root.mainloop()
    
