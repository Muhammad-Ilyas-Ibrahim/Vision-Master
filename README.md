# <center> Vision Master <center>


## Description
This project involves various image processing operations implemented using Python and OpenCV. It provides a graphical user interface (GUI) where users can load an image and apply different operations such as adding noise, removing noise, blurring, edge detection, feature extraction, and more.

## Authors
- [Muhammad Ilyas](https://github.com/Muhammad-Ilyas-Ibrahim)

## Contributors
- [Moavia Hassan](https://github.com/Moavia-Hassan)

## Prerequisites
- Python 3.x
- OpenCV
- tkinter
- numpy
- PIL
- matplotlib
- resizeimage
- opencv-contrib-python

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Muhammad-Ilyas-Ibrahim/Vision-Master.git
   ```
2. Change working directory:
   ```bash
   cd Vision-Master
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the `main.py` file:
   ```bash
   python main.py
   ```

2. The GUI will appear.
3. Click on the "Browse" button to select an image from your local system.
4. Choose an operation from the available buttons.
5. View the output image in the GUI.

## Operations Available
- Add Noise
- Remove Noise
- Blur
- Remove Blur
- SIFT Feature Extraction
- Harris Corner Detection
- Canny Edge Detection
- Histogram of Oriented Gradients (HOG)
- Laplacian Edge Detection
- Marr-Hildreth Edge Detection
- Prewitt Edge Detection
- Sobel Edge Detection
- K-Means Clustering

## File Structure
- `main.py`: Main Python script containing the GUI and image processing functions.
- `imgs/`: Directory containing sample input and output images.
- `buttons/`: Directory containing image files used for buttons in the GUI.
