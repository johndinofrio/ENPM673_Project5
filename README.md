# ENPM673_Project5

Must have Python 3.x installed.

Need OpenCV verision 3.4.2.16 in order for SIFT to work.

```
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
```

## Files
The code base consists of two executable files
- main.py - runs our homemade functions
- builtins.py - similar pipeline using built in functions

## Media Data
It is important to ensure the media is stored in the 
Oxford_dataset/stereo/centre folder. The Oxford_dataset folder
should be at the sme level as the code.

## Running the Code
From the top directory containing the code, run the following commands to run the code.

`python3 main.py` or `python main.py`

`python3 builtins.py` or `python builtins.py`

There are nearly 4000 frames and the image processing is relatively slow. These will run for a long time.
You can quit by selecting the video window and pressing `q` or `esc`.