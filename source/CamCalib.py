import os
import sys
import numpy as np
import cv2 as cv
import glob

from gooey import Gooey
from gooey import GooeyParser

# Draw and display the corners
def drawAndDisplayTheCorners (img, corners, ret):
    cv.drawChessboardCorners(img, (6,9), corners, ret)
    cv.imshow('img',img)
    cv.waitKey(0)

# Write calibration parameters into file
def write(matrix, dist, rvecs, tvecs, mapx, mapy):
    cv_file_json = cv.FileStorage("calibParams.json", cv.FILE_STORAGE_WRITE)
    cv_file_json.write("Camera_matrix", matrix)
    cv_file_json.write("Distortion_coefficient", dist)
    cv_file_json.write("Rotation_vectors", np.array(rvecs))
    cv_file_json.write("Translation_vectors", np.array(tvecs))
    cv_file_json.write("Undistortion", mapx)
    cv_file_json.write("Rectification", mapy)
    cv_file_json.release()

    cv_file_xml = cv.FileStorage("calibParams.xml", cv.FILE_STORAGE_WRITE)
    cv_file_xml.write("Camera_matrix", matrix)
    cv_file_xml.write("Distortion_coefficient", dist)
    cv_file_xml.write("Rotation_vectors", np.array(rvecs))
    cv_file_xml.write("Translation_vectors", np.array(tvecs))
    cv_file_xml.write("Undistortion", mapx)
    cv_file_xml.write("Rectification", mapy)
    cv_file_xml.release()

# Print calibration parameters into command line
def calibPrint (matrix, dist, rvecs, tvecs, mapx, mapy):
        print("Camera matrix:")
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=4)
        print(matrix)

        print("\nDistortion coefficient:")
        print(dist)

        print("\nRotation Vectors:")
        print(np.array(rvecs))

        print("\nTranslation Vectors:")
        print(np.array(tvecs))

        print("\nUndistortion:")
        print(mapx)

        print("\nRectification:")
        print(mapy)

def Calibration(inputDir, photo, outputDir):
    try:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((9*6,3), np.float32)
        objp[:,:2] = np.mgrid[0:6,0:9].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane
        # os.listdir(inputDir)
        os.chdir(inputDir)

        images = glob.glob('*.png')

        for filename in images:
            img = cv.imread(filename)

            height, width, channels = img.shape

            gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
            
            ret = False
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(img, (6,9), None)
            
            # If found, add object points, image points (after refining them)
            if ret == True:
                # add objectpoints            
                objpoints = [objp]
            
                corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

                # add imgpoints
                corners = np.array([[corner for [corner] in corners]])
                imgpoints = corners
            
            drawAndDisplayTheCorners (img, corners, ret)

        cv.destroyAllWindows()

        ret, matrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        img = cv.imread(photo)
        h,  w = img.shape[:2]

        newcameramtx, roi = cv.getOptimalNewCameraMatrix(matrix, dist, (w, h), 1, (w, h))

        mapx, mapy = cv.initUndistortRectifyMap(matrix, dist, None, newcameramtx, (w, h), 5)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

        os.chdir(outputDir)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imwrite('croppedResult.png', dst)

        calibPrint (matrix, dist, rvecs, tvecs, mapx, mapy)

        write(matrix, dist, rvecs, tvecs, mapx, mapy)

    except KeyboardInterrupt:
        print("\nProcess interrupted\n")
        exit(0)

    except :
        print("\nSomething went wrong\n")
        exit(1)


# Main function
@Gooey(
    program_name = 'Camera Calibrator',
    program_description = 'Program for camera calibration',
    default_size=(900, 700),
    clear_before_run = True
)
def main() -> int:
    parser = GooeyParser()
    calibrator = parser.add_argument_group("Camera Calibration")
    calibrator.add_argument("--CalibrationFilesPath",
        widget = 'DirChooser',
        help = 'Choose a directory with calibration photos',
        gooey_options={
            'required' : 'True',
            'type' : 'str',
            'help_color' : '#000000',
        }
    )

    calibrator.add_argument("--PhotoToEdit",
        widget = 'FileChooser',
        help = 'Select photo to edit with calibration',
        gooey_options={
            'wildcard' : 'Comma separated file (*.png)|*.png|',
            'required' : 'True',
            'type' : 'str',
            'help_color' : '#000000'
        }
    )

    calibrator.add_argument("--OutputPath",
        widget = 'DirChooser',
        help = 'Choose a directory to save outputs',
        gooey_options={
            'required' : 'True',
            'type' : 'str',
            'help_color' : '#000000',
        }
    )
    
    args = parser.parse_args()
    Calibration(args.CalibrationFilesPath, args.PhotoToEdit, args.OutputPath)
    return 0

if __name__ == '__main__':
    sys.exit(main())